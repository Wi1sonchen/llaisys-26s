#include "linear_nvidia.cuh"

#include "../../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

void checkCuda(
    cudaError_t status,
    const char *what
) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(what) +
            ": " +
            cudaGetErrorString(status)
        );
    }
}

void checkCublas(
    cublasStatus_t status,
    const char *what
) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(what) +
            ": cuBLAS status = " +
            std::to_string(
                static_cast<int>(status)
            )
        );
    }
}


// ------------------------------------------------------------
// cuBLAS handle
// ------------------------------------------------------------
//
// 当前作业先按每个线程 lazy 创建一个 handle。
// Context 本身也是 thread-local 的，因此这个模式足够先把
// NVIDIA 单卡推理打通。
//
cublasHandle_t getCublasHandle() {
    struct HandleHolder {
        cublasHandle_t handle = nullptr;

        HandleHolder() {
            checkCublas(
                cublasCreate(&handle),
                "cublasCreate"
            );
        }

        ~HandleHolder() {
            if (handle != nullptr) {
                cublasDestroy(handle);
            }
        }
    };

    static thread_local HandleHolder holder;

    return holder.handle;
}


// ------------------------------------------------------------
// CUDA datatype conversion helpers
// ------------------------------------------------------------

template <typename T>
struct CudaCast;

template <>
struct CudaCast<float> {
    __device__ static float toFloat(float value) {
        return value;
    }

    __device__ static float fromFloat(float value) {
        return value;
    }
};

template <>
struct CudaCast<__half> {
    __device__ static float toFloat(__half value) {
        return __half2float(value);
    }

    __device__ static __half fromFloat(float value) {
        return __float2half_rn(value);
    }
};

template <>
struct CudaCast<__nv_bfloat16> {
    __device__ static float toFloat(
        __nv_bfloat16 value
    ) {
        return __bfloat162float(value);
    }

    __device__ static __nv_bfloat16 fromFloat(
        float value
    ) {
        return __float2bfloat16_rn(value);
    }
};


// ------------------------------------------------------------
// Bias kernel
// ------------------------------------------------------------
//
// GEMM:
// out[M, N] = in[M, K] * weight[N, K]^T
//
// 然后：
// out[i, j] += bias[j]
//
template <typename T>
__global__ void addBiasKernel(
    T *out,
    const T *bias,
    size_t M,
    size_t N
) {
    const size_t idx =
        static_cast<size_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;

    const size_t numel = M * N;

    if (idx >= numel) {
        return;
    }

    const size_t j = idx % N;

    const float out_val =
        CudaCast<T>::toFloat(out[idx]);

    const float bias_val =
        CudaCast<T>::toFloat(bias[j]);

    out[idx] =
        CudaCast<T>::fromFloat(
            out_val + bias_val
        );
}


template <typename T>
void launchBias(
    T *out,
    const T *bias,
    size_t M,
    size_t N
) {
    if (bias == nullptr) {
        return;
    }

    const size_t numel = M * N;

    if (numel == 0) {
        return;
    }

    constexpr int BLOCK_SIZE = 256;

    const int grid_size =
        static_cast<int>(
            (numel + BLOCK_SIZE - 1)
            / BLOCK_SIZE
        );

    addBiasKernel<T><<<grid_size, BLOCK_SIZE>>>(
        out,
        bias,
        M,
        N
    );

    checkCuda(
        cudaGetLastError(),
        "addBiasKernel"
    );
}


// ------------------------------------------------------------
// GEMM
// ------------------------------------------------------------
//
// LLAISYS row-major:
//
// in     [M, K]
// weight [N, K]
// out    [M, N]
//
// wanted:
//
// out = in * weight^T
//
// cuBLAS column-major trick:
//
// out(row-major MxN)
//    memory ==
// out^T(column-major NxM)
//
// weight(row-major NxK)
//    memory ==
// weight^T(column-major KxN)
//
// in(row-major MxK)
//    memory ==
// in^T(column-major KxM)
//
// Therefore:
//
// C[N,M] = op(A)[N,K] * B[K,M]
//
// A memory = weight^T[K,N]
// op(A) = transpose → weight[N,K]
//
// B memory = in^T[K,M]
//
template <typename T>
void launchLinear(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    cudaDataType_t data_type,
    size_t M,
    size_t N,
    size_t K
) {
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    cublasHandle_t handle =
        getCublasHandle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    checkCublas(
        cublasGemmEx(
            handle,

            // weight memory viewed as column-major [K, N]
            // transpose it to obtain [N, K]
            CUBLAS_OP_T,

            // in memory viewed directly as column-major [K, M]
            CUBLAS_OP_N,

            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),

            &alpha,

            weight,
            data_type,
            static_cast<int>(K),

            in,
            data_type,
            static_cast<int>(K),

            &beta,

            out,
            data_type,
            static_cast<int>(N),

            // F16/BF16 also accumulate using FP32

#ifdef ENABLE_ILUVATAR_API
            CUDA_R_32F,
#else
            CUBLAS_COMPUTE_32F,
#endif
            CUBLAS_GEMM_DEFAULT
        ),
        "cublasGemmEx"
    );

    // bias is optional
    if (bias != nullptr) {
        launchBias<T>(
            out,
            bias,
            M,
            N
        );
    }
}

} // namespace


namespace llaisys::ops::nvidia {

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t M,
    size_t N,
    size_t K
) {
    switch (type) {

    case LLAISYS_DTYPE_F32:
        return launchLinear<float>(
            reinterpret_cast<float *>(out),

            reinterpret_cast<const float *>(
                in
            ),

            reinterpret_cast<const float *>(
                weight
            ),

            bias == nullptr
                ? nullptr
                : reinterpret_cast<const float *>(
                      bias
                  ),

            CUDA_R_32F,

            M,
            N,
            K
        );


    case LLAISYS_DTYPE_F16:
        return launchLinear<__half>(
            reinterpret_cast<__half *>(out),

            reinterpret_cast<const __half *>(
                in
            ),

            reinterpret_cast<const __half *>(
                weight
            ),

            bias == nullptr
                ? nullptr
                : reinterpret_cast<
                      const __half *
                  >(bias),

            CUDA_R_16F,

            M,
            N,
            K
        );


    case LLAISYS_DTYPE_BF16:
        return launchLinear<__nv_bfloat16>(
            reinterpret_cast<
                __nv_bfloat16 *
            >(out),

            reinterpret_cast<
                const __nv_bfloat16 *
            >(in),

            reinterpret_cast<
                const __nv_bfloat16 *
            >(weight),

            bias == nullptr
                ? nullptr
                : reinterpret_cast<
                      const __nv_bfloat16 *
                  >(bias),

            CUDA_R_16BF,

            M,
            N,
            K
        );


    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia