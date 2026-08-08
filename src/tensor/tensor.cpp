#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // // 从最后一维开始，期望步长为 1
    // for (size_t i = _meta.shape.size(); i-- > 0;) {
    //     // 计算当前维度的期望 stride
    //     size_t expected_stride = 1;
    //     for (size_t j = i + 1; j < _meta.shape.size(); ++j) {
    //         expected_stride *= _meta.shape[j];
    //     }

    //     // 比较实际 stride 和期望 stride
    //     if (_meta.strides[i] != expected_stride) {
    //         return false;
    //     }
    // }
    ptrdiff_t expected_stride = 1;

    for (size_t i = _meta.shape.size(); i-- > 0;) {
        if (_meta.strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= _meta.shape[i];
    }

    // 全部符合后返回 true
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // 1. 检查 order 长度
    if (order.size() != this->shape().size()) {
        throw std::runtime_error("Permute order length does not match tensor dimensions.");
    }

    // 2. 检查 order 是合法排列
    std::vector<bool> visited(order.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] >= order.size() || visited[order[i]]) {
            throw std::runtime_error("Invalid permute order.");
        }
        visited[order[i]] = true;
    }

    // 3. 创建 new_shape / new_strides
    std::vector<size_t> new_shape(order.size());
    std::vector<ptrdiff_t> new_strides(order.size());

    // 4. 按 order 重排 shape 和 strides
    for (size_t i = 0; i < order.size(); ++i) {
        new_shape[i] = this->shape()[order[i]];
        new_strides[i] = this->strides()[order[i]];
    }

    // 5. 构造新 Tensor，共享 storage，保留 offset
    return std::shared_ptr<Tensor>(new Tensor(
        TensorMeta{
            this->dtype(),
            new_shape,
            new_strides
        },
        this->_storage,
        this->_offset
    ));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 1. 检查新旧 shape 的元素总数是否一致
    size_t new_numel = std::accumulate(
        shape.begin(),
        shape.end(),
        size_t(1),
        std::multiplies<size_t>());

    if (new_numel != this->numel()) {
        throw std::runtime_error(
            "View shape does not match the number of elements.");
    }

    const auto &old_shape = this->shape();
    const auto &old_strides = this->strides();

    std::vector<ptrdiff_t> new_strides(shape.size());

    // 标量情况
    if (old_shape.empty()) {
        ptrdiff_t stride = 1;

        for (size_t i = shape.size(); i-- > 0;) {
            new_strides[i] = stride;
            stride *= static_cast<ptrdiff_t>(shape[i]);
        }

        return std::shared_ptr<Tensor>(
            new Tensor(
                TensorMeta{
                    this->dtype(),
                    shape,
                    new_strides
                },
                this->_storage,
                this->_offset
            )
        );
    }

    // 空张量：没有实际元素，可以按照连续布局生成新 stride
    if (this->numel() == 0) {
        ptrdiff_t stride = 1;

        for (size_t i = shape.size(); i-- > 0;) {
            new_strides[i] = stride;
            stride *= static_cast<ptrdiff_t>(shape[i]);
        }

        return std::shared_ptr<Tensor>(
            new Tensor(
                TensorMeta{
                    this->dtype(),
                    shape,
                    new_strides
                },
                this->_storage,
                this->_offset
            )
        );
    }

    /*
     * 从右向左处理。
     *
     * tensor_numel:
     *     当前原 Tensor 连续块包含多少元素
     *
     * chunk_base_stride:
     *     当前连续块最里面那个维度的 stride
     *
     * view_numel:
     *     新 shape 当前已经拿多少维度来匹配这个连续块
     */

    ptrdiff_t view_d =
        static_cast<ptrdiff_t>(shape.size()) - 1;

    ptrdiff_t chunk_base_stride =
        old_strides.back();

    size_t tensor_numel = 1;
    size_t view_numel = 1;

    for (ptrdiff_t tensor_d =
             static_cast<ptrdiff_t>(old_shape.size()) - 1;
         tensor_d >= 0;
         --tensor_d) {

        size_t td = static_cast<size_t>(tensor_d);

        // 当前原维度加入当前连续块
        tensor_numel *= old_shape[td];

        /*
         * 判断当前位置是不是一个连续块的左边界。
         *
         * 如果前一个维度不能和当前块连续连接，
         * 当前块就在这里结束。
         *
         * shape == 1 的维度不真正产生地址移动，
         * 因此可以忽略它的 stride。
         */
        bool chunk_end =
            tensor_d == 0 ||
            (
                old_shape[td - 1] != 1 &&
                old_strides[td - 1] !=
                    static_cast<ptrdiff_t>(tensor_numel)
                        * chunk_base_stride
            );

        if (chunk_end) {

            /*
             * 用新 shape 从右向左匹配当前原连续块。
             */
            while (
                view_d >= 0 &&
                (
                    view_numel < tensor_numel ||
                    shape[static_cast<size_t>(view_d)] == 1
                )
            ) {
                size_t vd = static_cast<size_t>(view_d);

                // 当前新维度在这个连续块中的 stride
                new_strides[vd] =
                    static_cast<ptrdiff_t>(view_numel)
                    * chunk_base_stride;

                // 当前新维度加入这个块
                view_numel *= shape[vd];

                --view_d;
            }

            /*
             * 如果新 shape 无法恰好组成当前连续块，
             * 说明它试图跨过一个不连续边界。
             */
            if (view_numel != tensor_numel) {
                throw std::runtime_error(
                    "View shape is not compatible with "
                    "the original tensor layout.");
            }

            /*
             * 当前块处理完毕。
             * 如果左边还有原维度，开始处理下一个块。
             */
            if (tensor_d > 0) {
                chunk_base_stride =
                    old_strides[td - 1];

                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }

    /*
     * 原来的所有连续块都处理完以后，
     * 新 shape 也必须刚好全部消费完。
     */
    if (view_d != -1) {
        throw std::runtime_error(
            "View shape is not compatible with "
            "the original tensor layout.");
    }

    // 共享 storage，不复制数据，并保留原 offset
    return std::shared_ptr<Tensor>(
        new Tensor(
            TensorMeta{
                this->dtype(),
                shape,
                new_strides
            },
            this->_storage,
            this->_offset
        )
    );
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 1. 检查 dim、start、end 是否合法
    if (dim >= this->shape().size()) {
        throw std::runtime_error("Dimension out of bounds.");
    }
    if (start >= end || end > this->shape()[dim]) {
        throw std::runtime_error("Invalid slice indices.");
    }

    // 2. 复制原 shape 得到 new_shape
    auto new_shape = this->shape();
    new_shape[dim] = end - start;


    // 3. strides 基本保持原样
    auto new_strides = this->strides();

    // 4. 计算 new_offset
    //    old_offset + start * stride[dim] * elementSize()
    auto new_offset = this->_offset + start * new_strides[dim] * this->elementSize();

    // 5. 构造新的 Tensor
    //    新 meta
    //    共享 _storage
    //    使用 new_offset
    return std::shared_ptr<Tensor>(new Tensor(
        TensorMeta{
            this->dtype(),
            new_shape,
            new_strides
        },
        this->_storage,
        new_offset
    ));
}

void Tensor::load(const void *src_) {
    // 切换到 Tensor 所在的设备
    core::context().setDevice(this->deviceType(), this->deviceId());

    // 计算字节数
    size_t bytes = this->numel() * this->elementSize();

    // 根据 deviceType() 选择 H2H 或 H2D
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            bytes,
            LLAISYS_MEMCPY_H2H);
    } else {
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src_,
            bytes,
            LLAISYS_MEMCPY_H2D);
    }

        // 调用 memcpy_sync
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
