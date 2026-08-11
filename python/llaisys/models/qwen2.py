from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType, LlaisysQwen2Meta

from ctypes import addressof, c_char, c_int, c_int64, c_size_t, c_void_p
from pathlib import Path
import json
import mmap
import struct


_DTYPE_MAP = {
    "float32": DataType.F32,
    "float": DataType.F32,
    "fp32": DataType.F32,
    "float16": DataType.F16,
    "half": DataType.F16,
    "fp16": DataType.F16,
    "bfloat16": DataType.BF16,
    "bf16": DataType.BF16,
}


def _safetensor_entries(path: Path):
    """Yield (name, pointer, nbytes) while the file mmap stays alive."""
    file_obj = path.open("rb")
    mm = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_COPY)
    try:
        if len(mm) < 8:
            raise ValueError(f"Invalid safetensors file: {path}")

        header_len = struct.unpack_from("<Q", mm, 0)[0]
        data_base = 8 + header_len
        if data_base > len(mm):
            raise ValueError(f"Invalid safetensors header length in: {path}")

        header = json.loads(bytes(mm[8:data_base]).decode("utf-8"))
        for name, info in header.items():
            if name == "__metadata__":
                continue

            start, end = info["data_offsets"]
            start = int(start)
            end = int(end)
            if start < 0 or end < start or data_base + end > len(mm):
                raise ValueError(f"Invalid tensor offsets for {name} in {path}")

            nbytes = end - start
            # ACCESS_COPY gives ctypes a writable buffer view without changing
            # the checkpoint file. The backend copies the bytes synchronously.
            raw = (c_char * nbytes).from_buffer(mm, data_base + start)
            try:
                yield name, c_void_p(addressof(raw)), nbytes
            finally:
                del raw
    finally:
        mm.close()
        file_obj.close()


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self._model = None
        self._device = device
        self._device_id = 0

        model_path = Path(model_path)
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Qwen2 config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        dtype_name = str(config.get("torch_dtype", config.get("dtype", "bfloat16"))).lower()
        if dtype_name not in _DTYPE_MAP:
            raise ValueError(f"Unsupported Qwen2 checkpoint dtype: {dtype_name}")

        hidden_size = int(config["hidden_size"])
        num_heads = int(config["num_attention_heads"])
        head_dim = int(config.get("head_dim", hidden_size // num_heads))

        eos = config.get("eos_token_id", -1)
        if isinstance(eos, (list, tuple)):
            eos = eos[0] if eos else -1
        self._end_token = int(eos)

        meta = LlaisysQwen2Meta(
            int(_DTYPE_MAP[dtype_name]),
            int(config["num_hidden_layers"]),
            hidden_size,
            num_heads,
            int(config.get("num_key_value_heads", num_heads)),
            head_dim,
            int(config["intermediate_size"]),
            int(config["max_position_embeddings"]),
            int(config["vocab_size"]),
            float(config.get("rms_norm_eps", 1e-6)),
            float(config.get("rope_theta", 10000.0)),
            self._end_token,
        )

        device_ids = (c_int * 1)(self._device_id)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            meta,
            int(device),
            device_ids,
            1,
        )
        if not self._model:
            raise RuntimeError("Failed to create LLAISYS Qwen2 model")

        recognized = 0
        try:
            files = sorted(model_path.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"No .safetensors files found in {model_path}")

            for file in files:
                for name, data_ptr, nbytes in _safetensor_entries(file):
                    loaded = LIB_LLAISYS.llaisysQwen2ModelLoadWeight(
                        self._model,
                        name.encode("utf-8"),
                        data_ptr,
                        c_size_t(nbytes),
                    )
                    recognized += int(bool(loaded))

            if not LIB_LLAISYS.llaisysQwen2ModelAllWeightsLoaded(self._model):
                raise RuntimeError(
                    "Qwen2 checkpoint loading is incomplete. "
                    f"Recognized {recognized} tensors; one or more required weights are missing."
                )
        except Exception:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
            raise

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # Assignment #3 requires argmax generation. These sampling arguments are
        # accepted for compatibility with test_infer.py but are intentionally
        # not implemented in Python.
        del top_k, top_p, temperature

        tokens = [int(x) for x in inputs]
        if not tokens:
            return []

        if max_new_tokens is None:
            max_new_tokens = 128
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        LIB_LLAISYS.llaisysQwen2ModelResetCache(self._model)

        output = list(tokens)
        pending = tokens

        for _ in range(max_new_tokens):
            token_buf = (c_int64 * len(pending))(*pending)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model,
                    token_buf,
                    c_size_t(len(pending)),
                )
            )
            output.append(next_token)

            if next_token == self._end_token:
                break

            # KV-cache stays inside C++; after prefill only the newly generated
            # token is passed through the decoder.
            pending = [next_token]

        return output
