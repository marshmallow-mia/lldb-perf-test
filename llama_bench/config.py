"""Configuration and search space definitions for llama-bench."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Curated flag metadata
# ---------------------------------------------------------------------------

CURATED_FLAGS: dict[str, dict[str, Any]] = {
    "--n-gpu-layers": {
        "type": int,
        "default": 45,
        "description": "Number of layers to offload to GPU (0 = CPU only).",
        "range": [0, 128],
    },
    "--split-mode": {
        "type": str,
        "default": "none",
        "description": "How to split the model across multiple GPUs.",
        "valid_values": ["none", "layer", "row"],
    },
    "--tensor-split": {
        "type": str,
        "default": "",
        "description": "Fraction of the model to offload to each GPU (comma-separated).",
    },
    "--main-gpu": {
        "type": int,
        "default": 0,
        "description": "Index of the primary GPU.",
        "range": [0, 15],
    },
    "--flash-attn": {
        "type": bool,
        "default": True,
        "description": "Enable Flash Attention (reduces VRAM usage, increases throughput).",
    },
    "--batch-size": {
        "type": int,
        "default": 1536,
        "description": "Logical maximum batch size for prompt processing.",
        "range": [32, 8192],
    },
    "--ubatch-size": {
        "type": int,
        "default": 512,
        "description": "Physical (micro) batch size used internally during prompt eval.",
        "range": [32, 4096],
    },
    "--kv-unified": {
        "type": bool,
        "default": True,
        "description": "Use a single unified KV cache pool across all slots.",
    },
    "--cache-type-k": {
        "type": str,
        "default": "q8_0",
        "description": "Quantisation type for K cache entries.",
        "valid_values": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
    },
    "--cache-type-v": {
        "type": str,
        "default": "q8_0",
        "description": "Quantisation type for V cache entries.",
        "valid_values": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
    },
    "--kv-offload": {
        "type": bool,
        "default": True,
        "description": "Offload KV cache to GPU memory.",
    },
    "--cache-reuse": {
        "type": int,
        "default": 512,
        "description": "Minimum prefix length (tokens) to trigger KV cache reuse.",
        "range": [0, 8192],
    },
    "--cont-batching": {
        "type": bool,
        "default": True,
        "description": "Enable continuous batching (process multiple requests simultaneously).",
    },
    "--threads": {
        "type": int,
        "default": 8,
        "description": "Number of CPU threads for inference.",
        "range": [1, 256],
    },
    "--threads-batch": {
        "type": int,
        "default": 8,
        "description": "Number of CPU threads for batch processing.",
        "range": [1, 256],
    },
    "-np": {
        "type": int,
        "default": 1,
        "description": "Number of parallel request slots.",
        "range": [1, 64],
    },
    "-c": {
        "type": int,
        "default": 49152,
        "description": "Total KV-cache context size (tokens, divided across all slots).",
        "range": [512, 1048576],
    },
}

# Cache quantisation types that are known to be incompatible with flash attention.
# These are sub-8-bit quants that lack the precision required by FA implementations.
_FLASH_ATTN_INCOMPATIBLE_CACHE_TYPES: frozenset[str] = frozenset(
    {"q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"}
)


# ---------------------------------------------------------------------------
# parse_range
# ---------------------------------------------------------------------------

def parse_range(s: str) -> list[int]:
    """Parse a range/list specification into a list of ints.

    Formats accepted:
      - ``"1-4"``      → ``[1, 2, 3, 4]``
      - ``"1,2,3"``    → ``[1, 2, 3]``
      - ``"49152"``    → ``[49152]``

    Raises ValueError for anything else.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty range string")

    # Range syntax: two integers separated by a dash (no commas allowed)
    if "-" in s and "," not in s:
        parts = s.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid range spec: {s!r}")
        try:
            lo, hi = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(f"Non-integer bounds in range spec: {s!r}")
        if lo > hi:
            raise ValueError(f"Lower bound {lo} > upper bound {hi} in range spec: {s!r}")
        return list(range(lo, hi + 1))

    # Comma-separated list
    if "," in s:
        try:
            return [int(x.strip()) for x in s.split(",")]
        except ValueError:
            raise ValueError(f"Non-integer value in comma list: {s!r}")

    # Single integer
    try:
        return [int(s)]
    except ValueError:
        raise ValueError(f"Invalid range spec (not an integer): {s!r}")


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------

@dataclass
class SearchSpace:
    np_values: list[int] = field(default_factory=lambda: [1])
    ctx_values: list[int] = field(default_factory=lambda: [49152])
    n_gpu_layers_values: list[int] = field(default_factory=lambda: [45])
    flash_attn_values: list[bool] = field(default_factory=lambda: [True])
    batch_size_values: list[int] = field(default_factory=lambda: [1536])
    ubatch_size_values: list[int] = field(default_factory=lambda: [512])
    kv_cache_type_k_values: list[str] = field(default_factory=lambda: ["q8_0"])
    kv_cache_type_v_values: list[str] = field(default_factory=lambda: ["q8_0"])
    kv_unified_values: list[bool] = field(default_factory=lambda: [True])
    cache_reuse_values: list[int] = field(default_factory=lambda: [512])
    cont_batching_values: list[bool] = field(default_factory=lambda: [True])
    split_mode_values: list[str] = field(default_factory=lambda: ["none"])
    threads_values: list[int] = field(default_factory=lambda: [8])
    threads_batch_values: list[int] = field(default_factory=lambda: [8])


def default_search_space() -> SearchSpace:
    """Return the v1 defaults tuned for reverse-engineering workloads."""
    return SearchSpace(
        np_values=[1],
        ctx_values=[49152],
        n_gpu_layers_values=[45],
        flash_attn_values=[True],
        batch_size_values=[1536],
        ubatch_size_values=[512],
        kv_cache_type_k_values=["q8_0"],
        kv_cache_type_v_values=["q8_0"],
        kv_unified_values=[True],
        cache_reuse_values=[512],
        cont_batching_values=[True],
        split_mode_values=["none"],
        threads_values=[8],
        threads_batch_values=[8],
    )


# ---------------------------------------------------------------------------
# BenchConfig
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    # Model / server
    model_path: str = ""
    server_binary: str = "./llama-server"
    host: str = "0.0.0.0"
    port: int = 5001
    use_sudo: bool = True
    vk_visible_devices: Optional[str] = None  # None = don't set GGML_VK_VISIBLE_DEVICES

    # Resolved device name(s) to pass as --device to llama-server.
    # e.g. "Vulkan0,Vulkan1" or "none" (CPU). None = don't pass --device.
    device: Optional[str] = None

    # Engine selection
    engine: str = "vulkan"  # "vulkan" | "cpu"

    # Slots / context
    np: int = 1
    ctx: int = 49152

    # GPU offload
    n_gpu_layers: int = 45
    split_mode: str = "none"
    tensor_split: str = ""
    main_gpu: int = 0

    # Attention / batching
    flash_attn: bool = True
    batch_size: int = 1536
    ubatch_size: int = 512

    # KV cache
    kv_unified: bool = True
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    kv_offload: bool = True
    cache_reuse: int = 512

    # Threading / continuity
    cont_batching: bool = True
    threads: int = 8
    threads_batch: int = 8


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_config(cfg: BenchConfig) -> list[str]:
    """Return a list of warning/error strings for the given config."""
    warnings: list[str] = []

    if cfg.ubatch_size > cfg.batch_size:
        warnings.append(
            f"ubatch_size ({cfg.ubatch_size}) > batch_size ({cfg.batch_size}): "
            "ubatch_size should be <= batch_size."
        )

    min_ctx = cfg.np * 4096
    if cfg.ctx < min_ctx:
        warnings.append(
            f"ctx ({cfg.ctx}) < np * 4096 ({min_ctx}): "
            "context size may be too small for the number of parallel slots."
        )

    if cfg.n_gpu_layers < 0:
        warnings.append(f"n_gpu_layers ({cfg.n_gpu_layers}) is negative.")

    if cfg.flash_attn:
        if cfg.cache_type_k in _FLASH_ATTN_INCOMPATIBLE_CACHE_TYPES:
            warnings.append(
                f"flash_attn=True may be incompatible with cache_type_k={cfg.cache_type_k!r}. "
                "Consider using q8_0 or f16."
            )
        if cfg.cache_type_v in _FLASH_ATTN_INCOMPATIBLE_CACHE_TYPES:
            warnings.append(
                f"flash_attn=True may be incompatible with cache_type_v={cfg.cache_type_v!r}. "
                "Consider using q8_0 or f16."
            )

    return warnings


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def generate_configs(space: SearchSpace, base: BenchConfig) -> list[BenchConfig]:
    """Return the cartesian product of all search-space values applied to *base*."""
    import copy

    keys_and_values = [
        ("np", space.np_values),
        ("ctx", space.ctx_values),
        ("n_gpu_layers", space.n_gpu_layers_values),
        ("flash_attn", space.flash_attn_values),
        ("batch_size", space.batch_size_values),
        ("ubatch_size", space.ubatch_size_values),
        ("cache_type_k", space.kv_cache_type_k_values),
        ("cache_type_v", space.kv_cache_type_v_values),
        ("kv_unified", space.kv_unified_values),
        ("cache_reuse", space.cache_reuse_values),
        ("cont_batching", space.cont_batching_values),
        ("split_mode", space.split_mode_values),
        ("threads", space.threads_values),
        ("threads_batch", space.threads_batch_values),
    ]

    keys = [k for k, _ in keys_and_values]
    value_lists = [v for _, v in keys_and_values]

    configs: list[BenchConfig] = []
    for combo in itertools.product(*value_lists):
        cfg = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            setattr(cfg, k, v)
        configs.append(cfg)

    return configs


# ---------------------------------------------------------------------------
# configs_from_args
# ---------------------------------------------------------------------------

def configs_from_args(
    *,
    server: str = "./llama-server",
    model: str,
    host: str = "0.0.0.0",
    port: int = 5001,
    np: int = 1,
    ctx: int = 49152,
    n_gpu_layers: int = 45,
    flash_attn: bool = True,
    batch_size: int = 1536,
    ubatch_size: int = 512,
    cache_type_k: str = "q8_0",
    cache_type_v: str = "q8_0",
    kv_unified: bool = True,
    cache_reuse: int = 512,
    cont_batching: bool = True,
    threads: int = 8,
    threads_batch: int = 8,
    split_mode: str = "none",
    vk_devices: Optional[str] = None,
    use_sudo: bool = True,
    engine: str = "vulkan",
    device: Optional[str] = None,
    **_kwargs: Any,
) -> BenchConfig:
    """Build a :class:`BenchConfig` from CLI keyword arguments."""
    return BenchConfig(
        model_path=model,
        server_binary=server,
        host=host,
        port=port,
        use_sudo=use_sudo,
        vk_visible_devices=vk_devices,
        device=device,
        engine=engine,
        np=np,
        ctx=ctx,
        n_gpu_layers=n_gpu_layers,
        flash_attn=flash_attn,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        kv_unified=kv_unified,
        cache_reuse=cache_reuse,
        cont_batching=cont_batching,
        threads=threads,
        threads_batch=threads_batch,
        split_mode=split_mode,
    )
