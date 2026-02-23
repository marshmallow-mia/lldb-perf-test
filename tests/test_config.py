"""Tests for llama_bench.config."""
import pytest

from llama_bench.config import (
    BenchConfig,
    SearchSpace,
    default_search_space,
    generate_configs,
    parse_range,
    validate_config,
)


# ---------------------------------------------------------------------------
# parse_range
# ---------------------------------------------------------------------------

def test_parse_range_dash():
    assert parse_range("1-2") == [1, 2]


def test_parse_range_dash_multi():
    assert parse_range("1-4") == [1, 2, 3, 4]


def test_parse_range_comma():
    assert parse_range("1,2,3") == [1, 2, 3]


def test_parse_range_single():
    assert parse_range("49152") == [49152]


def test_parse_range_invalid():
    with pytest.raises(ValueError):
        parse_range("invalid")


def test_parse_range_empty():
    with pytest.raises(ValueError):
        parse_range("")


def test_parse_range_bad_dash():
    with pytest.raises(ValueError):
        parse_range("1-2-3")


def test_parse_range_reversed():
    with pytest.raises(ValueError):
        parse_range("5-2")


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

def _make_valid_cfg(**kwargs) -> BenchConfig:
    defaults = dict(
        model_path="/tmp/model.gguf",
        np=1,
        ctx=49152,
        n_gpu_layers=45,
        ubatch_size=512,
        batch_size=1536,
        flash_attn=True,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
    )
    defaults.update(kwargs)
    return BenchConfig(**defaults)


def test_validate_config_ubatch_gt_batch():
    cfg = _make_valid_cfg(ubatch_size=2000, batch_size=1000)
    warnings = validate_config(cfg)
    assert any("ubatch_size" in w for w in warnings)


def test_validate_config_ctx_too_small_for_np():
    cfg = _make_valid_cfg(np=4, ctx=4096)  # need at least 4*4096=16384
    warnings = validate_config(cfg)
    assert any("context" in w.lower() or "ctx" in w.lower() for w in warnings)


def test_validate_config_negative_ngl():
    cfg = _make_valid_cfg(n_gpu_layers=-1)
    warnings = validate_config(cfg)
    assert any("n_gpu_layers" in w for w in warnings)


def test_validate_config_flash_attn_bad_cache():
    cfg = _make_valid_cfg(flash_attn=True, cache_type_k="q4_0")
    warnings = validate_config(cfg)
    assert any("flash_attn" in w for w in warnings)


def test_validate_config_clean():
    cfg = _make_valid_cfg()
    warnings = validate_config(cfg)
    assert warnings == []


# ---------------------------------------------------------------------------
# generate_configs
# ---------------------------------------------------------------------------

def test_generate_configs_count():
    space = SearchSpace(
        np_values=[1, 2],
        ctx_values=[49152],
        n_gpu_layers_values=[40, 45],
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
    base = _make_valid_cfg()
    configs = generate_configs(space, base)
    # 2 np * 1 ctx * 2 ngl = 4
    assert len(configs) == 4


def test_generate_configs_values_applied():
    space = SearchSpace(
        np_values=[3],
        ctx_values=[8192],
        n_gpu_layers_values=[10],
        flash_attn_values=[False],
        batch_size_values=[512],
        ubatch_size_values=[256],
        kv_cache_type_k_values=["f16"],
        kv_cache_type_v_values=["f16"],
        kv_unified_values=[False],
        cache_reuse_values=[0],
        cont_batching_values=[False],
        split_mode_values=["layer"],
        threads_values=[4],
        threads_batch_values=[4],
    )
    base = _make_valid_cfg()
    configs = generate_configs(space, base)
    assert len(configs) == 1
    c = configs[0]
    assert c.np == 3
    assert c.ctx == 8192
    assert c.n_gpu_layers == 10
    assert c.flash_attn is False
    assert c.cache_type_k == "f16"
    assert c.kv_unified is False
    assert c.split_mode == "layer"


def test_generate_configs_does_not_mutate_base():
    space = default_search_space()
    base = _make_valid_cfg(np=1)
    generate_configs(space, base)
    assert base.np == 1
