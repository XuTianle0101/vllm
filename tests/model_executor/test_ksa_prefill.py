# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests: bounded config, internal rows and causal GQA attention."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.ksa_prefill import (
    MAX_PREFILL_TOKENS,
    expand_prefill_sequence,
    parse_layer_windows,
    prefill_attention,
    prefill_layout,
    validate_ksa_config,
    visibility_mask,
)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "[1].real",
        "[x for x in [1]]",
        "[True]",
        "[-1]",
        "[1.0]",
        "[1] * 1000000",
        "[1] * (2 + 2)",
        "[[1]]",
        "[1] / 2",
        "[1] ** 2",
        "[1]" * 2000,
    ],
)
def test_parse_windows_rejects_unsafe_or_unbounded_expressions(expression):
    with pytest.raises(ValueError):
        parse_layer_windows(expression)


def test_released_config_expression():
    assert parse_layer_windows("([128]*3+[16768]*1)*9") == [128, 128, 128, 16768] * 9
    assert parse_layer_windows("([1])*36") == [1] * 36


@pytest.mark.parametrize("length", [1, 7, 8, 9, 15, 16, 17, 1023, 1024, 1025, 1032])
def test_summary_positions_and_text_mapping_at_boundaries(length):
    pos, rows, summary = prefill_layout(length)
    expanded, mapped = expand_prefill_sequence(torch.arange(length))
    assert torch.equal(rows, mapped)
    assert pos[rows].tolist() == list(range(length))
    assert pos[summary].tolist() == list(range(7, length, 8))
    assert expanded[rows].tolist() == list(range(length))
    assert (expanded[summary] == 151936).all()


@pytest.mark.parametrize("length", [0, MAX_PREFILL_TOKENS + 1])
def test_explicit_mask_rejects_unsupported_length(length):
    with pytest.raises(ValueError, match="supports lengths"):
        prefill_layout(length)


@pytest.mark.parametrize("window", [0, 1, 128, 16768])
def test_all_window_types_match_independent_visibility_predicate(window):
    pos, rows, summary = prefill_layout(1041)
    mask = visibility_mask(pos, summary, window)
    # Check all keys for block/window boundary queries and every summary row.
    queries = sorted(
        set(
            rows[[0, 7, 8, 15, 16, 1023, 1024, 1031, 1032, 1040]].tolist()
            + summary.nonzero().flatten().tolist()
        )
    )
    for q in queries:
        qb = q // 9
        expected = []
        for k in range(len(pos)):
            kb, ks = k // 9, k % 9 == 8
            visible = (
                (kb == qb)
                if q % 9 == 8
                else ((not ks and qb - kb <= window) or (ks and qb - kb > window))
            )
            expected.append(k <= q and visible)
        assert mask[q].tolist() == expected


@pytest.mark.parametrize("reference", [False, True])
def test_summary_sees_self_without_leaking_into_text_or_future(reference):
    pos, rows, summary = prefill_layout(17)
    mask = visibility_mask(pos, summary, 0)
    q = torch.zeros(len(pos), 4, 8)
    k = torch.zeros(len(pos), 2, 8)
    v = torch.zeros_like(k)
    v[8] = 9
    output = prefill_attention(q, k, v, mask, reference)
    torch.testing.assert_close(output[8], torch.ones_like(output[8]))
    assert not output[rows[:8]].any()
    v[18:] = 10000
    changed = prefill_attention(q, k, v, mask, reference)
    torch.testing.assert_close(output[:18], changed[:18], rtol=0, atol=0)


def test_sdpa_matches_fp32_gqa_oracle():
    torch.manual_seed(0)
    pos, _, summary = prefill_layout(17)
    q, k, v = torch.randn(3, len(pos), 4, 8)
    mask = visibility_mask(pos, summary, 1)
    expected = prefill_attention(q, k[:, :2], v[:, :2], mask, True)
    actual = prefill_attention(q, k[:, :2], v[:, :2], mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_ksa_configuration_validates_shared_projection_contract():
    config = SimpleNamespace(
        use_summary_attention=True,
        summary_sliding_chunk_num=[0, 1],
        summary_token_begin=31,
        vocab_size=32,
    )
    assert validate_ksa_config(config, 2) == [0, 1]
    config.mix_coeff = 1
    with pytest.raises(ValueError, match="mix_coeff"):
        validate_ksa_config(config, 2)


@pytest.mark.parametrize("ksa", [False, True])
def test_registry_keeps_plain_qwen3_on_its_original_path(monkeypatch, ksa):
    from vllm.model_executor.models.registry import ModelRegistry

    config = SimpleNamespace(
        hf_config=SimpleNamespace(use_summary_attention=ksa), model_impl="vllm"
    )
    monkeypatch.setattr(
        type(ModelRegistry), "_normalize_arch", lambda self, arch, cfg: arch
    )
    monkeypatch.setattr(
        type(ModelRegistry), "_try_inspect_model_cls", lambda self, arch: arch
    )
    monkeypatch.setattr(
        type(ModelRegistry), "_try_load_model_cls", lambda self, arch: arch
    )
    expected = "KSAForCausalLM" if ksa else "Qwen3ForCausalLM"
    assert ModelRegistry.inspect_model_cls(["Qwen3ForCausalLM"], config) == (
        expected,
        expected,
    )
    assert ModelRegistry.resolve_model_cls(["Qwen3ForCausalLM"], config) == (
        expected,
        expected,
    )


@pytest.fixture
def tiny_model(tmp_path):
    """Real Qwen projections/norms/MLP and loader; CPU tensors, no checkpoint."""
    import json

    from vllm.config import (
        CompilationConfig,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.models.ksa import KSAForCausalLM

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "num_hidden_layers": 4,
                "vocab_size": 64,
                "max_position_embeddings": 2048,
                "use_summary_attention": True,
                "summary_sliding_chunk_num": [0, 1, 128, 16768],
                "summary_token_begin": 63,
                "tie_word_embeddings": True,
            }
        )
    )
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="float32",
            enforce_eager=True,
        ),
        compilation_config=CompilationConfig(mode=0, custom_ops=["none"]),
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="gloo",
            distributed_init_method=f"file://{tmp_path}/store",
        )
        initialize_model_parallel(1, 1)
        try:
            model = KSAForCausalLM(vllm_config=config)
            with torch.no_grad():
                torch.manual_seed(42)
                for parameter in model.parameters():
                    parameter.normal_(0, 0.1)
            yield model
        finally:
            cleanup_dist_env_and_memory()


def unpacked_weights(model):
    for name, parameter in model.named_parameters():
        if ".qkv_proj." in name:
            for shard, tensor in zip(
                ("q_proj", "k_proj", "v_proj"), parameter.split([32, 16, 16])
            ):
                yield name.replace("qkv_proj", shard), tensor.detach().clone()
        elif ".gate_up_proj." in name:
            for shard, tensor in zip(("gate_proj", "up_proj"), parameter.chunk(2)):
                yield name.replace("gate_up_proj", shard), tensor.detach().clone()
        else:
            yield name, parameter.detach().clone()


def test_model_initialization_loader_and_text_only_forward(tiny_model):
    model = tiny_model
    weights = list(unpacked_weights(model))
    assert model.load_weights(iter(weights))
    ids = torch.arange(17)
    with torch.inference_mode():
        expected = model(ids, torch.arange(17))
        embedded = model(
            None, torch.arange(17), inputs_embeds=model.embed_input_ids(ids)
        )
        torch.testing.assert_close(embedded, expected)
        assert expected.shape == (17, 32)
        assert model.compute_logits(expected).shape == (17, 63)
        assert model.text_row_indices.tolist() == list(range(8)) + list(
            range(9, 17)
        ) + [18]
        changed = ids.clone()
        changed[9:] += 17
        torch.testing.assert_close(model(changed, torch.arange(17))[:9], expected[:9])
        model.reference_attention = True
        torch.testing.assert_close(
            model(ids, torch.arange(17)), expected, atol=1e-6, rtol=1e-5
        )
    with pytest.raises(ValueError, match="missing.*q_proj"):
        model.load_weights(
            (n, t) for n, t in weights if n != "model.layers.0.self_attn.q_proj.weight"
        )
    with pytest.raises(ValueError, match="unexpected.*q_proj_summary"):
        model.load_weights(
            iter(
                weights
                + [("model.layers.0.self_attn.q_proj_summary.weight", weights[0][1])]
            )
        )
    with pytest.raises(ValueError, match="duplicate"):
        model.load_weights(iter(weights + weights[:1]))
    with pytest.raises(ValueError, match="0..N-1"):
        model(ids, torch.arange(1, 18))
    with pytest.raises(ValueError, match="reserved"):
        model(torch.tensor([63]), torch.arange(1))


@pytest.mark.parametrize("length", [7, 8, 9, 1023])
def test_cached_decode_matches_full_prefill_across_blocks(tiny_model, length):
    """Catch summary timing, stale KV and text/internal-position confusion."""
    from vllm.model_executor.models.ksa_decode import KSACache

    model = tiny_model
    ids = torch.arange(length + 25) % 63
    cache = KSACache()
    with torch.inference_mode():
        expected = model(ids, torch.arange(len(ids)))
        prompt = model(ids[:length], torch.arange(length), cache=cache)
        torch.testing.assert_close(prompt, expected[:length], atol=2e-6, rtol=1e-5)
        for position in range(length, len(ids)):
            actual = model(
                ids[position : position + 1], torch.tensor([position]), cache=cache
            )
            torch.testing.assert_close(
                actual, expected[position : position + 1], atol=2e-6, rtol=1e-5
            )
            assert cache.text_tokens == position + 1
            assert cache.internal_rows == position + 1 + (position + 1) // 8
            from vllm.model_executor.models.ksa_decode import retained_layout

            assert all(
                k.shape[0] == len(retained_layout(cache.text_tokens, window)[0])
                for (k, _), window in zip(cache.layers, model.windows)
            )
            assert actual.shape == (1, 32)
        old_count = cache.text_tokens
        with pytest.raises(ValueError, match="contiguous"):
            model(ids[:1], torch.tensor([old_count - 1]), cache=cache)
        with pytest.raises(ValueError, match="exactly one"):
            model(ids[:2], torch.arange(old_count, old_count + 2), cache=cache)
        assert cache.text_tokens == old_count
        cache.clear()
        torch.testing.assert_close(
            model(ids[:8], torch.arange(8), cache=cache), expected[:8]
        )


def test_retained_kv_growth_and_release():
    """Window text is bounded while summary storage grows one row per block."""
    from vllm.model_executor.models.ksa_decode import KSACache, retained_layout

    for length in (1023, 1024, 1025, 1032, 4096, 8192):
        for window in (0, 1, 128, 16768):
            positions, summary, _ = retained_layout(length, window)
            expected_text = min(length, (window + 1) * 8 + length % 8)
            assert len(positions) <= length // 8 + expected_text
            assert int(summary.sum()) == length // 8
            assert positions[~summary].unique().numel() == (~summary).sum()
    cache = KSACache(text_tokens=8192, layers=[(torch.ones(1), torch.ones(1))])
    cache.clear()
    assert (
        cache.text_tokens == 0
        and cache.layers == []
        and cache.layouts == {}
        and cache.owner is None
    )


def test_cached_decode_accepts_noncontiguous_kv_storage(tiny_model):
    """KV reads must honor tensor strides after a cache page is remapped."""
    from vllm.model_executor.models.ksa_decode import KSACache

    ids = torch.arange(17) % 63
    cache = KSACache()
    with torch.inference_mode():
        expected = tiny_model(ids, torch.arange(17))[-1:]
        tiny_model(ids[:16], torch.arange(16), cache=cache)
        cache.layers = [
            (torch.stack((k, k), 1)[:, 0], torch.stack((v, v), 1)[:, 0])
            for k, v in cache.layers
        ]
        assert all(not k.is_contiguous() for k, _ in cache.layers)
        actual = tiny_model(ids[16:], torch.tensor([16]), cache=cache)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_generation_stops_counts_and_filters_summary_rows(tiny_model, monkeypatch):
    """A block-end prompt must sample the final text row; EOS counts once."""
    from vllm.model_executor.models.ksa_decode import KSAPythonRunner

    runner = KSAPythonRunner(tiny_model)
    ids = torch.arange(8)
    with torch.inference_mode():
        expected = (
            tiny_model.compute_logits(tiny_model(ids, torch.arange(8))[-1:])
            .argmax()
            .item()
        )
    result = runner.generate(ids, max_tokens=25, ignore_eos=True)
    assert result.token_ids[0] == expected
    assert result.prompt_tokens == 8 and result.completion_tokens == 25
    assert result.finish_reason == "length" and max(result.token_ids) < 63
    assert runner.generate(ids, max_tokens=25, ignore_eos=True) == result
    stopped = runner.generate(ids, max_tokens=25, eos_token_ids=[expected])
    assert stopped.token_ids == [expected] and stopped.finish_reason == "stop"
    assert stopped.completion_tokens == 1
    assert runner.generate(ids, max_tokens=0).token_ids == []
    with pytest.raises(ValueError, match="1-D"):
        runner.generate(ids[None], max_tokens=1)
    with pytest.raises(ValueError, match="exceeds"):
        runner.generate(ids, max_tokens=8192)
    with pytest.raises(ValueError, match="EOS"):
        runner.generate(ids, max_tokens=1, eos_token_ids=[63])
    with pytest.raises(TypeError):
        runner.generate(ids, max_tokens=1, temperature=1)

    def fail(*args, **kwargs):
        raise RuntimeError("injected layer failure")

    from vllm.model_executor.models.ksa_decode import KSACache

    cache = KSACache()
    with torch.inference_mode():
        tiny_model(ids, torch.arange(8), cache=cache)
        old_layers = cache.layers
        old_layouts = cache.layouts
        monkeypatch.setattr(tiny_model.model.layers[-1].mlp, "forward", fail)
        with pytest.raises(RuntimeError, match="injected"):
            tiny_model(ids[:1], torch.tensor([8]), cache=cache)
        assert cache.layers is old_layers and cache.layouts is old_layouts
        assert cache.text_tokens == 8
