# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.ksa import (
    KSAAttentionBackend,
    KSACache,
    KSAForCausalLM,
    KSAKernelMetadata,
    KSAStandardPagePool,
    KSASummaryAttentionBackend,
    KSASummarySpec,
    cached_layout,
    paged_attention,
    prefill_attention,
    prefill_layout,
    retained_layout,
    visibility_mask,
)
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.worker.utils import AttentionGroup


@pytest.mark.parametrize("summary", [False, True])
def test_runner_creates_ksa_metadata_builder(summary, default_vllm_config):
    """Both cache groups must instantiate builders during engine startup."""
    common = dict(num_kv_heads=1, head_size=16, dtype=torch.float32)
    spec = (
        KSASummarySpec(block_size=64, **common)
        if summary
        else SlidingWindowSpec(block_size=8, sliding_window=16, **common)
    )
    group = AttentionGroup(
        backend=KSASummaryAttentionBackend if summary else KSAAttentionBackend,
        layer_names=["ksa.summary" if summary else "ksa.text"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    group.create_metadata_builders(
        default_vllm_config, torch.device("cpu"), kernel_block_size=spec.block_size
    )
    builder = group.get_metadata_builder()
    common_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1]),
        seq_lens=torch.tensor([1]),
        block_table_tensor=torch.tensor([[1]]),
        slot_mapping=torch.tensor([8]),
        positions=torch.tensor([0]),
        num_actual_tokens=1,
    )
    metadata = builder.build(0, common_metadata)
    assert metadata.block_table is common_metadata.block_table_tensor
    assert metadata.query_start_loc is common_metadata.query_start_loc
    assert metadata.num_actual_tokens == 1


@pytest.mark.parametrize("length", [1, 7, 8, 9, 63, 64, 65])
def test_prefill_summary_order(length):
    positions, rows, summary = prefill_layout(length)
    assert positions[rows].tolist() == list(range(length))
    assert positions[summary].tolist() == list(range(7, length, 8))
    assert rows.tolist() == [n + n // 8 for n in range(length)]


def make_pool(device):
    layers, metadata = [], {}
    for kind, blocks in (("text", 16), ("summary", 2)):
        name = f"test.model.layers.0.self_attn.ksa_{kind}"
        layers.append(
            SimpleNamespace(
                layer_name=name,
                kv_cache=torch.full((blocks + 1, 8, 1, 32), torch.nan, device=device),
            )
        )
        metadata[name] = SimpleNamespace(
            block_table=torch.arange(1, blocks + 1, device=device).unsqueeze(0)
        )
    return KSAStandardPagePool(layers, metadata, [1])


@pytest.mark.parametrize("length", [7, 8, 9, 63, 64, 65])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.inference_mode()
def test_compressed_pages_match_reference_attention(length, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pool = make_pool(device)
    request = SimpleNamespace(index=0, num_computed_tokens=0)
    positions, _, summary = cached_layout(0, length, device)
    generator = torch.Generator(device=device).manual_seed(length)
    keys = torch.randn(len(positions), 1, 16, device=device, generator=generator)
    values = torch.randn(keys.shape, device=device, generator=generator)
    pool.commit(request, length, [(keys, values)], {})
    old_pos, old_summary, keep = retained_layout(length, 1, device)
    cache = KSACache(
        text_tokens=length,
        layouts={1: (old_pos, old_summary)},
        page_pool=pool,
        request=request,
    )
    actual_k, actual_v = pool.read(request, 0, cache.layouts[1])
    torch.testing.assert_close(actual_k, keys[keep])
    torch.testing.assert_close(actual_v, values[keep])
    assert torch.isnan(pool.layers[0].kv_cache[0]).all()
    assert torch.isnan(pool.layers[1].kv_cache[0]).all()
    assert request.num_computed_tokens == length
    if device == "cpu":
        return

    pool.prepare_request(request, cache)
    pos, _, flags = cached_layout(length, 1, device)
    q = torch.randn(len(pos), 2, 16, device=device, generator=generator)
    k = torch.randn(len(pos), 1, 16, device=device, generator=generator)
    v = torch.randn(k.shape, device=device, generator=generator)
    state = dict(pos=pos, cache=cache, old_lengths={1: len(old_pos)})
    metadata = KSAKernelMetadata.from_states([1], [state])
    actual = paged_attention(q, k, v, pool.storage, metadata, 0, pos, flags, len(pos))
    mask = visibility_mask(
        pos, flags, 1, torch.cat((old_pos, pos)), torch.cat((old_summary, flags))
    )
    expected = prefill_attention(
        q, torch.cat((actual_k, k)), torch.cat((actual_v, v)), mask, reference=True
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_standard_forward_rejects_unsupported_embeddings():
    # Match the serving config contract instead of silently ignoring embeddings.
    with pytest.raises(ValueError, match="prompt embeddings"):
        KSAForCausalLM._forward_standard(
            None, None, torch.arange(1), None, torch.zeros(1, 32), None
        )
