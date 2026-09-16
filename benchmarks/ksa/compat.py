# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit compatibility repairs; this is not the unmodified official profile."""


def repair_block_mask(original, *args, **kwargs):
    """Keep partial/full tiles disjoint and retain padded BlockMask index tables."""
    import torch
    from torch.nn.attention.flex_attention import BlockMask

    m = original(*args, **kwargs)
    nk = (m.seq_lengths[1] + 127) // 128

    def dense(count, idx):
        valid = torch.arange(idx.shape[-1], device=idx.device) < count[..., None]
        out = torch.zeros((*idx.shape[:-1], nk), dtype=torch.int32, device=idx.device)
        out.scatter_add_(-1, idx.long(), valid.int())
        return out.bool()

    any_ = dense(m.kv_num_blocks, m.kv_indices)
    full = dense(m.full_kv_num_blocks, m.full_kv_indices)

    def compact(x):
        return x.sum(-1).int(), x.int().argsort(
            dim=-1, descending=True, stable=True
        ).int()

    return BlockMask.from_kv_blocks(
        *compact(any_ & ~full),
        *compact(full),
        BLOCK_SIZE=m.BLOCK_SIZE,
        mask_mod=m.mask_mod,
        seq_lengths=m.seq_lengths,
    )


def install():
    import hashlib
    from pathlib import Path

    import summary_attn
    from summary_attn import interface
    from transformers.utils import generic

    source_hash = hashlib.sha256(Path(interface.__file__).read_bytes()).hexdigest()
    if (
        source_hash
        != "e131def6596be5fe6888f1fca4728c8a220e3a3745c6dc9ca72d6b56ad899b10"
    ):
        raise RuntimeError("Compatibility repairs require the pinned official wheel")
    if getattr(interface, "_ksa_repaired", False):
        return
    original_decorator = generic.check_model_inputs

    def decorator(func=None):
        return original_decorator(func) if func is not None else original_decorator

    generic.check_model_inputs = decorator
    original_mask = interface._create_block_mask_no_materialize
    interface._create_block_mask_no_materialize = lambda *a, **kw: repair_block_mask(
        original_mask, *a, **kw
    )
    original_attention = interface.summary_attn_func

    def attention(*args, **kwargs):
        positions = kwargs.get("summary_pos")
        if positions is not None:
            kwargs["summary_pos"] = positions.reshape(-1)
        return original_attention(*args, **kwargs)

    interface.summary_attn_func = attention
    summary_attn.summary_attn_func = attention
    interface._ksa_repaired = True


def install_fp32_calibration():
    """Use a bounded dense FP32 oracle only for the three calibration cases."""
    import torch
    from summary_attn import interface

    def reference(q, k, v, block_mask, enable_gqa=False):
        if q.shape[-2] > 4700 or q.dtype != torch.float32:
            raise ValueError("Dense calibration is limited to short FP32 sequences")
        if enable_gqa:
            groups = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        query = torch.arange(q.shape[-2], device=q.device)[:, None]
        key = torch.arange(k.shape[-2], device=q.device)[None, :]
        mask = block_mask.mask_mod(0, 0, query, key)
        scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5)
        return scores.masked_fill_(~mask, float("-inf")).softmax(-1) @ v

    interface._compiled_flex_attention = reference
