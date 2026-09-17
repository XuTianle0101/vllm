# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed KSA attention over physical history slots and uncommitted new KV."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _attention(
    Q,
    K,
    V,
    Pool,
    SummaryPool,
    Slots,
    OldPos,
    OldSummary,
    Lengths,
    Bases,
    Offsets,
    Pos,
    Summary,
    Out,
    Maxima,
    Sums,
    TOTAL: tl.constexpr,
    SPLITS: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    tile, head = tl.program_id(0), tl.program_id(1)
    request = tl.program_id(2) // SPLITS
    split = tl.program_id(2) % SPLITS
    begin = tl.load(Offsets + request)
    end = tl.load(Offsets + request + 1)
    if tile * BM >= (end - begin) * G:
        return
    old = tl.load(Lengths + request)
    base = tl.load(Bases + request)
    m = tile * BM + tl.arange(0, BM)
    row = begin + m // G
    group = m % G
    d = tl.arange(0, BD)
    qp = tl.load(Pos + row, row < end, 0)
    qs = tl.load(Summary + row, row < end, 0)
    q = tl.load(
        Q + row[:, None] * QS + (head * G + group[:, None]) * D + d[None, :],
        (row[:, None] < end) & (d[None, :] < D),
        0,
    )
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, BD), tl.float32)
    total_keys = old + tl.minimum(tl.cdiv((tile + 1) * BM, G), end - begin)
    blocks = tl.cdiv(total_keys, BN * SPLITS)
    for block in range(
        split * blocks, tl.minimum((split + 1) * blocks, tl.cdiv(total_keys, BN))
    ):
        n = block * BN + tl.arange(0, BN)
        history = n < old
        valid = n < total_keys
        slot = tl.load(Slots + base + n, history, 0)
        kp_old = tl.load(OldPos + base + n, history, 0)
        ks_old = tl.load(OldSummary + base + n, history, 0)
        new_row = begin + n - old
        kp = tl.where(history, kp_old, tl.load(Pos + new_row, ~history & valid, 0))
        ks = tl.where(history, ks_old, tl.load(Summary + new_row, ~history & valid, 0))
        history_pool = tl.where(ks_old, SummaryPool, Pool)
        pk = tl.load(
            history_pool[None, :]
            + slot[None, :] * (2 * HK * D)
            + head * D
            + d[:, None],
            history[None, :] & (d[:, None] < D),
            0,
        )
        nk = tl.load(
            K + new_row[None, :] * KS + head * D + d[:, None],
            (~history & valid)[None, :] & (d[:, None] < D),
            0,
        )
        k = tl.where(history[None, :], pk, nk)
        distance = qp[:, None] // 8 - kp[None, :] // 8
        causal = (kp[None, :] < qp[:, None]) | (
            (kp[None, :] == qp[:, None]) & (qs[:, None] | ~ks[None, :])
        )
        visible = tl.where(
            qs[:, None],
            distance == 0,
            (~ks[None, :] & (distance <= WINDOW)) | (ks[None, :] & (distance > WINDOW)),
        )
        score = tl.dot(q, k, input_precision="ieee") * (D**-0.5)
        score = tl.where(
            causal & visible & valid[None, :] & (row[:, None] < end),
            score,
            -float("inf"),
        )
        next_max = tl.maximum(maximum, tl.max(score, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = tl.exp(maximum - safe_max)
        probability = tl.exp(score - safe_max[:, None])
        pv = tl.load(
            history_pool[:, None]
            + slot[:, None] * (2 * HK * D)
            + HK * D
            + head * D
            + d[None, :],
            history[:, None] & (d[None, :] < D),
            0,
        )
        nv = tl.load(
            V + new_row[:, None] * VS + head * D + d[None, :],
            (~history & valid)[:, None] & (d[None, :] < D),
            0,
        )
        value = tl.where(history[:, None], pv, nv)
        accumulator = accumulator * alpha[:, None] + tl.dot(
            probability.to(value.dtype), value, input_precision="ieee"
        )
        denominator = denominator * alpha + tl.sum(probability, 1)
        maximum = next_max
    if SPLITS == 1:
        result = accumulator / denominator[:, None]
    else:
        result = accumulator
        scalar_offset = split * TOTAL * H + row * H + head * G + group
        tl.store(Maxima + scalar_offset, maximum, row < end)
        tl.store(Sums + scalar_offset, denominator, row < end)
    tl.store(
        Out
        + split * TOTAL * H * D
        + row[:, None] * (H * D)
        + (head * G + group[:, None]) * D
        + d[None, :],
        result,
        (row[:, None] < end) & (d[None, :] < D),
    )


@triton.jit
def _merge(
    Partial,
    Maxima,
    Sums,
    Out,
    TOTAL: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BD: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    split = tl.arange(0, SPLITS)
    d = tl.arange(0, BD)
    offset = split * TOTAL * H + row * H + head
    maximum = tl.load(Maxima + offset)
    sums = tl.load(Sums + offset)
    weight = tl.exp(maximum - tl.max(maximum, 0))
    value = tl.load(Partial + offset[:, None] * D + d[None, :], d[None, :] < D, 0)
    result = tl.sum(value * weight[:, None], 0) / tl.sum(sums * weight, 0)
    tl.store(Out + (row * H + head) * D + d, result, d < D)


class KSAAttentionMetadata:
    """Linear-size slot metadata; history KV stays in its scheduler-owned pool.

    Buffers can be staged repeatedly without changing addresses during capture.
    Slot maps include both text and summary pages in their retained causal order.
    """

    def __init__(self, windows, batch, capacity, device):
        self.capacity = capacity
        self.windows = windows
        self.offsets = torch.empty(batch + 1, dtype=torch.int32, device=device)
        shape = (len(windows) * batch * capacity,)
        self.slots = torch.zeros(shape, dtype=torch.int64, device=device)
        self.positions = torch.zeros(shape, dtype=torch.int64, device=device)
        self.summary = torch.zeros(shape, dtype=torch.bool, device=device)
        self.lengths = torch.zeros(
            (len(windows), batch), dtype=torch.int32, device=device
        )

        self.bases = torch.empty_like(self.lengths)

    def stage(self, states, *, padded=False):
        offsets = [0]
        lengths = []
        for state in states:
            offsets.append(offsets[-1] + (2 if padded else len(state["pos"])))
        self.offsets.copy_(
            torch.tensor(offsets, dtype=torch.int32, device=self.offsets.device)
        )
        slot_parts, pos_parts, summary_parts, bases = [], [], [], []
        offset = 0
        slot_orders = {}
        for layer, window in enumerate(self.windows):
            counts, starts = [], []
            for request_index, state in enumerate(states):
                cache = state["cache"]
                count = state["old_lengths"].get(window, 0)
                counts.append(count)
                starts.append(offset)
                offset += count
                if count:
                    slots = cache.page_pool.read_slots[cache.request.request_id][layer]
                    if isinstance(slots, tuple):
                        text_slots, summary_slots, flags = slots
                        layout_key = (request_index, window)
                        if layout_key not in slot_orders:
                            rank = flags.cumsum(0)
                            slot_orders[layout_key] = torch.where(
                                flags,
                                rank + len(text_slots) - 1,
                                torch.arange(count, device=flags.device) - rank,
                            )
                        slots = torch.cat((text_slots, summary_slots)).index_select(
                            0, slot_orders[layout_key]
                        )
                    slot_parts.append(slots)
                    positions, summary = cache.layouts[window]
                    pos_parts.append(positions)
                    summary_parts.append(summary)
            lengths.append(counts)
            bases.append(starts)
        for target, parts in (
            (self.slots, slot_parts),
            (self.positions, pos_parts),
            (self.summary, summary_parts),
        ):
            if parts:
                target[:offset].copy_(torch.cat(parts))
        self.lengths.copy_(
            torch.tensor(lengths, dtype=torch.int32, device=self.lengths.device)
        )
        self.bases.copy_(
            torch.tensor(bases, dtype=torch.int32, device=self.bases.device)
        )

    @classmethod
    def from_states(cls, windows, states):
        capacity = max(
            1, max((n for s in states for n in s["old_lengths"].values()), default=0)
        )
        result = cls(windows, len(states), capacity, states[0]["pos"].device)
        result.stage(states)
        return result


def paged_attention(q, k, v, pool, metadata, layer, positions, summary, max_rows):
    """One batched launch, one joint online softmax for local text and summaries."""
    heads, dim = q.shape[1:]
    kv_heads = k.shape[1]
    group = heads // kv_heads
    wide_prefill = max_rows > 2 and q.dtype == torch.bfloat16
    bm = 128 if wide_prefill else 32
    output = torch.empty_like(q)
    if isinstance(pool, dict):
        text_pool = pool[f"layer.{layer}.text"]
        summary_pool = pool[f"layer.{layer}.summary"]
    else:
        text_pool = summary_pool = pool
    splits = 8 if max_rows <= 2 else 1
    partial = (
        torch.empty((splits, *q.shape), device=q.device, dtype=torch.float32)
        if splits > 1
        else output
    )
    maxima = torch.empty(
        (splits, q.shape[0], heads), device=q.device, dtype=torch.float32
    )
    sums = torch.empty_like(maxima)
    _attention[
        (
            triton.cdiv(max_rows * group, bm),
            kv_heads,
            (metadata.offsets.numel() - 1) * splits,
        )
    ](
        q,
        k,
        v,
        text_pool,
        summary_pool,
        metadata.slots,
        metadata.positions,
        metadata.summary,
        metadata.lengths[layer],
        metadata.bases[layer],
        metadata.offsets,
        positions,
        summary,
        partial,
        maxima,
        sums,
        q.shape[0],
        splits,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        heads,
        kv_heads,
        dim,
        group,
        metadata.windows[layer],
        bm,
        64,
        triton.next_power_of_2(dim),
        num_warps=8 if wide_prefill else 4,
        num_stages=2 if wide_prefill else 3,
    )
    if splits > 1:
        _merge[(q.shape[0], heads)](
            partial,
            maxima,
            sums,
            output,
            q.shape[0],
            heads,
            dim,
            splits,
            triton.next_power_of_2(dim),
        )
    return output
