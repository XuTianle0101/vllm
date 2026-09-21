# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KSA model, attention contracts, compressed KV cache, and decode graphs."""

import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch.profiler import record_function

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.config.ksa import (
    MAX_CACHED_TEXT_TOKENS,
    MAX_PREFILL_TOKENS,
    internal_row_count,
    validate_ksa_config,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.kv_cache_spec_registry import register_kv_cache_spec

from .qwen3 import Qwen3ForCausalLM

logger = init_logger(__name__)


def prefill_layout(length: int, block_size: int = 8, device=None):
    """Return rotary positions, text indices and summary flags for internal rows."""
    if not 1 <= length <= MAX_PREFILL_TOKENS:
        raise ValueError(
            f"explicit KSA prefill supports lengths 1..{MAX_PREFILL_TOKENS}"
        )
    if block_size != 8:
        raise ValueError("T01 requires block_size=8")
    return cached_layout(0, length, device)


def expand_summary_positions(length: int, block_size: int = 8):
    positions, text_rows, summary = prefill_layout(length, block_size)
    return positions[text_rows], positions[summary]


def expand_prefill_sequence(input_ids, block_size=8, summary_token=151936):
    if input_ids.ndim != 1:
        raise ValueError("input_ids must be one-dimensional")
    positions, text_rows, _ = prefill_layout(
        input_ids.numel(), block_size, input_ids.device
    )
    expanded = input_ids.new_full(positions.shape, summary_token)
    expanded[text_rows] = input_ids
    return expanded, text_rows


def visibility_mask(positions, summary, window, key_positions=None, key_summary=None):
    """Official predicates for square prefill or rectangular cached queries."""
    if key_positions is None:
        key_positions, key_summary = positions, summary
    distance = positions[:, None] // 8 - key_positions[None, :] // 8
    # A block's last text and summary share RoPE position but not causal order.
    causal = (key_positions[None, :] < positions[:, None]) | (
        (key_positions[None, :] == positions[:, None])
        & (summary[:, None] | ~key_summary[None, :])
    )
    text_query = ((~key_summary[None, :]) & (distance <= window)) | (
        key_summary[None, :] & (distance > window)
    )
    return causal & torch.where(summary[:, None], distance == 0, text_query)


def build_expanded_prefill_mask(length, windows, block_size=8):
    positions, text_rows, summary = prefill_layout(length, block_size)
    if not windows:
        raise ValueError("at least one layer window is required")
    return torch.stack(
        [visibility_mask(positions, summary, w) for w in windows]
    ), text_rows


def build_prefill_mask(length, windows, block_size=8):
    masks, rows = build_expanded_prefill_mask(length, windows, block_size)
    return masks[:, rows][:, :, rows]


def prefill_attention(q, k, v, mask, reference=False):
    """Consume [rows, heads, dim]; FP32 oracle and SDPA share one mask."""
    repeats = q.shape[1] // k.shape[1]
    q = q.transpose(0, 1)
    k = k.repeat_interleave(repeats, dim=1).transpose(0, 1)
    v = v.repeat_interleave(repeats, dim=1).transpose(0, 1)
    if reference:
        scores = q.float() @ k.float().transpose(-1, -2) * q.shape[-1] ** -0.5
        probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        output = (probs @ v.float()).to(q.dtype)
    else:
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return output.transpose(0, 1).contiguous()


@dataclass
class KSACache:
    """Request-owned cache; text count excludes internal summary rows."""

    text_tokens: int = 0
    layers: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    layouts: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    owner: object | None = None
    page_pool: object | None = None
    request: object | None = None

    @property
    def internal_rows(self):
        return internal_row_count(0, self.text_tokens)

    def clear(self):
        if self.page_pool is not None and self.request is not None:
            self.page_pool.free(self.request)
        self.text_tokens = 0
        self.layers.clear()
        self.layouts.clear()
        self.owner = None


def retained_layout(text_tokens, window, device=None):
    """Positions of completed summaries and text needed by the next query."""
    positions, _, summary = cached_layout(0, text_tokens, device)
    first_block = max(0, text_tokens // 8 - window)
    keep = summary | (positions // 8 >= first_block)
    return positions[keep], summary[keep], keep


def cached_layout(start, length, device=None):
    """Expand only new text, placing summaries at absolute block ends."""
    if start == 0:
        rows = torch.arange(internal_row_count(start, length), device=device)
        summary = rows % 9 == 8
        positions = rows - rows // 9 - summary.long()
        return positions, rows[~summary], summary
    if length == 1:
        positions = torch.full((1 + (start % 8 == 7),), start, device=device)
        rows = torch.zeros(1, dtype=torch.long, device=device)
        summary = torch.zeros(len(positions), dtype=torch.bool, device=device)
        if len(positions) == 2:
            summary[1] = True
        return positions, rows, summary
    text = torch.arange(start, start + length, device=device)
    counts = 1 + ((text + 1) % 8 == 0).long()
    positions = text.repeat_interleave(counts)
    rows = counts.cumsum(0) - counts
    summary = torch.ones_like(positions, dtype=torch.bool)
    summary[rows] = False
    return positions, rows, summary


@dataclass(frozen=True)
class KSASummarySpec(FullAttentionSpec):
    """One stored summary per eight scheduler text positions."""

    @property
    def storage_block_size(self):
        return self.block_size // 8

    @property
    def real_page_size_bytes(self):
        return super().real_page_size_bytes // 8


def register_ksa_kv_cache_specs(vllm_config):
    """Register KSA specs after built-in specs have been initialized."""
    KVCacheSpecRegistry = __import__(
        "vllm.v1.kv_cache_spec_registry", fromlist=["KVCacheSpecRegistry"]
    ).KVCacheSpecRegistry
    KVCacheSpecRegistry._ensure_registered(vllm_config)
    register_kv_cache_spec(
        manager_class=FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )(KSASummarySpec)


@dataclass
class KSAAttentionMetadata(AttentionMetadata):
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    positions: torch.Tensor | None
    num_actual_tokens: int


class KSAAttentionMetadataBuilder(AttentionMetadataBuilder[KSAAttentionMetadata]):
    """Preserve scheduler boundaries and block tables without reinterpretation."""

    _cudagraph_support = AttentionCGSupport.NEVER
    supports_update_block_table = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        # The base constructor is abstract even though it initializes fields.
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> KSAAttentionMetadata:
        del common_prefix_len, fast_build
        return KSAAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            positions=common_attn_metadata.positions,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
        )

    def update_block_table(self, metadata, block_table, slot_mapping=None):
        # Earlier groups still reference the cached metadata object.
        return replace(
            metadata,
            block_table=block_table,
            slot_mapping=metadata.slot_mapping
            if slot_mapping is None
            else slot_mapping,
        )


class _KSAImpl(AttentionImpl):
    """Placeholder implementation; KSA executes the packed model loop."""

    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads=None,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype

    def forward(self, *args, **kwargs):
        raise RuntimeError("KSA attention is executed by KSAForCausalLM")


class KSAAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "float16",
    ]
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        return "KSA"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        return _KSAImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return KSAAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [8]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return (num_blocks, block_size, num_kv_heads, 2 * head_size)

    @classmethod
    def get_kv_cache_block_dim(
        cls, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ) -> int:
        del block_size, num_kv_heads, head_size, cache_dtype_str
        return 0

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension=False,
    ) -> tuple[int, ...]:
        return (1, 0, 2, 3, 4) if include_num_layers_dimension else (0, 1, 2, 3)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return True

    @classmethod
    def supports_combination(cls, *args, **kwargs) -> str | None:
        return None


class KSASummaryAttentionBackend(KSAAttentionBackend):
    """Backend view where eight logical kernel blocks share one summary page."""

    @staticmethod
    def get_name() -> str:
        return "KSA_SUMMARY"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # The scheduler block is 64 logical positions; storage has eight rows.
        # Selecting a 64-token kernel block prevents the generic runner from
        # multiplying the compressed page by eight a second time.
        return [64]


class KSAAttentionLayer(torch.nn.Module, AttentionLayerBase):
    """An AttentionLayerBase-compatible cache endpoint for one KSA page type."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        num_kv_heads: int,
        head_dim: int,
        window: int,
        summary: bool,
    ):
        super().__init__()
        self.layer_name = prefix
        self.num_heads = num_kv_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_dim
        self.attn_type = AttentionType.DECODER
        self.kv_cache = torch.tensor([])
        if summary:
            self.kv_cache_spec = KSASummarySpec(
                block_size=64,
                num_kv_heads=num_kv_heads,
                head_size=head_dim,
                dtype=_cache_dtype(vllm_config),
            )
        else:
            self.kv_cache_spec = SlidingWindowSpec(
                block_size=8,
                sliding_window=window * 8 + 8,
                num_kv_heads=num_kv_heads,
                head_size=head_dim,
                dtype=_cache_dtype(vllm_config),
            )
        vllm_config.compilation_config.static_forward_context[prefix] = self

    def get_attn_backend(self):
        return (
            KSASummaryAttentionBackend
            if isinstance(self.kv_cache_spec, KSASummarySpec)
            else KSAAttentionBackend
        )

    def get_kv_cache_spec(self, vllm_config):
        del vllm_config
        return self.kv_cache_spec

    def forward(self, *args, **kwargs):
        raise RuntimeError("KSA attention layers are called by KSAForCausalLM")


def _cache_dtype(vllm_config):
    dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)
    return dtype if isinstance(dtype, torch.dtype) else torch.bfloat16


class KSAStandardPagePool:
    """Adapter over the tensors and block tables allocated by the V1 runner."""

    def __init__(self, layers, metadata_by_name, windows):
        self.layers = layers
        self.metadata_by_name = metadata_by_name
        self.windows = windows
        self.storage = {}
        for layer_index in range(len(layers) // 2):
            self.storage[f"layer.{layer_index}.text"] = layers[2 * layer_index].kv_cache
            self.storage[f"layer.{layer_index}.summary"] = layers[
                2 * layer_index + 1
            ].kv_cache
        self.read_slots = {}
        self.graph_identity = tuple(
            tensor.data_ptr() for tensor in self.storage.values()
        )

    @staticmethod
    def _rows(tensor, slots, rows):
        # V1 KSA pages use the standard NHD layout: [block, row, head, 2D].
        tensor[slots // 8, slots % 8] = rows

    def _table(self, request, layer, summary):
        endpoint = self.layers[layer * 2 + summary]
        metadata = self.metadata_by_name[endpoint.layer_name]
        return metadata.block_table[request.index], endpoint.kv_cache

    def prepare_request(self, request, cache):
        """Materialize logical layouts from scheduler block tables for one step."""
        slots_by_layer = []
        for layer, window in enumerate(self.windows):
            if window not in cache.layouts:
                slots_by_layer.append(
                    (
                        torch.empty(0, dtype=torch.long),
                        torch.empty(0, dtype=torch.long),
                        torch.empty(0, dtype=torch.bool),
                    )
                )
                continue
            positions, summary = cache.layouts[window]
            text_table, _ = self._table(request, layer, False)
            summary_table, _ = self._table(request, layer, True)
            positions = positions.to(text_table.device)
            summary = summary.to(text_table.device)
            text_slots = (
                text_table[positions[~summary] // 8].to(torch.long) * 8
                + positions[~summary] % 8
            )
            summary_slots = (
                summary_table[positions[summary] // 64].to(torch.long) * 8
                + (positions[summary] // 8) % 8
            )
            slots_by_layer.append((text_slots, summary_slots, summary))
        self.read_slots[request.index] = slots_by_layer
        request.request_id = request.index

    def read(self, request, layer, layout):
        positions, summary = layout
        text_table, text_cache = self._table(request, layer, False)
        summary_table, summary_cache = self._table(request, layer, True)
        positions = positions.to(text_cache.device)
        summary = summary.to(text_cache.device)
        result = text_cache.new_empty(
            (len(positions), text_cache.shape[-2], text_cache.shape[-1])
        )
        if (~summary).any():
            p = positions[~summary]
            slots = text_table[p // 8].to(torch.long) * 8 + p % 8
            result[~summary] = text_cache[slots // 8, slots % 8]
        if summary.any():
            p = positions[summary]
            slots = summary_table[p // 64].to(torch.long) * 8 + (p // 8) % 8
            result[summary] = summary_cache[slots // 8, slots % 8]
        head_dim = result.shape[-1] // 2
        return result[..., :head_dim], result[..., head_dim:]

    def commit(self, request, end, pending, retained):
        del retained
        if not pending:
            return
        positions, _, summary = cached_layout(
            request.num_computed_tokens,
            end - request.num_computed_tokens,
            pending[0][0].device,
        )
        # Only validate blocks touched by this chunk. Other entries may be
        # unallocated padding or old text pages freed by the sliding window.
        for layer in range(len(pending)):
            table, text_cache = self._table(request, layer, False)
            summary_table, summary_cache = self._table(request, layer, True)
            text_blocks = positions[~summary] // 8
            summary_blocks = positions[summary] // 64
            if (text_blocks.numel() and text_blocks.max() >= table.numel()) or (
                summary_blocks.numel() and summary_blocks.max() >= summary_table.numel()
            ):
                raise RuntimeError("KSA write exceeds the scheduler block table")
            if text_blocks.numel() and torch.any(table[text_blocks] <= 0):
                raise RuntimeError("KSA attempted to write a freed scheduler page")
            if summary_blocks.numel() and torch.any(summary_table[summary_blocks] <= 0):
                raise RuntimeError("KSA attempted to write a freed scheduler page")
            if text_blocks.numel() and torch.any(
                table[text_blocks] >= text_cache.shape[0]
            ):
                raise RuntimeError("KSA write exceeds the allocated text cache")
            if summary_blocks.numel() and torch.any(
                summary_table[summary_blocks] >= summary_cache.shape[0]
            ):
                raise RuntimeError("KSA write exceeds the allocated summary cache")
        for layer, (key, value) in enumerate(pending):
            # Each head stores contiguous K and V halves, not interleaved values.
            rows = torch.cat((key, value), dim=-1)
            text_table, text_cache = self._table(request, layer, False)
            summary_table, summary_cache = self._table(request, layer, True)
            if (~summary).any():
                p = positions[~summary]
                slots = text_table[p // 8].to(torch.long) * 8 + p % 8
                self._rows(text_cache, slots, rows[~summary])
            if summary.any():
                p = positions[summary]
                slots = summary_table[p // 64].to(torch.long) * 8 + (p // 8) % 8
                self._rows(summary_cache, slots, rows[summary])
        request.num_computed_tokens = end

    def free(self, request):
        del request


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
    PB: tl.constexpr,
    PR: tl.constexpr,
    PH: tl.constexpr,
    PV: tl.constexpr,
    SB: tl.constexpr,
    SR: tl.constexpr,
    SH: tl.constexpr,
    SV: tl.constexpr,
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
        history_offset = (
            slot // 8 * tl.where(ks_old, SB, PB)
            + slot % 8 * tl.where(ks_old, SR, PR)
            + head * tl.where(ks_old, SH, PH)
        )
        pk = tl.load(
            history_pool[None, :] + history_offset[None, :] + d[:, None],
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
            + history_offset[:, None]
            + tl.where(ks_old, SV, PV)[:, None]
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


class KSAKernelMetadata:
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

    def cache_strides(tensor):
        if tensor.ndim == 4:
            # Standard V1: [block, row, head, K|V].
            return (*tensor.stride()[:3], dim)
        if tensor.ndim == 5:
            # Standalone scheduler pool: [block, row, K/V, head, dim].
            return (
                tensor.stride(0),
                tensor.stride(1),
                tensor.stride(3),
                tensor.stride(2),
            )
        # No history is read for an uncached prefill.
        return (0, 0, 0, 0)

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
        *cache_strides(text_pool),
        *cache_strides(summary_pool),
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


class _DecodeBuffers:
    def __init__(self, model, batch, capacity):
        self.model = model
        self.capacity = capacity
        parameter = next(model.parameters())
        device = parameter.device
        self.ids = torch.full(
            (batch, 2),
            model.config.summary_token_begin,
            dtype=torch.long,
            device=device,
        )
        self.states = []
        self.history = []
        attn = model.model.layers[0].self_attn
        for _ in range(batch):
            history = [
                (
                    parameter.new_zeros(capacity, attn.num_kv_heads, attn.head_dim),
                    parameter.new_zeros(capacity, attn.num_kv_heads, attn.head_dim),
                )
                for _ in model.model.layers
            ]
            masks = {
                window: torch.zeros(2, capacity + 2, dtype=torch.bool, device=device)
                for window in set(model.windows)
            }
            for mask in masks.values():
                mask[:, capacity] = True
            self.history.append(history)
            self.states.append(
                dict(
                    hidden=None,
                    pos=torch.zeros(2, dtype=torch.long, device=device),
                    rows=torch.zeros(1, dtype=torch.long, device=device),
                    summary=torch.tensor([False, True], device=device),
                    cache=KSACache(),
                    start=1,
                    end=2,
                    masks=masks,
                    retained={w: None for w in masks},
                    old_lengths={w: capacity for w in masks},
                    pending_layouts={},
                    pending=[],
                )
            )

    def run(self):
        hidden = self.model.embed_input_ids(self.ids.flatten()).view(
            len(self.states), 2, -1
        )
        for state, history, value in zip(self.states, self.history, hidden):
            state["hidden"] = value
            state["cache"].layers = history
            state["pending"] = []
        self.output = self.model._forward_states(self.states)
        self.new_kv = [
            [(k[self.capacity :], v[self.capacity :]) for k, v in s["cache"].layers]
            for s in self.states
        ]

    def stage(self, requests, prepared):
        for i, ((ids, _, cache), source, target, history) in enumerate(
            zip(requests, prepared, self.states, self.history)
        ):
            self.ids[i, :1].copy_(ids)
            target["pos"].fill_(source["start"])
            count = len(source["pos"])
            for window, mask in target["masks"].items():
                old = source["old_lengths"][window]
                mask.zero_()
                mask[:count, :old].copy_(source["masks"][window][:, :old])
                mask[:count, self.capacity : self.capacity + count].copy_(
                    source["masks"][window][:, old:]
                )
                # The unused summary row must stay finite but is never committed.
                if count == 1:
                    mask[1, self.capacity] = True
            for layer, (key, value) in enumerate(history):
                old_k, old_v = (
                    cache.layers[layer]
                    if cache.page_pool is None
                    else cache.page_pool.read(
                        cache.request, layer, cache.layouts[self.model.windows[layer]]
                    )
                )
                # Reset padding as well: a previous request may have occupied it.
                key.zero_()
                value.zero_()
                key[: len(old_k)].copy_(old_k)
                value[: len(old_v)].copy_(old_v)


class _PagedDecodeBuffers:
    """Graph inputs contain only new rows and linear-size page metadata."""

    def __init__(self, model, batch, capacity):
        self.model = model
        parameter = next(model.parameters())
        self.ids = torch.full(
            (batch, 2),
            model.config.summary_token_begin,
            dtype=torch.long,
            device=parameter.device,
        )
        self.metadata = KSAKernelMetadata(
            model.windows, batch, capacity, parameter.device
        )
        self.states = [
            dict(
                use_triton=True,
                hidden=None,
                pos=torch.zeros(2, dtype=torch.long, device=parameter.device),
                rows=torch.zeros(1, dtype=torch.long, device=parameter.device),
                summary=torch.tensor([False, True], device=parameter.device),
                cache=None,
                start=1,
                retained={},
                pending=[],
            )
            for _ in range(batch)
        ]

    def stage(self, requests, prepared):
        self.metadata.stage(prepared, padded=True)
        for i, ((ids, _, cache), source, target) in enumerate(
            zip(requests, prepared, self.states)
        ):
            self.ids[i, :1].copy_(ids)
            target["pos"].fill_(source["start"])
            target["cache"] = cache

    def run(self):
        hidden = self.model.embed_input_ids(self.ids.flatten()).view(
            len(self.states), 2, -1
        )
        for state, value in zip(self.states, hidden):
            state["hidden"] = value
            state["pending"] = []
        self.output = self.model._forward_states(
            self.states, kernel_metadata=self.metadata, commit=False
        )
        self.new_kv = [state["pending"] for state in self.states]


class KSADecodeGraphs:
    """Bounded lazy graph cache, keyed by batch size and padded history length.

    Only single-token cached decode is captured. Prefill/mixed prefill batches
    use the existing eager path. Returned hidden states own their storage.
    Request state and physical pages are never owned by a graph slot.
    """

    def __init__(self, model, *, enabled=True, max_graphs=4):
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        if enabled and next(model.parameters()).device.type != "cuda":
            raise ValueError("CUDA graphs require a CUDA model")
        self.model = model
        self.enabled = enabled
        self.max_graphs = max_graphs
        self.graphs = OrderedDict()
        self.startup = []
        self.replays = 0

    @torch.inference_mode()
    def forward_batch(self, requests):
        if not requests or any(
            len(ids) != 1 or cache is None or not cache.text_tokens
            for ids, _, cache in requests
        ):
            return self.model.forward_batch(requests)
        if len({id(c) for _, _, c in requests}) != len(requests):
            raise ValueError("a KSA cache may occur only once in a batch")
        with record_function("ksa.metadata"):
            prepared = [
                self.model._prepare(ids, pos, cache=cache)
                for ids, pos, cache in requests
            ]
            length = max(n for s in prepared for n in s["old_lengths"].values())
            capacity = max(16, 1 << (length - 1).bit_length())
            paged = all(s.get("use_triton", False) for s in prepared)
            page_pool = requests[0][2].page_pool if paged else None
            # The standard path creates a short-lived adapter around the
            # runner's tensors for each step. Key graphs by the backing cache
            # addresses so that adapter recreation does not force recapture.
            pool_id = (
                getattr(page_pool, "graph_identity", id(page_pool)) if paged else None
            )
            key = (len(requests), capacity, paged, pool_id)
        if key not in self.graphs:
            if len(self.graphs) == self.max_graphs:
                self.graphs.popitem(last=False)
            begin = time.perf_counter()
            buffer_type = _PagedDecodeBuffers if paged else _DecodeBuffers
            buffers = buffer_type(self.model, len(requests), capacity)
            buffers.stage(requests, prepared)
            graph = None
            if self.enabled:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        buffers.run()
                torch.cuda.current_stream().wait_stream(stream)
                torch.accelerator.synchronize()
                warmup = time.perf_counter()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    buffers.run()
                torch.accelerator.synchronize()
                self.startup.append(
                    dict(
                        batch=key[0],
                        capacity=capacity,
                        allocation_warmup_ms=(warmup - begin) * 1000,
                        capture_ms=(time.perf_counter() - warmup) * 1000,
                    )
                )
                logger.info(
                    "Captured KSA decode graph: batch=%d capacity=%d paged=%s",
                    key[0],
                    capacity,
                    paged,
                )
            self.graphs[key] = (buffers, graph)
        self.graphs.move_to_end(key)
        buffers, graph = self.graphs[key]
        with record_function("ksa.kv_stage"):
            buffers.stage(requests, prepared)
        with record_function("ksa.graph_replay" if graph else "ksa.buffered_eager"):
            if graph:
                graph.replay()
                self.replays += 1
                logger.debug("Replayed KSA decode graph: key=%s", key)
            else:
                buffers.run()
        with record_function("ksa.kv_commit"):
            for source, layers in zip(prepared, buffers.new_kv):
                cache = source["cache"]
                count = len(source["pos"])
                pending = [(k[:count], v[:count]) for k, v in layers]
                if cache.page_pool is not None:
                    try:
                        cache.page_pool.commit(
                            cache.request, source["end"], pending, source["retained"]
                        )
                    except BaseException:
                        if cache.request.num_computed_tokens != source["start"]:
                            cache.clear()
                        raise
                else:
                    updated = []
                    for layer, ((old_k, old_v), (k, v)) in enumerate(
                        zip(cache.layers, pending)
                    ):
                        k, v = torch.cat((old_k, k)), torch.cat((old_v, v))
                        keep = source["retained"][self.model.windows[layer]]
                        updated.append((k, v) if keep is None else (k[keep], v[keep]))
                    cache.layers = updated
                cache.layouts = source["pending_layouts"]
                cache.text_tokens = source["end"]
                cache.owner = self.model
        return [value.clone() for value in buffers.output]


class KSAForCausalLM(Qwen3ForCausalLM):
    """Eager packed projections with request-isolated compressed KV attention."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.windows = validate_ksa_config(config, config.num_hidden_layers)
        if (
            vllm_config.parallel_config.pipeline_parallel_size != 1
            or vllm_config.parallel_config.tensor_parallel_size != 1
        ):
            raise ValueError("KSA Python prototype requires TP=PP=1")
        if vllm_config.quant_config is not None:
            raise ValueError("KSA Python prototype requires unquantized weights")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Register the two physical KSA page contracts with the unmodified V1
        # cache allocator. They are metadata endpoints; packed attention still
        # runs once per transformer layer below.
        register_ksa_kv_cache_specs(vllm_config)
        self.ksa_cache_layers = []
        for index, window in enumerate(self.windows):
            layer = self.model.layers[index].self_attn
            # The Qwen attention modules remain projection containers, while
            # these endpoints own the scheduler-visible K/V pages.
            vllm_config.compilation_config.static_forward_context.pop(
                layer.attn.layer_name, None
            )
            for summary in (False, True):
                suffix = "summary" if summary else "text"
                self.ksa_cache_layers.append(
                    KSAAttentionLayer(
                        vllm_config=vllm_config,
                        prefix=(
                            f"{prefix}.model.layers.{index}.self_attn.ksa_{suffix}"
                        ),
                        num_kv_heads=layer.num_kv_heads,
                        head_dim=layer.head_dim,
                        window=window,
                        summary=summary,
                    )
                )
        self.text_row_indices = None
        self.reference_attention = False
        self.triton_attention = True
        self.return_full_vocab_logits = False
        self.layer_observer = None
        self._standard_graphs = None
        limit = getattr(config, "truncate_predict_nums", config.summary_token_begin)
        self.text_vocab_size = min(
            limit if limit > 0 else config.summary_token_begin,
            config.summary_token_begin,
        )

    def _use_triton(self, cache, device):
        return (
            self.triton_attention
            and not self.reference_attention
            and device.type == "cuda"
            and (
                cache is None
                or (cache.page_pool is not None and hasattr(cache.page_pool, "storage"))
            )
        )

    def _prepare(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        *,
        cache: KSACache | None = None,
    ):
        if intermediate_tensors is not None:
            raise ValueError(
                "KSA Python prototype does not support pipeline parallelism"
            )
        start = cache.text_tokens if cache is not None else 0
        if (
            positions.ndim != 1
            or positions.dtype not in (torch.int32, torch.int64)
            or not torch.equal(
                positions,
                torch.arange(start, start + positions.numel(), device=positions.device),
            )
        ):
            raise ValueError(
                "requires contiguous text positions 0..N-1 then cached decode"
            )
        if cache is not None and cache.owner is not None and cache.owner is not self:
            raise ValueError("KSA cache belongs to a different model")
        if not 1 <= positions.numel() <= MAX_PREFILL_TOKENS:
            raise ValueError("KSA chunks require 1..4096 text tokens")
        end = start + positions.numel()
        if end > min(MAX_CACHED_TEXT_TOKENS, self.config.max_position_embeddings):
            raise ValueError("T02 cache exceeds maximum text length")
        if start:
            if cache.page_pool is None and len(cache.layers) != len(self.model.layers):
                raise ValueError("incomplete KSA layer cache")
            if cache.layouts.keys() != set(self.windows):
                raise ValueError("incomplete KSA cache layouts")
            pos, rows, summary = cached_layout(
                start, positions.numel(), positions.device
            )
        else:
            pos, rows, summary = prefill_layout(
                positions.numel(), device=positions.device
            )
        self.text_row_indices = rows
        if inputs_embeds is None:
            if (
                input_ids is None
                or input_ids.shape != positions.shape
                or input_ids.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("input_ids must match text positions")
            if torch.any(
                (input_ids < 0) | (input_ids >= self.config.summary_token_begin)
            ):
                raise ValueError("text input contains a reserved summary token")
            ids = positions.new_full(pos.shape, self.config.summary_token_begin)
            ids[rows] = input_ids.to(ids.dtype)
            hidden = self.embed_input_ids(ids)
        else:
            if inputs_embeds.shape != (positions.numel(), self.config.hidden_size):
                raise ValueError("inputs_embeds must contain exactly the text rows")
            ids = positions.new_full(pos.shape, self.config.summary_token_begin)
            hidden = self.embed_input_ids(ids)
            hidden[rows] = inputs_embeds
        use_triton = self._use_triton(cache, positions.device)
        masks = {}
        retained = {}
        old_lengths = {}
        pending_layouts = {}
        for window in set(self.windows):
            if start:
                old_pos, old_summary = cache.layouts[window]
                old_lengths[window] = len(old_pos)
                key_pos = torch.cat((old_pos, pos))
                key_summary = torch.cat((old_summary, summary))
                if not use_triton:
                    masks[window] = visibility_mask(
                        pos, summary, window, key_pos, key_summary
                    )
            else:
                key_pos, key_summary = pos, summary
                if not use_triton:
                    masks[window] = visibility_mask(pos, summary, window)
            if cache is not None:
                first_block = max(0, end // 8 - window)
                old_first_block = max(0, start // 8 - window)
                if first_block == old_first_block:
                    retained[window] = None
                else:
                    retained[window] = key_summary | (key_pos // 8 >= first_block)
                keep = retained[window]
                pending_layouts[window] = (
                    (key_pos, key_summary)
                    if keep is None
                    else (key_pos[keep], key_summary[keep])
                )
        return dict(
            use_triton=use_triton,
            hidden=hidden,
            pos=pos,
            rows=rows,
            summary=summary,
            cache=cache,
            start=start,
            end=end,
            masks=masks,
            retained=retained,
            old_lengths=old_lengths,
            pending_layouts=pending_layouts,
            pending=[],
        )

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        *,
        cache: KSACache | None = None,
    ):
        # The V1 runner owns KV allocation and publishes its metadata through
        # forward_context. Keep the compressed reference path available for
        # standalone numerical tests, but use the standard model path when the
        # engine is executing a scheduled batch.
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if is_forward_context_available() and cache is None:
            context = get_forward_context()
            if not context.attn_metadata:
                # Profiling and warmup intentionally run without scheduler KV
                # pages. Use the same reference path as standalone eager
                # execution and never write synthetic cache rows.
                previous_triton = self.triton_attention
                self.triton_attention = False
                try:
                    # GPUModelRunner profiling may use arbitrary synthetic
                    # positions; there is no history to preserve here.
                    profile_positions = torch.arange(
                        positions.numel(), device=positions.device
                    )
                    state = self._prepare(
                        input_ids,
                        profile_positions,
                        intermediate_tensors,
                        inputs_embeds,
                    )
                    return self._forward_states([state])[0]
                finally:
                    self.triton_attention = previous_triton
            # Standard runner inputs are packed and the cache layer metadata is
            # available by name. KSA consumes those boundaries while preserving
            # its internal summary-row expansion.
            metadata = next(
                (
                    value
                    for value in context.attn_metadata.values()
                    if value is not None
                ),
                None,
            )
            if metadata is None:
                raise RuntimeError("KSA standard forward requires attention metadata")
            return self._forward_standard(
                input_ids, positions, intermediate_tensors, inputs_embeds, metadata
            )
        state = self._prepare(
            input_ids, positions, intermediate_tensors, inputs_embeds, cache=cache
        )
        return self._forward_states([state])[0]

    def _forward_standard(
        self, input_ids, positions, intermediate_tensors, inputs_embeds, metadata
    ):
        """Run KSA with scheduler packed inputs and standard metadata.

        The page adapter uses scheduler block tables without persistent
        request caches.
        """
        if intermediate_tensors is not None:
            raise ValueError("KSA standard path requires TP=PP=1")
        if inputs_embeds is not None:
            raise ValueError("KSA standard path does not support prompt embeddings")
        if positions.ndim != 1 or positions.numel() == 0:
            raise ValueError(
                "KSA standard path requires packed one-dimensional positions"
            )
        # The scheduler only exposes a flat batch; reconstruct request ranges
        # from query_start_loc and use the existing numerically checked kernel.
        from vllm.forward_context import get_forward_context

        context = get_forward_context()
        pool = KSAStandardPagePool(
            self.ksa_cache_layers, context.attn_metadata, self.windows
        )
        starts = metadata.query_start_loc.tolist()
        requests = []
        for request_index, (begin, end) in enumerate(zip(starts, starts[1:])):
            ids = None if input_ids is None else input_ids[begin:end]
            start = int(positions[begin].item()) if end > begin else 0
            request = SimpleNamespace(index=request_index, num_computed_tokens=start)
            cache = KSACache(text_tokens=start, page_pool=pool, request=request)
            if start:
                for window in set(self.windows):
                    pos, summary, _ = retained_layout(start, window, positions.device)
                    cache.layouts[window] = (pos, summary)
                pool.prepare_request(request, cache)
            requests.append((ids, positions[begin:end], cache))

        # The local graph only handles one-token cached decode. Prefill and
        # mixed batches deliberately stay on the eager reference path.
        graph_enabled = (
            self._standard_graphs is not None and self._standard_graphs.enabled
        )
        if self._standard_graphs is None:
            extra = self.vllm_config.additional_config
            enabled = isinstance(extra, dict) and extra.get("ksa_cudagraph", False)
            self._standard_graphs = KSADecodeGraphs(self, enabled=enabled)
            graph_enabled = enabled
        if graph_enabled and all(len(pos) == 1 for _, pos, _ in requests):
            return torch.cat(self._standard_graphs.forward_batch(requests))

        previous_triton = self.triton_attention
        self.triton_attention = False
        try:
            return torch.cat(self.forward_batch(requests))
        finally:
            self.triton_attention = previous_triton

    def forward_batch(self, requests):
        """Run packed projections; attention and cache ownership stay per request.

        Each entry is (text IDs, absolute text positions, request cache). All
        attention finishes before any page is recycled for the new chunks.
        Returned tensors contain text rows only, in the supplied request order.
        """
        if not requests:
            return []
        caches = [cache for _, _, cache in requests if cache is not None]
        if len({id(cache) for cache in caches}) != len(caches):
            raise ValueError("a KSA cache may occur only once in a batch")
        states = [self._prepare(ids, pos, cache=cache) for ids, pos, cache in requests]
        return self._forward_states(states)

    def _forward_states(self, states, *, kernel_metadata=None, commit=True):
        hidden = torch.cat([state["hidden"] for state in states])
        pos = torch.cat([state["pos"] for state in states])
        sizes = [len(state["pos"]) for state in states]
        use_triton = all(s.get("use_triton", False) for s in states)
        if use_triton:
            if kernel_metadata is None:
                kernel_metadata = KSAKernelMetadata.from_states(self.windows, states)
            summary = torch.cat([state["summary"] for state in states])
            pools = {
                id(s["cache"].page_pool): s["cache"].page_pool
                for s in states
                if s["cache"] is not None
            }
            if len(pools) > 1:
                raise ValueError("batched KSA attention requires a shared page pool")
            pool = next(iter(pools.values())).storage if pools else hidden
        elif any(s.get("use_triton", False) for s in states):
            raise ValueError("cannot mix reference and Triton caches in one batch")
        residual = None
        for index, layer in enumerate(self.model.layers):
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)
            attn = layer.self_attn
            with record_function("ksa.projection_qkv_rope"):
                qkv, _ = attn.qkv_proj(hidden)
                q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], -1)
                q = attn.q_norm(q.reshape(-1, attn.num_heads, attn.head_dim))
                k = attn.k_norm(k.reshape(-1, attn.num_kv_heads, attn.head_dim))
                q, k = attn.rotary_emb(pos, q.flatten(1), k.flatten(1))
                q = q.reshape(-1, attn.num_heads, attn.head_dim)
                k = k.reshape(-1, attn.num_kv_heads, attn.head_dim)
                v = v.reshape(-1, attn.num_kv_heads, attn.head_dim)
            window = self.windows[index]
            if use_triton:
                with record_function("ksa.attention"):
                    attention_output = paged_attention(
                        q, k, v, pool, kernel_metadata, index, pos, summary, max(sizes)
                    )
                for state, key, value in zip(states, k.split(sizes), v.split(sizes)):
                    state["pending"].append((key, value))
            else:
                outputs = []
                for state, query, key, value in zip(
                    states, q.split(sizes), k.split(sizes), v.split(sizes)
                ):
                    cache = state["cache"]
                    keep = state["retained"].get(window)
                    if cache is not None and cache.page_pool is not None:
                        state["pending"].append((key, value))
                    if state["start"]:
                        old_k, old_v = (
                            cache.layers[index]
                            if cache.page_pool is None
                            else cache.page_pool.read(
                                cache.request, index, cache.layouts[window]
                            )
                        )
                        if (
                            old_k.shape[0] != state["old_lengths"][window]
                            or old_v.shape != old_k.shape
                        ):
                            raise ValueError(
                                "KSA cache row count does not match text count"
                            )
                        key, value = torch.cat((old_k, key)), torch.cat((old_v, value))
                    if cache is not None and cache.page_pool is None:
                        state["pending"].append(
                            (key, value) if keep is None else (key[keep], value[keep])
                        )
                    with record_function("ksa.attention"):
                        outputs.append(
                            prefill_attention(
                                query,
                                key,
                                value,
                                state["masks"][window],
                                self.reference_attention,
                            )
                        )
                attention_output = torch.cat(outputs)
            with record_function("ksa.projection_output_mlp"):
                hidden, _ = attn.o_proj(attention_output.flatten(1))
                hidden, residual = layer.post_attention_layernorm(hidden, residual)
                hidden = layer.mlp(hidden)
            if self.layer_observer is not None:
                for state, value in zip(states, (hidden + residual).split(sizes)):
                    self.layer_observer(index, value, state["rows"], state["summary"])
        hidden, _ = self.model.norm(hidden, residual)
        for state in states if commit else []:
            cache = state["cache"]
            if cache is None:
                continue
            if cache.page_pool is not None:
                try:
                    cache.page_pool.commit(
                        cache.request, state["end"], state["pending"], state["retained"]
                    )
                except BaseException:
                    if cache.request.num_computed_tokens != state["start"]:
                        cache.clear()
                    raise
            else:
                cache.layers = state["pending"]
            cache.layouts = state["pending_layouts"]
            cache.text_tokens = state["end"]
            cache.owner = self
        return [
            value[state["rows"]] for state, value in zip(states, hidden.split(sizes))
        ]

    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        if self.return_full_vocab_logits:
            logits[..., self.text_vocab_size :] = -float("inf")
            return logits
        return logits[..., : self.text_vocab_size]

    def load_weights(self, weights):
        """Validate every unpacked checkpoint shard before AutoWeightsLoader."""
        expected = set()
        for name, _ in self.named_parameters():
            for packed, shards in self.packed_modules_mapping.items():
                if f".{packed}." in name:
                    expected.update(
                        name.replace(f".{packed}.", f".{s}.") for s in shards
                    )
                    break
            else:
                expected.add(name)
        seen = set()

        def checked():
            for name, tensor in weights:
                if name == "lm_head.weight" and self.config.tie_word_embeddings:
                    continue
                if name not in expected or name in seen:
                    raise ValueError(
                        f"unexpected or duplicate KSA checkpoint parameter: {name}"
                    )
                seen.add(name)
                yield name, tensor

        loaded = super().load_weights(checked())
        if missing := expected - seen:
            raise ValueError(f"missing KSA checkpoint parameters: {sorted(missing)}")
        return loaded
