# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KSA page storage backed by the scheduler's KVCacheManager.

The eager runner commits already computed KV after attention has finished. This
allows prefill to discard old text without overwriting keys its early queries
still need. Chunked execution uses the same delayed commit protocol.
"""

from dataclasses import dataclass
from itertools import count

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


@dataclass(frozen=True)
class KSASummarySpec(FullAttentionSpec):
    """One stored summary per eight scheduler text positions."""

    @property
    def storage_block_size(self):
        return self.block_size // 8

    @property
    def real_page_size_bytes(self):
        return super().real_page_size_bytes // 8


class KSAPagePool:
    """Equal byte pages shared by separate text/summary groups for each layer."""

    def __init__(
        self, windows, heads, head_dim, dtype, device, max_tokens, *, num_blocks=None
    ):
        self.windows = windows
        self.max_tokens = max_tokens
        self.serial = count()
        self.read_slots = {}
        self.batch_writes = True
        self.compact_prefill = True
        groups = []
        capacity = 1  # vLLM reserves block zero as a null page.
        for index, window in enumerate(windows):
            common = dict(num_kv_heads=heads, head_size=head_dim, dtype=dtype)
            text = SlidingWindowSpec(
                block_size=8, sliding_window=window * 8 + 1, **common
            )
            summary = KSASummarySpec(block_size=64, **common)
            for kind, spec in (("text", text), ("summary", summary)):
                groups.append(KVCacheGroupSpec([f"layer.{index}.{kind}"], spec))
            capacity += min((max_tokens + 7) // 8, window + 1)
            capacity += (max_tokens + 63) // 64
        # One extra page per group permits capacity checks before evicting old
        # pages, so allocation failure cannot damage the committed request.
        capacity += len(groups)
        num_blocks = capacity if num_blocks is None else num_blocks
        page_bytes = groups[0].kv_cache_spec.page_size_bytes
        names = [name for group in groups for name in group.layer_names]
        self.config = KVCacheConfig(
            num_blocks, [KVCacheTensor(num_blocks * page_bytes, names)], groups
        )
        self.manager = KVCacheManager(
            self.config,
            max_model_len=max_tokens,
            scheduler_block_size=64,
            hash_block_size=8,
            enable_caching=False,
        )
        self.storage = torch.empty(
            (num_blocks, 8, 2, heads, head_dim), dtype=dtype, device=device
        )

    def new_request(self):
        return Request(str(next(self.serial)), [], SamplingParams(max_tokens=1), None)

    def read(self, request, layer, layout):
        slots = self.read_slots[request.request_id][layer]
        values = self.storage.flatten(0, 1).index_select(0, slots)
        return values[:, 0], values[:, 1]

    def commit(self, request, end, pending, retained):
        """Commit per-layer KV and masks after a successful model forward.

        With compact_prefill enabled, initial KV contains only retained rows;
        subsequent KV contains all new text and summary rows in the chunk.
        Masks select the next layout from previous rows followed by new rows.
        Allocation failure preserves committed state. A write failure releases
        the request because partially overwritten pages cannot be rolled back.
        """
        if not request.num_computed_tokens < end <= self.max_tokens:
            raise ValueError("invalid KSA page commit length")
        # Conservative preflight before allocate_slots can free skipped pages.
        needed = self.required_blocks(request, end)
        if needed > self.manager.block_pool.get_num_free_blocks():
            raise MemoryError("KSA KV page pool exhausted")
        # A running sliding-window manager normally grows by a small decode
        # step. A completed chunk may jump over entire pages that were never
        # stored: pad those gaps before its length-based allocation fast path.
        for manager in self.manager.coordinator.single_type_managers:
            if request.request_id in manager.num_cached_block:
                table = manager.req_to_blocks[request.request_id]
                skipped = manager.get_num_skipped_tokens(end) // manager.block_size
                table.extend(
                    [self.manager.block_pool.null_block] * max(0, skipped - len(table))
                )
        # Already computed KV enters through the existing external-KV allocation
        # path: it sizes all groups and skips expired text before allocating.
        allocated = self.manager.allocate_slots(
            request,
            num_new_tokens=0,
            num_external_computed_tokens=end - request.num_computed_tokens,
        )
        if allocated is None:
            raise MemoryError("KSA KV page pool exhausted")
        try:
            tables = self.manager.get_blocks(request.request_id).get_block_ids()
            new_positions = [
                (p, False) for p in range(request.num_computed_tokens, end)
            ]
            new_positions = [
                row
                for p, _ in new_positions
                for row in ([(p, False), (p, True)] if p % 8 == 7 else [(p, False)])
            ]
            previous = self.read_slots.get(request.request_id)
            next_slots = []
            slot_ids = []
            for layer in range(len(pending)):
                text_table, summary_table = tables[2 * layer : 2 * layer + 2]
                slot_ids.extend(
                    summary_table[p // 64] * 8 + (p // 8) % 8
                    if summary
                    else text_table[p // 8] * 8 + p % 8
                    for p, summary in new_positions
                )
            slots = torch.tensor(slot_ids, dtype=torch.long, device=self.storage.device)
            valid = [i for i, slot in enumerate(slot_ids) if slot >= 8]
            indices = torch.tensor(valid, dtype=torch.long, device=self.storage.device)
            write_slots = slots.index_select(0, indices)
            if self.batch_writes:
                keys = torch.cat([k for k, _ in pending])
                values = torch.cat([v for _, v in pending])
                rows = torch.stack((keys, values), dim=1)
                if previous is not None or not self.compact_prefill:
                    rows = rows.index_select(0, indices)
                self.storage.flatten(0, 1).index_copy_(0, write_slots, rows)
            else:
                for layer, (k, v) in enumerate(pending):
                    layer_ids = slot_ids[
                        layer * len(new_positions) : (layer + 1) * len(new_positions)
                    ]
                    keep = [i for i, slot in enumerate(layer_ids) if slot >= 8]
                    row_indices = torch.tensor(keep, dtype=torch.long, device=k.device)
                    layer_slots = slots.view(len(pending), -1)[layer].index_select(
                        0, row_indices
                    )
                    if previous is not None or not self.compact_prefill:
                        k, v = (
                            k.index_select(0, row_indices),
                            v.index_select(0, row_indices),
                        )
                    self.storage[layer_slots // 8, layer_slots % 8, 0] = k
                    self.storage[layer_slots // 8, layer_slots % 8, 1] = v
            for layer, slots_for_layer in enumerate(slots.view(len(pending), -1)):
                combined = (
                    slots_for_layer
                    if previous is None
                    else torch.cat((previous[layer], slots_for_layer))
                )
                keep = retained[self.windows[layer]]
                next_slots.append(combined if keep is None else combined[keep])
            self.read_slots[request.request_id] = next_slots
            request.num_computed_tokens = end
        except BaseException:
            self.free(request)
            raise

    def required_blocks(self, request, end):
        """Conservative admission cost before recycling any historical pages."""
        return sum(
            max(
                0,
                (end + manager.block_size - 1) // manager.block_size
                - max(
                    len(table),
                    manager.get_num_skipped_tokens(end) // manager.block_size,
                ),
            )
            for manager, table in zip(
                self.manager.coordinator.single_type_managers,
                self.manager.get_blocks(request.request_id).blocks,
            )
        )

    def free(self, request):
        self.read_slots.pop(request.request_id, None)
        self.manager.free(request)
        request.num_computed_tokens = 0

    def occupancy(self, request, text_tokens):
        tables = self.manager.get_blocks(request.request_id).blocks
        pages = [sum(not b.is_null for b in table) for table in tables]
        rows = [len(slots) for slots in self.read_slots.get(request.request_id, [])]
        row_bytes = self.config.kv_cache_groups[0].kv_cache_spec.page_size_bytes // 8
        return dict(
            layer_rows=rows,
            group_pages=pages,
            effective_kv_bytes=sum(rows) * row_bytes,
            allocated_kv_bytes=sum(pages) * row_bytes * 8,
            pool_bytes=self.storage.numel() * self.storage.element_size(),
        )
