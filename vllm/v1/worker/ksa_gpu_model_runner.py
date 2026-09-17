# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KSA execution with the V1 scheduler, input batch and sampler.

Only the scheduler owns page allocation. The worker maps its block tables to
text/summary storage; it never runs a second allocator or scheduling loop.
"""

from types import SimpleNamespace

import numpy as np
import torch

from vllm.model_executor.models.ksa_cache import KSASummarySpec
from vllm.model_executor.models.ksa_decode import (
    KSACache,
    cached_layout,
    retained_layout,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.worker.gpu_model_runner import ExecuteModelState, GPUModelRunner


def ksa_cache_specs(windows, heads, head_dim, dtype):
    specs: dict[str, KVCacheSpec] = {}
    for layer, window in enumerate(windows):
        common = dict(num_kv_heads=heads, head_size=head_dim, dtype=dtype)
        # Preserve the entire earliest visible text block before this chunk.
        specs[f"layer.{layer}.text"] = SlidingWindowSpec(
            block_size=8, sliding_window=window * 8 + 8, **common
        )
        specs[f"layer.{layer}.summary"] = KSASummarySpec(block_size=64, **common)
    return specs


class KSASchedulerPagePool:
    """Views of V1-owned pages, with request-local row maps only."""

    compact_prefill = False

    def __init__(self, config, windows, device):
        self.windows = windows
        self.read_slots = {}
        self.tables = {}
        self.storage = {}
        self.groups = {}
        for gid, group in enumerate(config.kv_cache_groups):
            spec = group.kv_cache_spec
            for name in group.layer_names:
                self.groups[name] = gid
        for tensor in config.kv_cache_tensors:
            name = tensor.shared_by[0]
            spec = config.kv_cache_groups[self.groups[name]].kv_cache_spec
            storage = torch.empty(
                tensor.size // torch.empty((), dtype=spec.dtype).element_size(),
                device=device,
                dtype=spec.dtype,
            ).view(config.num_blocks, 8, 2, spec.num_kv_heads, spec.head_size)
            for name in tensor.shared_by:
                self.storage[name] = storage

    def read(self, request, layer, layout):
        text_slots, summary_slots, summary = self.read_slots[request.request_id][layer]
        text = self.storage[f"layer.{layer}.text"].flatten(0, 1)[text_slots]
        summaries = self.storage[f"layer.{layer}.summary"].flatten(0, 1)[summary_slots]
        values = text.new_empty((len(summary), *text.shape[1:]))
        values[~summary] = text
        values[summary] = summaries
        return values[:, 0], values[:, 1]

    def _slots(self, req_id, layer, positions, summary):
        slots = []
        for kind, mask, block_size in (("text", ~summary, 8), ("summary", summary, 64)):
            name = f"layer.{layer}.{kind}"
            table = torch.tensor(
                self.tables[req_id][self.groups[name]],
                device=positions.device,
                dtype=torch.long,
            )
            pos = positions[mask]
            blocks = table[pos // block_size]
            if torch.any(blocks == 0):
                raise RuntimeError("KSA attempted to access a freed scheduler page")
            offset = pos % 8 if kind == "text" else (pos // 8) % 8
            slots.append(blocks * 8 + offset)
        return (*slots, summary)

    def commit(self, request, end, pending, retained):
        positions, _, summary = cached_layout(
            request.num_computed_tokens,
            end - request.num_computed_tokens,
            pending[0][0].device,
        )
        next_slots = []
        for layer, (key, value) in enumerate(pending):
            text_slots, summary_slots, _ = self._slots(
                request.request_id, layer, positions, summary
            )
            rows = torch.stack((key, value), dim=1)
            for kind, slots, mask in (
                ("text", text_slots, ~summary),
                ("summary", summary_slots, summary),
            ):
                self.storage[f"layer.{layer}.{kind}"].flatten(0, 1).index_copy_(
                    0, slots, rows[mask]
                )
            pos, is_summary, _ = retained_layout(end, self.windows[layer], key.device)
            next_slots.append(self._slots(request.request_id, layer, pos, is_summary))
        self.read_slots[request.request_id] = next_slots
        request.num_computed_tokens = end

    def free(self, request):
        self.read_slots.pop(request.request_id, None)
        self.tables.pop(request.request_id, None)
        request.num_computed_tokens = 0


class _KSAProfilePool:
    """Profile historical gathers without counting scheduler KV as activations."""

    compact_prefill = False

    def __init__(self, heads, head_dim, dtype, device):
        self.shape = (2, heads, head_dim)
        self.dtype = dtype
        self.device = device

    def read(self, request, layer, layout):
        # Real reads materialize gathered text/summary and a merged KV buffer.
        values = torch.zeros(
            (len(layout[0]), *self.shape), dtype=self.dtype, device=self.device
        ).clone()
        return values[:, 0], values[:, 1]

    def commit(self, request, end, pending, retained):
        pass


class KSAGPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        self.ksa_caches = {}
        self.ksa_graphs = None

    def load_model(self, load_dummy_weights=False):
        super().load_model(load_dummy_weights)
        self.model.return_full_vocab_logits = True
        extra = self.vllm_config.additional_config
        if isinstance(extra, dict) and extra.get("ksa_cudagraph", False):
            from vllm.model_executor.models.ksa_graph import KSADecodeGraphs

            self.ksa_graphs = KSADecodeGraphs(self.model)

    def get_kv_cache_spec(self):
        attn = self.model.model.layers[0].self_attn
        return ksa_cache_specs(
            self.model.windows, attn.num_kv_heads, attn.head_dim, self.dtype
        )

    def initialize_kv_cache(self, kv_cache_config, is_profiling=False):
        self.kv_cache_config = kv_cache_config
        self.ksa_pool = KSASchedulerPagePool(
            kv_cache_config, self.model.windows, self.device
        )
        self._kernel_block_sizes = [
            group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
        ]
        self.may_reinitialize_input_batch(kv_cache_config, self._kernel_block_sizes)

    @torch.inference_mode()
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        if self.execute_model_state is not None:
            raise RuntimeError("sample_tokens must follow KSA execute_model")
        discarded = (
            scheduler_output.finished_req_ids
            | scheduler_output.scheduled_cached_reqs.resumed_req_ids
            | (scheduler_output.preempted_req_ids or set())
        )
        for req_id in discarded:
            if cache := self.ksa_caches.pop(req_id, None):
                cache.clear()
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT
        counts = np.array(
            [
                scheduler_output.num_scheduled_tokens[r]
                for r in self.input_batch.req_ids
            ],
            dtype=np.int32,
        )
        logits_indices, _ = self._prepare_inputs(scheduler_output, counts)
        requests = []
        offset = 0
        for req_id, length in zip(self.input_batch.req_ids, counts.tolist()):
            state = self.requests[req_id]
            cache = self.ksa_caches.get(req_id)
            if cache is None:
                cache = KSACache(
                    page_pool=self.ksa_pool,
                    request=SimpleNamespace(request_id=req_id, num_computed_tokens=0),
                )
                self.ksa_caches[req_id] = cache
            if cache.text_tokens != state.num_computed_tokens:
                raise RuntimeError("KSA worker cache is out of sync with V1 scheduler")
            self.ksa_pool.tables[req_id] = state.block_ids
            requests.append(
                (
                    self.input_ids.gpu[offset : offset + length],
                    self.positions[offset : offset + length],
                    cache,
                )
            )
            offset += length
        executor = self.model
        if self.ksa_graphs is not None and 2 * len(requests) <= self.max_num_tokens:
            executor = self.ksa_graphs
        hidden = torch.cat(executor.forward_batch(requests))
        sample_hidden = hidden[logits_indices]
        logits = self.model.compute_logits(sample_hidden)
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            None,
            None,
            hidden,
            sample_hidden,
            None,
            None,
            None,
            None,
        )
        return None

    @torch.inference_mode()
    def _dummy_run(self, num_tokens, **kwargs):
        # Exercise the actual eager path, including hidden summary rows. Profiling
        # also includes the largest historical gather and pending per-layer KV.
        profile = kwargs.get("is_profile", False)
        num_reqs = min(self.max_num_reqs, max(1, num_tokens // 2))
        text_tokens = max(1, num_tokens * 8 // 9 - num_reqs)
        chunks = [
            text_tokens // num_reqs + (i < text_tokens % num_reqs)
            for i in range(num_reqs)
        ]
        chunks = [min(4096, self.max_model_len, n) for n in chunks if n]
        attn = self.model.model.layers[0].self_attn
        profile_pool = _KSAProfilePool(
            attn.num_kv_heads, attn.head_dim, self.dtype, self.device
        )
        requests = []
        for length in chunks:
            start = self.max_model_len - length if profile else 0
            cache = KSACache(
                text_tokens=start, page_pool=profile_pool if profile else None
            )
            if start:
                for window in set(self.model.windows):
                    pos, summary, _ = retained_layout(start, window, self.device)
                    cache.layouts[window] = (pos, summary)
            requests.append(
                (
                    torch.zeros(length, dtype=torch.long, device=self.device),
                    torch.arange(start, start + length, device=self.device),
                    cache,
                )
            )
        outputs = self.model.forward_batch(requests)
        return torch.cat(outputs), torch.stack([h[-1] for h in outputs])

    def profile_run(self):
        hidden, last = self._dummy_run(self.max_num_tokens, is_profile=True)
        self._dummy_sampler_run(hidden_states=last)
        self._sync_device()
        del hidden, last

    def shutdown(self):
        for cache in self.ksa_caches.values():
            cache.clear()
        self.ksa_caches.clear()
        if self.ksa_graphs is not None:
            self.ksa_graphs.graphs.clear()
        super().shutdown()
