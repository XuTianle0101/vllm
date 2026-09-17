# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape decode graphs; paging and validation remain outside capture."""

import time
from collections import OrderedDict

import torch
from torch.profiler import record_function

from .ksa_decode import KSACache


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
            key = (len(requests), capacity)
        if key not in self.graphs:
            if len(self.graphs) == self.max_graphs:
                self.graphs.popitem(last=False)
            begin = time.perf_counter()
            buffers = _DecodeBuffers(self.model, *key)
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
            self.graphs[key] = (buffers, graph)
        self.graphs.move_to_end(key)
        buffers, graph = self.graphs[key]
        with record_function("ksa.kv_stage"):
            buffers.stage(requests, prepared)
        with record_function("ksa.graph_replay" if graph else "ksa.buffered_eager"):
            if graph:
                graph.replay()
                self.replays += 1
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
