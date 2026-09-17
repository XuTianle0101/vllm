# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager KSA continuous batching over the shared vLLM compressed page allocator.

This bounded Python runner deliberately owns its scheduling loop: the generic
runner allocates dense Qwen KV and cannot represent KSA's two cache groups yet.
"""

from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from vllm.model_executor.models.ksa_decode import MAX_CACHED_TEXT_TOKENS, KSACache


def internal_rows(start: int, length: int) -> int:
    return length + (start + length) // 8 - start // 8


def text_budget(start: int, available: int, rows: int) -> int:
    """Largest contiguous text chunk whose summaries also fit the budget."""
    low, high = 0, min(available, rows)
    while low < high:
        mid = (low + high + 1) // 2
        if internal_rows(start, mid) <= rows:
            low = mid
        else:
            high = mid - 1
    return low


@dataclass
class KSARequest:
    request_id: str
    prompt: list[int]
    max_tokens: int
    eos: frozenset[int]
    cache: KSACache
    prompt_logprobs: list[float | None] | None
    output: list[int] = field(default_factory=list)
    preemptions: int = 0

    @property
    def history(self):
        return self.prompt + self.output


@dataclass
class KSAOutput:
    request_id: str
    token_ids: list[int]
    prompt_tokens: int
    finish_reason: str | None
    prompt_logprobs: list[float | None] | None = None
    error: str | None = None

    @property
    def usage(self):
        return dict(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=len(self.token_ids),
            total_tokens=self.prompt_tokens + len(self.token_ids),
        )


class KSABatchedRunner:
    """FCFS continuous batching, greedy sampling, chunking and recomputation.

    All methods must be called on the same execution thread. Each step returns
    cumulative text-token outputs only for requests that emitted or finished.
    Cancelled requests release pages immediately and produce no further output.
    Page pressure preempts later requests, retaining their text history. CUDA
    workspace OOM fails the selected requests and frees their pages; it is not
    retried indefinitely. A request that cannot fit alone fails explicitly.
    """

    def __init__(
        self,
        model,
        *,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        chunk_size=256,
        enable_prefix_caching=False,
        kv_cache_num_blocks=None,
    ):
        if enable_prefix_caching:
            raise ValueError("KSA prefix caching is unsupported")
        if any(
            type(v) is not int or v < 1
            for v in (max_num_seqs, max_num_batched_tokens, chunk_size)
        ):
            raise ValueError("KSA scheduling limits must be positive integers")
        if max_num_batched_tokens < 2 or chunk_size > 4096:
            raise ValueError("KSA requires a row budget >=2 and chunk_size <=4096")
        self.model = model
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.chunk_size = chunk_size
        self.requests: OrderedDict[str, KSARequest] = OrderedDict()
        self.device = next(model.parameters()).device
        model.new_cache().clear()
        if kv_cache_num_blocks is not None:
            from vllm.model_executor.models.ksa_cache import KSAPagePool

            old = model.page_pool
            if old.read_slots:
                raise ValueError("cannot resize a KSA pool with active requests")
            if type(kv_cache_num_blocks) is not int or kv_cache_num_blocks < 2:
                raise ValueError("kv_cache_num_blocks must be an integer >=2")
            model.page_pool = KSAPagePool(
                model.windows,
                old.storage.shape[-2],
                old.storage.shape[-1],
                old.storage.dtype,
                old.storage.device,
                old.max_tokens,
                num_blocks=kv_cache_num_blocks,
            )
        self.pool = model.page_pool
        self.last_step_rows = 0
        self.last_step_text_tokens = 0
        self.num_preemptions = 0

    def add_request(
        self,
        request_id,
        input_ids,
        *,
        max_tokens,
        eos_token_ids=(),
        ignore_eos=False,
        prompt_logprobs=False,
    ):
        if request_id in self.requests:
            raise ValueError("duplicate KSA request ID")
        ids = list(input_ids)
        limit = min(MAX_CACHED_TEXT_TOKENS, self.model.config.max_position_embeddings)
        if not ids or any(
            type(t) is not int or not 0 <= t < self.model.text_vocab_size for t in ids
        ):
            raise ValueError("KSA requires nonempty text token IDs")
        if (
            type(max_tokens) is not int
            or max_tokens < 0
            or len(ids) + max_tokens > limit
        ):
            raise ValueError(f"KSA prompt plus max_tokens must fit {limit} text tokens")
        eos = frozenset(eos_token_ids)
        if any(
            type(t) is not int or not 0 <= t < self.model.text_vocab_size for t in eos
        ):
            raise ValueError("EOS IDs must belong to the text prediction vocabulary")
        self.requests[request_id] = KSARequest(
            request_id,
            ids,
            max_tokens,
            frozenset() if ignore_eos else eos,
            self.model.new_cache(),
            [None] * len(ids) if prompt_logprobs else None,
        )

    def abort_request(self, request_id):
        request = self.requests.pop(request_id, None)
        if request is not None:
            request.cache.clear()

    def close(self):
        for request_id in list(self.requests):
            self.abort_request(request_id)

    def preempt(self, request_id):
        request = self.requests[request_id]
        request.cache.clear()
        request.preemptions += 1
        self.num_preemptions += 1

    def _output(self, request, reason=None, error=None):
        output = KSAOutput(
            request.request_id,
            list(request.output),
            len(request.prompt),
            reason,
            None if request.prompt_logprobs is None else list(request.prompt_logprobs),
            error,
        )
        if reason is not None:
            self.abort_request(request.request_id)
        return output

    @torch.inference_mode()
    def step(self):
        outputs = []
        selected: list[tuple[KSARequest, int, int]] = []
        rows = self.max_num_batched_tokens
        reserved = 0
        self.last_step_rows = self.last_step_text_tokens = 0
        for request in list(self.requests.values()):
            if len(selected) >= self.max_num_seqs:
                break
            if request.max_tokens == 0 and request.prompt_logprobs is None:
                outputs.append(self._output(request, "length"))
                continue
            start = request.cache.text_tokens
            length = text_budget(
                start, min(self.chunk_size, len(request.history) - start), rows
            )
            if not length:
                continue
            needed = self.pool.required_blocks(request.cache.request, start + length)
            free = self.pool.manager.block_pool.get_num_free_blocks()
            if needed + reserved > free and not selected:
                # Oldest request has priority. Recompute later requests only
                # when their resident pages prevent this request progressing.
                for victim in reversed(list(self.requests.values())):
                    if victim is request or not victim.cache.text_tokens:
                        continue
                    self.preempt(victim.request_id)
                    free = self.pool.manager.block_pool.get_num_free_blocks()
                    if needed <= free:
                        break
                while needed > free and length > 1:
                    length = max(1, length // 2)
                    needed = self.pool.required_blocks(
                        request.cache.request, start + length
                    )
                if needed > free:
                    outputs.append(
                        self._output(
                            request,
                            "error",
                            "KSA KV page pool exhausted for one request",
                        )
                    )
                    continue
            elif needed + reserved > free:
                continue
            selected.append((request, start, length))
            reserved += needed
            rows -= internal_rows(start, length)
        self.last_step_rows = self.max_num_batched_tokens - rows
        self.last_step_text_tokens = sum(length for _, _, length in selected)
        if not selected:
            return outputs
        try:
            hidden = self.model.forward_batch(
                [
                    (
                        torch.tensor(
                            request.history[start : start + length], device=self.device
                        ),
                        torch.arange(start, start + length, device=self.device),
                        request.cache,
                    )
                    for request, start, length in selected
                ]
            )
            for (request, start, length), values in zip(selected, hidden):
                end = start + length
                need_prompt = (
                    request.prompt_logprobs is not None
                    and start < len(request.prompt) - 1
                )
                logits = self.model.compute_logits(
                    values if need_prompt else values[-1:]
                )
                if need_prompt:
                    assert request.prompt_logprobs is not None
                    count = min(end, len(request.prompt) - 1) - start
                    targets = torch.tensor(
                        request.prompt[start + 1 : start + count + 1],
                        device=self.device,
                    )
                    logprobs = (
                        logits[:count]
                        .float()
                        .log_softmax(-1)
                        .gather(1, targets[:, None])[:, 0]
                        .tolist()
                    )
                    request.prompt_logprobs[start + 1 : start + count + 1] = logprobs
                if end < len(request.history):
                    continue
                if request.max_tokens == 0:
                    outputs.append(self._output(request, "length"))
                    continue
                token = logits[-1].argmax().item()
                request.output.append(token)
                reason = (
                    "stop"
                    if token in request.eos
                    else "length"
                    if len(request.output) >= request.max_tokens
                    else None
                )
                outputs.append(self._output(request, reason))
        except (MemoryError, torch.OutOfMemoryError) as exc:
            outputs.extend(
                self._output(request, "error", str(exc))
                for request, _, _ in selected
                if request.request_id in self.requests
            )
        except BaseException:
            # A partial batch commit must never survive an execution failure.
            for request, _, _ in selected:
                self.abort_request(request.request_id)
            raise
        return outputs
