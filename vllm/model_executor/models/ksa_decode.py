# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eager, single-request KSA generation with windowed KV storage."""

from dataclasses import dataclass, field

import torch

MAX_CACHED_TEXT_TOKENS = 8192


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
        return self.text_tokens + self.text_tokens // 8

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
        rows = torch.arange(length + length // 8, device=device)
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


@dataclass
class KSAGenerationOutput:
    token_ids: list[int]
    prompt_tokens: int
    finish_reason: str

    @property
    def completion_tokens(self):
        return len(self.token_ids)


class KSAPythonRunner:
    """T02 greedy runner, outside the scheduler. One request per invocation.

    EOS is counted in token_ids, as a generated text-vocabulary token. Decode
    these IDs with skip_special_tokens=True for display. Internal summaries
    never enter token_ids or either usage count.
    """

    def __init__(self, model, *, paged=True):
        self.model = model
        self.paged = paged

    @torch.inference_mode()
    def generate(self, input_ids, *, max_tokens, eos_token_ids=(), ignore_eos=False):
        if input_ids.ndim != 1 or input_ids.numel() == 0:
            raise ValueError("T02 requires one nonempty 1-D text request")
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if input_ids.numel() > 4096:
            raise ValueError("T02 prefill supports at most 4096 text tokens")
        limit = min(MAX_CACHED_TEXT_TOKENS, self.model.config.max_position_embeddings)
        if input_ids.numel() + max_tokens > limit:
            raise ValueError(f"T02 prompt plus max_tokens exceeds {limit}")
        if input_ids.dtype not in (torch.int32, torch.int64) or torch.any(
            (input_ids < 0) | (input_ids >= self.model.config.summary_token_begin)
        ):
            raise ValueError("text input contains invalid or reserved token IDs")
        eos = set(eos_token_ids)
        if any(
            type(t) is not int or not 0 <= t < self.model.text_vocab_size for t in eos
        ):
            raise ValueError("EOS IDs must belong to the text prediction vocabulary")
        output = KSAGenerationOutput([], input_ids.numel(), "length")
        if max_tokens == 0:
            return output
        cache = self.model.new_cache() if self.paged else KSACache()
        tokens = input_ids
        try:
            for _ in range(max_tokens):
                positions = torch.arange(
                    cache.text_tokens,
                    cache.text_tokens + tokens.numel(),
                    device=tokens.device,
                )
                hidden = self.model(tokens, positions, cache=cache)
                # forward returns only text rows, including at a block boundary.
                tokens = self.model.compute_logits(hidden[-1:]).argmax(-1)
                token = tokens.item()
                output.token_ids.append(token)
                if not ignore_eos and token in eos:
                    output.finish_reason = "stop"
                    break
        finally:
            cache.clear()
        return output
