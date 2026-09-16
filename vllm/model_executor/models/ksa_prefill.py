# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, explicit-mask Python KSA prefill (one complete text request)."""

import ast

import torch
import torch.nn.functional as F

MAX_PREFILL_TOKENS = 4096
MAX_LAYERS = 256


def parse_layer_windows(expression: str | list[int] | int) -> list[int]:
    """Parse bounded integer lists, concatenation and list repetition only."""

    def integer(value):
        if type(value) is not int or not 0 <= value <= 1_000_000:
            raise ValueError("window values must be non-negative bounded integers")
        return value

    def parse(node):
        if isinstance(node, ast.List):
            if not all(isinstance(x, ast.Constant) for x in node.elts):
                raise ValueError("only flat integer lists are allowed")
            result = [integer(x.value) for x in node.elts]
        elif isinstance(node, ast.Constant):
            result = [integer(node.value)]
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            result = parse(node.left) + parse(node.right)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if not isinstance(node.right, ast.Constant):
                raise ValueError("repeat count must be an integer literal")
            count = integer(node.right.value)
            left = parse(node.left)
            if count > MAX_LAYERS or len(left) * count > MAX_LAYERS:
                raise ValueError("window expression exceeds layer limit")
            result = left * count
        else:
            raise ValueError(
                "only integer lists, concatenation and repeats are allowed"
            )
        if len(result) > MAX_LAYERS:
            raise ValueError("window expression exceeds layer limit")
        return result

    if isinstance(expression, list):
        if len(expression) > MAX_LAYERS:
            raise ValueError("too many layer windows")
        return [integer(x) for x in expression]
    if type(expression) is int:
        return [integer(expression)]
    if not isinstance(expression, str) or len(expression) > 4096:
        raise ValueError("invalid window expression")
    try:
        return parse(ast.parse(expression, mode="eval").body)
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("invalid window expression") from exc


def validate_ksa_config(config: object, expected_layers: int = 36) -> list[int]:
    if not getattr(config, "use_summary_attention", False):
        raise ValueError("KSA configuration must enable summary attention")
    required = {
        "summary_chunk_size": 8,
        "summary_token_num": 1,
        "mix_coeff": 0,
        "summary_chunk_position_ids_type": "origin",
        "summary_token_position_ids_type": "last_chunk_slice_right",
        "summary_independent_attention_layernorm": False,
    }
    for name, value in required.items():
        if getattr(config, name, value) != value:
            raise ValueError(f"T01 requires {name}={value!r}")
    windows = parse_layer_windows(getattr(config, "summary_sliding_chunk_num", None))
    frequency = parse_layer_windows(
        getattr(config, "summary_layer_freq", [1] * expected_layers)
    )
    if len(windows) != expected_layers or frequency != [1] * expected_layers:
        raise ValueError(
            f"expected {expected_layers} windows and summary in every layer"
        )
    token = getattr(config, "summary_token_begin", -1)
    if not 0 <= token < config.vocab_size:
        raise ValueError("summary_token_begin must be inside the vocabulary")
    return windows


def prefill_layout(length: int, block_size: int = 8, device=None):
    """Return rotary positions, text indices and summary flags for internal rows."""
    if not 1 <= length <= MAX_PREFILL_TOKENS:
        raise ValueError(
            f"explicit KSA prefill supports lengths 1..{MAX_PREFILL_TOKENS}"
        )
    if block_size != 8:
        raise ValueError("T01 requires block_size=8")
    rows = torch.arange(length + length // block_size, device=device)
    summary = rows % (block_size + 1) == block_size
    positions = rows - rows // (block_size + 1) - summary.long()
    return positions, rows[~summary], summary


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


def visibility_mask(positions, summary, window):
    """Official predicates: local text, distant summaries, same-block summary Q."""
    blocks = positions // 8
    distance = blocks[:, None] - blocks[None, :]
    rows = torch.arange(positions.numel(), device=positions.device)
    causal = rows[:, None] >= rows[None, :]
    text_query = ((~summary[None, :]) & (distance <= window)) | (
        summary[None, :] & (distance > window)
    )
    # Released T00 semantics include the summary query's own KV.
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
