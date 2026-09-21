# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KSA checkpoint semantics and supported V1 execution contract.

This module is CPU-only: registry, config and scheduler must not import model
or worker implementations to identify KSA or account for internal rows.
"""

import ast
from typing import Any, cast

from vllm import envs

SUMMARY_CHUNK_SIZE = 8
MAX_PREFILL_TOKENS = 4096
MAX_CACHED_TEXT_TOKENS = 131072
MAX_LAYERS = 256


def uses_summary_attention(hf_config: object) -> bool:
    """Identify KSA by the checkpoint flag, independently of architecture name."""
    return bool(getattr(hf_config, "use_summary_attention", False))


def is_ksa(config) -> bool:
    model = config.model_config
    return model is not None and uses_summary_attention(model.hf_config)


def ksa_architectures(architectures: list[str], model_config) -> list[str]:
    """Route released Qwen3-named checkpoints; preserve explicit transformers."""
    if (
        uses_summary_attention(model_config.hf_config)
        and "Qwen3ForCausalLM" in architectures
        and model_config.model_impl != "transformers"
    ):
        return ["KSAForCausalLM"]
    return architectures


def internal_row_count(start: int, length: int) -> int:
    """Rows for a non-negative text span, including each completed summary."""
    return length + (start + length) // SUMMARY_CHUNK_SIZE - start // SUMMARY_CHUNK_SIZE


def fit_text_tokens(start: int, length: int, budget: int) -> int:
    """Largest text span fitting the row budget and the per-request chunk cap.

    Inputs are non-negative text offset, available text length and internal-row
    budget. Zero means defer this request (even one text token may need two rows).
    This CPU-only admission check does not allocate pages or advance a prefix.
    """
    low, high = 0, min(length, budget, MAX_PREFILL_TOKENS)
    while low < high:
        mid = (low + high + 1) // 2
        if internal_row_count(start, mid) <= budget:
            low = mid
        else:
            high = mid - 1
    return low


def parse_layer_windows(expression: object) -> list[int]:
    """Parse bounded integer lists, concatenation and list repetition only."""

    def integer(value):
        if type(value) is not int or not 0 <= value <= 1_000_000:
            raise ValueError("window values must be non-negative bounded integers")
        return value

    def parse(node):
        if isinstance(node, ast.List):
            if not all(isinstance(x, ast.Constant) for x in node.elts):
                raise ValueError("only flat integer lists are allowed")
            result = [integer(cast(ast.Constant, x).value) for x in node.elts]
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


def validate_ksa_config(config: Any, expected_layers: int = 36) -> list[int]:
    if not uses_summary_attention(config):
        raise ValueError("KSA configuration must enable summary attention")
    required = {
        "summary_chunk_size": SUMMARY_CHUNK_SIZE,
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


def configure_ksa(config) -> None:
    """Validate KSA before generic config normalization, then select sync HMA.

    Raise one ValueError listing unsupported settings before mutating anything.
    ``ksa_cudagraph`` remains a worker-local decode option with enforce_eager;
    it is deliberately independent of standard compilation/cudagraph settings.
    """
    model = config.model_config
    scheduler = config.scheduler_config
    parallel = config.parallel_config
    row_budget = (
        scheduler.max_num_batched_tokens
        if scheduler.max_num_scheduled_tokens is None
        else scheduler.max_num_scheduled_tokens
    )
    unsupported = {
        "--enforce-eager is required": not model.enforce_eager,
        "--no-enable-prefix-caching is required": (
            config.cache_config.enable_prefix_caching
        ),
        "TP=PP=DP=1 is required": any(
            size != 1
            for size in (
                parallel.tensor_parallel_size,
                parallel.pipeline_parallel_size,
                parallel.data_parallel_size,
                parallel.decode_context_parallel_size,
                parallel.prefill_context_parallel_size,
            )
        ),
        "V2 model runner": envs.VLLM_USE_V2_MODEL_RUNNER is True,
        "async scheduling": scheduler.async_scheduling is True,
        "custom scheduler": scheduler.scheduler_cls is not None,
        "disabled hybrid KV manager": scheduler.disable_hybrid_kv_cache_manager is True,
        "unchunked prefill": not scheduler.enable_chunked_prefill,
        "speculative decoding": config.speculative_config is not None,
        "LoRA": config.lora_config is not None,
        "quantization": model.quantization is not None,
        "KV transfer": config.kv_transfer_config is not None,
        "KV offloading": config.cache_config.kv_offloading_size is not None,
        "sleep mode": model.enable_sleep_mode,
        "KV sharing": config.cache_config.kv_sharing_fast_prefill,
        "non-auto KV dtype": config.cache_config.cache_dtype != "auto",
        "prompt embeddings": model.enable_prompt_embeds,
        "pooling": model.runner_type != "generate",
        "max_model_len > 131072": model.max_model_len > MAX_CACHED_TEXT_TOKENS,
        "row budget < 2": row_budget < 2,
        "scheduled budget exceeds worker capacity": (
            row_budget > scheduler.max_num_batched_tokens
        ),
    }
    errors = [name for name, enabled in unsupported.items() if enabled]
    if errors:
        raise ValueError("Unsupported KSA V1 configuration: " + "; ".join(errors))
    scheduler.async_scheduling = False
    scheduler.disable_hybrid_kv_cache_manager = False
