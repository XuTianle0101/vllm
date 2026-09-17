# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Supported configuration for the eager KSA V1 path."""

from vllm import envs


def is_ksa(config) -> bool:
    model = config.model_config
    return model is not None and bool(
        getattr(model.hf_config, "use_summary_attention", False)
    )


def configure_ksa(config) -> None:
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
        "max_model_len > 131072": model.max_model_len > 131072,
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
