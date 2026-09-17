# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KSA eager prefill and cached decode using Qwen3 components and PyTorch attention."""

import torch
from torch.profiler import record_function

from .ksa_decode import MAX_CACHED_TEXT_TOKENS, KSACache, cached_layout
from .ksa_prefill import (
    prefill_attention,
    prefill_layout,
    validate_ksa_config,
    visibility_mask,
)
from .qwen3 import Qwen3ForCausalLM


class KSAForCausalLM(Qwen3ForCausalLM):
    """Eager packed projections with request-isolated compressed KV attention."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.max_cached_text_tokens = min(
            MAX_CACHED_TEXT_TOKENS,
            config.max_position_embeddings,
            vllm_config.model_config.max_model_len,
        )
        self.windows = validate_ksa_config(config, config.num_hidden_layers)
        if (
            vllm_config.parallel_config.pipeline_parallel_size != 1
            or vllm_config.parallel_config.tensor_parallel_size != 1
        ):
            raise ValueError("KSA Python prototype requires TP=PP=1")
        if vllm_config.quant_config is not None:
            raise ValueError("KSA Python prototype requires unquantized weights")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.text_row_indices = None
        self.reference_attention = False
        self.triton_attention = True
        self.return_full_vocab_logits = False
        self.layer_observer = None
        limit = getattr(config, "truncate_predict_nums", config.summary_token_begin)
        self.text_vocab_size = min(
            limit if limit > 0 else config.summary_token_begin,
            config.summary_token_begin,
        )

    def new_cache(self):
        from .ksa_cache import KSAPagePool

        if not hasattr(self, "page_pool"):
            attn = self.model.layers[0].self_attn
            parameter = next(self.parameters())
            self.page_pool = KSAPagePool(
                self.windows,
                attn.num_kv_heads,
                attn.head_dim,
                parameter.dtype,
                parameter.device,
                self.max_cached_text_tokens,
            )
        return KSACache(page_pool=self.page_pool, request=self.page_pool.new_request())

    def _use_triton(self, cache, device):
        return (
            self.triton_attention
            and not self.reference_attention
            and device.type == "cuda"
            and (cache is None or cache.page_pool is not None)
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
        from vllm.forward_context import is_forward_context_available

        if is_forward_context_available():
            raise RuntimeError(
                "KSA requires KSAGPUModelRunner or the standalone eager runner; "
                "dense attention metadata cannot represent summary attention"
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
        if not 1 <= positions.numel() <= 4096:
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
                elif start:
                    retained[window] = key_summary | (key_pos // 8 >= first_block)
                else:
                    retained[window] = summary | (pos // 8 >= first_block)
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
        state = self._prepare(
            input_ids, positions, intermediate_tensors, inputs_embeds, cache=cache
        )
        return self._forward_states([state])[0]

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
            from .ksa_attention import KSAAttentionMetadata, paged_attention

            if kernel_metadata is None:
                kernel_metadata = KSAAttentionMetadata.from_states(self.windows, states)
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
                    cache = state["cache"]
                    keep = state["retained"].get(window)
                    if (
                        cache is not None
                        and not state["start"]
                        and cache.page_pool.compact_prefill
                        and keep is not None
                    ):
                        key, value = key[keep], value[keep]
                    state["pending"].append((key, value))
            else:
                outputs = []
                window = self.windows[index]
                for state, query, key, value in zip(
                    states, q.split(sizes), k.split(sizes), v.split(sizes)
                ):
                    cache = state["cache"]
                    keep = state["retained"].get(window)
                    if cache is not None and cache.page_pool is not None:
                        if state["start"] or not cache.page_pool.compact_prefill:
                            state["pending"].append((key, value))
                        else:
                            state["pending"].append(
                                (key, value.clone())
                                if keep is None
                                else (key[keep], value[keep])
                            )
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
