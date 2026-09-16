# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KSA eager prefill using Qwen3 components and PyTorch attention."""

import torch

from .ksa_prefill import (
    expand_prefill_sequence,
    prefill_attention,
    prefill_layout,
    validate_ksa_config,
    visibility_mask,
)
from .qwen3 import Qwen3ForCausalLM


class KSAForCausalLM(Qwen3ForCausalLM):
    """Single-request, uncached prefill; invoke through the T01 launcher."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.windows = validate_ksa_config(config, config.num_hidden_layers)
        if (
            vllm_config.parallel_config.pipeline_parallel_size != 1
            or vllm_config.parallel_config.tensor_parallel_size != 1
        ):
            raise ValueError("T01 requires tensor and pipeline parallel sizes of one")
        if vllm_config.quant_config is not None:
            raise ValueError("T01 requires unquantized weights")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.text_row_indices = None
        self.reference_attention = False
        self.layer_observer = None

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if intermediate_tensors is not None:
            raise ValueError("T01 only supports complete, uncached prefill")
        if positions.ndim != 1 or not torch.equal(
            positions, torch.arange(positions.numel(), device=positions.device)
        ):
            raise ValueError("T01 requires one full request with text positions 0..N-1")
        # The normal serving runner would retain an empty KV cache for decode.
        from vllm.forward_context import is_forward_context_available

        if is_forward_context_available():
            raise RuntimeError("T01 is prefill-only; use benchmarks/ksa/prefill.py")
        pos, rows, summary = prefill_layout(positions.numel(), device=positions.device)
        self.text_row_indices = rows
        if inputs_embeds is None:
            if input_ids is None or input_ids.shape != positions.shape:
                raise ValueError("input_ids must match text positions")
            if torch.any(
                (input_ids < 0) | (input_ids >= self.config.summary_token_begin)
            ):
                raise ValueError("text input contains a reserved summary token")
            ids, _ = expand_prefill_sequence(
                input_ids, summary_token=self.config.summary_token_begin
            )
            hidden = self.embed_input_ids(ids)
        else:
            if inputs_embeds.shape != (positions.numel(), self.config.hidden_size):
                raise ValueError("inputs_embeds must contain exactly the text rows")
            ids = positions.new_full(pos.shape, self.config.summary_token_begin)
            hidden = self.embed_input_ids(ids)
            hidden[rows] = inputs_embeds
        masks = {w: visibility_mask(pos, summary, w) for w in set(self.windows)}
        residual = None
        for index, layer in enumerate(self.model.layers):
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)
            attn = layer.self_attn
            qkv, _ = attn.qkv_proj(hidden)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], -1)
            q = attn.q_norm(q.reshape(-1, attn.num_heads, attn.head_dim))
            k = attn.k_norm(k.reshape(-1, attn.num_kv_heads, attn.head_dim))
            q, k = attn.rotary_emb(pos, q.flatten(1), k.flatten(1))
            mask = masks[self.windows[index]]
            output = prefill_attention(
                q.reshape(-1, attn.num_heads, attn.head_dim),
                k.reshape(-1, attn.num_kv_heads, attn.head_dim),
                v.reshape(-1, attn.num_kv_heads, attn.head_dim),
                mask,
                self.reference_attention,
            )
            hidden, _ = attn.o_proj(output.flatten(1))
            hidden, residual = layer.post_attention_layernorm(hidden, residual)
            hidden = layer.mlp(hidden)
            if self.layer_observer is not None:
                self.layer_observer(index, hidden + residual, rows, summary)
        hidden, _ = self.model.norm(hidden, residual)
        return hidden[rows]

    def compute_logits(self, hidden_states):
        logits = super().compute_logits(hidden_states)
        # Match the released model's text-only prediction vocabulary.
        limit = getattr(
            self.config, "truncate_predict_nums", self.config.summary_token_begin
        )
        return logits[..., :limit] if limit > 0 else logits

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
