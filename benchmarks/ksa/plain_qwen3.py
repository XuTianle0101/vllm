# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Qwen3 regression with deterministic tiny weights, no external download."""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from baseline import command, write_json
from prefill import ROOT


def inspect_worker(worker):
    runner = worker.model_runner
    return dict(model=type(runner.model).__name__, runner=type(runner).__name__)


def run(args):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path.insert(0, str(ROOT))
    from vllm import LLM, SamplingParams

    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        eos_token_id=None,
        pad_token_id=0,
        bos_token_id=1,
        tie_word_embeddings=False,
    )
    prompts = [[11, 42, 71, 8], [5] * 17]
    with tempfile.TemporaryDirectory() as directory:
        model = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
        model.save_pretrained(directory)
        model = model.to("cuda")
        with torch.inference_mode():
            expected = [
                model.generate(
                    torch.tensor([ids], device="cuda"),
                    attention_mask=torch.ones(
                        1, len(ids), device="cuda", dtype=torch.long
                    ),
                    max_new_tokens=16,
                    do_sample=False,
                )[0, len(ids) :].tolist()
                for ids in prompts
            ]
        model = model.to("cpu")
        torch.accelerator.empty_cache()
        llm = LLM(
            model=directory,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=128,
            max_num_seqs=2,
            max_num_batched_tokens=128,
            enforce_eager=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.2,
            max_logprobs=256,
        )
        try:
            runtime = llm.collective_rpc(inspect_worker)[0]
            outputs = llm.generate(
                [{"prompt_token_ids": ids} for ids in prompts],
                SamplingParams(
                    temperature=0, max_tokens=16, ignore_eos=True, logprobs=256
                ),
            )
            actual = [o.outputs[0].token_ids for o in outputs]
            model = model.to("cuda")
            comparisons = []
            with torch.inference_mode():
                for prompt, output, generated in zip(prompts, outputs, actual):
                    ids = torch.tensor([prompt + generated[:-1]], device="cuda")
                    logits = model(ids).logits[0, len(prompt) - 1 :].float()
                    reference = logits.log_softmax(-1).cpu()
                    observed = torch.tensor(
                        [
                            [step[token].logprob for token in range(256)]
                            for step in output.outputs[0].logprobs
                        ]
                    )
                    margin = (
                        (
                            logits.max(-1).values
                            - logits.gather(
                                1, torch.tensor(generated, device="cuda")[:, None]
                            )[:, 0]
                        )
                        .max()
                        .item()
                    )
                    error = (reference - observed).abs().max().item()
                    comparisons.append(
                        dict(
                            max_logprob_error=error,
                            max_mismatch_margin=margin,
                            status="pass"
                            if error <= 0.05 and margin <= 0.02
                            else "fail",
                        )
                    )
            passed = (
                all(c["status"] == "pass" for c in comparisons)
                and runtime["model"] == "Qwen3ForCausalLM"
                and runtime["runner"] == "GPUModelRunner"
            )
            write_json(
                args.output / "results.json",
                dict(
                    status="pass" if passed else "fail",
                    runtime=runtime,
                    expected=expected,
                    actual=actual,
                    same_prefix=comparisons,
                    exact_tokens=actual == expected,
                    logprob_atol=0.05,
                    mismatch_margin_atol=0.02,
                    prompts=prompts,
                    git_sha=command("git", "rev-parse", "HEAD"),
                    seed=17,
                    scope="Tiny deterministic native Qwen3 HF/vLLM greedy regression; "
                    "not a pretrained Qwen3 model quality evaluation",
                ),
            )
            if not passed:
                raise AssertionError("Native Qwen3 regression failed")
        finally:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
