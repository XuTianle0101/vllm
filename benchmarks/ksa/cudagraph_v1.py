# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate KSA graph replay on real V1-owned pages and scheduler slots."""

import argparse
import os
import sys
from pathlib import Path

import torch
from baseline import write_json
from decode import validate
from prefill import ROOT, compare


def set_attention(worker, enabled):
    worker.model_runner.model.triton_attention = enabled


def set_mode(worker, mode):
    runner = worker.model_runner
    if not hasattr(runner, "ksa_saved_graphs"):
        runner.ksa_saved_graphs = runner.ksa_graphs
    graphs = runner.ksa_saved_graphs
    runner.ksa_graphs = graphs if mode == "graph" else None
    executor = runner.ksa_graphs or runner.model
    runner.ksa_observed_executor = executor
    runner.ksa_original_forward = executor.forward_batch
    runner.ksa_observed = {}

    def observed(requests):
        outputs = runner.ksa_original_forward(requests)
        for (_, positions, cache), hidden in zip(requests, outputs):
            prompt = runner.requests[cache.request.request_id].prompt_token_ids
            if cache.text_tokens < len(prompt):
                continue
            logits = runner.model.compute_logits(hidden[-1:])
            runner.ksa_observed.setdefault(tuple(prompt), {})[int(positions[-1])] = (
                logits[0, : runner.model.text_vocab_size].float().cpu()
            )
        return outputs

    executor.forward_batch = observed


def export(worker, path):
    runner = worker.model_runner
    torch.save(runner.ksa_observed, path)
    runner.ksa_observed_executor.forward_batch = runner.ksa_original_forward
    del runner.ksa_original_forward
    del runner.ksa_observed_executor
    del runner.ksa_observed
    return dict(
        replays=runner.ksa_saved_graphs.replays,
        captures=len(runner.ksa_saved_graphs.startup),
    )


def run(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path.insert(0, str(ROOT))
    from vllm import LLM, SamplingParams

    args.output.mkdir(parents=True, exist_ok=False)
    baseline, _, inputs = validate(args)
    names = ["length-7", "length-8", "length-1023", "length-1024"]
    prompts = [{"prompt_token_ids": inputs[n]["input_ids"]} for n in names]
    llm = LLM(
        model=str(args.model),
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=8192,
        max_num_seqs=4,
        max_num_batched_tokens=257,
        gpu_memory_utilization=0.5,
        compilation_config={"mode": 0, "custom_ops": ["none"]},
        additional_config={"ksa_cudagraph": True},
    )
    results, captures, runtime = [], [], []
    try:
        for repeat, mode in enumerate(("eager", "graph", "graph")):
            if args.compare_t05:
                llm.collective_rpc(set_attention, args=(mode != "eager",))
            llm.collective_rpc(set_mode, args=(mode,))
            order = list(range(4)) if repeat % 2 == 0 else list(reversed(range(4)))
            outputs = llm.generate(
                [prompts[i] for i in order],
                SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
            )
            results.append(
                {i: list(out.outputs[0].token_ids) for i, out in zip(order, outputs)}
            )
            path = (args.output / f"{mode}-{repeat}.pt").resolve()
            runtime.append(llm.collective_rpc(export, args=(str(path),))[0])
            captures.append(torch.load(path, weights_only=True))
        rows = []
        for repeat in (1, 2):
            for i, prompt in enumerate(prompts):
                tokens = prompt["prompt_token_ids"]
                expected, actual = results[0][i], results[repeat][i]
                first = next(
                    (j for j, (a, b) in enumerate(zip(expected, actual)) if a != b),
                    len(expected) - 1,
                )
                positions = list(range(len(tokens) - 1, len(tokens) + first))
                ref = torch.stack([captures[0][tuple(tokens)][p] for p in positions])
                got = torch.stack(
                    [captures[repeat][tuple(tokens)][p] for p in positions]
                )
                rows.append(
                    dict(
                        case=names[i],
                        repeat=repeat,
                        exact_tokens=expected == actual,
                        shared_prefix_steps=len(positions),
                        **compare(torch, ref, got, positions, baseline["thresholds"]),
                    )
                )
        status = (
            "pass"
            if (
                all(r["status"] == "pass" for r in rows)
                and runtime[2]["replays"] > runtime[1]["replays"] > 0
            )
            else "fail"
        )
        write_json(
            args.output / "results.json",
            dict(
                status=status,
                git_sha=args.expected_sha,
                reference="T05 attention" if args.compare_t05 else "eager",
                cases=rows,
                runtime=runtime,
                generations=results,
            ),
        )
        if status != "pass":
            raise AssertionError("V1 graph validation failed")
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument(
        "--compare-t05",
        action="store_true",
        help="Compare Triton graphs to T05 SDPA on real V1 pages",
    )
    run(parser.parse_args())
