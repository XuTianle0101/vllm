# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate KSA graph replay on real V1-owned pages and scheduler slots."""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from hf_reference import ROOT, compare, validate, write_json


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


def set_timing_mode(worker, mode):
    runner = worker.model_runner
    runner.model.triton_attention = mode != "T05_eager"
    runner.ksa_graphs = runner.ksa_saved_graphs if mode == "triton_graph" else None
    return len(runner.ksa_saved_graphs.startup)


def run(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path.insert(0, str(ROOT))
    from vllm import LLM, SamplingParams

    args.output.mkdir(parents=True, exist_ok=False)
    baseline, _, inputs = validate(args)
    names = ["length-7", "length-8", "length-1023", "length-1024"]
    if args.all_cases:
        names = list(inputs)
    prompts = [{"prompt_token_ids": inputs[n]["input_ids"]} for n in names]
    llm = LLM(
        model=str(args.model),
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=131072 if args.all_cases else 8192,
        max_num_seqs=4,
        max_num_batched_tokens=4096 if args.all_cases else 257,
        gpu_memory_utilization=0.5,
        compilation_config={"mode": 0, "custom_ops": ["none"]},
        additional_config={"ksa_cudagraph": True},
    )
    results, captures, runtime, retrieval = [], [], [], []
    try:
        for repeat, mode in enumerate(("eager", "graph", "graph")):
            if args.compare_t05:
                llm.collective_rpc(set_attention, args=(mode != "eager",))
            llm.collective_rpc(set_mode, args=(mode,))
            order = list(range(len(prompts)))
            if repeat % 2:
                order.reverse()
            outputs = llm.generate(
                [prompts[i] for i in order],
                SamplingParams(
                    temperature=0,
                    max_tokens=128 if args.all_cases else 32,
                    ignore_eos=True,
                ),
            )
            results.append(
                {i: list(out.outputs[0].token_ids) for i, out in zip(order, outputs)}
            )
            for i, out in zip(order, outputs):
                answer = inputs[names[i]].get("expected_answer")
                if answer is not None:
                    retrieval.append(
                        dict(
                            case=names[i],
                            mode=mode,
                            repeat=repeat,
                            text=out.outputs[0].text,
                            expected_answer=answer,
                            status="pass" if answer in out.outputs[0].text else "fail",
                        )
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
                and all(r["status"] == "pass" for r in retrieval)
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
                retrieval=retrieval,
            ),
        )
        if status != "pass":
            raise AssertionError("V1 graph validation failed")
        if args.timing_repeats:
            timings = []
            tokens = inputs["length-1024"]["input_ids"]
            timed_prompts = [
                {"prompt_token_ids": tokens[: len(tokens) - i]} for i in range(4)
            ]
            for mode in ("T05_eager", "triton_eager", "triton_graph"):
                for repeat in range(-1, args.timing_repeats):
                    before = llm.collective_rpc(set_timing_mode, args=(mode,))[0]
                    begin = time.perf_counter()
                    output = llm.generate(
                        timed_prompts,
                        SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
                        use_tqdm=False,
                    )
                    elapsed = (time.perf_counter() - begin) * 1000
                    after = llm.collective_rpc(set_timing_mode, args=(mode,))[0]
                    if [len(o.outputs[0].token_ids) for o in output] != [32] * 4:
                        raise AssertionError("V1 timed generation truncated")
                    timings.append(
                        dict(
                            mode=mode,
                            repetition=repeat,
                            elapsed_ms=elapsed,
                            new_captures=after - before,
                        )
                    )
                    write_json(args.output / "timing.json", timings)
                print(f"V1 timing {mode}", flush=True)

    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument(
        "--compare-t05",
        action="store_true",
        help="Compare Triton graphs to T05 SDPA on real V1 pages",
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=0,
        help="Optional uninstrumented 1K/batch=4 service-core timings",
    )
    args = parser.parse_args()
    if args.timing_repeats and args.timing_repeats < 5:
        parser.error("require zero or at least five timing repeats")
    run(args)
