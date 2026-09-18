# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard V1 performance using the fixed refactor baseline protocol."""

import argparse
import csv
import importlib.metadata as metadata
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from hf_reference import (
    BASELINE_ID,
    FIXTURES,
    ROOT,
    TOLERANCE_ID,
    command,
    read,
    validate,
    write_json,
)


def telemetry(worker, reset=False):
    import torch

    runner = worker.model_runner
    graphs = runner.ksa_graphs
    result = dict(
        captures=[] if graphs is None else list(graphs.startup),
        peak_memory_bytes=torch.accelerator.max_memory_allocated(),
    )
    if reset:
        torch.accelerator.reset_peak_memory_stats()
    return result


def worker(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path.insert(0, str(ROOT))
    from vllm import LLM, SamplingParams

    args.output.mkdir(parents=True, exist_ok=False)
    _, _, inputs = validate(args)
    tokens = inputs[f"length-{args.length}"]["input_ids"]
    begin = time.perf_counter()
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=args.length + 128,
        max_num_seqs=args.batch,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.85,
        compilation_config={"mode": 0, "custom_ops": ["none"]},
        additional_config={"ksa_cudagraph": args.mode == "graph"},
    )
    engine = llm.llm_engine
    scheduler = engine.engine_core.engine_core.scheduler
    pool = scheduler.kv_cache_manager.block_pool
    free_initial = pool.get_num_free_blocks()
    startup = dict(load_and_profile_ms=(time.perf_counter() - begin) * 1000)
    rows = []
    try:
        engine.add_request(
            "kv-probe",
            {"prompt_token_ids": tokens},
            SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
        )
        while True:
            if engine.step():
                break
        request_id = next(iter(scheduler.requests))
        manager = scheduler.kv_cache_manager
        blocks = manager.get_blocks(request_id).get_block_ids()
        groups = scheduler.kv_cache_config.kv_cache_groups
        write_json(
            args.output / "kv.json",
            dict(
                text_tokens=args.length,
                groups=[
                    dict(
                        layers=g.layer_names,
                        spec=type(g.kv_cache_spec).__name__,
                        block_size=g.kv_cache_spec.block_size,
                        live_pages=sum(b != 0 for b in ids),
                        page_size_bytes=g.kv_cache_spec.page_size_bytes,
                    )
                    for g, ids in zip(groups, blocks)
                ],
                allocated_pool_bytes=sum(
                    t.size for t in scheduler.kv_cache_config.kv_cache_tensors
                ),
            ),
        )
        while engine.has_unfinished_requests():
            engine.step()
        for rep in range(-1, 5):
            before = llm.collective_rpc(telemetry, args=(True,))[0]
            arrival = [[] for _ in range(args.batch)]
            max_used = 0
            # Token IDs and length are identical to the official reference.
            params = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)
            begin = time.perf_counter()
            for i in range(args.batch):
                engine.add_request(str(i), {"prompt_token_ids": tokens}, params)
            while engine.has_unfinished_requests():
                outputs = engine.step()
                now = (time.perf_counter() - begin) * 1000
                max_used = max(max_used, free_initial - pool.get_num_free_blocks())
                for output in outputs:
                    times = arrival[int(output.request_id)]
                    count = len(output.outputs[0].token_ids)
                    if count != len(times) + 1:
                        raise AssertionError("expected one observed token per step")
                    times.append(now)
            elapsed = (time.perf_counter() - begin) * 1000
            after = llm.collective_rpc(telemetry)[0]
            if any(len(times) != 128 for times in arrival):
                raise AssertionError("generation truncated")
            if pool.get_num_free_blocks() != free_initial:
                raise AssertionError("scheduler pages leaked")
            decode = [[b - a for a, b in zip(t, t[1:])] for t in arrival]
            boundary = [(args.length + i + 1) % 8 == 0 for i in range(127)]
            captures = len(after["captures"]) - len(before["captures"])
            row = dict(
                backend="vllm",
                mode=args.mode,
                length=args.length,
                batch=args.batch,
                repetition=rep,
                status="warmup" if rep == -1 else "pass",
                ttft_ms=statistics.mean(t[0] for t in arrival),
                decode_ms=decode,
                mean_decode_ms=statistics.mean(x for t in decode for x in t),
                tpot_steady_ms=statistics.mean(
                    x for t in decode for x, b in zip(t, boundary) if not b
                ),
                tpot_boundary_ms=statistics.mean(
                    x for t in decode for x, b in zip(t, boundary) if b
                ),
                output_tokens_per_s=args.batch * 128000 / elapsed,
                elapsed_ms=elapsed,
                max_used_pages=max_used,
                free_pages_after=pool.get_num_free_blocks(),
                expected_free_pages=free_initial,
                new_captures=captures,
                peak_memory_bytes=after["peak_memory_bytes"],
            )
            if rep >= 0 and captures:
                row["status"] = "capture_contaminated"
            rows.append(row)
            write_json(args.output / "timing.json", rows)
            startup["captures"] = after["captures"]
            write_json(args.output / "startup.json", startup)
            print(
                f"{args.mode} {args.length} x {args.batch} rep {rep}: "
                f"TTFT {row['ttft_ms']:.1f}, decode {row['mean_decode_ms']:.2f}",
                flush=True,
            )
        if any(r["status"] != "pass" for r in rows if r["repetition"] >= 0):
            raise AssertionError("Graph capture contaminated steady measurements")
    finally:
        engine.engine_core.shutdown()


LENGTHS = [4096, 16384, 65536]
METRICS = (
    "ttft_ms",
    "mean_decode_ms",
    "tpot_steady_ms",
    "tpot_boundary_ms",
    "output_tokens_per_s",
    "peak_memory_bytes",
)


def compare_metric(old, new, metric):
    """Apply ticket 01's mean AND disjoint-range regression rule."""
    ratio = statistics.mean(new) / statistics.mean(old)
    disjoint = min(new) > max(old) or max(new) < min(old)
    worse = ratio < 0.95 if metric == "output_tokens_per_s" else ratio > 1.05
    if metric == "peak_memory_bytes":
        status = "retest" if worse else "pass"
    else:
        # A range change only counts if it moved in the adverse direction.
        adverse_range = (
            max(new) < min(old)
            if metric == "output_tokens_per_s"
            else min(new) > max(old)
        )
        status = (
            "regression"
            if worse and disjoint
            else ("retest" if worse or adverse_range else "pass")
        )
    return dict(status=status, mean_ratio=ratio, before=old, after=new)


def summarize(args):
    with args.before.open() as stream:
        previous = list(csv.DictReader(stream))
    comparisons = []
    for length in LENGTHS:
        for mode in ("eager", "graph"):
            path = args.output / f"{mode}-{length}-1" / "timing.json"
            old = [
                r
                for r in previous
                if int(r["length"]) == length
                and r["mode"] == mode
                and int(r["batch"]) == 1
                and int(r["repetition"]) >= 0
            ]
            new = (
                [r for r in read(path) if r["repetition"] >= 0] if path.exists() else []
            )
            complete = (
                len(old) == len(new) == 5
                and all(
                    {int(r["repetition"]) for r in rows} == set(range(5))
                    for rows in (old, new)
                )
                and all(
                    math.isfinite(float(r[m])) and float(r[m]) > 0
                    for r in old + new
                    for m in METRICS
                )
                and all(
                    r["status"] == "pass"
                    and int(r["new_captures"]) == 0
                    and int(r["free_pages_after"]) == int(r["expected_free_pages"])
                    for r in old + new
                )
            )
            row = dict(
                length=length,
                mode=mode,
                status="invalid" if path.exists() else "not_run",
            )
            if complete:
                metrics = {
                    m: compare_metric(
                        [float(r[m]) for r in old], [float(r[m]) for r in new], m
                    )
                    for m in METRICS
                }
                states = {r["status"] for r in metrics.values()}
                row.update(
                    metrics=metrics,
                    status=(
                        "regression"
                        if "regression" in states
                        else "retest"
                        if "retest" in states
                        else "pass"
                    ),
                )
            comparisons.append(row)
    result = dict(
        baseline_id=BASELINE_ID,
        tolerance_id=TOLERANCE_ID,
        comparisons=comparisons,
        status="pass" if all(r["status"] == "pass" for r in comparisons) else "fail",
        note="Performance only; HF accuracy and service checks are separate gates.",
    )
    write_json(args.output / "summary.json", result)
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    _, lock, _ = validate(args)
    previous = read(args.before.with_name("environment.json"))
    identity = dict(
        git_sha=args.expected_sha,
        model_hashes=lock["model_hashes"],
        packages={name: metadata.version(name) for name in previous["packages"]},
        command=sys.argv,
        gpu=command(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader",
        ),
        scope="V1 scheduler + sampling + model, no HTTP",
        output_tokens=128,
        repetitions=5,
        warmups=1,
    )
    compatible = (
        identity["gpu"] == previous["gpu"].rsplit(", ", 1)[0]
        and identity["model_hashes"] == previous["model_hashes"]
        and all(
            identity["packages"].get(k) == v for k, v in previous["packages"].items()
        )
    )
    identity["baseline_environment_matches"] = compatible
    write_json(args.output / "environment.json", identity)
    if not compatible:
        raise ValueError("Performance hardware/dependencies differ from ticket 01")
    processes = []
    for length in LENGTHS:
        for mode in ("eager", "graph"):
            name = f"{mode}-{length}-1"
            argv = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--length",
                str(length),
                "--batch",
                "1",
                "--mode",
                mode,
                "--baseline",
                str(args.baseline),
                "--model",
                str(args.model),
                "--expected-sha",
                args.expected_sha,
                "--output",
                str(args.output / name),
            ]
            print(f"Running {name}", flush=True)
            start = time.time()
            with (args.output / f"{name}.log").open("w") as log:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
            processes.append(
                dict(
                    job=name,
                    returncode=result.returncode,
                    command=argv,
                    elapsed_s=time.time() - start,
                )
            )
            write_json(args.output / "processes.json", processes)
    result = summarize(args)
    if result["status"] != "pass" or any(r["returncode"] for r in processes):
        raise AssertionError("Performance comparison requires review; see summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=FIXTURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument(
        "--before",
        type=Path,
        default=ROOT / "docs/ksa/results/refactor-01/performance.csv",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--length", type=int, choices=LENGTHS)
    parser.add_argument("--batch", type=int, choices=[1])
    parser.add_argument("--mode", choices=["eager", "graph"])
    args = parser.parse_args()
    if args.worker and any(x is None for x in (args.length, args.batch, args.mode)):
        parser.error("--worker requires --length, --batch and --mode")
    try:
        worker(args) if args.worker else run(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "failure.json",
            dict(
                status="oom" if "out of memory" in str(exc).lower() else "error",
                traceback=traceback.format_exc(),
                command=sys.argv,
            ),
        )
        raise


if __name__ == "__main__":
    main()
