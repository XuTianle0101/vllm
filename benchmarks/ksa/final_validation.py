# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T07 isolated, resumable HF / V1 performance matrix and conservative gates."""

import argparse
import importlib.metadata as metadata
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from baseline import LONG, command, write_json
from prefill import BASELINE_ID, ROOT, TOLERANCE_ID, read


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
    inputs = read(args.baseline / "inputs.json")
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


def summarize(output):
    processes = read(output / "processes.json")
    comparisons = []
    for length in LONG:
        hf = []
        for rep in range(5):
            path = output / f"hf-{length}" / "outputs" / f"timing-{length}-{rep}.json"
            if path.exists():
                hf.append(read(path))
        for mode in ("eager", "graph"):
            path = output / f"{mode}-{length}-1" / "timing.json"
            vl = (
                [r for r in read(path) if r["repetition"] >= 0] if path.exists() else []
            )
            complete = len(hf) == len(vl) == 5 and all(
                r["status"] == "pass" for r in vl
            )
            row = dict(length=length, mode=mode, status="not_run")
            if complete:
                h = [statistics.mean(r["decode_ms"]) for r in hf]
                v = [r["mean_decode_ms"] for r in vl]
                row.update(
                    status="pass" if max(v) < min(h) else "fail",
                    hf_decode_ms=h,
                    vllm_decode_ms=v,
                    decode_speedup=statistics.mean(h) / statistics.mean(v),
                    hf_ttft_ms=[r["ttft_ms"] for r in hf],
                    vllm_ttft_ms=[r["ttft_ms"] for r in vl],
                )
            comparisons.append(row)
    required = [
        r
        for r in comparisons
        if r["length"] in (16384, 32768, 65536) and r["mode"] == "graph"
    ]
    result = dict(
        baseline_id=BASELINE_ID,
        tolerance_id=TOLERANCE_ID,
        comparisons=comparisons,
        processes=processes,
        decode_gate="pass" if all(r["status"] == "pass" for r in required) else "fail",
        final_acceptance="not_evaluated",
        note="Performance only. Accuracy, retrieval, service, lifecycle and KV "
        "compression evidence must be reviewed separately. HF batch>1 is N/A.",
    )
    write_json(output / "summary.json", result)
    return result


def run(args):
    from decode import validate

    args.output.mkdir(parents=True, exist_ok=True)
    _, lock, _ = validate(args)
    packages = subprocess.check_output(
        [
            str(args.hf_python),
            "-c",
            (
                "import importlib.metadata as m,json; "
                "print(json.dumps({d.metadata['Name']: d.version "
                "for d in m.distributions()}))"
            ),
        ],
        text=True,
    )
    if json.loads(packages) != lock["packages"]:
        raise ValueError("HF packages differ from frozen T00")
    identity = dict(
        ticket="T07",
        git_sha=args.expected_sha,
        baseline_id=BASELINE_ID,
        tolerance_id=TOLERANCE_ID,
        model_hashes=lock["model_hashes"],
        packages={d.metadata["Name"]: d.version for d in metadata.distributions()},
        hf_packages=json.loads(packages),
        command=sys.argv,
        gpu=command(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader",
        ),
        scope="V1 scheduler + sampling + model, no HTTP; HF official cached generation",
        output_tokens=128,
        repetitions=5,
        warmups=1,
    )
    env = args.output / "environment.json"
    if env.exists():
        previous = read(env)
        if any(previous[k] != identity[k] for k in identity if k != "command"):
            raise ValueError("Cannot resume a different runtime/source matrix")
    else:
        write_json(env, identity)
    path = args.output / "processes.json"
    processes = read(path) if path.exists() else []
    done = {r["job"] for r in processes}
    for length in args.lengths:
        jobs = [
            (
                f"hf-{length}",
                [
                    str(args.hf_python),
                    str(Path(__file__).with_name("baseline.py")),
                    "--mode",
                    "performance",
                    "--case",
                    f"length-{length}",
                ],
            )
        ]
        jobs += [
            (
                f"{mode}-{length}-{batch}",
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--length",
                    str(length),
                    "--batch",
                    str(batch),
                    "--mode",
                    mode,
                    "--baseline",
                    str(args.baseline),
                ],
            )
            for batch in args.batches
            for mode in args.modes
        ]
        for name, argv in jobs:
            if name in done:
                continue
            argv += ["--model", str(args.model), "--output", str(args.output / name)]
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
            write_json(path, processes)
            summarize(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha")
    parser.add_argument(
        "--hf-python", type=Path, default=ROOT / ".venv-ksa-hf/bin/python"
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=LONG)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"]
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--length", type=int, choices=LONG)
    parser.add_argument("--batch", type=int, choices=[1, 4, 8])
    parser.add_argument("--mode", choices=["eager", "graph"])
    args = parser.parse_args()
    if not set(args.lengths) <= set(LONG) or not set(args.batches) <= {1, 4, 8}:
        parser.error("Unsupported matrix dimensions")
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
