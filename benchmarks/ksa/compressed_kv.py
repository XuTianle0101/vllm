# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T03 windowed KV accuracy, occupancy and latency on one GPU."""

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
from baseline import command, digest, file_hash
from decode import GENERATION_CASES, timed_vllm
from prefill import BASELINE_ID, ROOT, TOLERANCE_ID, compare, read


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def teacher(model, ids, continuation, *, paged=True):
    from vllm.model_executor.models.ksa_decode import KSACache, retained_layout

    cache = model.new_cache() if paged else KSACache()
    logits = []
    occupancy = []
    try:
        for tokens in [ids] + [ids.new_tensor([t]) for t in continuation]:
            positions = torch.arange(
                cache.text_tokens, cache.text_tokens + len(tokens), device=ids.device
            )
            hidden = model(tokens, positions, cache=cache)
            logits.append(model.compute_logits(hidden[-1:]).float().cpu()[0])
            usage = (
                cache.page_pool.occupancy(cache.request, cache.text_tokens)
                if paged
                else dict(
                    layer_rows=[k.shape[0] for k, _ in cache.layers],
                    effective_kv_bytes=sum(
                        k.numel() * k.element_size() * 2 for k, _ in cache.layers
                    ),
                )
            )
            if paged:
                tables = cache.page_pool.manager.get_blocks(
                    cache.request.request_id
                ).get_block_ids()
                usage["noncontiguous_pages"] = any(
                    any(b != a + 1 for a, b in zip(t, t[1:]) if a and b) for t in tables
                )
            rows = usage["layer_rows"]
            expected = [
                len(retained_layout(cache.text_tokens, w, ids.device)[0])
                for w in model.windows
            ]
            if rows != expected:
                raise AssertionError(f"KV occupancy mismatch: {rows} != {expected}")
            occupancy.append(
                {
                    "text_tokens": cache.text_tokens,
                    **usage,
                }
            )
        return torch.stack(logits), occupancy
    finally:
        cache.clear()
        if paged and cache.page_pool.manager.block_pool.get_num_free_blocks() != (
            cache.page_pool.config.num_blocks - 1
        ):
            raise AssertionError("KSA page leak after request release")


def run(args):
    sys.path.insert(0, str(ROOT))
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model

    sha = command("git", "rev-parse", "HEAD")
    dirty = command("git", "status", "--porcelain", "--untracked-files=all")
    if args.expected_sha and (sha != args.expected_sha or dirty):
        raise ValueError("T03 requires the exact clean source SHA")
    baseline = read(args.baseline / "correctness.json")
    lock = read(args.baseline / "baseline-lock.json")
    calibration = read(args.baseline / "calibration.json")
    if (
        baseline["status"] != "pass"
        or baseline["baseline_id"] != BASELINE_ID
        or baseline["tolerance_id"] != TOLERANCE_ID
        or "T00-" + digest(lock)[:20] != BASELINE_ID
        or "T00-tol-" + digest(calibration)[:20] != TOLERANCE_ID
    ):
        raise ValueError("Frozen T00 baseline or tolerance mismatch")
    for name, expected in lock["model_hashes"].items():
        if file_hash(args.model / name) != expected:
            raise ValueError(f"Model hash mismatch: {name}")
    for filename, key in (
        ("baseline.py", "harness_sha256"),
        ("compat.py", "compat_sha256"),
    ):
        if file_hash(Path(__file__).with_name(filename)) != lock[key]:
            raise ValueError(f"Frozen HF reference source changed: {filename}")
    t02_hashes = {}
    if args.t02_results:
        environment = read(args.t02_results / "environment.json")
        if (
            environment["git_sha"] != "4f2845da8a3565d8bbedec082fdbf2fc40319099"
            or environment["baseline_id"] != BASELINE_ID
            or environment["tolerance_id"] != TOLERANCE_ID
            or environment["model_hashes"] != lock["model_hashes"]
        ):
            raise ValueError("Frozen T02 identity mismatch")
        for file in args.t02_results.iterdir():
            if file.suffix in (".json", ".pt"):
                t02_hashes[file.name] = file_hash(file)
    inputs = read(args.baseline / "inputs.json")
    if digest(inputs) != lock["input_hash"]:
        raise ValueError("Frozen input hash mismatch")
    config = EngineArgs(
        model=str(args.model),
        enforce_eager=True,
        max_model_len=8192,
        dtype="bfloat16",
        compilation_config={"mode": 0, "custom_ops": ["none"]},
    ).create_engine_config()
    results = {
        "baseline_id": baseline["baseline_id"],
        "tolerance_id": baseline["tolerance_id"],
        "git_sha": sha,
        "t02_artifact_hashes": t02_hashes,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dirty": bool(dirty),
        "command": sys.argv,
        "cache_mode": args.cache_mode,
        "page_writes": args.page_writes,
        "compact_prefill": not args.retain_prefill_projections,
        "cases": [],
        "generation": [],
        "timing": [],
        "gpu": torch.get_device_module("cuda").get_device_name(),
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    with tempfile.TemporaryDirectory() as temp, set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="nccl",
            distributed_init_method=f"file://{temp}/store",
        )
        initialize_model_parallel(1, 1)
        try:
            model = get_model(vllm_config=config)
            if args.cache_mode == "paged":
                model.new_cache().clear()
                model.page_pool.batch_writes = args.page_writes == "batched"
                model.page_pool.compact_prefill = not args.retain_prefill_projections
            with torch.inference_mode():
                for name, case in inputs.items():
                    length = len(case["input_ids"])
                    if length > 4096:
                        continue
                    ids = torch.tensor(case["input_ids"], device="cuda")
                    print(f"Teacher {name}", flush=True)
                    actual, occupancy = teacher(
                        model,
                        ids,
                        case["teacher_ids"],
                        paged=args.cache_mode == "paged",
                    )
                    reference = torch.load(
                        args.baseline
                        / f"correctness-{name}"
                        / "outputs"
                        / f"{name}.pt",
                        weights_only=True,
                    )
                    row = compare(
                        torch,
                        reference["decode_logits"],
                        actual,
                        list(range(length - 1, length + len(case["teacher_ids"]))),
                        baseline["thresholds"],
                    )
                    row.update(case_id=name, occupancy=occupancy)
                    results["cases"].append(row)
                    write(args.output / "results.json", results)
                if args.t02_results:
                    from vllm.model_executor.models.ksa_decode import KSAPythonRunner

                    runner = KSAPythonRunner(model, paged=args.cache_mode == "paged")
                    old_generation = read(args.t02_results / "generation.json")
                    for name in GENERATION_CASES:
                        print(f"Generate {name}", flush=True)
                        ids = torch.tensor(inputs[name]["input_ids"], device="cuda")
                        output = runner.generate(ids, max_tokens=128, ignore_eos=True)
                        repeat = runner.generate(ids, max_tokens=128, ignore_eos=True)
                        old = next(r for r in old_generation if r["case_id"] == name)
                        row = dict(
                            case_id=name,
                            **vars(output),
                            repeat_equal=output == repeat,
                            exact_t02_tokens=output.token_ids == old["token_ids"],
                            exact_hf_tokens=output.token_ids == old["hf_token_ids"],
                        )
                        # Frozen T02 exports contain HF logits under this prefix.
                        # Fail explicitly if it changes rather than comparing
                        # unrelated free-running positions.
                        if row["exact_t02_tokens"]:
                            actual, _ = teacher(
                                model,
                                ids,
                                output.token_ids[:-1],
                                paged=args.cache_mode == "paged",
                            )
                            for backend, filename in (("hf", "hf"), ("t02", "vllm")):
                                ref = torch.load(
                                    args.t02_results
                                    / f"{filename}-generated-{name}.pt",
                                    weights_only=True,
                                )
                                row[backend + "_same_prefix"] = compare(
                                    torch,
                                    ref,
                                    actual,
                                    list(range(len(ids) - 1, len(ids) + 127)),
                                    baseline["thresholds"],
                                )
                        row["status"] = (
                            "pass"
                            if (
                                row["repeat_equal"]
                                and row["exact_t02_tokens"]
                                and all(
                                    row[b + "_same_prefix"]["status"] == "pass"
                                    for b in ("hf", "t02")
                                )
                            )
                            else "fail"
                        )
                        results["generation"].append(row)
                        write(args.output / "results.json", results)
                for length in (1024, 4096):
                    ids = torch.tensor(
                        inputs[f"length-{length}"]["input_ids"], device="cuda"
                    )
                    for rep in range(-1, 5):
                        torch.get_device_module("cuda").reset_peak_memory_stats()
                        row = timed_vllm(
                            model,
                            ids,
                            cache_factory=(
                                model.new_cache if args.cache_mode == "paged" else None
                            ),
                        )
                        results["timing"].append(
                            {"length": length, "repetition": rep, **row}
                        )
                        write(args.output / "results.json", results)
            del model
            config.compilation_config.static_forward_context.clear()
        finally:
            cleanup_dist_env_and_memory()
    results["status"] = (
        "pass"
        if all(r["status"] == "pass" for r in results["cases"] + results["generation"])
        else "fail"
    )
    write(args.output / "results.json", results)
    for length in (1024, 4096):
        rows = [
            r
            for r in results["timing"]
            if r["length"] == length and r["repetition"] >= 0
        ]
        print(
            length,
            "mean TPOT ms",
            statistics.mean(r["tpot_steady_ms"] for r in rows),
            "peak GiB",
            max(r["peak_memory_bytes"] for r in rows) / 2**30,
        )
    if results["status"] != "pass":
        raise RuntimeError("T03 precision gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha")
    parser.add_argument("--cache-mode", choices=("paged", "tensor"), default="paged")
    parser.add_argument(
        "--page-writes", choices=("batched", "per-layer"), default="batched"
    )
    parser.add_argument("--retain-prefill-projections", action="store_true")
    parser.add_argument("--t02-results", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    run(args)
    print("Elapsed seconds", time.perf_counter() - start)


if __name__ == "__main__":
    main()
