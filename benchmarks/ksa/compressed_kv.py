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
from baseline import digest, file_hash
from decode import timed_vllm
from prefill import BASELINE_ID, ROOT, TOLERANCE_ID, compare, read


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def teacher(model, ids, continuation):
    from vllm.model_executor.models.ksa_decode import KSACache, retained_layout

    cache = KSACache()
    logits = []
    occupancy = []
    try:
        for tokens in [ids] + [ids.new_tensor([t]) for t in continuation]:
            positions = torch.arange(
                cache.text_tokens, cache.text_tokens + len(tokens), device=ids.device
            )
            hidden = model(tokens, positions, cache=cache)
            logits.append(model.compute_logits(hidden[-1:]).float().cpu()[0])
            rows = [k.shape[0] for k, _ in cache.layers]
            expected = [
                len(retained_layout(cache.text_tokens, w, ids.device)[0])
                for w in model.windows
            ]
            if rows != expected:
                raise AssertionError(f"KV occupancy mismatch: {rows} != {expected}")
            occupancy.append(
                {
                    "text_tokens": cache.text_tokens,
                    "layer_rows": rows,
                    "effective_kv_bytes": sum(
                        k.numel() * k.element_size() + v.numel() * v.element_size()
                        for k, v in cache.layers
                    ),
                }
            )
        return torch.stack(logits), occupancy
    finally:
        cache.clear()


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
        "cases": [],
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
            with torch.inference_mode():
                for name, case in inputs.items():
                    length = len(case["input_ids"])
                    if length > 4096:
                        continue
                    ids = torch.tensor(case["input_ids"], device="cuda")
                    actual, occupancy = teacher(model, ids, case["teacher_ids"])
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
                for length in (1024, 4096):
                    ids = torch.tensor(
                        inputs[f"length-{length}"]["input_ids"], device="cuda"
                    )
                    for rep in range(-1, 5):
                        torch.get_device_module("cuda").reset_peak_memory_stats()
                        row = timed_vllm(model, ids)
                        results["timing"].append(
                            {"length": length, "repetition": rep, **row}
                        )
                        write(args.output / "results.json", results)
            del model
            config.compilation_config.static_forward_context.clear()
        finally:
            cleanup_dist_env_and_memory()
    results["status"] = (
        "pass" if all(r["status"] == "pass" for r in results["cases"]) else "fail"
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    run(args)
    print("Elapsed seconds", time.perf_counter() - start)


if __name__ == "__main__":
    main()
