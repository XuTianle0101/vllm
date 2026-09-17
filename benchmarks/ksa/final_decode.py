# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T07 cached teacher-forced decode at every frozen length, eager and graph."""

import argparse
import sys
import tempfile
from pathlib import Path

from baseline import file_hash, write_json
from decode import validate
from prefill import BASELINE_ID, ROOT, TOLERANCE_ID, compare


def teacher(torch, model, executor, cases):
    caches = {name: model.new_cache() for name in cases}
    rows = {name: [] for name in cases}
    try:
        for name, case in cases.items():
            ids = torch.tensor(case["input_ids"], device="cuda")
            for start in range(0, len(ids), 4096):
                end = min(start + 4096, len(ids))
                hidden = model(
                    ids[start:end],
                    torch.arange(start, end, device="cuda"),
                    cache=caches[name],
                )
            rows[name].append(model.compute_logits(hidden[-1:]).float().cpu())
        for step in range(max(len(c["teacher_ids"]) for c in cases.values())):
            names = [n for n in cases if step < len(cases[n]["teacher_ids"])]
            if step % 2:
                names.reverse()
            outputs = executor.forward_batch(
                [
                    (
                        torch.tensor([cases[n]["teacher_ids"][step]], device="cuda"),
                        torch.tensor([caches[n].text_tokens], device="cuda"),
                        caches[n],
                    )
                    for n in names
                ]
            )
            for name, hidden in zip(names, outputs):
                rows[name].append(model.compute_logits(hidden).float().cpu())
        return {name: torch.cat(values) for name, values in rows.items()}
    finally:
        for cache in caches.values():
            cache.clear()


def run(args):
    import torch

    sys.path.insert(0, str(ROOT))
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.ksa_cache import KSAPagePool
    from vllm.model_executor.models.ksa_graph import KSADecodeGraphs

    args.output.mkdir(parents=True, exist_ok=False)
    baseline, lock, inputs = validate(args)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = EngineArgs(
        model=str(args.model),
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=131072,
        dtype="bfloat16",
        compilation_config={"mode": 0, "custom_ops": ["none"]},
    ).create_engine_config()
    result = dict(
        git_sha=args.expected_sha,
        harness_sha256=file_hash(Path(__file__)),
        baseline_id=BASELINE_ID,
        tolerance_id=TOLERANCE_ID,
        thresholds=baseline["thresholds"],
        model_hashes=lock["model_hashes"],
        command=sys.argv,
        cases=[],
        status="running",
    )
    with tempfile.TemporaryDirectory() as directory, set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            backend="nccl",
            distributed_init_method=f"file://{directory}/store",
        )
        initialize_model_parallel(1, 1)
        try:
            model = get_model(vllm_config=config)
            model.new_cache().clear()
            pool = model.page_pool
            attn = model.model.layers[0].self_attn
            model.page_pool = KSAPagePool(
                model.windows,
                attn.num_kv_heads,
                attn.head_dim,
                torch.bfloat16,
                "cuda",
                131072,
                num_blocks=pool.config.num_blocks * 3,
            )
            del pool
            graphs = KSADecodeGraphs(model)
            names = list(inputs)
            with torch.inference_mode():
                for start in range(0, len(names), 3):
                    cases = {name: inputs[name] for name in names[start : start + 3]}
                    for mode, executor in (("eager", model), ("graph", graphs)):
                        actual = teacher(torch, model, executor, cases)
                        torch.save(actual, args.output / f"{mode}-{start}.pt")
                        for name, case in cases.items():
                            ref = torch.load(
                                args.baseline
                                / f"correctness-{name}"
                                / "outputs"
                                / f"{name}.pt",
                                weights_only=True,
                            )
                            positions = list(
                                range(
                                    len(case["input_ids"]) - 1,
                                    len(case["input_ids"]) + len(case["teacher_ids"]),
                                )
                            )
                            row = compare(
                                torch,
                                ref["decode_logits"],
                                actual[name],
                                positions,
                                baseline["thresholds"],
                            )
                            result["cases"].append(dict(case=name, mode=mode, **row))
                        write_json(args.output / "results.json", result)
                        print(f"Teacher decode {mode} {list(cases)}", flush=True)
            result["free_pages"] = (
                model.page_pool.manager.block_pool.get_num_free_blocks()
            )
            result["expected_free_pages"] = model.page_pool.config.num_blocks - 1
            result["graph_replays"] = graphs.replays
            passed = (
                len(result["cases"]) == 2 * len(inputs)
                and all(r["status"] == "pass" for r in result["cases"])
                and result["free_pages"] == result["expected_free_pages"]
                and graphs.replays > 0
            )
            result["status"] = "pass" if passed else "fail"
            write_json(args.output / "results.json", result)
            if not passed:
                raise AssertionError("Frozen cached decode gate failed")
        finally:
            cleanup_dist_env_and_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "failure.json", dict(error=repr(exc)))
        raise
