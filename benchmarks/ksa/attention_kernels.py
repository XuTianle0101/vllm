# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T06 frozen accuracy and matched reference/Triton full-model measurements."""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import torch
from baseline import command, write_json
from cudagraph import profile, sync, teacher, timing
from decode import validate
from prefill import BASELINE_ID, ROOT, TOLERANCE_ID, compare

sys.path.insert(0, str(ROOT))


@torch.inference_mode()
def prefill_timing(model, tokens, batch):
    caches = [model.new_cache() for _ in range(batch)]
    try:
        requests = []
        for i, cache in enumerate(caches):
            ids = torch.tensor(tokens[: len(tokens) - i], device="cuda")
            requests.append((ids, torch.arange(len(ids), device="cuda"), cache))
        sync()
        begin = time.perf_counter()
        outputs = model.forward_batch(requests)
        model.compute_logits(torch.cat([h[-1:] for h in outputs])).argmax(-1)
        sync()
        return (time.perf_counter() - begin) * 1000
    finally:
        for cache in caches:
            cache.clear()


def run(args):
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

    report, lock, inputs = validate(args)
    write_json(
        args.output / "environment.json",
        dict(
            ticket="T06",
            git_sha=args.expected_sha,
            baseline_id=BASELINE_ID,
            tolerance_id=TOLERANCE_ID,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            model_hashes=lock["model_hashes"],
            driver=command(
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
            ),
            command=sys.argv,
            custom_ops="none",
            dtype="bfloat16",
            scope="model + logits + greedy; prefill separate; no scheduler/network",
        ),
    )
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = EngineArgs(
        model=str(args.model),
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=8192,
        dtype="bfloat16",
        compilation_config={"mode": 0, "custom_ops": ["none"]},
    ).create_engine_config()
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
            begin = time.perf_counter()
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
                8192,
                num_blocks=pool.config.num_blocks * 8,
            )
            del pool
            sync()
            load_ms = (time.perf_counter() - begin) * 1000
            graphs = KSADecodeGraphs(model)
            cases = {n: c for n, c in inputs.items() if len(c["input_ids"]) <= 4096}
            correctness = []
            names = list(cases)
            for start in range(0, len(names), 4):
                group = {n: cases[n] for n in names[start : start + 4]}
                model.triton_attention = False
                eager = teacher(model, model, group)
                model.triton_attention = True
                for repeat, executor in enumerate((model, graphs)):
                    actual = teacher(model, executor, group, reverse=bool(repeat))
                    for name, case in group.items():
                        length = len(case["input_ids"])
                        positions = list(
                            range(length - 1, length + len(case["teacher_ids"]))
                        )
                        hf = torch.load(
                            args.baseline
                            / f"correctness-{name}"
                            / "outputs"
                            / f"{name}.pt",
                            weights_only=True,
                        )
                        for ref_name, ref in (
                            ("eager", eager[name]),
                            ("HF", hf["decode_logits"]),
                        ):
                            row = compare(
                                torch,
                                ref,
                                actual[name],
                                positions,
                                report["thresholds"],
                            )
                            correctness.append(
                                dict(
                                    case=name,
                                    mode="triton_graph" if repeat else "triton",
                                    reference=ref_name,
                                    **row,
                                )
                            )
                    write_json(args.output / "correctness.json", correctness)
                print(f"Accuracy {list(group)}", flush=True)
            if any(row["status"] != "pass" for row in correctness):
                raise AssertionError("T06 frozen accuracy gate failed")
            timings = []
            for length in args.lengths:
                tokens = inputs[f"length-{length}"]["input_ids"]
                for batch in args.batches:
                    for name, executor, triton_enabled in (
                        ("T05_eager", model, False),
                        ("T05_graph", graphs, False),
                        ("triton_eager", model, True),
                        ("triton_graph", graphs, True),
                    ):
                        model.triton_attention = triton_enabled
                        for repetition in range(-1, args.repeats):
                            torch.accelerator.reset_peak_memory_stats()
                            row = timing(model, executor, tokens, batch, args.steps)
                            if executor is model:
                                row["prefill_ms"] = prefill_timing(model, tokens, batch)
                            if repetition >= 0 and row["new_captures"]:
                                raise AssertionError(
                                    "capture contaminated steady timing"
                                )
                            timings.append(
                                dict(
                                    mode=name,
                                    length=length,
                                    batch=batch,
                                    repetition=repetition,
                                    **row,
                                )
                            )
                            write_json(args.output / "timing.json", timings)
                        print(
                            f"Timing {name} length={length} batch={batch}", flush=True
                        )
            model.triton_attention = True
            if not args.skip_trace:
                tokens = inputs["length-1024"]["input_ids"]
                timing(model, graphs, tokens, 4, 8)
                for name, executor in (("eager", model), ("graph", graphs)):
                    profile(model, executor, tokens, args.output, name)
            write_json(
                args.output / "startup.json",
                dict(
                    model_load_ms=load_ms,
                    captures=graphs.startup,
                    replays=graphs.replays,
                    compilation="JIT/capture excluded from steady timing",
                ),
            )
            write_json(
                args.output / "status.json",
                dict(
                    status="pass",
                    checks=len(correctness),
                    thresholds=report["thresholds"],
                    free_pages=model.page_pool.manager.block_pool.get_num_free_blocks(),
                    expected_free_pages=model.page_pool.config.num_blocks - 1,
                ),
            )
        finally:
            cleanup_dist_env_and_memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-trace", action="store_true")
    args = parser.parse_args()
    if args.steps < 16 or args.repeats < 5:
        parser.error("need >=16 decode steps and >=5 steady repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as exc:
        write_json(args.output / "status.json", dict(status="fail", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
