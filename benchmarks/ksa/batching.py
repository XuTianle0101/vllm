# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T04 frozen-HF chunk/batch accuracy and independent concurrency throughput."""

import argparse
import gc
import importlib.metadata as metadata
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from baseline import command, digest, file_hash, write_json
from prefill import BASELINE_ID, ROOT, TOLERANCE_ID, compare, read


@torch.inference_mode()
def teacher_batch(model, cases, scheme):
    caches = {name: model.new_cache() for name in cases}
    collected = {name: [] for name in cases}
    step = 0
    try:
        while caches:
            names, batch, ranges = [], [], []
            for index, (name, cache) in enumerate(caches.items()):
                case = cases[name]
                tokens = case["input_ids"] + case["teacher_ids"]
                start = cache.text_tokens
                if scheme == "whole":
                    length = len(case["input_ids"]) if start == 0 else 1
                elif scheme == "phase":
                    length = [7, 1, 9, 255, 513][(step + index) % 5]
                else:
                    length = 257
                end = min(start + length, len(tokens))
                names.append(name)
                batch.append(
                    (
                        torch.tensor(tokens[start:end], device="cuda"),
                        torch.arange(start, end, device="cuda"),
                        cache,
                    )
                )
                ranges.append((start, end))
            # Reorder every step without tying a cache to a batch slot.
            if step % 2:
                names.reverse()
                batch.reverse()
                ranges.reverse()
            values = model.forward_batch(batch)
            for name, hidden, (start, end) in zip(names, values, ranges):
                prompt = len(cases[name]["input_ids"])
                first = max(start, prompt - 1)
                if first < end:
                    collected[name].append(
                        model.compute_logits(hidden[first - start :]).float().cpu()
                    )
                if end == prompt + len(cases[name]["teacher_ids"]):
                    caches.pop(name).clear()
            step += 1
        return {name: torch.cat(rows) for name, rows in collected.items()}
    finally:
        for cache in caches.values():
            cache.clear()


def pressure_recovery(model, ids):
    from vllm.model_executor.models.ksa_decode import KSAPythonRunner
    from vllm.v1.worker.ksa_model_runner import KSABatchedRunner

    expected = KSAPythonRunner(model).generate(
        torch.tensor(ids, device="cuda"), max_tokens=17, ignore_eos=True
    )
    saved_pool = model.page_pool
    try:
        runner = KSABatchedRunner(
            model, chunk_size=8, max_num_batched_tokens=32, kv_cache_num_blocks=400
        )
        for name in ("a", "b", "c"):
            runner.add_request(name, ids, max_tokens=17, ignore_eos=True)
        done = []
        for _ in range(200):
            done.extend(out for out in runner.step() if out.finish_reason is not None)
            if not runner.requests:
                break
        assert len(done) == 3 and runner.num_preemptions > 0
        assert all(
            out.token_ids == expected.token_ids and not out.error for out in done
        )
        assert runner.pool.manager.block_pool.get_num_free_blocks() == 399
        held = runner.pool.manager.block_pool.get_new_blocks(399)
        runner.add_request("oom", ids, max_tokens=1)
        failure = runner.step()[0]
        assert failure.finish_reason == "error" and not runner.requests
        runner.pool.manager.block_pool.free_blocks(held)
        torch.accelerator.synchronize()
        return dict(
            status="pass",
            preemptions=runner.num_preemptions,
            oom=failure.error,
            free_pages=399,
        )
    finally:
        if "runner" in locals():
            runner.close()
        model.page_pool = saved_pool


def timed(runner, ids, concurrency, max_tokens):
    runner.close()
    runner.num_preemptions = 0
    torch.get_device_module("cuda").reset_peak_memory_stats()
    torch.accelerator.synchronize()
    start = time.perf_counter()
    arrivals = {str(i): [] for i in range(concurrency)}
    boundaries = {str(i): [] for i in range(concurrency)}
    for name in arrivals:
        runner.add_request(name, ids, max_tokens=max_tokens, ignore_eos=True)
    steps = []
    while runner.requests:
        outputs = runner.step()
        torch.accelerator.synchronize()
        now = time.perf_counter()
        steps.append(
            dict(
                internal_rows=runner.last_step_rows,
                text_tokens=runner.last_step_text_tokens,
            )
        )
        for output in outputs:
            if output.error:
                raise MemoryError(output.error)
            arrivals[output.request_id].append(now - start)
            boundaries[output.request_id].append(
                (len(ids) + len(output.token_ids) - 1) % 8 == 0
            )
    elapsed = time.perf_counter() - start
    steady, boundary = [], []
    for name, times in arrivals.items():
        assert len(times) == max_tokens
        for i in range(1, len(times)):
            (boundary if boundaries[name][i] else steady).append(
                (times[i] - times[i - 1]) * 1000
            )
    assert max(step["internal_rows"] for step in steps) <= runner.max_num_batched_tokens
    assert (
        runner.pool.manager.block_pool.get_num_free_blocks()
        == runner.pool.config.num_blocks - 1
    )
    return dict(
        elapsed_s=elapsed,
        output_tokens_per_s=concurrency * max_tokens / elapsed,
        ttft_ms=statistics.mean(times[0] for times in arrivals.values()) * 1000,
        tpot_steady_ms=statistics.mean(steady),
        tpot_boundary_ms=statistics.mean(boundary),
        peak_memory_bytes=torch.get_device_module("cuda").max_memory_allocated(),
        pool_bytes=runner.pool.storage.numel() * runner.pool.storage.element_size(),
        preemptions=runner.num_preemptions,
        steps=steps,
    )


def export_hf(args):
    from baseline import teacher
    from compat import install
    from transformers import AutoConfig, AutoModelForCausalLM

    lock = read(args.baseline / "baseline-lock.json")
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    if packages != lock["packages"]:
        raise ValueError("HF packages differ from frozen T00")
    install()
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config.rope_parameters = {"rope_type": "default", "rope_theta": config.rope_theta}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    inputs = read(args.baseline / "inputs.json")
    with torch.inference_mode():
        for row in read(args.output / "results.json")["generation"]:
            name = row["case_id"]
            ids = torch.tensor([inputs[name]["input_ids"]], device="cuda")
            logits = teacher(torch, model, ids, row["tokens"][:-1])
            torch.save(logits, args.output / f"hf-generated-{name}.pt")


def run(args):
    sys.path.insert(0, str(ROOT))
    from vllm.entrypoints.ksa import load_model
    from vllm.model_executor.models.ksa_decode import KSAPythonRunner
    from vllm.v1.worker.ksa_model_runner import KSABatchedRunner

    sha = command("git", "rev-parse", "HEAD")
    dirty = command("git", "status", "--porcelain", "--untracked-files=all")
    if args.expected_sha and (args.expected_sha != sha or dirty):
        raise ValueError("T04 requires the exact clean source SHA")
    baseline, lock, calibration = [
        read(args.baseline / filename)
        for filename in ("correctness.json", "baseline-lock.json", "calibration.json")
    ]
    inputs = read(args.baseline / "inputs.json")
    if (
        baseline["baseline_id"] != BASELINE_ID
        or baseline["status"] != "pass"
        or baseline["tolerance_id"] != TOLERANCE_ID
        or "T00-" + digest(lock)[:20] != BASELINE_ID
        or "T00-tol-" + digest(calibration)[:20] != TOLERANCE_ID
        or digest(inputs) != lock["input_hash"]
    ):
        raise ValueError("frozen baseline identity mismatch")
    for filename, key in (
        ("baseline.py", "harness_sha256"),
        ("compat.py", "compat_sha256"),
    ):
        if file_hash(Path(__file__).with_name(filename)) != lock[key]:
            raise ValueError(f"frozen reference source changed: {filename}")
    for name, expected in lock["model_hashes"].items():
        if file_hash(args.model / name) != expected:
            raise ValueError(f"model hash mismatch: {name}")
    t03_path = ROOT / "docs/ksa/results/T03/a100-268933f716/summary.json"
    result = dict(
        git_sha=sha,
        dirty=bool(dirty),
        baseline_id=BASELINE_ID,
        tolerance_id=TOLERANCE_ID,
        gpu=torch.get_device_module("cuda").get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        command=sys.argv,
        t03_reference_sha256=file_hash(t03_path),
        t03_reference=read(t03_path),
        cases=[],
        generation=[],
        timing=[],
        status="running",
    )
    output = args.output / "results.json"
    write_json(output, result)
    cases = {
        name: case for name, case in inputs.items() if len(case["input_ids"]) <= 4096
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with load_model(str(args.model)) as model:
            model.new_cache().clear()
            runner = KSABatchedRunner(
                model, kv_cache_num_blocks=model.page_pool.config.num_blocks * 8
            )
            for scheme, concurrency in [
                ("whole", 1),
                ("phase", 1),
                ("257", 1),
                ("phase", 8),
            ]:
                items = list(cases.items())
                for offset in range(0, len(items), concurrency):
                    group = dict(items[offset : offset + concurrency])
                    print(
                        f"Teacher {scheme} batch={len(group)} {list(group)}", flush=True
                    )
                    actual = teacher_batch(model, group, scheme)
                    for name, logits in actual.items():
                        reference = torch.load(
                            args.baseline
                            / f"correctness-{name}"
                            / "outputs"
                            / f"{name}.pt",
                            weights_only=True,
                        )
                        length = len(cases[name]["input_ids"])
                        row = compare(
                            torch,
                            reference["decode_logits"],
                            logits,
                            list(range(length - 1, length - 1 + len(logits))),
                            baseline["thresholds"],
                        )
                        result["cases"].append(
                            dict(case_id=name, scheme=scheme, batch=len(group), **row)
                        )
                    write_json(output, result)
            for name in ("length-7", "length-8", "length-9", "english", "chinese"):
                ids = cases[name]["input_ids"]
                expected = KSAPythonRunner(model).generate(
                    torch.tensor(ids, device="cuda"), max_tokens=32, ignore_eos=True
                )
                runner.add_request(
                    name, ids, max_tokens=32, ignore_eos=True, prompt_logprobs=True
                )
                runner.add_request("cancel", ids, max_tokens=32)
                runner.step()
                runner.abort_request("cancel")
                runner.preempt(name)
                final = None
                while runner.requests:
                    for final in runner.step():
                        if final.error:
                            raise RuntimeError(final.error)
                assert final is not None
                generated_case = {
                    name: dict(input_ids=ids, teacher_ids=final.token_ids[:-1])
                }
                whole = teacher_batch(model, generated_case, "whole")[name]
                chunked = teacher_batch(model, generated_case, "phase")[name]
                positions = list(
                    range(len(ids) - 1, len(ids) - 1 + len(final.token_ids))
                )
                same_prefix = compare(
                    torch, whole, chunked, positions, baseline["thresholds"]
                )
                torch.save(chunked, args.output / f"vllm-generated-{name}.pt")
                result["generation"].append(
                    dict(
                        case_id=name,
                        status=same_prefix["status"],
                        exact_single_request_tokens=final.token_ids
                        == expected.token_ids,
                        single_request_same_prefix=same_prefix,
                        tokens=final.token_ids,
                        prompt_logprobs=final.prompt_logprobs,
                        usage=final.usage,
                    )
                )
            assert not runner.pool.read_slots
            result["pressure_recovery"] = pressure_recovery(
                model, inputs["length-1024"]["input_ids"][:40]
            )
            for length in args.lengths:
                for concurrency in (1, 4, 8):
                    for repetition in range(-1, args.repetitions):
                        print(
                            f"Timing length={length} concurrency={concurrency} "
                            f"repeat={repetition}",
                            flush=True,
                        )
                        row = timed(
                            runner,
                            inputs[f"length-{length}"]["input_ids"],
                            concurrency,
                            args.max_tokens,
                        )
                        result["timing"].append(
                            dict(
                                length=length,
                                concurrency=concurrency,
                                repetition=repetition,
                                **row,
                            )
                        )
                        write_json(output, result)
        del model, runner
        gc.collect()
        torch.accelerator.empty_cache()
        write_json(output, result)
        subprocess.run(
            [
                str(args.hf_python),
                str(Path(__file__).resolve()),
                "--export-hf",
                "--model",
                str(args.model.resolve()),
                "--baseline",
                str(args.baseline.resolve()),
                "--output",
                str(args.output.resolve()),
            ],
            check=True,
        )
        for row in result["generation"]:
            name = row["case_id"]
            reference = torch.load(
                args.output / f"hf-generated-{name}.pt", weights_only=True
            )
            actual = torch.load(
                args.output / f"vllm-generated-{name}.pt", weights_only=True
            )
            length = len(inputs[name]["input_ids"])
            row["hf_same_prefix"] = compare(
                torch,
                reference,
                actual,
                list(range(length - 1, length - 1 + len(actual))),
                baseline["thresholds"],
            )
            row["status"] = (
                "pass"
                if row["status"] == row["hf_same_prefix"]["status"] == "pass"
                else "fail"
            )
        result["status"] = (
            "pass"
            if all(
                row["status"] == "pass"
                for row in result["cases"] + result["generation"]
            )
            else "fail"
        )
    except BaseException as exc:
        result.update(status="error", error=repr(exc))
        raise
    finally:
        write_json(output, result)
    if result["status"] != "pass":
        raise SystemExit("T04 accuracy gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument(
        "--hf-python", type=Path, default=ROOT / ".venv-ksa-hf/bin/python"
    )
    parser.add_argument("--export-hf", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.export_hf:
        export_hf(args)
        raise SystemExit(0)
    if args.repetitions < 1 or args.max_tokens < 9:
        parser.error("require repetitions >=1 and max-tokens >=9")
    args.output.mkdir(parents=True, exist_ok=False)
    run(args)
