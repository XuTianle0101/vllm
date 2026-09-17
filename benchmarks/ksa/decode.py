# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T02 cached decode accuracy and matched HF/vLLM generation measurements."""

import argparse
import importlib.metadata as metadata
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from baseline import command, digest, file_hash, write_json
from prefill import BASELINE_ID, ROOT, TOLERANCE_ID, compare, read

GENERATION_CASES = ("length-7", "length-8", "length-9", "english", "chinese")


def validate(args):
    report = read(args.baseline / "correctness.json")
    lock = read(args.baseline / "baseline-lock.json")
    calibration = read(args.baseline / "calibration.json")
    if (
        report["baseline_id"] != BASELINE_ID
        or report["status"] != "pass"
        or "T00-" + digest(lock)[:20] != BASELINE_ID
        or report["tolerance_id"] != TOLERANCE_ID
        or "T00-tol-" + digest(calibration)[:20] != TOLERANCE_ID
    ):
        raise ValueError("T00 identity or frozen tolerance mismatch")
    if command("git", "rev-parse", "HEAD") != args.expected_sha:
        raise ValueError("T02 full commit SHA mismatch")
    if command(
        "git",
        "diff",
        "HEAD",
        "--",
        "vllm",
        "benchmarks/ksa",
        "tests/model_executor/test_ksa_prefill.py",
    ):
        raise ValueError("T02 source differs from the requested commit")
    for name, expected in lock["model_hashes"].items():
        if file_hash(args.model / name) != expected:
            raise ValueError(f"Model hash mismatch: {name}")
    for filename, key in (
        ("baseline.py", "harness_sha256"),
        ("compat.py", "compat_sha256"),
    ):
        if file_hash(Path(__file__).with_name(filename)) != lock[key]:
            raise ValueError(f"Frozen reference source changed: {filename}")
    inputs = read(args.baseline / "inputs.json")
    if digest(inputs) != lock["input_hash"]:
        raise ValueError("Frozen input hash mismatch")
    write_json(args.output / "inputs.json", inputs)
    return report, lock, inputs


def export_hf(args):
    import torch
    from baseline import teacher, timed_generation
    from compat import install
    from transformers import AutoConfig, AutoModelForCausalLM

    lock = read(args.baseline / "baseline-lock.json")
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    if packages != lock["packages"]:
        raise ValueError("HF packages differ from frozen T00")
    install()
    torch.backends.cuda.matmul.allow_tf32 = False
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
    timings = []
    generated = read(args.output / "generation.json")
    with torch.inference_mode():
        for case in generated:
            name = case["case_id"]
            ids = torch.tensor([inputs[name]["input_ids"]], device="cuda")
            # Compare each chosen token under the SAME generated prefix. Divergent
            # free-running sequences cannot be compared by timestep alone.
            logits = teacher(torch, model, ids, case["token_ids"][:-1])
            torch.save(logits, args.output / f"hf-generated-{name}.pt")
        for length in (1024, 4096):
            ids = torch.tensor([inputs[f"length-{length}"]["input_ids"]], device="cuda")
            for rep in range(-1, 5):
                torch.get_device_module("cuda").reset_peak_memory_stats()
                row = timed_generation(torch, model, ids)
                timings.append(dict(backend="HF", length=length, repetition=rep, **row))
                write_json(args.output / "hf-timing.json", timings)


def teacher_vllm(model, ids, continuation):
    import torch

    from vllm.model_executor.models.ksa_decode import KSACache

    cache = KSACache()
    rows = []
    try:
        for tokens in [ids] + [ids.new_tensor([token]) for token in continuation]:
            positions = torch.arange(
                cache.text_tokens, cache.text_tokens + len(tokens), device=ids.device
            )
            hidden = model(tokens, positions, cache=cache)
            rows.append(model.compute_logits(hidden[-1:]).float().cpu()[0])
        if cache.text_tokens != len(ids) + len(continuation):
            raise AssertionError("text count differs from consumed tokens")
        if any(k.shape[0] != cache.internal_rows for k, v in cache.layers):
            raise AssertionError("internal KV count differs from expanded rows")
        return torch.stack(rows)
    finally:
        cache.clear()


def timed_vllm(model, ids, *, cache_factory=None):
    import torch

    from vllm.model_executor.models.ksa_decode import KSACache

    cache = KSACache() if cache_factory is None else cache_factory()
    times, boundary = [], []
    tokens = ids
    torch.accelerator.synchronize()
    try:
        for step in range(128):
            start = time.perf_counter()
            positions = torch.arange(
                cache.text_tokens, cache.text_tokens + len(tokens), device=ids.device
            )
            hidden = model(tokens, positions, cache=cache)
            tokens = model.compute_logits(hidden[-1:]).argmax(-1)
            torch.accelerator.synchronize()
            times.append((time.perf_counter() - start) * 1000)
            if step:
                boundary.append(cache.text_tokens % 8 == 0)
        return {
            "ttft_ms": times[0],
            "tpot_steady_ms": statistics.mean(
                t for t, b in zip(times[1:], boundary) if not b
            ),
            "tpot_boundary_ms": statistics.mean(
                t for t, b in zip(times[1:], boundary) if b
            ),
            "output_tokens_per_s": 128000 / sum(times),
            "peak_memory_bytes": torch.get_device_module("cuda").max_memory_allocated(),
            "decode_ms": times[1:],
            "boundary": boundary,
        }
    finally:
        cache.clear()


def run(args):
    import torch

    sys.path.insert(0, str(ROOT))
    from transformers import AutoTokenizer

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.ksa import KSAForCausalLM
    from vllm.model_executor.models.ksa_decode import KSAPythonRunner

    report, lock, inputs = validate(args)
    write_json(
        args.output / "environment.json",
        {
            "ticket": "T02",
            "git_sha": args.expected_sha,
            "baseline_id": BASELINE_ID,
            "tolerance_id": TOLERANCE_ID,
            "gpu": torch.get_device_module("cuda").get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": sys.version,
            "packages": {
                d.metadata["Name"]: d.version for d in metadata.distributions()
            },
            "driver": command(
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
            ),
            "model_hashes": lock["model_hashes"],
            "command": sys.argv,
            "precision": "bfloat16",
            "prefill_limit": 4096,
            "total_text_limit": 8192,
            "cache": "full-history Python KV; no paging or eviction",
            "unsupported": [
                "serving",
                "batching",
                "chunked prefill",
                "prefix caching",
                "graphs",
                "speculation",
                "sampling",
            ],
        },
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
    correctness = {
        "baseline_id": BASELINE_ID,
        "tolerance_id": TOLERANCE_ID,
        "thresholds": report["thresholds"],
        "cases": [],
        "stopping": [],
    }
    generated, timing = [], []
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
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
            if not isinstance(model, KSAForCausalLM):
                raise TypeError("KSA model route was not selected")
            runner = KSAPythonRunner(model)
            with torch.inference_mode():
                for name, case in inputs.items():
                    length = len(case["input_ids"])
                    if length > 4096:
                        continue
                    print(f"Teacher forced {name}", flush=True)
                    ids = torch.tensor(case["input_ids"], device="cuda")
                    reference = torch.load(
                        args.baseline
                        / f"correctness-{name}"
                        / "outputs"
                        / f"{name}.pt",
                        weights_only=True,
                    )
                    actual = teacher_vllm(model, ids, case["teacher_ids"])
                    positions = list(
                        range(length - 1, length + len(case["teacher_ids"]))
                    )
                    row = compare(
                        torch,
                        reference["decode_logits"],
                        actual,
                        positions,
                        report["thresholds"],
                    )
                    row["case_id"] = name
                    row["repeat_max_abs_error"] = max(
                        (teacher_vllm(model, ids, case["teacher_ids"]) - actual)
                        .abs()
                        .max()
                        .item()
                        for _ in range(2)
                    )
                    if row["repeat_max_abs_error"] != 0:
                        row["status"] = "fail"
                    correctness["cases"].append(row)
                    write_json(args.output / "correctness.json", correctness)
                for name in GENERATION_CASES:
                    print(f"Generate {name}", flush=True)
                    ids = torch.tensor(inputs[name]["input_ids"], device="cuda")
                    output = runner.generate(ids, max_tokens=128, ignore_eos=True)
                    repeat = runner.generate(ids, max_tokens=128, ignore_eos=True)
                    baseline_case = next(
                        r for r in report["cases"] if r["case_id"] == name
                    )
                    expected = baseline_case["generated_ids"]
                    actual = teacher_vllm(model, ids, output.token_ids[:-1])
                    torch.save(actual, args.output / f"vllm-generated-{name}.pt")
                    generated.append(
                        {
                            "case_id": name,
                            **vars(output),
                            "completion_tokens": output.completion_tokens,
                            "text": tokenizer.decode(
                                output.token_ids, skip_special_tokens=True
                            ),
                            "hf_token_ids": expected,
                            "exact_hf_tokens": output.token_ids == expected,
                            "first_token_difference": next(
                                (
                                    i
                                    for i, (a, b) in enumerate(
                                        zip(expected, output.token_ids)
                                    )
                                    if a != b
                                ),
                                None,
                            ),
                            "repeat_equal": output == repeat,
                            "no_summary_leak": all(
                                0 <= t < model.text_vocab_size for t in output.token_ids
                            ),
                        }
                    )
                    eos = output.token_ids[0]
                    stop = runner.generate(ids, max_tokens=128, eos_token_ids=[eos])
                    zero = runner.generate(ids, max_tokens=0)
                    one = runner.generate(ids, max_tokens=1, ignore_eos=True)
                    correctness["stopping"].append(
                        {
                            "case_id": name,
                            "injected_eos_id": eos,
                            "status": "pass"
                            if stop.token_ids == [eos]
                            and stop.finish_reason == "stop"
                            and zero.completion_tokens == 0
                            and one.completion_tokens == 1
                            and one.finish_reason == "length"
                            and output.completion_tokens == 128
                            and output.prompt_tokens == len(ids)
                            and output.finish_reason == "length"
                            else "fail",
                        }
                    )
                    write_json(args.output / "generation.json", generated)
                for length in (1024, 4096):
                    print(f"Timing {length}", flush=True)
                    ids = torch.tensor(
                        inputs[f"length-{length}"]["input_ids"], device="cuda"
                    )
                    for rep in range(-1, 5):
                        torch.get_device_module("cuda").reset_peak_memory_stats()
                        row = timed_vllm(model, ids)
                        timing.append(
                            dict(backend="vLLM", length=length, repetition=rep, **row)
                        )
                        write_json(args.output / "timing.json", timing)
            del runner, model
            config.compilation_config.static_forward_context.clear()
        finally:
            cleanup_dist_env_and_memory()
    write_json(args.output / "correctness.json", correctness)
    with (args.output / "hf-reference.log").open("w") as log:
        subprocess.run(
            [
                str(args.hf_python),
                str(Path(__file__).resolve()),
                "--export-hf",
                "--model",
                str(args.model),
                "--baseline",
                str(args.baseline),
                "--output",
                str(args.output),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    for row in generated:
        name = row["case_id"]
        expected = torch.load(
            args.output / f"hf-generated-{name}.pt", weights_only=True
        )
        actual = torch.load(
            args.output / f"vllm-generated-{name}.pt", weights_only=True
        )
        length = len(inputs[name]["input_ids"])
        row["same_prefix_comparison"] = compare(
            torch,
            expected,
            actual,
            list(range(length - 1, length + 127)),
            report["thresholds"],
        )
        row["chosen_tokens_match_logits"] = (
            actual.argmax(-1).tolist() == row["token_ids"]
        )
        row["status"] = (
            "pass"
            if (
                row["same_prefix_comparison"]["status"] == "pass"
                and row["repeat_equal"]
                and row["no_summary_leak"]
                and row["chosen_tokens_match_logits"]
            )
            else "fail"
        )
    write_json(args.output / "generation.json", generated)
    timing += read(args.output / "hf-timing.json")
    write_json(args.output / "timing.json", timing)
    passed = all(
        r["status"] == "pass"
        for r in correctness["cases"] + correctness["stopping"] + generated
    )
    correctness["status"] = "pass" if passed else "fail"
    write_json(args.output / "correctness.json", correctness)
    lines = [
        "# T02 cached decode",
        "",
        f"Status: {correctness['status']}",
        "",
        f"SHA: {args.expected_sha}",
        f"Baseline: {BASELINE_ID}; tolerance: {TOLERANCE_ID}",
        "",
        (
            "BF16, batch=1, 128 output tokens, EOS disabled for timing; "
            "one warmup and five repetitions."
        ),
        (
            "Generation comparisons use the frozen T00 margin <= 0.5 rule "
            "under identical prefixes; exact IDs are separately reported."
        ),
        "",
        (
            "| Backend | Prompt | TTFT ms | Steady TPOT ms | "
            "Boundary TPOT ms | Tokens/s | Peak GiB |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for backend in ("HF", "vLLM"):
        for length in (1024, 4096):
            rows = [
                r
                for r in timing
                if r["backend"] == backend
                and r["length"] == length
                and r["repetition"] >= 0
            ]
            values = [
                statistics.mean(r[k] for r in rows)
                for k in (
                    "ttft_ms",
                    "tpot_steady_ms",
                    "tpot_boundary_ms",
                    "output_tokens_per_s",
                )
            ]
            lines.append(
                f"| {backend} | {length} | "
                + " | ".join(f"{v:.3f}" for v in values)
                + f" | {max(r['peak_memory_bytes'] for r in rows) / 2**30:.3f} |"
            )
    lines += [
        "",
        (
            "Full-history Python KV and dense prefill; no serving, concurrency, "
            "graphs, paging, or long-context performance claim."
        ),
        (
            "TTFT includes prefill and argmax; TPOT includes preparation, forward, "
            "argmax and device synchronization. "
            "Load/compile is outside warm measurements."
        ),
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    if not passed:
        raise RuntimeError("T02 accuracy gates failed; retain all diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha")
    parser.add_argument(
        "--hf-python", type=Path, default=ROOT / ".venv-ksa-hf/bin/python"
    )
    parser.add_argument("--export-hf", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.export_hf)
    try:
        if args.export_hf:
            export_hf(args)
        else:
            run(args)
    except Exception:
        write_json(
            args.output / "failure.json",
            {"error": traceback.format_exc(), "command": sys.argv},
        )
        raise


if __name__ == "__main__":
    main()
