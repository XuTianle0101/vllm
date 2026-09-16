# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T01 single-request prefill acceptance against the frozen T00 artifacts."""

import argparse
import importlib.metadata as metadata
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from baseline import command, digest, file_hash, metrics, write_json

ROOT = Path(__file__).resolve().parents[2]
BASELINE_ID = "T00-305bf8f9dc429f1af60e"
TOLERANCE_ID = "T00-tol-fbd0eb931ac1c7153403"
CALIBRATION_LENGTHS = (17, 1025)


def read(path):
    return json.loads(path.read_text())


def export_hf(args):
    """Separate pinned HF process; export all internal layer rows in FP32."""
    import torch
    from compat import install, install_fp32_calibration
    from transformers import AutoConfig, AutoModelForCausalLM

    lock = read(args.baseline / "baseline-lock.json")
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    if packages != lock["packages"]:
        raise ValueError("HF reference packages differ from frozen T00 environment")
    install()
    install_fp32_calibration()
    torch.backends.cuda.matmul.allow_tf32 = False
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config.rope_parameters = {"rope_type": "default", "rope_theta": config.rope_theta}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map="cuda",
    ).eval()
    inputs = read(args.baseline / "inputs.json")
    layers = {}
    handles = [
        layer.register_forward_hook(
            lambda module, inp, out, index=i: layers.__setitem__(
                index, out[0].float().cpu()
            )
        )
        for i, layer in enumerate(model.model.layers)
    ]
    with torch.inference_mode():
        for length in CALIBRATION_LENGTHS:
            ids = torch.tensor([inputs[f"length-{length}"]["input_ids"]], device="cuda")
            logits = (
                model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0].cpu()
            )
            torch.save(
                {"layers": dict(layers), "logits": logits},
                args.output / f"hf-fp32-{length}.pt",
            )
    for handle in handles:
        handle.remove()


def tensor_error(torch, expected, actual):
    delta = (expected.float() - actual.float()).abs()
    row_errors = delta.reshape(-1, delta.shape[-1]).amax(-1)
    anomalies = (row_errors > 0.005).nonzero().flatten()
    return {
        "worst_row_index": row_errors.argmax().item(),
        "first_row_over_0_005": anomalies[0].item() if len(anomalies) else None,
        "feature_argmax_agreement": (expected.argmax(-1) == actual.argmax(-1))
        .float()
        .mean()
        .item(),
        "max_abs_error": delta.max().item(),
        "max_abs_error_over_reference_peak": (
            delta.max() / expected.float().abs().max().clamp_min(1.0)
        ).item(),
        "max_relative_error": (delta / expected.float().abs().clamp_min(1e-6))
        .max()
        .item(),
        "finite": bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()),
    }


def compare(torch, expected, actual, positions, thresholds):
    result = metrics(torch, expected, actual)
    result.update(tensor_error(torch, expected, actual))
    result["positions"] = positions
    result["per_position"] = [
        {
            "text_position": p,
            **metrics(torch, a[None], b[None]),
            **tensor_error(torch, a, b),
        }
        for p, a, b in zip(positions, expected, actual)
    ]
    gates = (
        "logits_max_abs_error",
        "logits_rmse",
        "logprobs_max_abs_error",
        "max_mismatch_margin",
    )
    result["status"] = (
        "pass"
        if result["finite"] and all(result[key] <= thresholds[key] for key in gates)
        else "fail"
    )
    result["first_anomaly_position"] = next(
        (
            r["text_position"]
            for r in result["per_position"]
            if not r["finite"] or any(r[k] > thresholds[k] for k in gates)
        ),
        None,
    )
    return result


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
    from vllm.model_executor.models.ksa import KSAForCausalLM
    from vllm.model_executor.models.ksa_prefill import MAX_PREFILL_TOKENS

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
        raise ValueError("T01 full commit SHA mismatch")
    if command(
        "git",
        "diff",
        "HEAD",
        "--",
        "vllm",
        "benchmarks/ksa",
        "tests/model_executor/test_ksa_prefill.py",
    ):
        raise ValueError("T01 source differs from the requested commit")
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
    write_json(
        args.output / "environment.json",
        {
            "ticket": "T01",
            "git_sha": args.expected_sha,
            "baseline_id": BASELINE_ID,
            "tolerance_id": TOLERANCE_ID,
            "gpu": torch.cuda.get_device_name(),
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
            "max_text_tokens": MAX_PREFILL_TOKENS,
            "attention": "PyTorch SDPA explicit mask",
            "decode": "unsupported",
            "command": sys.argv,
            "summary_self_kv": "visible (frozen T00 semantics)",
        },
    )
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
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = EngineArgs(
        model=str(args.model),
        enforce_eager=True,
        max_model_len=MAX_PREFILL_TOKENS,
        dtype="bfloat16",
        compilation_config={"mode": 0, "custom_ops": ["none"]},
    ).create_engine_config()
    correctness = {
        "baseline_id": BASELINE_ID,
        "tolerance_id": TOLERANCE_ID,
        "thresholds": report["thresholds"],
        "cases": [],
        "layers": [],
        "layer_policy": "Unnormalized hidden states are diagnostic; "
        "require finite values. "
        "T00 FP32 atol=0.005 applies to logits only.",
    }
    timing = {"mode": "prefill_only", "warmups": 1, "repetitions": 5, "cases": []}
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
                raise TypeError(f"Wrong model route: {type(model)}")
            with torch.inference_mode():
                for name, case in inputs.items():
                    length = len(case["input_ids"])
                    if length > MAX_PREFILL_TOKENS:
                        continue
                    print(f"Comparing {name}", flush=True)
                    reference = torch.load(
                        args.baseline
                        / f"correctness-{name}"
                        / "outputs"
                        / f"{name}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    # At the mask limit compare the final prompt row only.
                    tokens = (case["input_ids"] + case["teacher_ids"])[
                        :MAX_PREFILL_TOKENS
                    ]
                    ids = torch.tensor(tokens, device="cuda")
                    positions = torch.arange(length - 1, len(tokens), device="cuda")
                    hidden = model(ids, torch.arange(len(tokens), device="cuda"))
                    actual = model.compute_logits(hidden[positions]).float().cpu()
                    expected = reference["prefill_logits"][: len(positions)]
                    result = compare(
                        torch,
                        expected,
                        actual,
                        positions.tolist(),
                        report["thresholds"],
                    )
                    # Re-run the exact unextended prompt to cover each block boundary.
                    prompt_hidden = model(
                        ids[:length], torch.arange(length, device="cuda")
                    )
                    prompt_logits = (
                        model.compute_logits(prompt_hidden[-1:]).float().cpu()
                    )
                    result["prompt_only"] = compare(
                        torch,
                        expected[:1],
                        prompt_logits,
                        [length - 1],
                        report["thresholds"],
                    )
                    repeated = (
                        model.compute_logits(
                            model(ids, torch.arange(len(tokens), device="cuda"))[
                                positions
                            ]
                        )
                        .float()
                        .cpu()
                    )
                    result["repeat_max_abs_error"] = (
                        (repeated - actual).abs().max().item()
                    )
                    result["case_id"] = name
                    if (
                        result["prompt_only"]["status"] != "pass"
                        or result["repeat_max_abs_error"] != 0
                    ):
                        result["status"] = "fail"
                    correctness["cases"].append(result)
                    write_json(args.output / "correctness.json", correctness)
                for length in (1024, 4096, 16384):
                    case = inputs[f"length-{length}"]
                    ids = torch.tensor(case["input_ids"], device="cuda")
                    positions = torch.arange(length, device="cuda")
                    for repetition in range(-1, 5):
                        try:
                            torch.accelerator.reset_peak_memory_stats()
                            torch.accelerator.synchronize()
                            start = time.perf_counter()
                            logits = model.compute_logits(model(ids, positions)[-1:])
                            logits.argmax(-1)
                            torch.accelerator.synchronize()
                            seconds = time.perf_counter() - start
                            timing["cases"].append(
                                {
                                    "length": length,
                                    "repetition": repetition,
                                    "status": "warmup" if repetition == -1 else "pass",
                                    "ttft_ms": seconds * 1000,
                                    "input_tokens_per_second": length / seconds,
                                    "peak_memory_bytes": (
                                        torch.accelerator.max_memory_allocated()
                                    ),
                                }
                            )
                        except Exception:
                            timing["cases"].append(
                                {
                                    "length": length,
                                    "repetition": repetition,
                                    "status": "unsupported"
                                    if length > MAX_PREFILL_TOKENS
                                    else "error",
                                    "error": traceback.format_exc(),
                                }
                            )
                            if length > MAX_PREFILL_TOKENS:
                                break
                        write_json(args.output / "timing.json", timing)
                write_json(args.output / "timing.json", timing)
                del model
                config.compilation_config.static_forward_context.clear()
                config.model_config.dtype = torch.float32
                model = get_model(vllm_config=config)
                model.reference_attention = True
                for length in CALIBRATION_LENGTHS:
                    print(f"FP32 layers {length}", flush=True)
                    reference = torch.load(
                        args.output / f"hf-fp32-{length}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )

                    def observe(
                        index, hidden, rows, summary, reference=reference, length=length
                    ):
                        expected = reference["layers"][index]
                        for role, selected in (
                            ("text", ~summary),
                            ("summary", summary),
                        ):
                            error = tensor_error(
                                torch, expected[selected.cpu()], hidden[selected].cpu()
                            )
                            correctness["layers"].append(
                                {
                                    "length": length,
                                    "layer": index,
                                    "window": model.windows[index],
                                    "cycle_slot": index % 4,
                                    "role": role,
                                    "internal_row_indices": selected.nonzero()
                                    .flatten()
                                    .cpu()
                                    .tolist(),
                                    **error,
                                    "status": "finite" if error["finite"] else "fail",
                                }
                            )

                    model.layer_observer = observe
                    ids = torch.tensor(
                        inputs[f"length-{length}"]["input_ids"], device="cuda"
                    )
                    actual = model.compute_logits(
                        model(ids, torch.arange(length, device="cuda"))[-1:]
                    ).cpu()
                    error = tensor_error(torch, reference["logits"], actual)
                    correctness.setdefault("fp32_logits", []).append(
                        {
                            "length": length,
                            **error,
                            "status": "pass"
                            if error["finite"] and error["max_abs_error"] <= 0.005
                            else "fail",
                        }
                    )
                    write_json(args.output / "correctness.json", correctness)
        finally:
            cleanup_dist_env_and_memory()
    passed = all(
        r["status"] == "pass"
        for key in ("cases", "fp32_logits")
        for r in correctness[key]
    )
    passed &= all(r["status"] == "finite" for r in correctness["layers"])
    passed &= all(
        r["status"] in ("warmup", "pass", "unsupported") for r in timing["cases"]
    )
    correctness["status"] = "pass" if passed else "fail"
    write_json(args.output / "correctness.json", correctness)
    (args.output / "summary.md").write_text(
        f"# T01 prefill\n\nStatus: {correctness['status']}\n\n"
        f"SHA: {args.expected_sha}\n\n"
        f"Baseline: {BASELINE_ID}\n\nTolerance: {TOLERANCE_ID}\n\n"
        "Single-request, uncached prefill only; maximum 4096 text tokens. "
        "16K is explicitly unsupported. No decode or serving performance claim.\n"
    )
    if not passed:
        raise RuntimeError("T01 frozen acceptance gates failed; see correctness.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha")
    parser.add_argument(
        "--hf-python", type=Path, default=ROOT / ".venv-ksa-hf/bin/python"
    )
    parser.add_argument("--export-hf", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.export_hf:
        for name, value in {
            "environment.json": {"ticket": "T01", "git_sha": args.expected_sha},
            "inputs.json": {},
            "correctness.json": {"status": "not_run", "cases": []},
            "timing.json": {"status": "not_run", "cases": []},
        }.items():
            path = args.output / name
            if path.exists():
                raise ValueError(f"Refusing to overwrite existing result: {path}")
            write_json(path, value)
    try:
        if args.export_hf:
            export_hf(args)
        else:
            run(args)
    except Exception:
        error = traceback.format_exc()
        write_json(args.output / "failure.json", {"error": error, "command": sys.argv})
        if not (args.output / "summary.md").exists():
            (args.output / "summary.md").write_text(
                "# T01\n\nStatus: failed\n\n" + error
            )
        raise


if __name__ == "__main__":
    main()
