# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validation-only HF reference, frozen inputs, and numerical acceptance gates."""

import argparse
import gzip
import hashlib
import importlib.metadata as metadata
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).with_name("fixtures") / "hf_reference"
BASELINE_ID = "T00-305bf8f9dc429f1af60e"
TOLERANCE_ID = "T00-tol-fbd0eb931ac1c7153403"

SHORT = [1, 7, 8, 9, 15, 16, 17, 1023, 1024, 1025, 1031, 1032, 1033]
LONG = [4096, 16384, 32768, 65536, 130944]
PROMPT = "The archive records that the secret code is 73921. Remember this fact. "


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def make_inputs(tokenizer):
    unit = tokenizer.encode(PROMPT, add_special_tokens=False)
    cases = {}
    for length in SHORT + LONG:
        ids = (unit * ((length + 32) // len(unit) + 1))[: length + 24]
        cases[f"length-{length}"] = {
            "input_ids": ids[:length],
            "teacher_ids": ids[length:],
        }
    for name, prompt in [
        ("english", "The capital of France is"),
        ("chinese", "请简要介绍长上下文模型。"),
    ]:
        cases[name] = {
            "input_ids": tokenizer.encode(prompt, add_special_tokens=False),
            "teacher_ids": (unit * (24 // len(unit) + 1))[:24],
        }
    # Deterministic retrieval input; the expected answer is recorded, not assumed.
    needle = tokenizer.encode("\nThe secret code is 73921.\n", add_special_tokens=False)
    filler = tokenizer.encode("An ordinary archive entry. ", add_special_tokens=False)
    question = tokenizer.encode(
        "\nWhat is the secret code? Answer:", add_special_tokens=False
    )
    for length in [4096, 32768, 65536]:
        count = length - len(needle) - len(question)
        body = (filler * (count // len(filler) + 1))[:count]
        split = count // 4
        cases[f"retrieval-{length}"] = {
            "input_ids": body[:split] + needle + body[split:] + question,
            "teacher_ids": (unit * (24 // len(unit) + 1))[:24],
            "expected_answer": "73921",
        }
    return cases


def metrics(torch, expected, actual):
    a, b = expected.float(), actual.float()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return {
            "finite": False,
            "logits_max_abs_error": None,
            "logits_rmse": None,
            "logprobs_max_abs_error": None,
            "top1_agreement": None,
            "first_mismatch": None,
            "min_top1_margin": None,
            "max_mismatch_margin": None,
        }
    delta = a - b
    top_a, top_b = a.argmax(-1), b.argmax(-1)
    mismatches = (top_a != top_b).nonzero()
    first_mismatch = None
    if len(mismatches):
        index = tuple(mismatches[0].tolist())
        first_mismatch = {
            "row_index": list(index),
            "expected_token": top_a[index].item(),
            "actual_token": top_b[index].item(),
            "top1_margin": a[index].topk(2).values.diff().abs().item(),
        }
    margins = a.topk(2).values.diff(dim=-1).abs().squeeze(-1)
    return {
        "max_mismatch_margin": margins[top_a != top_b].max().item()
        if len(mismatches)
        else 0.0,
        "logits_max_abs_error": delta.abs().max().item(),
        "logits_rmse": delta.square().mean().sqrt().item(),
        "logprobs_max_abs_error": (a.log_softmax(-1) - b.log_softmax(-1))
        .abs()
        .max()
        .item(),
        "top1_agreement": (top_a == top_b).float().mean().item(),
        "first_mismatch": first_mismatch,
        "min_top1_margin": (a.topk(2).values.diff(dim=-1).abs().min().item()),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }


def teacher(torch, model, ids, continuation):
    output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    rows = [output.logits[0, -1].float().cpu()]
    cache = output.past_key_values
    for token in continuation:
        token = torch.tensor([[token]], device="cuda")
        prepared = model.prepare_inputs_for_generation(
            token, past_key_values=cache, use_cache=True
        )
        output = model(**prepared, logits_to_keep=1)
        cache = output.past_key_values
        rows.append(output.logits[0, -1].float().cpu())
    return torch.stack(rows)


def read(path):
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        return json.loads(
            gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())
        )
    return json.loads(path.read_text())


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


def load_reference(directory=FIXTURES):
    """Load the frozen contract without requiring historical experiment outputs."""
    report = read(directory / "correctness.json")
    lock = read(directory / "baseline-lock.json")
    calibration = read(directory / "calibration.json")
    inputs = read(directory / "inputs.json")
    thresholds = {
        "logits_max_abs_error": 20.33521270751953,
        "logits_rmse": 3.7059452533721924,
        "logprobs_max_abs_error": 8.020130157470703,
        "repeat_max_abs_error": 0.0,
        "max_mismatch_margin": 0.5,
    }
    if (
        report["baseline_id"] != BASELINE_ID
        or report["status"] != "pass"
        or "T00-" + digest(lock)[:20] != BASELINE_ID
        or report["tolerance_id"] != TOLERANCE_ID
        or "T00-tol-" + digest(calibration)[:20] != TOLERANCE_ID
        or report["thresholds"] != thresholds
    ):
        raise ValueError("Frozen reference identity or tolerance mismatch")
    if digest(inputs) != lock["input_hash"]:
        raise ValueError("Frozen input hash mismatch")
    frozen = read(FIXTURES / "correctness.json")
    trajectories = {c["case_id"]: c["generated_ids"] for c in report["cases"]}
    if trajectories != {c["case_id"]: c["generated_ids"] for c in frozen["cases"]}:
        raise ValueError("Frozen HF generation mismatch")
    return report, lock, inputs


def validate(args):
    """Verify inputs/model identity and record the actual validation source."""
    report, lock, inputs = load_reference(args.baseline)
    if command("git", "rev-parse", "HEAD") != args.expected_sha:
        raise ValueError("Reference full commit SHA mismatch")
    for name, expected in lock["model_hashes"].items():
        if file_hash(args.model / name) != expected:
            raise ValueError(f"Model hash mismatch: {name}")
    if file_hash(Path(__file__).with_name("compat.py")) != lock["compat_sha256"]:
        raise ValueError("Frozen reference compatibility source changed")
    write_json(args.output / "inputs.json", inputs)
    write_json(
        args.output / "reference-source.json",
        {
            "git_sha": args.expected_sha,
            "dirty": bool(command("git", "status", "--porcelain")),
            "source_diff_sha256": digest(command("git", "diff", "HEAD")),
            "reference_sha256": file_hash(Path(__file__)),
            "compat_sha256": lock["compat_sha256"],
            "baseline_id": BASELINE_ID,
            "tolerance_id": TOLERANCE_ID,
            "input_hash": lock["input_hash"],
        },
    )
    return report, lock, inputs


def load_model(model_path, lock):
    """Load official BF16 HF weights in the independently frozen environment."""
    import torch
    from compat import install
    from transformers import AutoConfig, AutoModelForCausalLM

    packages = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    if packages != lock["packages"]:
        raise ValueError("HF packages differ from frozen reference")
    install()
    torch.backends.cuda.matmul.allow_tf32 = False
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config.rope_parameters = {"rope_type": "default", "rope_theta": config.rope_theta}
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()


def export(args):
    """Regenerate teacher logits consumed by V1, without the old matrix runner."""
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    report, lock, inputs = validate(args)
    model = load_model(args.model, lock)
    for name in ("correctness.json", "baseline-lock.json", "calibration.json"):
        write_json(args.output / name, read(args.baseline / name))
    with torch.inference_mode():
        for name, case in inputs.items():
            if args.cases and name not in args.cases:
                continue
            ids = torch.tensor([case["input_ids"]], device="cuda")
            logits = teacher(torch, model, ids, case["teacher_ids"])
            target = args.output / f"correctness-{name}" / "outputs"
            target.mkdir(parents=True)
            torch.save(
                {
                    "decode_logits": logits,
                    "text_positions": torch.arange(
                        len(case["input_ids"]) - 1,
                        len(case["input_ids"]) + len(case["teacher_ids"]),
                    ),
                },
                target / f"{name}.pt",
            )
            print(f"HF teacher: {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=FIXTURES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--cases", nargs="+", choices=list(load_reference()[2]))
    export(parser.parse_args())
