# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze tolerances from independent FP32 calibration, then enforce T00 gates."""

import argparse
import csv
import json
from pathlib import Path

from baseline import LONG, SHORT, digest, file_hash, metrics, write_json

# Chosen before the acceptance matrix. Never fitted to BF16 prefill/decode errors.
POLICY = {
    "version": "ksa-bf16-calibration-v1",
    "noise_multiplier": 2.0,
    "floors": {
        "logits_max_abs_error": 0.5,
        "logits_rmse": 0.125,
        "logprobs_max_abs_error": 0.5,
    },
    "ceilings": {
        "logits_max_abs_error": 24.0,
        "logits_rmse": 4.0,
        "logprobs_max_abs_error": 12.0,
    },
    "fp32_max_abs_error": 0.005,
    "repeat_max_abs_error": 0.0,
    # Two BF16 ULPs at a logit magnitude of 32; ties may choose different tokens.
    "max_mismatch_margin": 0.5,
}


def read(path):
    return json.loads(path.read_text())


def assess(output):
    import torch
    from run_matrix import CALIBRATION_CASES

    report = read(output / "correctness.json")
    failures, calibration = [], []
    noise = dict.fromkeys(POLICY["floors"], 0.0)
    baseline_id = report.get("baseline_id")
    expected_cases = {f"length-{n}" for n in SHORT + LONG} | {
        "english",
        "chinese",
        "retrieval-4096",
        "retrieval-32768",
        "retrieval-65536",
    }
    if {r["case_id"] for r in report["cases"]} != expected_cases:
        failures.append("Incomplete correctness matrix")
    for case in CALIBRATION_CASES:
        fp_dir = output / f"calibration-{case}"
        bf_dir = output / f"correctness-{case}"
        try:
            fp_report = read(fp_dir / "correctness.json")
            fp_row = fp_report["cases"][0]
            if (
                not fp_row.get("finite")
                or fp_row["logits_max_abs_error"] > POLICY["fp32_max_abs_error"]
                or fp_row.get("generation_repeat_equal") is not True
            ):
                raise ValueError("FP32 equivalence or generation repeat failed")
            fp_lock = read(fp_dir / "baseline-lock.json")
            bf_lock = read(output / "baseline-lock.json")
            if {**fp_lock, "dtype": "bfloat16"} != bf_lock:
                raise ValueError("Calibration does not match baseline identity")
            fp_path = fp_dir / "outputs" / f"{case}.pt"
            bf_path = bf_dir / "outputs" / f"{case}.pt"
            fp = torch.load(fp_path, weights_only=True, map_location="cpu")
            bf = torch.load(bf_path, weights_only=True, map_location="cpu")
            comparisons = {
                key: metrics(torch, fp[key], bf[key])
                for key in ["decode_logits", "prefill_logits"]
            }
            for value in comparisons.values():
                if not value["finite"]:
                    raise ValueError("Nonfinite calibration")
                for key in noise:
                    noise[key] = max(noise[key], value[key])
            calibration.append(
                {
                    "case_id": case,
                    "fp32_baseline_id": fp_report["baseline_id"],
                    "fp32_equivalence": fp_row["logits_max_abs_error"],
                    "bf16_vs_fp32": comparisons,
                    "tensor_hashes": {
                        "fp32": file_hash(fp_path),
                        "bf16": file_hash(bf_path),
                    },
                }
            )
        except (OSError, ValueError, KeyError, IndexError) as exc:
            failures.append(f"Calibration {case}: {exc}")
    thresholds = {
        key: max(POLICY["floors"][key], POLICY["noise_multiplier"] * value)
        for key, value in noise.items()
    }
    if any(thresholds[k] > POLICY["ceilings"][k] for k in thresholds):
        failures.append("Calibration exceeds predeclared noise budget")
    calibrated = not failures
    evidence = {
        "baseline_id": baseline_id,
        "policy": POLICY,
        "noise": noise,
        "cases": calibration,
        "thresholds": thresholds if calibrated else None,
        "assessment_sha256": file_hash(Path(__file__)),
    }
    write_json(output / "calibration.json", evidence)
    report["thresholds"] = (
        {
            **thresholds,
            "repeat_max_abs_error": 0.0,
            "max_mismatch_margin": POLICY["max_mismatch_margin"],
        }
        if calibrated
        else None
    )
    report["tolerance_id"] = "T00-tol-" + digest(evidence)[:20] if calibrated else None
    unavailable = []
    for row in report["cases"]:
        name = row["case_id"]
        if row["status"] == "oom" and name == "length-130944":
            unavailable.append(name)
            continue
        if row["status"] in {"error", "oom", "fail"}:
            failures.append(f"Correctness {name}: {row['status']}")
            continue
        if not calibrated:
            continue
        passed = (
            row.get("finite")
            and row.get("generation_repeat_equal") is True
            and all(row[k] <= limit for k, limit in thresholds.items())
            and row.get("max_mismatch_margin", float("inf"))
            <= POLICY["max_mismatch_margin"]
            and len(row.get("repeat_metrics", [])) == 2
            and all(
                r["finite"] and r["logits_max_abs_error"] == 0.0
                for r in row["repeat_metrics"]
            )
        )
        row["status"] = "pass" if passed else "fail"
        row["reason"] = (
            "Independent FP32-calibrated BF16 envelope; exact repeats"
            if passed
            else "Frozen tolerance gate failed"
        )
        if not passed:
            failures.append(f"Correctness {name}: frozen tolerance gate failed")
    with (output / "performance.csv").open() as stream:
        performance = list(csv.DictReader(stream))
    for length in LONG:
        rows = [r for r in performance if r["case_id"] == f"length-{length}"]
        if len(rows) != 6 or {int(r["repetition"]) for r in rows} != set(range(-1, 5)):
            failures.append(f"Performance {length}: incomplete repetitions")
        for row in rows:
            if row["status"] == "oom" and length == 130944:
                unavailable.append("performance-length-130944")
            elif row["status"] != (
                "warmup" if int(row["repetition"]) == -1 else "pass"
            ):
                failures.append(f"Performance {length}: {row['status']}")
    for worker in read(output / "semantics.json")["workers"]:
        result = worker["result"]
        if not result or any(r["status"] != "pass" for r in result["cases"]):
            failures.append(f"Semantics {worker['worker']} failed")
    report["status"] = "pass" if not failures else "blocked"
    report["failures"] = failures
    report["unavailable_for_comparison"] = sorted(set(unavailable))
    write_json(output / "correctness.json", report)
    (output / "summary.md").write_text(
        f"# T00\n\nStatus: {report['status'].upper()}\n\n"
        f"Baseline: {baseline_id}\n\nTolerance: {report['tolerance_id']}\n\n"
        f"Unavailable: {report['unavailable_for_comparison']}\n\nFailures: {failures}\n"
    )
    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if assess(args.output) else 1)
