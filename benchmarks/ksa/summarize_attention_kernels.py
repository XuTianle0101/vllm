# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize T06 accuracy and five-repeat performance gates without traces."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(raw):
    timing = read(raw / "timing.json")
    if {(x["length"], x["batch"]) for x in timing} != {
        (length, batch) for length in (1024, 4096) for batch in (1, 4, 8)
    }:
        raise ValueError("incomplete T06 performance matrix")
    comparisons = []
    for length, batch in sorted({(x["length"], x["batch"]) for x in timing}):
        for metric, reference, candidate in (
            ("mean_ms", "T05_eager", "triton_eager"),
            ("mean_ms", "T05_graph", "triton_graph"),
            ("prefill_ms", "T05_eager", "triton_eager"),
        ):
            samples = {}
            for mode in (reference, candidate):
                rows = [
                    x
                    for x in timing
                    if x["length"] == length
                    and x["batch"] == batch
                    and x["mode"] == mode
                    and x["repetition"] >= 0
                ]
                if len(rows) < 5 or any(x["new_captures"] for x in rows):
                    raise ValueError("need five steady repetitions without capture")
                values = [x[metric] for x in rows]
                samples[mode] = dict(
                    median_ms=statistics.median(values),
                    min_ms=min(values),
                    max_ms=max(values),
                    stdev_ms=statistics.stdev(values),
                    values_ms=values,
                )
            ref, new = samples[reference], samples[candidate]
            comparisons.append(
                dict(
                    length=length,
                    batch=batch,
                    metric=metric,
                    reference=reference,
                    candidate=candidate,
                    samples=samples,
                    speedup=ref["median_ms"] / new["median_ms"],
                    exceeds_observed_variation=new["max_ms"] < ref["min_ms"],
                )
            )
    correctness = read(raw / "correctness.json")
    status = read(raw / "status.json")
    passed = (
        status["status"] == "pass"
        and len(correctness) == 68
        and all(x["status"] == "pass" for x in correctness)
        and status["free_pages"] == status["expected_free_pages"]
        and all(x["exceeds_observed_variation"] for x in comparisons)
    )
    return dict(
        status="pass" if passed else "needs_fix",
        environment=read(raw / "environment.json"),
        correctness=[
            {
                key: row[key]
                for key in (
                    "case",
                    "mode",
                    "reference",
                    "status",
                    "finite",
                    "logits_max_abs_error",
                    "logits_rmse",
                    "logprobs_max_abs_error",
                    "max_mismatch_margin",
                    "top1_agreement",
                    "first_anomaly_position",
                )
            }
            for row in correctness
        ],
        runtime=read(raw / "startup.json"),
        page_status=status,
        comparisons=comparisons,
        acceptance_scope="measured GPU/shapes only; no 5090 or service claim",
        raw_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(raw.glob("*.json"))
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["status"] != "pass":
        raise SystemExit("T06 accuracy or performance gate needs investigation")


if __name__ == "__main__":
    main()
