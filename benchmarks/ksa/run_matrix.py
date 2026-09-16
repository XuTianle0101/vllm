# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate CUDA failures, collect every case, and evaluate a calibrated baseline."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from baseline import LONG, SHORT, write_json

CALIBRATION_CASES = ["length-17", "length-1025", "length-4096"]


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [f"length-{n}" for n in SHORT + LONG]
    cases += [
        "english",
        "chinese",
        "retrieval-4096",
        "retrieval-32768",
        "retrieval-65536",
    ]
    jobs = [("calibration", case) for case in CALIBRATION_CASES]
    jobs += [("performance", f"length-{n}") for n in LONG]
    jobs += [("correctness", case) for case in cases]
    correctness = {
        "schema_version": 1,
        "cases": [],
        "status": "awaiting_calibration_review",
        "thresholds": None,
        "tolerance_id": None,
    }
    performance, processes, semantics = [], [], []
    identity = None
    for mode, case in jobs:
        directory = args.output / f"{mode}-{case}"
        command = [
            sys.executable,
            str(Path(__file__).with_name("baseline.py")),
            "--model",
            str(args.model),
            "--output",
            str(directory),
            "--mode",
            "correctness" if mode == "calibration" else mode,
            "--case",
            case,
            "--precision",
            "float32" if mode == "calibration" else "bfloat16",
        ]
        print(f"Running {mode} {case}", flush=True)
        with (args.output / f"{mode}-{case}.log").open("w") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        processes.append(
            {"mode": mode, "case_id": case, "returncode": result.returncode}
        )
        write_json(args.output / "processes.json", processes)
        semantics.append(
            {
                "worker": directory.name,
                "result": read_json(directory / "semantics.json"),
            }
        )
        write_json(args.output / "semantics.json", {"workers": semantics})
        lock = read_json(directory / "baseline-lock.json")
        if lock:
            normalized = {**lock, "dtype": "bfloat16"}
            if identity is not None and normalized != identity:
                raise RuntimeError("Baseline identity changed between workers")
            identity = normalized
            if mode != "calibration":
                for name in ["baseline-lock.json", "environment.json", "inputs.json"]:
                    if not (args.output / name).exists():
                        (args.output / name).write_bytes(
                            (directory / name).read_bytes()
                        )
                env = read_json(directory / "environment.json")
                correctness.update(
                    {k: env[k] for k in ["baseline_id", "git_sha", "ticket"]}
                )
        if mode == "correctness":
            report = read_json(directory / "correctness.json", {})
            rows = report.get("cases", [])
            correctness["cases"].extend(
                rows
                or [
                    {
                        "case_id": case,
                        "status": "error",
                        "reason": f"Worker exited {result.returncode}; see worker log",
                    }
                ]
            )
            write_json(args.output / "correctness.json", correctness)
        elif mode == "performance":
            rows = []
            if (directory / "performance.csv").exists():
                with (directory / "performance.csv").open() as stream:
                    rows = [
                        r for r in csv.DictReader(stream) if r.get("case_id") == case
                    ]
            seen = {int(r["repetition"]) for r in rows}
            for repetition in range(-1, 5):
                if repetition not in seen:
                    rows.append(
                        {
                            "case_id": case,
                            "repetition": repetition,
                            "status": "error",
                            "reason": f"Worker exited {result.returncode}",
                        }
                    )
            performance.extend(rows)
            columns = list(dict.fromkeys(k for row in performance for k in row))
            with (args.output / "performance.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(performance)
    from assess import assess

    return assess(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    sys.exit(0 if run(parser.parse_args()) else 1)
