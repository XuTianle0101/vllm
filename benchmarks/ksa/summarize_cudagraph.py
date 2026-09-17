# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact, hash-linked T05 evidence from an existing completed experiment."""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from baseline import file_hash, write_json
from prefill import read


def mean_or_none(values):
    values = list(values)
    return statistics.mean(values) if values else None


def summarize_trace(path, mode):
    events = read(path)["traceEvents"]
    ranges = sorted(
        (
            e
            for e in events
            if e.get("ph") == "X"
            and e.get("cat") == "user_annotation"
            and e.get("name", "").startswith("ksa.")
        ),
        key=lambda e: e["ts"],
    )
    if mode == "graph":
        first = next(e["ts"] for e in ranges if e["name"] == "ksa.metadata")
    else:
        qkv = [e for e in ranges if e["name"] == "ksa.projection_qkv_rope"]
        # Harness profiles four sequential prefills followed by eight decode steps.
        if len(qkv) % 12:
            raise ValueError("unexpected trace shape")
        first = qkv[len(qkv) // 12 * 4]["ts"]
    syncs = [
        e["ts"] + e["dur"]
        for e in events
        if e.get("name") == "cudaDeviceSynchronize"
        and "dur" in e
        and e["ts"] + e["dur"] <= first
    ]
    start = max(syncs) if syncs else first
    cpu = defaultdict(float)
    gpu = defaultdict(float)
    for event in events:
        if event.get("ph") != "X" or event["ts"] < start:
            continue
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith(
            "ksa."
        ):
            cpu[event["name"]] += event["dur"] / 8000
        if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu[event["name"]] += event["dur"] / 8000
    return dict(
        steps=8,
        batch=4,
        length=1024,
        cpu_range_ms_per_step=dict(cpu),
        gpu_activity_ms_per_step=sum(gpu.values()),
        gpu_hotspots_ms_per_step=dict(
            sorted(gpu.items(), key=lambda p: p[1], reverse=True)[:20]
        ),
        note="prefill excluded; inclusive CPU ranges overlap GPU time, do not add",
    )


def summarize(raw):
    status = read(raw / "status.json")
    if status["status"] != "pass":
        raise ValueError("cannot summarize a failed experiment as accepted")
    if status["free_pages"] != status["expected_free_pages"]:
        raise ValueError("page leak after experiment")
    correctness = read(raw / "correctness.json")
    timing = read(raw / "timing.json")
    rows = []
    for length, batch in sorted({(r["length"], r["batch"]) for r in timing}):
        row = dict(length=length, batch=batch)
        for mode in ("eager", "graph"):
            selected = [
                r
                for r in timing
                if r["length"] == length
                and r["batch"] == batch
                and r["mode"] == mode
                and r["repetition"] >= 0
            ]
            if len(selected) < 5 or any(r["new_captures"] for r in selected):
                raise ValueError("need five uncontaminated steady repetitions")
            means = [r["mean_ms"] for r in selected]
            phase_samples = [
                (ms, 7 in phases)
                for r in selected
                for ms, phases in zip(r["step_ms"], r["phases"])
            ]
            row[mode] = dict(
                mean_ms=statistics.mean(means),
                stdev_ms=statistics.stdev(means),
                min_ms=min(means),
                max_ms=max(means),
                ordinary_ms=mean_or_none(ms for ms, b in phase_samples if not b),
                with_summary_ms=mean_or_none(ms for ms, b in phase_samples if b),
                tokens_per_s=statistics.mean(r["tokens_per_s"] for r in selected),
                peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in selected),
                peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in selected),
            )
        row["speedup"] = row["eager"]["mean_ms"] / row["graph"]["mean_ms"]
        rows.append(row)
    gates = (
        "logits_max_abs_error",
        "logits_rmse",
        "logprobs_max_abs_error",
        "max_mismatch_margin",
    )
    accuracy = {
        ref: {
            key: max(r[key] for r in correctness if r["reference"] == ref)
            for key in gates
        }
        for ref in ("eager", "HF")
    }
    return dict(
        environment=read(raw / "environment.json"),
        status=status,
        accuracy=dict(
            checks=len(correctness),
            passed=sum(r["status"] == "pass" for r in correctness),
            maxima=accuracy,
        ),
        performance=rows,
        startup=read(raw / "startup.json"),
        raw_path=str(raw.resolve()),
        artifacts={p.name: file_hash(p) for p in sorted(raw.iterdir()) if p.is_file()},
        decode_traces={
            mode: summarize_trace(raw / f"{mode}-trace.json", mode)
            for mode in ("eager", "graph")
            if (raw / f"{mode}-trace.json").exists()
        },
        memory_note="live graph cache in shared process; peaks are not isolated deltas",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    write_json(args.output, summarize(args.raw))
