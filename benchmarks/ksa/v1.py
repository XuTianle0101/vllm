# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the actual V1 scheduler/runner against frozen HF teacher logits."""

import argparse
import json
import os
import sys
from pathlib import Path

from hf_reference import (
    BASELINE_ID,
    ROOT,
    TOLERANCE_ID,
    command,
    compare,
    file_hash,
    load_reference,
    write_json,
)


def install_capture(worker, prompt_lengths):
    """Observe scheduled text hidden states; never replace execution or sampling."""

    runner = worker.model_runner
    assert type(runner).__name__ == "KSAGPUModelRunner"
    original = runner.model.forward_batch
    runner.ksa_validation = {}
    runner.ksa_validation_max_rows = 0

    def observed(requests):
        rows = sum(
            len(pos) + (int(pos[-1]) + 1) // 8 - int(pos[0]) // 8
            for _, pos, _ in requests
        )
        assert rows <= runner.max_num_tokens
        runner.ksa_validation_max_rows = max(runner.ksa_validation_max_rows, rows)
        outputs = original(requests)
        for (_, positions, cache), hidden in zip(requests, outputs):
            req_id = cache.request.request_id
            prompt = runner.requests[req_id].prompt_token_ids
            case = prompt_lengths.get(tuple(prompt))
            if case is None:
                continue
            first, name = case
            selected = positions >= first - 1
            logits = runner.model.compute_logits(hidden[selected])
            logits = logits[:, : runner.model.text_vocab_size].float().cpu()
            target = runner.ksa_validation.setdefault(name, {})
            for position, row in zip(positions[selected].tolist(), logits):
                target[position] = row.clone()
        return outputs

    runner.ksa_validation_original_forward = original
    runner.model.forward_batch = observed


def export_capture(worker, path):
    import torch

    runner = worker.model_runner
    torch.save(runner.ksa_validation, path)
    runner.model.forward_batch = runner.ksa_validation_original_forward
    del runner.ksa_validation_original_forward
    del runner.ksa_validation
    return {"max_internal_rows": runner.ksa_validation_max_rows}


def run(args):
    import torch

    # Keep the diagnostic callable local; serving uses the normal MP engine.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path.insert(0, str(ROOT))
    from vllm import LLM, SamplingParams

    args.output.mkdir(parents=True, exist_ok=False)
    baseline, lock, inputs = load_reference(args.baseline)
    for name, expected in lock["model_hashes"].items():
        if file_hash(args.model / name) != expected:
            raise ValueError(f"Model hash mismatch: {name}")
    cases = {
        name: case
        for name, case in inputs.items()
        if len(case["input_ids"]) + len(case["teacher_ids"]) < args.max_model_len
        and (args.all_cases or len(case["input_ids"]) <= 4096)
    }
    prompts = [
        {"prompt_token_ids": c["input_ids"] + c["teacher_ids"]} for c in cases.values()
    ]
    llm = LLM(
        model=str(args.model),
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=args.max_model_len,
        max_num_seqs=8,
        max_num_batched_tokens=args.row_budget,
        gpu_memory_utilization=0.5,
        compilation_config={"mode": 0, "custom_ops": ["none"]},
    )
    llm.collective_rpc(
        install_capture,
        args=(
            {
                tuple(p["prompt_token_ids"]): (len(c["input_ids"]), name)
                for p, (name, c) in zip(prompts, cases.items())
            },
        ),
    )
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=0,
            max_tokens=1,
            ignore_eos=True,
            prompt_logprobs=1,
        ),
    )
    capture_path = (args.output / "scheduled-logits.pt").resolve()
    runtime = llm.collective_rpc(export_capture, args=(str(capture_path),))[0]
    captured = torch.load(capture_path, weights_only=True)
    report = {
        "git_sha": command("git", "rev-parse", "HEAD"),
        "dirty": bool(command("git", "status", "--porcelain")),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "baseline_id": BASELINE_ID,
        "tolerance_id": TOLERANCE_ID,
        "row_budget": args.row_budget,
        "runtime": runtime,
        "cases": [],
        "status": "pass",
    }
    for (name, case), output in zip(cases.items(), outputs):
        artifact = torch.load(
            args.baseline / f"correctness-{name}" / "outputs" / f"{name}.pt",
            weights_only=True,
        )
        positions = artifact["text_positions"].tolist()
        actual = torch.stack([captured[name][p] for p in positions])
        result = compare(
            torch, artifact["decode_logits"], actual, positions, baseline["thresholds"]
        )
        result["case"] = name
        assert len(output.prompt_logprobs) == len(
            case["input_ids"] + case["teacher_ids"]
        )
        report["cases"].append(result)
        if result["status"] != "pass":
            report["status"] = "fail"
    write_json(args.output / "results.json", report)
    print(json.dumps({"status": report["status"], "cases": len(cases), **runtime}))
    llm.llm_engine.engine_core.shutdown()
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-budget", type=int, default=257)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--all-cases", action="store_true")
    run(parser.parse_args())
