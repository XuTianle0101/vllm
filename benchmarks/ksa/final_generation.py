# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check final V1 greedy trajectories against official HF on identical prefixes.

Run in the frozen HF environment after cudagraph_v1.py has exited. Reuse HF
logits for identical trajectories, but evaluate every exported eager/graph row.
"""

import argparse
import gc
import importlib.metadata as metadata
from pathlib import Path

from baseline import teacher, write_json
from decode import validate
from prefill import BASELINE_ID, TOLERANCE_ID, compare, read


def run(args):
    import torch
    from compat import install
    from transformers import AutoConfig, AutoModelForCausalLM

    args.output.mkdir(parents=True, exist_ok=False)
    baseline, lock, inputs = validate(args)
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    if packages != lock["packages"]:
        raise ValueError("HF packages differ from frozen T00")
    generation = read(args.graphs / "results.json")
    if generation["status"] != "pass":
        raise ValueError("V1 generation/graph validation did not pass")
    names = list(inputs)
    if any(
        set(r) != {str(i) for i in range(len(names))} for r in generation["generations"]
    ):
        raise ValueError("Expected complete final V1 generation matrix")
    captured = [
        torch.load(args.graphs / f"{mode}-{rep}.pt", weights_only=True)
        for rep, mode in enumerate(("eager", "graph", "graph"))
    ]
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
    report = dict(
        git_sha=args.expected_sha,
        baseline_id=BASELINE_ID,
        tolerance_id=TOLERANCE_ID,
        thresholds=baseline["thresholds"],
        cases=[],
        status="running",
    )
    with torch.inference_mode():
        for index, name in enumerate(names):
            prompt = inputs[name]["input_ids"]
            ids = torch.tensor([prompt], device="cuda")
            trajectories = {}
            original = next(
                c["generated_ids"] for c in baseline["cases"] if c["case_id"] == name
            )
            for rep, mode in enumerate(("eager", "graph", "graph")):
                tokens = generation["generations"][rep][str(index)]
                if len(tokens) != 32:
                    raise ValueError("Expected 32 generated text tokens")
                key = tuple(tokens)
                if key not in trajectories:
                    reference = teacher(torch, model, ids, tokens[:-1])
                    trajectories[key] = reference
                    torch.save(reference, args.output / f"hf-{name}-{rep}.pt")
                reference = trajectories[key]
                positions = list(range(len(prompt) - 1, len(prompt) + 31))
                actual = torch.stack(
                    [captured[rep][tuple(prompt)][p] for p in positions]
                )
                row = compare(
                    torch, reference, actual, positions, baseline["thresholds"]
                )
                margin = (
                    (
                        reference.max(-1).values
                        - reference.gather(1, torch.tensor(tokens)[:, None])[:, 0]
                    )
                    .max()
                    .item()
                )
                if margin > baseline["thresholds"]["max_mismatch_margin"]:
                    row["status"] = "fail"
                report["cases"].append(
                    dict(
                        case=name,
                        mode=mode,
                        repeat=rep,
                        chosen_token_max_hf_margin=margin,
                        exact_frozen_hf_tokens=tokens == original[:32],
                        **row,
                    )
                )
                write_json(args.output / "results.json", report)
            gc.collect()
            torch.get_device_module("cuda").empty_cache()
            print(f"HF same-prefix generation: {name}", flush=True)
    report["status"] = (
        "pass"
        if (
            len(report["cases"]) == 3 * len(inputs)
            and all(c["status"] == "pass" for c in report["cases"])
        )
        else "fail"
    )
    write_json(args.output / "results.json", report)
    if report["status"] != "pass":
        raise AssertionError("Final same-prefix HF generation gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "failure.json", dict(error=repr(exc)))
        raise
