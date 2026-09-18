# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Force real V1 page pressure and verify recomputation, abort and reuse."""

import argparse
import json
import os
import sys
from pathlib import Path


def run(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    from hf_reference import compare, load_reference
    from v1 import export_capture, install_capture

    from vllm import LLM, SamplingParams

    args.output.mkdir(parents=True, exist_ok=False)
    llm = LLM(
        model=str(args.model),
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=512,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        num_gpu_blocks_override=330,
        gpu_memory_utilization=0.3,
    )
    params = SamplingParams(
        temperature=0, ignore_eos=True, max_tokens=80, allowed_token_ids=[11]
    )
    prompts = [{"prompt_token_ids": [token] * 257} for token in (42, 43)]
    names = ("pressure-a", "pressure-b")
    capture_map = {
        tuple(p["prompt_token_ids"]): (257, name) for p, name in zip(prompts, names)
    }
    llm.collective_rpc(install_capture, args=(capture_map,))
    expected = [llm.generate([p], params)[0].outputs[0].token_ids for p in prompts]
    reference_path = str((args.output / "serial.pt").resolve())
    llm.collective_rpc(export_capture, args=(reference_path,))
    llm.collective_rpc(install_capture, args=(capture_map,))
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    for req_id, prompt in zip(names, prompts):
        engine.add_request(req_id, prompt, params)
    requests = list(scheduler.requests.values())
    seen = {}
    steps = 0
    while engine.has_unfinished_requests():
        for output in engine.step():
            tokens = output.outputs[0].token_ids
            previous = seen.get(output.request_id, [])
            assert tokens[: len(previous)] == previous
            seen[output.request_id] = list(tokens)
        steps += 1
        assert steps < 1000, "V1 preemption failed to make progress"
    preemptions = sum(r.num_preemptions for r in requests)
    assert preemptions > 0
    assert len(seen) == 2
    assert [seen[name] for name in names] == expected
    actual_path = str((args.output / "pressure.pt").resolve())
    llm.collective_rpc(export_capture, args=(actual_path,))
    reference = torch.load(reference_path, weights_only=True)
    actual = torch.load(actual_path, weights_only=True)
    comparisons = {}
    thresholds = load_reference(args.baseline)[0]["thresholds"]
    for name in names:
        positions = sorted(reference[name])
        comparisons[name] = compare(
            torch,
            torch.stack([reference[name][p] for p in positions]),
            torch.stack([actual[name][p] for p in positions]),
            positions,
            thresholds,
        )
        assert comparisons[name]["status"] == "pass"
    prompt = prompts[0]
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == 329
    engine.add_request("abort", prompt, params)
    engine.step()
    assert pool.get_num_free_blocks() < 329
    engine.abort_request(["abort"])
    engine.step()
    assert pool.get_num_free_blocks() == 329
    recovered = llm.generate([prompt], params)[0].outputs[0].token_ids
    assert recovered == expected[0]
    report = {
        "status": "pass",
        "preemptions": preemptions,
        "steps": steps,
        "output_tokens_per_request": len(expected[0]),
        "teacher_forced_comparisons": comparisons,
        "free_blocks": 329,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    engine.engine_core.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
