# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen teacher-forced decode through the standard V1 scheduler and sampler."""

import argparse
import os
from pathlib import Path

import torch
from hf_reference import compare, validate, write_json

from vllm.v1.sample.logits_processor import LogitsProcessor
from vllm.v1.sample.logits_processor.interface import MoveDirectionality


class FrozenTeacher(LogitsProcessor):
    """Capture unmodified logits, then force the next frozen teacher token.

    Partial prefill samples are discarded by V1; their position-zero capture is
    overwritten by the completed prompt. Output lists track actual V1 progress.
    """

    def __init__(self, vllm_config, device, is_pin_memory):
        self.requests = {}
        self.rows = {}

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.requests.pop(index, None)
        for index, params, _, output in batch_update.added:
            extra = params.extra_args
            self.requests[index] = (extra["case"], extra["teacher_ids"], output)
        for source, target, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                self.requests[source], self.requests[target] = (
                    self.requests[target],
                    self.requests[source],
                )
            else:
                self.requests[target] = self.requests.pop(source)

    def apply(self, logits):
        for index, (name, teacher, output) in self.requests.items():
            step = len(output)
            self.rows.setdefault(name, {})[step] = logits[index].float().cpu().clone()
            if step < len(teacher):
                logits[index].fill_(-float("inf"))
                logits[index, teacher[step]] = 0
        return logits


def export(worker, path):
    runner = worker.model_runner
    processor = next(
        p for p in runner.input_batch.logitsprocs.all if isinstance(p, FrozenTeacher)
    )
    torch.save(
        {
            name: {
                step: row[: runner.model.text_vocab_size] for step, row in rows.items()
            }
            for name, rows in processor.rows.items()
        },
        path,
    )
    processor.rows.clear()
    graphs = runner.ksa_graphs
    return dict(
        runner=type(runner).__name__,
        graph_replays=0 if graphs is None else graphs.replays,
        graph_enabled=graphs is not None,
    )


def run(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams

    args.output.mkdir(parents=True, exist_ok=False)
    baseline, _, inputs = validate(args)
    report = dict(git_sha=args.expected_sha, cases=[], runtime=[], status="running")
    # Separate engines verify checkpoint recognition and the actual config key.
    for mode in ("eager", "graph"):
        llm = LLM(
            model=str(args.model),
            dtype="bfloat16",
            enforce_eager=True,
            enable_prefix_caching=False,
            max_model_len=131072,
            max_num_seqs=3,
            max_num_batched_tokens=args.row_budget,
            gpu_memory_utilization=0.5,
            compilation_config={"mode": 0, "custom_ops": ["none"]},
            additional_config={"ksa_cudagraph": mode == "graph"},
            logits_processors=[FrozenTeacher],
        )
        engine = llm.llm_engine
        pool = engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
        free = pool.get_num_free_blocks()
        try:
            names = list(inputs)
            for start in range(0, len(names), 3):
                batch = names[start : start + 3]
                if mode == "graph":
                    batch.reverse()
                outputs = llm.generate(
                    [{"prompt_token_ids": inputs[n]["input_ids"]} for n in batch],
                    [
                        SamplingParams(
                            temperature=0,
                            max_tokens=len(inputs[n]["teacher_ids"]) + 1,
                            ignore_eos=True,
                            extra_args=dict(
                                case=n, teacher_ids=inputs[n]["teacher_ids"]
                            ),
                        )
                        for n in batch
                    ],
                    use_tqdm=False,
                )
                path = (args.output / f"{mode}-{start}.pt").resolve()
                runtime = llm.collective_rpc(export, args=(str(path),))[0]
                captured = torch.load(path, weights_only=True)
                assert runtime["runner"] == "KSAGPUModelRunner"
                assert runtime["graph_enabled"] == (mode == "graph")
                for name, output in zip(batch, outputs):
                    case = inputs[name]
                    assert output.outputs[0].token_ids[:-1] == case["teacher_ids"]
                    ref = torch.load(
                        args.baseline
                        / f"correctness-{name}"
                        / "outputs"
                        / f"{name}.pt",
                        weights_only=True,
                    )
                    positions = ref["text_positions"].tolist()
                    assert set(captured[name]) == set(range(len(positions)))
                    actual = torch.stack(
                        [captured[name][i] for i in range(len(positions))]
                    )
                    row = compare(
                        torch,
                        ref["decode_logits"],
                        actual,
                        positions,
                        baseline["thresholds"],
                    )
                    report["cases"].append(dict(case=name, mode=mode, **row))
                assert pool.get_num_free_blocks() == free, "scheduler pages leaked"
                write_json(args.output / "results.json", report)
                print(f"V1 teacher {mode}: {batch}", flush=True)
            assert mode != "graph" or runtime["graph_replays"] > 0
            report["runtime"].append(dict(mode=mode, free_pages=free, **runtime))
        finally:
            engine.engine_core.shutdown()
            del llm, engine, pool
            from vllm.distributed import cleanup_dist_env_and_memory

            cleanup_dist_env_and_memory()
    report["status"] = (
        "pass" if all(r["status"] == "pass" for r in report["cases"]) else "fail"
    )
    write_json(args.output / "results.json", report)
    if report["status"] != "pass":
        raise AssertionError("Frozen cached decode gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--row-budget", type=int, default=4096)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "failure.json", dict(error=repr(exc)))
        raise
