# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests; run independently of vLLM's GPU pytest fixtures."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from baseline import (
    GENERATION,
    LONG,
    SHORT,
    digest,
    file_hash,
    generate,
    make_inputs,
    performance,
    write_json,
)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class BaselineContractTest(unittest.TestCase):
    def test_repaired_mask_visits_full_tiles_only_once(self):
        """BlockMask full tiles must not also appear among partial tiles."""
        import torch
        from compat import repair_block_mask
        from torch.nn.attention.flex_attention import BlockMask

        count = torch.tensor([[[2]]], dtype=torch.int32)
        indices = torch.tensor([[[[0, 1]]]], dtype=torch.int32)
        original = BlockMask.from_kv_blocks(
            count,
            indices,
            torch.ones_like(count),
            indices[..., :1],
            BLOCK_SIZE=128,
            seq_lengths=(128, 256),
        )
        fixed = repair_block_mask(lambda: original)
        self.assertEqual(fixed.kv_num_blocks.item(), 1)
        self.assertEqual(fixed.kv_indices[0, 0, 0, 0].item(), 1)
        self.assertEqual(fixed.full_kv_num_blocks.item(), 1)
        self.assertEqual(fixed.full_kv_indices.shape[-1], 2)

    def test_matrix_continues_after_worker_crash_and_records_missing_rows(self):
        import run_matrix

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            with (
                patch.object(run_matrix, "LONG", [4096]),
                patch.object(run_matrix, "SHORT", []),
                patch.object(run_matrix, "CALIBRATION_CASES", []),
                patch.object(
                    run_matrix.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=1),
                ) as worker,
                patch("assess.assess", return_value=False),
            ):
                self.assertFalse(
                    run_matrix.run(SimpleNamespace(output=output, model=Path("model")))
                )
            self.assertEqual(worker.call_count, 7)
            with (output / "performance.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["status"] == "error" for row in rows))
            report = json.loads((output / "correctness.json").read_text())
            self.assertEqual(len(report["cases"]), 6)
            self.assertTrue(all(row["status"] == "error" for row in report["cases"]))

    def test_acceptance_rejects_error_outside_independent_calibration(self):
        import assess
        import run_matrix
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            names = [
                "length-8",
                "english",
                "chinese",
                "retrieval-4096",
                "retrieval-32768",
                "retrieval-65536",
            ]
            row = {
                "status": "not_run",
                "finite": True,
                "generation_repeat_equal": True,
                "logits_max_abs_error": 0.0,
                "logits_rmse": 0.0,
                "logprobs_max_abs_error": 0.0,
                "max_mismatch_margin": 0.0,
                "repeat_metrics": [{"finite": True, "logits_max_abs_error": 0.0}] * 2,
            }
            report = {
                "baseline_id": "test",
                "cases": [{**row, "case_id": name} for name in names],
            }
            write_json(output / "correctness.json", report)
            write_json(output / "baseline-lock.json", {"dtype": "bfloat16"})
            write_json(
                output / "semantics.json",
                {
                    "workers": [
                        {"worker": "test", "result": {"cases": [{"status": "pass"}]}}
                    ]
                },
            )
            (output / "performance.csv").write_text(
                "case_id,repetition,status\n"
                + "".join(
                    f"length-8,{rep},{'warmup' if rep == -1 else 'pass'}\n"
                    for rep in range(-1, 5)
                )
            )
            for mode, dtype in [
                ("calibration", "float32"),
                ("correctness", "bfloat16"),
            ]:
                worker = output / f"{mode}-length-8"
                (worker / "outputs").mkdir(parents=True)
                write_json(worker / "baseline-lock.json", {"dtype": dtype})
                write_json(
                    worker / "correctness.json", {"baseline_id": "test", "cases": [row]}
                )
                torch.save(
                    {
                        "decode_logits": torch.tensor([[1.0, 2.0, 3.0]]),
                        "prefill_logits": torch.tensor([[1.0, 2.0, 3.0]]),
                    },
                    worker / "outputs/length-8.pt",
                )
            with (
                patch.object(assess, "LONG", [8]),
                patch.object(assess, "SHORT", []),
                patch.object(run_matrix, "CALIBRATION_CASES", ["length-8"]),
            ):
                self.assertTrue(assess.assess(output))
                report = json.loads((output / "correctness.json").read_text())
                self.assertIsNotNone(report["tolerance_id"])
                report["cases"][0]["logits_max_abs_error"] = 0.6
                write_json(output / "correctness.json", report)
                self.assertFalse(assess.assess(output))
                rejected = json.loads((output / "correctness.json").read_text())
                self.assertEqual(rejected["thresholds"]["logits_max_abs_error"], 0.5)
                self.assertEqual(rejected["cases"][0]["status"], "fail")

    def test_generate_supplies_the_official_ring_cache(self):
        model = Mock()
        factory = Mock()
        with patch.object(
            sys.modules[type(model).__module__],
            "Qwen3RingBufferCache",
            factory,
            create=True,
        ):
            generate(model, "input")
        factory.assert_called_once_with(model.config, model.model._sliding_chunk_nums)
        model.generate.assert_called_once_with(
            "input", past_key_values=factory.return_value, **GENERATION
        )

    def test_oom_row_survives_and_warmup_is_not_a_measured_repetition(self):
        torch = Mock()
        torch.cuda.OutOfMemoryError = MemoryError
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "outputs").mkdir()
            inputs = make_inputs(Tokenizer())
            with (
                patch(
                    "baseline.timed_generation",
                    side_effect=[MemoryError("test OOM"), *[{"ttft_ms": 1.0}] * 29],
                ),
                self.assertLogs(level="ERROR"),
            ):
                performance(
                    torch,
                    Mock(),
                    inputs,
                    SimpleNamespace(output=output),
                    {"ticket": "T00", "git_sha": "test", "baseline_id": "test"},
                    1,
                )
            with (output / "performance.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 30)
            self.assertEqual(rows[0]["status"], "oom")
            self.assertEqual(rows[0]["ttft_ms"], "")
            self.assertEqual(sum(r["status"] == "pass" for r in rows), 25)
            self.assertEqual(sum(r["status"] == "warmup" for r in rows), 4)

    def test_matrix_preserves_lengths_and_context_budget(self):
        cases = make_inputs(Tokenizer())
        for n in SHORT + LONG:
            case = cases[f"length-{n}"]
            self.assertEqual(len(case["input_ids"]), n)
            self.assertEqual(len(case["teacher_ids"]), 24)
            self.assertLessEqual(n + GENERATION["max_new_tokens"], 131072)
        for n in [4096, 32768, 65536]:
            case = cases[f"retrieval-{n}"]
            self.assertEqual(len(case["input_ids"]), n)
            self.assertEqual(bytes(case["input_ids"]).count(b"73921"), 1)

    def test_baseline_identity_tracks_input_and_code_changes(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        self.assertNotEqual(digest({"input_ids": [1]}), digest({"input_ids": [2]}))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            write_json(path, {"revision": "old"})
            old = file_hash(path)
            write_json(path, {"revision": "new"})
            self.assertNotEqual(old, file_hash(path))
            self.assertFalse(path.with_suffix(".json.tmp").exists())


class FinalValidationContractTest(unittest.TestCase):
    def test_capture_telemetry_preserves_the_before_snapshot(self):
        """In-process RPC must not hide new captures through a mutable alias."""
        import torch
        from final_validation import telemetry

        graphs = SimpleNamespace(startup=[{"batch": 1}])
        worker = SimpleNamespace(model_runner=SimpleNamespace(ksa_graphs=graphs))
        with patch.object(torch.accelerator, "max_memory_allocated", return_value=0):
            before = telemetry(worker)
            graphs.startup.append({"batch": 4})
            after = telemetry(worker)
        self.assertEqual(len(before["captures"]), 1)
        self.assertEqual(len(after["captures"]), 2)

    def test_final_decode_gate_rejects_missing_or_contaminated_measurements(self):
        """Missing jobs, graph capture and leaked pages invalidate timing."""
        from final_validation import LENGTHS, ROOT, summarize

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                before=ROOT / "docs/ksa/results/refactor-01/performance.csv",
                output=root,
            )
            self.assertEqual(summarize(args)["status"], "fail")
            with args.before.open() as stream:
                previous = list(csv.DictReader(stream))
            for length in LENGTHS:
                for mode in ("eager", "graph"):
                    rows = [
                        {
                            k: v if k in ("mode", "status") else float(v)
                            for k, v in row.items()
                        }
                        for row in previous
                        if int(row["length"]) == length and row["mode"] == mode
                    ]
                    target = root / f"{mode}-{length}-1"
                    target.mkdir()
                    write_json(target / "timing.json", rows)
            self.assertEqual(summarize(args)["status"], "pass")
            target = root / "graph-4096-1" / "timing.json"
            original = json.loads(target.read_text())
            for field, value in (
                ("status", "capture_contaminated"),
                ("new_captures", 1),
                ("free_pages_after", 0),
                ("repetition", 1),
                ("mean_decode_ms", float("nan")),
            ):
                rows = [dict(row) for row in original]
                next(r for r in rows if r["repetition"] == 0)[field] = value
                write_json(target, rows)
                self.assertEqual(summarize(args)["status"], "fail")


class HFReferenceContractTest(unittest.TestCase):
    """Frozen inputs and public numerical gates survive reference extraction."""

    def test_frozen_inputs_match_original_constructor_and_failure_prefixes(self):
        import hf_reference as reference

        _, lock, inputs = reference.load_reference()
        self.assertEqual(reference.digest(inputs), lock["input_hash"])
        self.assertEqual(reference.make_inputs(Tokenizer()), make_inputs(Tokenizer()))
        self.assertEqual(list(inputs), list(make_inputs(Tokenizer())))
        self.assertEqual(len(inputs), 23)
        failures = reference.read(reference.FIXTURES / "known-failures.json")
        self.assertEqual(
            {(r["case"], r["mode"], r["repeat"]) for r in failures},
            {
                (f"length-{n}", mode, rep)
                for n in (1023, 1024, 1031, 1032, 1033)
                for rep, mode in enumerate(("eager", "graph", "graph"))
            },
        )
        self.assertTrue(all(len(r["generated_ids"]) == 128 for r in failures))

    def test_reference_rejects_changed_tokens_and_relaxed_thresholds(self):
        import shutil

        import hf_reference as reference

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "reference"
            shutil.copytree(reference.FIXTURES, root)
            report, _, inputs = reference.load_reference(root)
            report["thresholds"]["logits_max_abs_error"] += 0.01
            write_json(root / "correctness.json", report)
            with self.assertRaisesRegex(ValueError, "tolerance"):
                reference.load_reference(root)
            shutil.copy(reference.FIXTURES / "correctness.json", root)
            inputs["length-8"]["input_ids"][0] += 1
            write_json(root / "inputs.json", inputs)
            with self.assertRaisesRegex(ValueError, "input hash"):
                reference.load_reference(root)

    def test_original_and_extracted_gates_agree_on_boundaries_and_nonfinite(self):
        import torch
        from hf_reference import compare, load_reference
        from prefill import compare as original_compare

        thresholds = load_reference()[0]["thresholds"]
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        for error in (0.0, 0.5, 20.33521270751953, 20.33521270751953 + 0.001):
            actual = expected + error
            self.assertEqual(
                compare(torch, expected, actual, [7, 8], thresholds),
                original_compare(torch, expected, actual, [7, 8], thresholds),
            )
        for value in (float("nan"), float("inf")):
            actual = torch.full_like(expected, value)
            result = compare(torch, expected, actual, [7, 8], thresholds)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["first_anomaly_position"], 7)

    def test_teacher_preserves_cache_order_and_logits_rows(self):
        import torch
        from baseline import teacher as original_teacher
        from hf_reference import teacher

        def run(function):
            model = Mock()
            model.side_effect = [
                SimpleNamespace(
                    logits=torch.tensor([[[float(i), 1.0]]]),
                    past_key_values=f"cache-{i}",
                )
                for i in range(3)
            ]
            model.prepare_inputs_for_generation.side_effect = lambda token, **kwargs: (
                dict(input_ids=token, **kwargs)
            )
            # Exercise the public teacher interface on CPU; CUDA equivalence is
            # measured separately with the actual frozen HF model.
            real_tensor = torch.tensor
            with patch.object(
                torch, "tensor", side_effect=lambda x, **kw: real_tensor(x)
            ):
                rows = function(torch, model, "prompt", [7, 8])
            calls = model.prepare_inputs_for_generation.call_args_list
            self.assertEqual(
                [c.kwargs["past_key_values"] for c in calls], ["cache-0", "cache-1"]
            )
            self.assertEqual([c.args[0].item() for c in calls], [7, 8])
            return rows

        torch.testing.assert_close(run(teacher), run(original_teacher), rtol=0, atol=0)


class StandardValidationTest(unittest.TestCase):
    """Guard diagnostic token forcing and the frozen performance decision rule."""

    def test_teacher_preserves_raw_logits_across_slot_moves_and_reuse(self):
        import torch
        from final_decode import FrozenTeacher

        from vllm import SamplingParams
        from vllm.v1.sample.logits_processor.interface import (
            BatchUpdate,
            MoveDirectionality,
        )

        processor = FrozenTeacher(None, None, False)
        output_a, output_b = [], []

        def added(index, name, teacher, output):
            return (
                index,
                SamplingParams(extra_args=dict(case=name, teacher_ids=teacher)),
                [42],
                output,
            )

        processor.update_state(
            BatchUpdate(
                2,
                [],
                [added(0, "a", [1, 2], output_a), added(1, "b", [3, 0], output_b)],
                [],
            )
        )
        raw = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        self.assertEqual(processor.apply(raw.clone()).argmax(-1).tolist(), [1, 3])
        torch.testing.assert_close(processor.rows["a"][0], raw[0])
        output_a.append(1)
        output_b.append(3)
        processor.update_state(
            BatchUpdate(2, [], [], [(0, 1, MoveDirectionality.SWAP)])
        )
        self.assertEqual(processor.apply(raw.clone()).argmax(-1).tolist(), [0, 2])
        torch.testing.assert_close(processor.rows["a"][1], raw[1])
        processor.update_state(
            BatchUpdate(1, [0], [], [(1, 0, MoveDirectionality.UNIDIRECTIONAL)])
        )
        output_a.append(2)
        torch.testing.assert_close(processor.apply(raw[:1].clone()), raw[:1])
        processor.update_state(BatchUpdate(1, [0], [added(0, "c", [0], [])], []))
        self.assertEqual(processor.apply(raw[:1].clone()).argmax(-1).item(), 0)
        self.assertEqual(set(processor.rows), {"a", "b", "c"})

    def test_performance_requires_both_mean_and_range_for_regression(self):
        from final_validation import compare_metric

        for old, new, metric, expected in (
            ([100] * 5, [106] * 5, "ttft_ms", "regression"),
            ([100] * 5, [102] * 5, "ttft_ms", "retest"),
            ([90, 90, 100, 110, 110], [99, 99, 109, 119, 119], "ttft_ms", "retest"),
            ([100] * 5, [94] * 5, "output_tokens_per_s", "regression"),
            ([100] * 5, [110] * 5, "peak_memory_bytes", "retest"),
            ([100] * 5, [90] * 5, "ttft_ms", "pass"),
        ):
            with self.subTest(metric=metric, new=new):
                self.assertEqual(compare_metric(old, new, metric)["status"], expected)


if __name__ == "__main__":
    unittest.main()
