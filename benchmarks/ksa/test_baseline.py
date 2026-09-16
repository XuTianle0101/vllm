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


if __name__ == "__main__":
    unittest.main()
