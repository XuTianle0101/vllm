# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests; run independently of vLLM's GPU pytest fixtures."""

import csv
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
