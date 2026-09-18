#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." || exit 1
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
OUT="${OUT:-../results/refactor-01-20260918}"
mkdir "$OUT" || exit 1
MODEL="${MODEL:-../models/KSA-4B-base}"
BASELINE="${BASELINE:-../results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw}"
SHA="${SHA:-81d2c510efbfdf9efdd801e6b6204abfaff97c23}"
test "$(git rev-parse HEAD)" = "$SHA" || exit 1
git diff --quiet HEAD -- vllm benchmarks/ksa tests/model_executor/test_ksa_prefill.py || exit 1
failures=0
run() {
  local name="$1"
  shift
  printf '%s START %s\n' "$(date -u +%FT%TZ)" "$name" | tee -a "$OUT/progress.log"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then failures=$((failures + 1)); fi
  printf '%s END %s rc=%s\n' "$(date -u +%FT%TZ)" "$name" "$rc" | tee -a "$OUT/progress.log"
}
run unit .venv/bin/python -m pytest tests/model_executor/test_ksa_prefill.py -q
run contracts .venv/bin/python -m unittest discover -s benchmarks/ksa -p test_baseline.py
run scheduler .venv/bin/python -m pytest tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py -q -k 'not pp and not pipeline'
run prefill .venv/bin/python benchmarks/ksa/v1.py --model "$MODEL" --baseline "$BASELINE" --row-budget 257 --output "$OUT/prefill"
run pressure .venv/bin/python benchmarks/ksa/v1_pressure.py --model "$MODEL" --baseline "$BASELINE" --output "$OUT/pressure"
run plain-qwen3 .venv/bin/python benchmarks/ksa/plain_qwen3.py --output "$OUT/plain-qwen3"
for length in 4096 16384 65536; do
  for mode in eager graph; do
    name="perf-$mode-$length"
    run "$name" .venv/bin/python benchmarks/ksa/final_validation.py --worker --model "$MODEL" --baseline "$BASELINE" --expected-sha "$SHA" --length "$length" --batch 1 --mode "$mode" --output "$OUT/$name"
  done
done
run graphs .venv/bin/python benchmarks/ksa/cudagraph_v1.py --model "$MODEL" --baseline "$BASELINE" --expected-sha "$SHA" --all-cases --output "$OUT/graphs"
run generation-hf .venv-ksa-hf/bin/python benchmarks/ksa/final_generation.py --model "$MODEL" --baseline "$BASELINE" --expected-sha "$SHA" --graphs "$OUT/graphs" --output "$OUT/generation-hf"

# The known HF accuracy failure is retained as a nonzero suite result.
test "$failures" -eq 0
