#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." || exit 1
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
OUT="${OUT:-../results/refactor-01-20260918}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
ROW_BUDGET="${ROW_BUDGET:-257}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
MODES="${MODES:-eager graph}"
failures=0
for mode in $MODES; do
  test ! -e "$OUT/http-$mode-server.log" || exit 1
  enabled=false
  if [ "$mode" = graph ]; then enabled=true; fi
  .venv/bin/vllm serve ../models/KSA-4B-base --served-model-name ksa --host 127.0.0.1 --port 18015 --dtype bfloat16 --enforce-eager --no-enable-prefix-caching --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 8 --max-num-batched-tokens "$ROW_BUDGET" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --additional-config "{\"ksa_cudagraph\":$enabled}" > "$OUT/http-$mode-server.log" 2>&1 &
  server_pid=$!
  trap 'kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
  ready=false
  for attempt in {1..180}; do
    if curl -fsS http://127.0.0.1:18015/health >/dev/null 2>&1; then ready=true; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    sleep 1
  done
  if [ "$ready" = true ]; then
    .venv/bin/python benchmarks/ksa/v1_serving.py --url http://127.0.0.1:18015 --model ksa --output "$OUT/http-$mode" > "$OUT/http-$mode.log" 2>&1
    rc=$?
  else
    rc=125
  fi
  if [ "$rc" -ne 0 ]; then failures=$((failures + 1)); fi
  printf '%s HTTP %s rc=%s\n' "$(date -u +%FT%TZ)" "$mode" "$rc" | tee -a "$OUT/progress.log"
  kill -TERM "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - EXIT
done

test "$failures" -eq 0
