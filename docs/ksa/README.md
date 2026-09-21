server_ksa.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONPATH="${SCRIPT_DIR}/vllm${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# KSA currently supports single-GPU eager serving, without prefix caching.
# Plain-text template for base-model evaluation; adds no chat special tokens.
exec "${SCRIPT_DIR}/vllm/.venv/bin/vllm" serve "${MODEL_PATH:-${SCRIPT_DIR}/models/KSA-4B-base}" \
    --served-model-name "${MODEL_NAME:-ksa}" \
    --host 0.0.0.0 \
    --port "${PORT:-8803}" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 257 \
    --gpu-memory-utilization 0.9 \
    --chat-template "{{ messages | map(attribute='content') | join('\n\n') }}" \
    --chat-template-content-format string \
    "$@"
```

smoke_ksa.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

# server_ksa.sh exposes the OpenAI-compatible text completions endpoint.
API_URL="${API_URL:-http://127.0.0.1:${PORT:-8803}/v1}"
MODEL_NAME="${MODEL_NAME:-ksa}"
PROMPT="${1:-${PROMPT:-What is 2 + 2? Answer briefly.}}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0}"
API_KEY="${API_KEY:-EMPTY}"

if [[ ! "$MAX_TOKENS" =~ ^[0-9]+$ ]]; then
    echo "MAX_TOKENS must be a non-negative integer" >&2
    exit 2
fi
if [[ ! "$TEMPERATURE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "TEMPERATURE must be a non-negative number" >&2
    exit 2
fi

# Escape the values that are allowed to contain arbitrary user text.
json_escape() {
    local value=$1
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//$'\n'/\\n}
    value=${value//$'\r'/\\r}
    value=${value//$'\t'/\\t}
    printf '%s' "$value"
}

model_json=$(json_escape "$MODEL_NAME")
prompt_json=$(json_escape "$PROMPT")

# This is intentionally the only inference request made by the smoke test.
curl --fail-with-body --silent --show-error \
    --connect-timeout "${CONNECT_TIMEOUT:-5}" \
    --max-time "${REQUEST_TIMEOUT:-300}" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${API_KEY}" \
    --data-binary @- \
    "${API_URL%/}/completions" <<EOF
{"model":"${model_json}","prompt":"${prompt_json}","max_tokens":${MAX_TOKENS},"temperature":${TEMPERATURE}}
EOF
```

eval_ksa.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# Install in a separate evaluation environment: uv pip install evalscope
# Full GSM8K test set by default; smoke test: ./eval_ksa.sh --limit 10
exec evalscope eval \
    --model "${MODEL_NAME:-ksa}" \
    --api-url "${API_URL:-http://127.0.0.1:${PORT:-8803}/v1}" \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets gsm8k \
    --dataset-args '{"gsm8k": {"few_shot_num": 4}}' \
    --eval-batch-size 8 \
    --generation-config '{"temperature": 0, "max_tokens": 2048}' \
    --work-dir "${SCRIPT_DIR}/results/evalscope/ksa-gsm8k" \
    "$@"
```
