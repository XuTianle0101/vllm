#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: bash benchmarks/ksa/run_server.sh EXPECTED_SHA MODEL_DIR RESULT_DIR" >&2
    exit 2
fi
expected_sha=$1
model_dir=$(realpath "$2")
result_dir=$(realpath -m "$3")
[[ $(git rev-parse HEAD) == "$expected_sha" ]]
[[ -z $(git status --porcelain) ]]
[[ ! -e "$result_dir" ]]
mkdir -p "$(dirname "$result_dir")"
exec > >(tee "${result_dir}.launcher.log") 2>&1
collect_results() {
    code=$?
    echo "launcher_exit_code=$code"
    if [[ -d "$result_dir" ]]; then
        tar -czf "${result_dir}.tar.gz" --exclude='*.pt' -C "$(dirname "$result_dir")" "$(basename "$result_dir")"
    fi
    exit "$code"
}
trap collect_results EXIT
uv venv --python 3.12 .venv-ksa-hf
uv pip sync --python .venv-ksa-hf/bin/python --torch-backend cu128 --require-hashes benchmarks/ksa/requirements.lock
uv pip freeze --python .venv-ksa-hf/bin/python > "${result_dir}.packages.txt"
.venv-ksa-hf/bin/python benchmarks/ksa/baseline.py --model "$model_dir" --output "$result_dir"
