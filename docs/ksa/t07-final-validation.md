# T07 最终验收与复现

T07 在 A100 80GB 上执行，沿用 T00 冻结环境、权重、输入和容差。
RTX 5090 未实测；A100 结果不得替代 SM120 结论。最终状态以 ticket 和结果报告为准。

## 安装、环境与启动

使用 Python 3.12、uv 和现有 CUDA 扩展；HF 使用独立的 `.venv-ksa-hf`，
严格核对其完整包清单与 T00 lock，不能在 HF 环境安装 vLLM 或升级依赖。
HF 安装和官方兼容修复见 [T00](t00-baseline.md)。

```bash
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python -e . \
  --torch-backend=auto --config-settings editable_mode=compat
.venv/bin/python -I -c 'import vllm; print(vllm.__file__)'
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1
```

标准服务（本地验证端口）：

```bash
.venv/bin/vllm serve ../models/KSA-4B-base \
  --served-model-name ksa --host 127.0.0.1 --port 18005 \
  --enforce-eager --no-enable-prefix-caching \
  --max-model-len 131072 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 --additional-config '{"ksa_cudagraph":true}'
```

日志必须显示 `KSAForCausalLM`，原始模型配置无须改写。
`--enforce-eager` 禁用通用图路径；`ksa_cudagraph` 独立启用 KSA decode 图。
单个 prefill chunk 仍至多 4096 文本 token，长输入由 scheduler 分块。
总文本上限 131072，包括输出预留；内部 summary 不计入 usage。
首版仅 BF16、单卡、TP=PP=DP=1。量化、LoRA、prefix caching、speculation、
KV 传输/卸载、V2 runner 等限制仍见 [V1 说明](v1-serving.md)。
base 模型的 chat 需要用户提供模板，验收接口为 completions。

## 完整实验命令

从 vLLM 仓库根目录运行。`T07_SHA` 设置为 ticket 记录的代码提交；
`OUT` 使用全新目录。矩阵允许在相同提交/环境下续跑，已记录的失败保留，
修复后须换结果目录。所有 GPU 命令顺序执行。

```bash
git fetch origin releases/v0.26.0-ksa
git checkout "$T07_SHA"
test "$(git rev-parse HEAD)" = "$T07_SHA"
BASELINE=../results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw
MODEL=../models/KSA-4B-base
OUT=../results/T07/final
mkdir -p "$OUT"
.venv/bin/python -m pytest tests/model_executor/test_ksa_prefill.py -q
.venv/bin/python -m unittest discover -s benchmarks/ksa -p test_baseline.py
.venv/bin/python benchmarks/ksa/plain_qwen3.py --output "$OUT/plain-qwen3"
.venv/bin/python benchmarks/ksa/v1.py --model "$MODEL" --baseline "$BASELINE" \
  --all-cases --max-model-len 131072 --row-budget 4096 --output "$OUT/accuracy"
.venv/bin/python benchmarks/ksa/cudagraph_v1.py --model "$MODEL" \
  --baseline "$BASELINE" --expected-sha "$T07_SHA" --all-cases \
  --output "$OUT/graphs-retrieval"
.venv/bin/python benchmarks/ksa/v1_pressure.py --model "$MODEL" \
  --baseline "$BASELINE" --output "$OUT/pressure"
.venv/bin/python benchmarks/ksa/final_validation.py --model "$MODEL" \
  --baseline "$BASELINE" --expected-sha "$T07_SHA" --output "$OUT/performance"
```

启动上述 HTTP 服务后运行，结束后停止服务再做其他 GPU 实验：

```bash
.venv/bin/python benchmarks/ksa/v1_serving.py --url http://127.0.0.1:18005 \
  --model ksa --output "$OUT/http"
```

## 指标定义与门禁

性能矩阵是 4096/16384/32768/65536/130944 prompt × 128 输出，vLLM 并发 1/4/8，
每个形状分别 eager/graph；HF 官方缓存路径只跑单请求。同卡子进程顺序执行，
每项一次预热、五次正式重复。失败子进程不会阻断其他形状，OOM 保留 traceback。

vLLM 时间包含标准 V1 scheduler、模型、采样和主机分发，不含 HTTP；
HF 包含官方 summary 插入、forward、argmax 和同步。TTFT 从提交请求到首个 token，
TPOT 从相邻 token 返回时间计算；并发排队、抢占和混合 prefill 会体现于用户可见延迟。
启动/profiling 与图捕获单列；正式重复内重捕获使该重复无效。
`kv.json` 是计时之外两 token 探针在 prompt 结束后的实际 scheduler 各组物理页，
计时每步另记录最大在用页并检查全部归还。

`summary.json` 只评价性能，不会自动宣称 T07 完成。
16K/32K/64K 的 graph 单请求 decode 最慢重复必须快于 HF 最快重复，
eager 同时披露。缺失、OOM、少于五次或重捕获均不得视为通过。
长距离检索检查原输入中的 `expected_answer`；生成轨迹差异只比较共同前缀。
冻结教师位置 logits 比较覆盖全输入集，避免导出长序列全部词表 logits。
页压缩、128K 无 OOM、精度、服务、生命周期、普通 Qwen3 回归及性能回退分析
需在结果报告中逐项汇总后，才能将 ticket 改为 DONE。

保留 `environment.json`、`processes.json`、每个子进程日志、HF 原始 timing、
vLLM `timing.json` / `startup.json` / `kv.json`、精度 logits 和 HTTP/压力报告。
大型 logits 留在工作区，版本库只提交报告、紧凑指标与文件哈希。
