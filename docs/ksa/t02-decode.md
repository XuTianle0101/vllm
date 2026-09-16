# T02 Python cached decode

状态：DONE，A100 单请求缓存 decode 精度与停止检查通过。
[精度与性能报告](results/T02/a100-20260916T071809Z/README.md) 保留生成分叉与相对 HF 的性能差距。

## 使用契约

通过 vLLM `get_model` 加载 `KSAForCausalLM` 后，使用
`KSAPythonRunner(model).generate(input_ids, max_tokens=128, eos_token_ids=[...])`。
`input_ids` 为模型设备上的一维整数 tensor。结果包含 `token_ids`、`prompt_tokens`、
`completion_tokens` 与 `finish_reason`（`stop` 或 `length`）。显示文本使用
`tokenizer.decode(result.token_ids, skip_special_tokens=True)`；EOS 计入 completion。
`max_tokens=0` 返回空 completion，`ignore_eos=True` 固定生成预算。

缓存由一次 generate 独占，结束或异常时释放。直接教师强制调用可传
`cache=KSACache()`，首次从文本位置 0 开始，以后每次仅传一个新文本 token，
位置等于 `cache.text_tokens`。各层 KV 包含历史文本与 summary；文本数与内部行数
分别记录。一个完整 forward 成功后才提交缓存，防止层内异常造成部分更新。

块末文本与 summary 同次 forward，summary 位置复用该块最后文本 RoPE 位置。
矩形 decode 掩码读取全历史 KV，维持 T00 可见性：summary 可见本块因果文本与自身，
文本可见窗口内文本及严格超窗的 summary。最后文本行用于预测，summary 不进入输出。

只支持单请求、贪心、无量化、TP=PP=1；首次 prefill 最大 4096 token，prompt 加生成
预算最大 8192（同时受模型位置上限约束）。普通 serving scheduler、随机采样、批量、
chunked prefill、prefix cache、分页、KV 压缩、推测解码、CUDA graphs 不支持。
这是 vLLM 模型组件之上的专用 Python 运行器，不是 `LLM.generate` 或 HTTP 服务。

## 复现

本机已有 `.venv` 和冻结 `.venv-ksa-hf`。新环境分别按 vLLM 开发安装流程以及
`benchmarks/ksa/requirements.lock` 准备，HF 环境不得安装额外依赖，否则身份校验失败。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/model_executor/test_ksa_prefill.py -q
.venv/bin/python -m pytest benchmarks/ksa/test_baseline.py -q
.venv/bin/pre-commit run --files \
  vllm/model_executor/models/ksa.py \
  vllm/model_executor/models/ksa_prefill.py \
  vllm/model_executor/models/ksa_decode.py \
  tests/model_executor/test_ksa_prefill.py benchmarks/ksa/decode.py
```

服务器运行命令（结果目录必须不存在）：

```bash
KSA_SHA=4f2845da8a3565d8bbedec082fdbf2fc40319099
test "$(git rev-parse HEAD)" = "$KSA_SHA"
KSA_RESULT="/workspace/volume/h20-data/xutianle/KSA/results/T02/a100-$(date -u +%Y%m%dT%H%M%SZ)"
set -o pipefail
.venv/bin/python benchmarks/ksa/decode.py \
  --model /workspace/volume/h20-data/xutianle/KSA/models/KSA-4B-base \
  --baseline /workspace/volume/h20-data/xutianle/KSA/results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw \
  --hf-python .venv-ksa-hf/bin/python \
  --expected-sha "$KSA_SHA" --output "$KSA_RESULT" \
  2>&1 | tee "${KSA_RESULT}.log"
```

先 `git fetch origin releases/v0.26.0-ksa` 并 `git switch --detach "$KSA_SHA"`；
本服务器已在本地提交。当前 GitHub 凭据失效，远端尚未同步。
恢复认证后执行 `git push origin HEAD:releases/v0.26.0-ksa`，禁止 force-push。

## 验证与报告

基线 `T00-305bf8f9dc429f1af60e`，容差 `T00-tol-fbd0eb931ac1c7153403`。
所有支持的冻结输入逐步教师强制比较 logits/logprobs，各重复三次；完整保留
prompt 尾部与 24 个续写位置。长度 4096 的 decode 不再被 prefill 上限截断。

固定 7/8/9 token、中英文提示各贪心生成 128 token 并重复。记录与 HF 的全部 ID 和
首个差异；自由生成可能因 BF16 低 margin 选择而分叉。为检验分叉后的每一步，
HF 另沿 vLLM 生成轨迹逐步导出 logits，以完全相同前缀应用冻结数值与 margin 门禁。
精确 token 一致性作为独立结果，不能将不同前缀的逐时刻 logits 当成同输入比较。

停止测试将首次实际生成 token 注入 EOS，要求计数为 1、原因 stop；同时检查 0/1/128
预算、ignore_eos、文本 usage 和 summary 过滤。CPU 测试补充缓存异常原子性和拒绝路径。

性能单独计时，不导出 logits；HF/vLLM 均为 BF16、batch=1、1K/4K prompt、128 输出，
EOS 关闭，预热一次、正式五次。TTFT 包括 prefill 与 argmax；TPOT 包括输入准备、
forward、argmax 和设备同步。边界按已消费文本数整除 8 判断。加载与冷编译不计入
热态指标；保留全部逐 token 延迟、峰值 allocated 显存与运行环境。

结果包含 `environment.json`、`inputs.json`、`correctness.json`、`generation.json`、
`timing.json`、`hf-timing.json`、`summary.md` 以及 HF 日志。原始生成轨迹 logits
保存在 `.pt`，不提交 Git。失败写入 `failure.json` 并返回非零，不推进下一 ticket。
