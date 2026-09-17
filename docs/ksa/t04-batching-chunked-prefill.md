# T04：批处理、分块预填充和基础服务

本轮实现为 TP=PP=1、BF16、eager Python 路径，复用 T03 的
`KVCacheManager` 和文本/summary 两组压缩页。入口是
`vllm.entrypoints.ksa`，不是通用 `vllm serve` / `LLM`。
通用 `GPUModelRunner` 的 dense Qwen KV 布局仍明确拒绝 KSA；本轮没有将
独立调度循环宣称为通用 V1 scheduler 接入。该架构差异仍需在后续原生框架接入中解决。

## 实现与语义

- `KSAForCausalLM.forward_batch` 合并各请求的投影、MLP 和归一化计算，attention
  按请求隔离。支持不同长度、不同 8-token 块相位、任意文本位置切分和批内重排。
- 当前 chunk 所有层的 attention 完成后才提交页；历史页不在早期 query 读取前淘汰。
  运行中请求跨过的整页补 null 映射，避免为已过期文本重新分配真实页。
- `KSABatchedRunner` 用 FCFS 调度，每步计算预算包含内部 summary：
  `text_length + floor((start + text_length)/8) - floor(start/8)`。
  返回 logits、采样、长度、EOS 和 usage 都只计文本。
- 请求按 ID 持有缓存，不依赖 batch 槽位。取消、正常结束、执行错误会释放全部组。
  页池压力优先抢占较晚请求，仅保留文本历史；恢复时重新计算文本和 summary，
  不重复向客户端发送已生成 token。单请求最小 chunk 仍装不下时返回明确错误。
- prompt logprobs 为输入 token 的条件 log probability；首 token 为 null，跨 chunk
  用前一个文本位置预测，不将 summary 位置当成 logits。HTTP 扩展字段
  `prompt_logprobs: true` 返回每个 prompt token 的选中-token logprob，未实现 top-k。
- HTTP 支持 `/v1/models`、`/v1/completions`、SSE 和 `/health`。SSE 最终返回
  finish reason、文本 usage 和 `[DONE]`；断连取消保护执行线程中排队的回收任务。
  输出队列保存累计 token，慢客户端也不会丢 token。

## 支持边界与 OOM

当前仅 greedy (`temperature=0`)、`n=1`、单个字符串或 token-ID prompt；未知字段、
非零 temperature、前缀缓存、stop strings 等直接拒绝。不支持量化、LoRA、推测解码、
图执行、多卡及 chat 模板。每个 chunk 最多 4096 文本 token，总文本长度最多 8192。
mask 和历史 KV gather 仍是 Python 正确性实现，不代表最终长上下文或高性能路径。

页池默认沿用 T03 的单请求容量，可用 `--kv-cache-num-blocks` 显式扩大；多请求共享
固定池，不因并发自动无限扩容。页耗尽触发抢占重算；不可恢复的单请求容量不足和
CUDA 临时工作区 OOM 以错误结束对应请求并释放页，不无限重试。
逻辑回收后预分配池继续存在，因此不能要求整个进程显存归零。

## 验证与复现

正式代码 SHA：`8160714ad8e9a38dc571508014ffc50c7635470f`；
[验收报告](results/T04/a100-8160714ad8/README.md)。命令从仓库根目录运行，输出目录必须不存在。
先设置 `T04_SHA` 为记录的完整代码 SHA；正式脚本拒绝错误 SHA 或 dirty checkout。

```bash
T04_SHA=8160714ad8e9a38dc571508014ffc50c7635470f
git checkout "$T04_SHA"
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python -e . --torch-backend=auto
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/model_executor/test_ksa_prefill.py -q
OMP_NUM_THREADS=1 .venv/bin/python benchmarks/ksa/batching.py \
  --expected-sha "$T04_SHA" \
  --model ../models/KSA-4B-base \
  --baseline ../results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw \
  --output ../results/T04/a100-final
```

上述脚本执行 17 个冻结教师强制案例 × 四种执行方式（整段、两种切分、混合 batch），
另测自由生成、取消后重算、页压力恢复与不可恢复 OOM。自由生成序列差异单列，
用相同生成前缀下的 logits 和冻结 T00 门禁判断；HF 导出在固定 `.venv-ksa-hf` 环境
单独执行。性能默认 1K/4K prompt、128 输出 token、并发 1/4/8，每组一次预热与五次
正式重复。输出 TTFT、稳态/块边界 TPOT、吞吐、峰值、池容量和抢占次数。
并发结果独立报告；不构造 batch=4/8 相对于 HF batch=1 的加速比。

在另一终端启动服务，再测试真实 HTTP、流式结束和三次断连回收：

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m vllm.entrypoints.ksa \
  --model ../models/KSA-4B-base --port 18004
```

```bash
.venv/bin/python benchmarks/ksa/serving.py \
  --url http://127.0.0.1:18004 \
  --baseline ../results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw \
  --output ../results/T04/http-final
```

CPU 测试契约：模型输入为文本 IDs、绝对位置及请求缓存，输出仅含文本行；最小随机模型
捕获 summary 错位、历史覆盖和跨请求污染。分配器测试使用真实物理页验证压力与回收；
HTTP 测试验证协议计数及拒绝行为，真实 socket 测试补充 ASGI 断连竞态覆盖。
