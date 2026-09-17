# T04：批处理与 chunked prefill

- 状态：AWAITING_SERVER
- 依赖：T03 DONE
- 目标：在压缩 KV 上支持基础 vLLM 服务调度。

## 实现范围

支持不同长度和不同块相位请求混合、任意文本位置切分、summary 计算预算、文本 logits 映射、基础服务、流式输出及 prompt logprobs。处理取消、完成、重排、请求槽复用和内存压力抢占重算。

防止当前 prefill 的新 KV 覆盖本批次早期 query 需要的历史数据。内部行数用于计算预算，对外 usage 保持文本语义。尚未支持的前缀缓存等功能明确拒绝。

## 服务器实验

整段与多个切分方案比较；batch=1 与混合 batch 结果比较；覆盖跨 8-token 和窗口边界。测试流式结束、取消、重排、反复复用、抢占重算和显存回收。并发 1/4/8 测量吞吐；HF 只支持 batch=1 时，并发结果单列。

## 门禁与反馈

所有调度变换下精度一致，无跨请求数据污染；预算、计数和回收正确。记录 OOM 场景和预期恢复行为。不得以批处理吞吐冒充同批量 HF 加速比。

## 交付记录

- 实现：[T04 运行说明](../t04-batching-chunked-prefill.md)。独立 KSA eager 调度与
  OpenAI completions 入口已实现；通用 GPUModelRunner/V1 scheduler 接入仍未实现。
- 提交 SHA：以 `git log --format=%H --grep='implement KSA batching and chunked prefill' -1` 查询代码提交。
- 本地检查：72 项 CPU 合约测试；Ruff 和 pre-commit 结果随正式验收记录更新。
- 服务器命令：见运行说明；`benchmarks/ksa/batching.py` 与 `benchmarks/ksa/serving.py`。
- 开发结果：`../results/T04/a100-dev/` 教师强制 68/68；自由生成有序列分歧，
  正在补充同前缀 HF 门禁。`../results/T04/http-dev3/` HTTP/SSE/断连回收通过。
- 结论：等待绑定代码 SHA 的正式服务器结果；不提前标记 DONE。
