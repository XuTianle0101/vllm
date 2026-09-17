# Ticket 看板

T00 已完成 A100 精度、校准与性能验收，基线与容差已冻结。T01/T02/T03 已完成 A100 验收；T03 为分页单请求 Python 原型，性能差距见交付记录。

| Ticket | 标题 | 依赖 | 状态 |
| --- | --- | --- | --- |
| [T00](T00-transformers-baseline.md) | 冻结 Transformers 官方基线 | 无 | DONE |
| [T01](T01-python-prefill.md) | Python 模型与 prefill | T00 | DONE |
| [T02](T02-python-decode.md) | Python 缓存 decode | T01 | DONE |
| [T03](T03-compressed-kv.md) | 压缩 KV 与 Python 优化 | T02 | DONE |
| [T04](T04-batching-chunked-prefill.md) | 批处理与 chunked prefill | T03 | AWAITING_SERVER |
| [T05](T05-cudagraph-profiling.md) | CUDA Graph 与瓶颈定位 | T04 | TODO |
| [T06](T06-attention-kernels.md) | 官方算子或 Triton | T05 | TODO |
| [T07](T07-final-validation.md) | 最终精度与性能验收 | T06 | TODO |

逐项遵循 [工作流](../workflow.md)。当前 ticket 的服务器结果通过后才启动下一项。每个文件中的“交付记录”在实际开发时填写，不能提前填写提交 SHA 或通过结论。
