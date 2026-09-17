# KSA vLLM 适配文档

当前状态：**T00 基线已冻结，T01 prefill、T02 单请求 decode 与 T03 分页压缩缓存已通过 A100 验收。**

- [T04 批处理与基础服务](t04-batching-chunked-prefill.md)：分块、预算、抢占重算及独立 HTTP 入口。
- [T03 分页缓存与验收](t03-compressed-kv.md)：真实页分配、压缩占用、独立性能对照与限制。
- [T02 运行与验收](t02-decode.md)：Python 缓存生成、精度与性能对照、当前限制。
- [总计划](ksa-vllm-plan.md)：目标、约束、语义、精度与性能门禁。
- [Ticket 看板](tickets/README.md)：T00–T07 的顺序与状态。
- [开发与服务器实验流程](workflow.md)：逐 ticket 的交付、push、实验、反馈和关闭规则。
- [实验反馈模板](templates/experiment-feedback.md)：每轮服务器反馈的填写格式。
- [结果文件约定](templates/result-contract.md)：结构化结果字段与示例。
- [实验结果目录](results/README.md)：包含本轮已验收并冻结的基线。

文档以当前用户确认的方案为准。目标是在已记录的 CUDA GPU 上先以 Python 实现 Transformers 精度对齐，再优化框架与缓存，最后评估官方 KSA 算子或 Triton。

根目录 `docs/` 是本地入口；本目录为随代码版本化的副本。T00 服务器命令、已知兼容性差异和验收步骤见 [运行说明](t00-baseline.md)。
