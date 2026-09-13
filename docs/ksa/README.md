# KSA vLLM 适配文档

当前状态：**T00 实验工具已实现，等待 5090 服务器运行与验收。**

- [总计划](ksa-vllm-plan.md)：目标、约束、语义、精度与性能门禁。
- [Ticket 看板](tickets/README.md)：T00–T07 的顺序与状态。
- [开发与服务器实验流程](workflow.md)：逐 ticket 的交付、push、实验、反馈和关闭规则。
- [实验反馈模板](templates/experiment-feedback.md)：每轮服务器反馈的填写格式。
- [结果文件约定](templates/result-contract.md)：结构化结果字段与示例。
- [实验结果目录](results/README.md)：未来存放真实结果；目前没有实验结果。

文档以当前用户确认的方案为准。目标是 RTX 5090 上先以 Python 实现 Transformers 精度对齐，再优化框架与缓存，最后评估官方 KSA 算子或 Triton。

根目录 `docs/` 是本地入口；本目录为随代码版本化的副本。T00 服务器命令、已知兼容性差异和验收步骤见 [运行说明](t00-baseline.md)。
