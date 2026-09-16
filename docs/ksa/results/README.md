# 实验结果

T00 已冻结的 A100 基线：[a100-repaired-20260916T032325Z-0eea77a7ba21](T00/a100-repaired-20260916T032325Z-0eea77a7ba21/README.md)。23 个精度案例与 25 次正式性能测量全部通过，校准与容差已绑定 baseline ID。

此前不完善的 T00 产物已清理，以本轮已冻结结果为准。

按 `Txx/<run-id>/` 保存服务器反馈，结构见 [结果约定](../templates/result-contract.md)。每轮绑定实际提交 SHA 和 baseline ID，失败、OOM、中断结果也保留。原始大体积 logits、trace 或模型数据留服务器，反馈中给出持久化位置和哈希。
