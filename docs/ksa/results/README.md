# 实验结果

已保存 T00 A100 兼容运行结果：见 [T00/A100 结果](T00/a100-compat-20260914T062343Z/)。

未来按 `Txx/<run-id>/` 保存每轮服务器反馈，结构见 [结果约定](../templates/result-contract.md)。每轮必须绑定实际提交 SHA 和 baseline ID，失败、OOM、中断结果也保留。

原始大体积 logits、trace 或模型数据默认保留在服务器，反馈中给出位置与必要摘要。结果中的失败和 OOM 状态必须保留，不能视为完整验收。
