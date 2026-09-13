# T01：Python 模型与 prefill

- 状态：TODO
- 依赖：T00 DONE
- 目标：以 Python 路径实现与 Transformers 对齐的 KSA prefill。

## 实现范围

自动识别 KSA 配置，原生加载现有权重；安全解析配置表达式，复用 Qwen3 公共组件。实现 summary 展开、位置编号、各层掩码和文本 logits 映射。当前 mix_coeff=0 使用共享投影。添加模型初始化及普通 Qwen3 回归检查。

建立小规模 FP32 显式掩码参考，BF16 路径优先使用 PyTorch 现成算子。允许短序列 dense mask，明确长度限制，不在此 ticket 开发自定义 Triton 或长上下文优化。

## 服务器实验

比较文本位置 logits/logprobs，覆盖长度 1/7/8/9/15/16/17、1024 窗口及其相邻位置、全部层类型。检查 summary 位置、self-KV 与未来信息不可见；对照 T00 固定数据与阈值。

## 门禁与反馈

模型权重正确加载；误差满足冻结阈值；首个异常位置可定位到输入、层或 attention 角色。普通 Qwen3 不回归。未完成 decode，性能仅记录 prefill，不宣称端到端可用。

## 交付记录

- 提交 SHA：待实现
- 本地检查：未执行
- 服务器命令：待脚本实现后填写
- 结果路径：未生成
- 结论：未验收
