# T06：官方算子或 Triton

- 状态：TODO
- 依赖：T05 DONE
- 目标：消除已定位的 attention 瓶颈，提升端到端性能。

## 实现范围

prefill 与 decode 分别决策。官方算子只有在许可证、RTX 5090 支持、KSA 掩码、分页接口、CUDA Graph 和端到端收益均符合要求时复用；否则为相应路径编写仓库内 Triton。决策先写入本 ticket 的实验记录。

不修改 C++/CUDA 源码。保留 Python 参考用于小规模验证。生产路径禁止平方级 mask、完整历史复制和逐请求 Python attention。分区或 split-K 结果用统一、稳定的 softmax 归一化合并；不能将局部与 summary 两个独立 softmax 输出直接相加。

## 服务器实验

FP32 小规模参考与 BF16 误差测试，覆盖分页、窗口边界、summary 自身、短尾及混合请求。回归 HF 和 T05；测量单算子与完整模型，分离官方接口所需 gather/转换开销。性能预热后至少重复 5 次。

## 门禁与反馈

精度和调度回归通过，生产路径满足内存复杂度要求。实测端到端收益成立，不能仅凭内核快就接受集成。未超过基线的路径继续在当前 ticket 优化或明确阻塞，不虚报通过。

## 交付记录

- prefill/decode 路径选择与依据：[T05 决策](../t06-kernel-decision.md)。
  均选仓库内 Triton；官方发布的 CuTe wheel 不接受 SM120，且接口未提供 KSA 双组分页。
  T05 trace 显示 copy/cast、math attention 和图外 KV 管理开销，应优先直接读页与融合。
  本 ticket 尚未启动，没有 Triton 实测收益。
- 提交 SHA：待实现
- 本地检查：未执行
- 服务器命令：待脚本实现后填写
- 结果路径：未生成
- 结论：未验收
