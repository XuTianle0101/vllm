# T00：冻结 Transformers 官方基线

- 状态：DONE（A100 完整精度、校准和五档性能矩阵通过）
- 依赖：无
- 目标：建立可在具备 CUDA 的受支持 GPU 上复现的官方精度与性能基准。

## 实现范围

核验模型 remote code、论文仓库推荐部署、依赖和 summary 算子兼容性；提供环境采集、官方推理、精度数据导出、性能测量及结果收集脚本。冻结模型/tokenizer 哈希、HF 与官方代码版本、输入 token IDs、生成配置、baseline ID。

检查 summary 自身可见性与 prefill/decode 语义是否一致。基准若存在冲突，先记录并解决，不能静默修补后仍称原始官方基线。官方 wheel 不兼容时，记录解决方法及官方源码版本；不得用低性能 dense 替代品作为最终性能对照。

根据重复运行、误差校准和现有测试规范确定精度阈值，在 T01 前冻结。建立分层实验集，包含短块边界、窗口边界、长上下文与固定生成案例。

## 服务器实验

1. 采集 GPU、驱动、运行时版本和文件哈希。
2. 验证官方 BF16 加载、prefill、decode 与重复运行。
3. 导出精度参考，性能计时路径不导出全量 logits。
4. 运行 4K/16K/32K/64K 基准，接近 128K 的场景保留生成空间；预热后至少重复 5 次，记录 OOM。

## 门禁与反馈

官方路径成功运行；版本和容差已冻结；summary 语义无未解决歧义；环境、正确性、性能及日志文件完整。长上下文无法运行必须记录原因，明确哪些结果不能作为后续比较。

## 交付记录

- 服务器代码快照 SHA：`0eea77a7ba216dfc79bf5d145ccf3ce76dd927da`；干净工作区。本地分支 `ksa-t00-baseline-snapshot`，未推送远端；原工作区的 T01 改动保留。
- 配置：`official-flex-cu128-repaired-v1`。显式修复官方 BlockMask 普通块/完整块重叠与索引布局、长度 1 的 summary 标量 mask；保留官方编译 FlexAttention 性能路径。FP32 短序列显式参考仅用于校准。详见 [兼容性和运行说明](../t00-baseline.md)。
- 本地检查：7 项 unittest、Ruff lint/format、Bash 语法通过；本轮代码变更的完整 pre-commit 钩子通过。44 项带哈希依赖锁同步通过。
- 服务器命令：已在独立干净 checkout 实际执行 `benchmarks/ksa/run_server.sh`，退出码 0。环境为 A100-SXM4-80GB、PyTorch 2.7.1/cu128、Transformers 4.57.1、tokenizers 0.22.1。
- 结果：[完整验收摘要](../results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/README.md)。持久化原始结果含 `.pt`：`/workspace/volume/h20-data/xutianle/KSA/results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw`；小包、源码 bundle 和校验清单同目录保存。
- baseline ID：`T00-305bf8f9dc429f1af60e`；tolerance ID：`T00-tol-fbd0eb931ac1c7153403`。模型、tokenizer、remote code、算子、兼容层、执行器、评估器、输入与生成配置均有哈希记录。
- 精度：23/23 案例通过，含 130944 和三个检索案例；重复 logits 最大差为 0，重复生成 IDs 一致。三组 FP32 校准的 prefill/decode 最大误差均小于 0.005。BF16 跨路径阈值由独立 FP32 噪声冻结，并保留 margin、有限值与重复一致性门禁；未按待验收最大误差调宽。
- 性能：4K/16K/32K/64K/130944 全部完成一次预热与五次正式重复，共 25/25 正式测量通过。无 OOM、error 或不可比较项；130944 的原非法访存未重现。
- 结论：T00 DONE。此前不完善的产物已清理；后续 ticket 必须引用本轮 baseline ID 和 tolerance ID。此结论仅覆盖已记录的 A100 环境，其他 GPU 需独立验收。
