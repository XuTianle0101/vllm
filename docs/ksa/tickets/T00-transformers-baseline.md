# T00：冻结 Transformers 官方基线

- 状态：AWAITING_SERVER
- 依赖：无
- 目标：建立能在 RTX 5090 上复现的官方精度与性能基准。

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

- 提交 SHA：见本 ticket 文件的最新 Git 提交（服务器启动器要求完整 SHA）
- 本地检查：4 项 unittest、Ruff lint/format、Bash 语法及 44 项 Linux 依赖锁安装解析通过；无依赖失败留档 smoke 通过。独立 Ruff、markdownlint、typos、SPDX、lazy-import、boolean-context 检查通过。完整 pre-commit 的工具环境初始化受网络阻塞未跑完，不能声称全套通过。
- 服务器命令：见 [T00 运行说明](../t00-baseline.md)，执行 benchmarks/ksa/run_server.sh。
- 结果路径 / baseline ID：服务器运行时生成，当前未产生真实 GPU 基线。
- 结论：脚本已实现，等待 5090 服务器实验、语义确认及容差冻结；未验收，不启动 T01。
