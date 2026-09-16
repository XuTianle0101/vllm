# T01：Python 模型与 prefill

- 状态：DONE（A100 Python prefill 验收通过）
- 依赖：T00 DONE
- 目标：以 Python 路径实现与 Transformers 对齐的 KSA prefill。

## 实现范围

自动识别 KSA 配置，原生加载现有权重；安全解析配置表达式，复用 Qwen3 公共组件。实现 summary 展开、位置编号、各层掩码和文本 logits 映射。当前 mix_coeff=0 使用共享投影。添加模型初始化及普通 Qwen3 回归检查。

建立小规模 FP32 显式掩码参考，BF16 路径优先使用 PyTorch 现成算子。允许短序列 dense mask，明确长度限制，不在此 ticket 开发自定义 Triton 或长上下文优化。

## 代码交付

实现应位于 vLLM 的模型执行层，并通过模型注册或配置字段选择 KSA 路径；不得修改
普通 `Qwen3ForCausalLM` 的默认行为。权重加载必须沿用 `AutoWeightsLoader`，并对
缺失或多余的 KSA 参数给出可定位的错误。配置表达式只允许解析整数、列表和有限的
重复语法，拒绝函数调用、属性访问和任意 Python 表达式。

forward 的输入输出契约如下：运行器传入文本位置和 hidden states，模型内部在每个
8-token 块末展开一个 summary 行；返回值只包含文本行的 hidden states/logits，并
保留文本行到内部行的索引映射。每层根据窗口类型构造因果可见集合，summary 行不
参与对外采样。FP32 参考路径与 BF16 路径必须共用位置编号和掩码生成逻辑，以便
逐层比较；长度超过显式 mask 上限时要显式报错。

建议新增最小单元测试覆盖配置解析拒绝危险语法、块边界位置编号、四类窗口掩码、
summary 不泄露未来 token，以及普通 Qwen3 配置仍走原路径。测试应使用小型随机
配置，不依赖 KSA 权重或 GPU。

## 服务器实验

服务器启动前先同步并核验 T00 的 baseline ID 与本 ticket 的完整提交 SHA。使用实现
后的 T01 专用启动器（不得复用不存在的命令）比较文本位置 logits/logprobs，覆盖
长度 1/7/8/9/15/16/17、1024 窗口及其相邻位置、全部层类型。检查 summary 位置与 self-KV 可见性（T00 已冻结为 summary 可见自身），
以及未来信息不可见；对照 T00 固定数据与阈值。性能只记录 prefill，至少
预热后重复 5 次，并分别记录 1K/4K/16K（或实际支持上限）的 TTFT、吞吐和峰值显存。

结果目录应遵循 `docs/ksa/results/T01/<环境>-<UTC 时间>/`，至少包含
`environment.json`、`inputs.json`、`correctness.json`、`timing.json`、`summary.md`
和失败时的 `failure.json`。`correctness.json` 记录每个长度、位置和层类型的最大
绝对误差、相对误差、top-1 一致率及首个异常位置。

## 门禁与反馈

模型权重正确加载；误差满足 T00 冻结阈值；首个异常位置可定位到输入、层或
attention 角色。普通 Qwen3 不回归。服务器结果和日志必须完整，OOM 或未支持的
长度保留原始错误。未完成 decode，性能仅记录 prefill，不宣称端到端可用。

## 交付记录

- 分支：`releases/v0.26.0-ksa`；本地已提交。推送因 GitHub 认证失败受阻（VS Code 凭据 socket 不可用），未声称远端已同步；恢复认证后执行 `git push origin HEAD:releases/v0.26.0-ksa`。
- 实现与实验提交：`a7a696efd57283d56098e3b91cb73aa9c47c3901`。自动选择 `KSAForCausalLM`，复用 Qwen3 组件与 AutoWeightsLoader；完成安全配置解析、summary 展开、文本行映射与词表裁剪、共享掩码 FP32/SDPA 路径。单请求完整 prefill 上限为 4096 文本 token。
- 基线：`T00-305bf8f9dc429f1af60e`；容差：`T00-tol-fbd0eb931ac1c7153403`。模型、输入、HF 依赖与参考代码通过哈希/版本核验；summary 查询可见自身 KV，遵循 T00。
- 本地检查：禁用 GPU 的 37 项 CPU 测试、7 项 T00 harness 回归检查、完整代码 pre-commit 钩子通过。涵盖真实小模型初始化、权重缺失/多余/重复诊断、四种窗口、未来信息隔离、FP32/SDPA 对照与普通 Qwen3 路由。
- 服务器：A100-SXM4-80GB 实际执行专用 `benchmarks/ksa/prefill.py`，退出码 0。[完整同步和运行命令](../t01-prefill.md)。
- 结果：[验收摘要](../results/T01/a100-20260916T063504Z/README.md)，目录包含环境、输入、逐位置正确性、逐层诊断、计时和日志。原始大张量与失败记录已持久化至 `/workspace/volume/h20-data/xutianle/KSA/results/T01/`；清单记录 SHA256。
- 精度：17/17 BF16 案例通过冻结门限，重复 logits 最大差为 0。两组 FP32 logits 最大误差为 0.000151；144 组内部层/角色诊断均有限，原始层误差保留，不把 logits 门限用于未归一化隐状态。
- 性能：1K/4K 各一次预热与五次正式重复，TTFT 均值 151.87/1592.05 ms，峰值 allocated 显存 8.12/13.93 GiB。4K 比 T00 官方路径约慢 4.99 倍，不宣称加速。16K 显式超限，原始错误保留。
- 结论：T01 DONE，仅覆盖该环境的完整无缓存 prefill；decode、分页、批处理、chunked prefill 和 serving runner 不支持。两轮失败及修复依据在验收摘要中完整保留。
