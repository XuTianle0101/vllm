# KSA 通用 V1：A100 验证

2026-09-17，在单张 NVIDIA A100-SXM4-80GB 上完成标准 V1 接入验证。
本记录对应基线提交 `a493692cfe` 之后的 V1 接入改动，验证时工作区尚未提交；
[summary.json](summary.json) 记录实际文件 SHA256、运行时和结构化结果。
启动与复现命令见 [V1 运行说明](../../../v1-serving.md)。

## 精度

冻结基线 `T00-305bf8f9dc429f1af60e`，容差 `T00-tol-fbd0eb931ac1c7153403`。
17 个案例包括短序列 1/7/8/9/15/16/17、1K 窗口与淘汰边界、4K、英文、中文和检索。
通过标准 V1 scheduler 执行，观察每例 25 个冻结文本位置；没有更改门禁。

| 指标（全部案例最坏值） | 257 行预算 | 31 行预算 | 冻结上限 |
| --- | ---: | ---: | ---: |
| logits 最大绝对误差 | 2.273438 | 4.500000 | 20.335213 |
| logits RMSE | 0.682465 | 1.419070 | 3.705945 |
| logprobs 最大绝对误差 | 1.093727 | 1.062517 | 8.020130 |
| top-1 分歧 margin | 0.250000 | 0.250000 | 0.500000 |
| 通过案例 | 17/17 | 17/17 | 全部 |
| 实测最大内部行数 | 257 | 31 | 不超过对应预算 |

这些是固定教师序列下的比较，不能据此声称任意自由生成逐 token 一致。
相对 T04：保留相同模型 attention 语义，改为标准 V1 负责调度和页分配；
结构化报告列出已冻结 T04 的精度最坏值供对照。未保存相同矩阵的 T04 全部 logits，
因此没有伪造逐元素的 T04→V1 差值。

## 请求生命周期和标准 HTTP

- 330 块页池（含 1 个 null block），两个不同的 257-token prompt，各执行
  80 个固定教师 token。触发 1 次真实 V1 抢占，共 154 个调度步骤；
  串行/抢占后的原始 logits 比较均通过冻结门禁，最大 logits 误差分别为
  0.4375/0.75，最大分歧 margin 为 0.0625/0.125。
- 累计输出无重复或倒退；完成及取消后均回收至 329 个可用块，随后请求复用正常。
- 实际执行 `.venv/bin/vllm serve`，日志确认 `KSAForCausalLM`、同步调度和 V1 引擎。
  `/health`、`/v1/models`、并发 1/4/8 completions、SSE `[DONE]` 和 usage、
  seed、非零 temperature、top-p、`n=2`、惩罚参数、echo/prompt logprobs、
  stop strings、非法 token 400 响应全部通过。
- 三次真实 socket 断连后，running、waiting、KV usage 三项指标均为零；新请求正常。
  此轮 native HTTP 并发输出没有近并列分歧。

HTTP smoke 的 65 输入/16 输出 token 耗时分别约 1.061/3.491/6.873 秒
（并发 1/4/8）。没有完整重复性能矩阵，不主张相对 HF 或 T04 的加速。
当前 Python attention 的逐请求开销仍然明显。

## 检查和环境

- KSA 单元/分配器测试：78 passed。
- 标准 scheduler 和 async scheduler 单卡适用回归：140 passed，5 个 PP 相关用例排除。
  未筛选的套件在 PP=2 构造阶段报告仅有 1 张 GPU，不作为代码回归通过项。
- Python 3.10/3.12 mypy 通过；其余 pre-commit 检查通过。
- 安装时发现 system-site-packages 遮蔽默认 editable finder，已改为 compat 安装，
  并使用 `python -I` 从仓库外确认源码路径。
  该问题修复前启动的上游 Qwen HTTP 结果全部废弃，不计入本报告。
- 开发中一次后台预编译安装改写正在加载的扩展，造成 bus error/segfault。
  已结束残留安装，恢复匹配的 v0.26.0 扩展并重跑成功项；报告记录扩展哈希。
  KSA 不使用 DeepGEMM；其生成式安装目录的 Python wrapper 也恢复为匹配版本。

原始数据位于工作区 `results/V1/a100-257-final`、`a100-31-native`、
`pressure-native`、`http-native`。大体积 logits 未加入版本控制。
CUDA Graph、异步调度、前缀缓存、多卡、128K 和最终性能验收不在本次通过范围内。
