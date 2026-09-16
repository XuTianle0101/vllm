# T02 A100 验收

**DONE**：2026-09-16 在 NVIDIA A100-SXM4-80GB 完成 Python 单请求缓存 decode 验收。
完整启动器退出码 0；不代表普通 serving 或长上下文优化已经实现。

- 分支：`releases/v0.26.0-ksa`。
- 实测实现 SHA：`4f2845da8a3565d8bbedec082fdbf2fc40319099`。
- 基线：`T00-305bf8f9dc429f1af60e`；容差：`T00-tol-fbd0eb931ac1c7153403`。
- [复现步骤](../../../t02-decode.md)、[运行日志](run.log)、[环境](environment.json)。
- [正确性](correctness.json)、[生成逐位置比较](generation.json)、[全部计时](timing.json)。
- 原始结果：`/workspace/volume/h20-data/xutianle/KSA/results/T02/a100-run1/`。
  [artifacts.json](artifacts.json) 记录所有原始文件的绝对路径、大小和 SHA256。
  十份生成轨迹 `.pt` 保留服务器，不纳入 Git；归档 summary 仅修正 Markdown 表格空格。

## 精度与停止

17/17 冻结教师强制案例通过，覆盖块末、部分块、多次跨块、窗口边界、4K 和检索输入。
每例包含 prompt 最后文本行及后续 24 个文本位置，各执行三次；重复 logits 最大差为 0。

| 指标 | 实测最大值 | 冻结门限 |
| --- | ---: | ---: |
| logits 最大绝对误差 | 3.5 | 20.33521270751953 |
| logits RMSE | 1.0058326721191406 | 3.7059452533721924 |
| logprobs 最大绝对误差 | 1.1184139251708984 | 8.020130157470703 |
| top1 不一致位置最大参考 margin | 0.25 | 0.5 |

5 组固定提示各生成 128 token，重复结果完全一致。两组与 HF 的全部 token 精确一致；
其余三组的首个分叉如下。序号从 0 开始，HF logits 均基于与 vLLM 相同的前缀。

| 案例 | 首个分叉序号 | HF token | vLLM token | HF top1 margin |
| --- | ---: | ---: | ---: | ---: |
| length-7 | 无 | — | — | — |
| length-8 | 2 | 15 | 17 | 0 |
| length-9 | 无 | — | — | — |
| english | 1 | 11 | 13 | 0 |
| chinese | 20 | 100006 | 67338 | 0.125 |

HF 沿五条 vLLM 生成轨迹逐步复算，全部 640 个位置通过冻结数值门限；各轨迹最大
logits 误差不超过 5.875，全部 top1 不一致位置的最大参考 margin 为 0.125。
因此按 T00 允许的低 margin 选择差异通过，**不声称所有自由生成 IDs 与 HF 完全一致**。
所有生成 ID 均与实际采样 logits 的 argmax 一致，无 summary 泄漏。

五组 EOS 注入测试全部通过：以第一步实际输出 ID 作为 EOS 时立即停止，计数为 1，
finish_reason=stop。0/1/128 token 预算、ignore_eos、prompt/completion 计数全部正确。
CPU 测试覆盖真实小模型与完整 prefill 对照、KV 行数、异常后原子性、请求重置、
错误位置、批量、词表及不支持的参数。

## 性能

同 checkpoint、BF16、batch=1、固定输出 128 token、EOS 禁用，1K/4K prompt 各一次
预热及五次正式测量。均为同步 wall time，包括输入准备、forward、argmax；性能路径
不导出 logits。峰值为 PyTorch allocated，冷加载/编译不在热态计时内。

| 后端 | Prompt | TTFT ms | 普通 TPOT ms | 块末 TPOT ms | token/s | 峰值 GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HF | 1024 | 97.435 | 29.800 | 37.630 | 32.004 | 7.898 |
| vLLM Python | 1024 | 152.463 | 41.042 | 41.575 | 23.824 | 8.620 |
| HF | 4096 | 311.517 | 29.720 | 37.539 | 30.453 | 8.643 |
| vLLM Python | 4096 | 1592.784 | 41.113 | 42.010 | 18.747 | 16.040 |

vLLM 原型的吞吐分别比 HF 低约 25.6% 和 38.4%；TTFT 分别约为 HF 的 1.56 倍和
5.11 倍。相对 T01，1K/4K TTFT（151.87/1592.05 ms）基本不变，但缓存使峰值显存
从 8.12/13.93 GiB 增至 8.62/16.04 GiB。T01 不支持生成，因此没有上一 ticket 的
端到端吞吐可比值。

本实现保留 dense prefill、全历史 K/V、逐步 K/V 拼接与 Python 调度，不宣称加速。
缓存压缩与性能优化留给 T03 及后续 ticket。16K 及以上、并发、分页 KV、CUDA Graph、
chunked prefill、普通 serving 均不在本次验收范围。

## 检查与交付

42 项 CPU 模型契约测试、7 项 T00 harness 回归测试、所有适用 pre-commit 钩子通过。
这是首轮 A100 运行，未调整 T00 阈值，未发生需要删除或掩盖的失败实验。

实现和结果已本地提交。GitHub 推送因凭据失败：VS Code credential socket 不存在，
服务端拒绝匿名写入；未声称远端已同步。源码 bundle 位于
`/workspace/volume/h20-data/xutianle/KSA/results/T02/source.bundle`，
以前置提交 `39ccd67fb529dd2dac2d7086af031d3e76838a9b` 为基点，包含前置 T00/T01 及 T02。
