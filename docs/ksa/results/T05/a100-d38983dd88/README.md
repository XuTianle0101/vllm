# T05 A100 图执行与瓶颈定位

核心图执行代码及性能实验提交：`d38983dd881a9766f0349254d5b0b6fda95b3753`。
原始结果：`/workspace/volume/h20-data/xutianle/KSA/results/T05/a100-d38983dd88/`。
[结构化摘要与原始文件哈希](summary.json)、[复现说明](../../../t05-cudagraph-profiling.md)。

## 精度与状态

A100-SXM4-80GB、BF16、TP=PP=1，基线 `T00-305bf8f9dc429f1af60e`，
容差 `T00-tol-fbd0eb931ac1c7153403`，未调整阈值。

- 17 个冻结教师强制案例 × 两轮 × eager/HF 两种参考，68/68 通过。
  批内每步反转请求，跨轮复用图槽位，覆盖普通文本、块末文本+summary 和混合相位。
- 对 eager 最大 logits 误差 1.53125、RMSE 0.398605；对 HF 分别为 3.5、1.014221。
  两种参考的最大 top-1 分歧 margin 都为 0.25（门禁 0.5）；不宣称逐位相同。
- 共 1408 次图重放，11 次捕获（包括形状变化与有界缓存淘汰后的重捕获）。
  实验结束 139039/139039 页全部归还。
- 最终完整回归 82/82 通过；包含四项缓冲区测试，其中两项在 CUDA 上真实捕获，
  检查取消/重建、slot 重排、窗口淘汰、每层 KV、旧输出存活与缓存容量上限。
- 适用 pre-commit 检查通过，包括 Ruff、mypy、文档与仓库专项检查。

## 标准 V1 页表验证

补充提交：`bc74fbe2c2b8099c987ee3d38eeda35027ef8418`，新增验证脚本和
图 padding 行数不超过 worker 预算的 eager fallback；核心图计算与性能脚本未更改。
[结构化 V1 结果](v1-summary.json)，原始目录 `../results/T05/v1-bc74fbe2c2/`。

标准 LLM/V1 调度执行 7/8/1023/1024-token prompt，每请求自由生成 32 token，
两轮 graph 与一轮 eager 比较。累计 62 次 replay、5 次捕获，8/8 同前缀门禁通过。
7/8 自由生成序列完全相同；反向提交批次中的 length-8 在第三个输出处分歧，
共同前缀（包含首次分歧预测）的最大 margin 0.25，未越过 0.5 门禁。
分歧后的 logits 不作跨轨迹比较；完整固定前缀正确性由上面的教师强制实验覆盖。
这不是生成轨迹逐位一致的声明，也不是 HTTP 吞吐测试。

## 稳态性能

1K/4K × batch 1/4/8，32 decode 步，一次预热和五次正式重复。
多请求长度分别为标称长度减去请求序号，故每步可包含不同块相位。
表中为五次均值 ± 标准差，单位 ms/批次 decode step；加速比为 eager/graph。
区间包含 metadata/KV、模型、LM head 和 greedy，不包含 prefill、scheduler、网络和捕获。

| 长度 | batch | eager ms | graph ms | eager/graph |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 1 | 54.123 ± 0.527 | 26.487 ± 0.078 | 2.043 |
| 1024 | 4 | 95.245 ± 1.382 | 75.960 ± 0.541 | 1.254 |
| 1024 | 8 | 146.827 ± 0.778 | 141.972 ± 0.290 | 1.034 |
| 4096 | 1 | 53.656 ± 1.185 | 56.537 ± 0.058 | 0.949 |
| 4096 | 4 | 95.854 ± 1.260 | 198.379 ± 0.199 | 0.483 |
| 4096 | 8 | 154.004 ± 0.956 | 386.155 ± 0.489 | 0.399 |

**图模式只在部分形状下获益，4K 全部退化，不能默认替代 eager。**
当前固定缓冲区使用跨层最大历史长度桶，并为每请求固定两行；
这减少 Python kernel launch，却增加短窗口层填充和普通 decode 的冗余计算。
保留两条路径是本轮结论的一部分，不把短序列收益推广到所有负载。

启动成本独立：权重加载及页池初始化 2678.07 ms；每桶分配+三次 warmup
163.17–1117.25 ms，捕获 71.76–260.35 ms；11 次合计分别为 4368.05、1511.90 ms。
未启用 torch.compile；首次 kernel 初始化计入 warmup，没有声称测得编译加速。
正式计时的 `new_captures` 全部为零。

同一进程保留图缓存，矩阵中最大 allocated 峰值为 52.322 GiB；
这不是图相对 eager 的独立增量。池、图历史缓冲区、图私有执行内存均需预算，
不能仅按压缩 KV 页池大小估算图模式显存。

## Trace 与 T06 优先级

保存 eager/graph Chrome trace、算子热点表、CPU/CUDA ranges。
完整 trace 包含四个请求的 eager prefill；摘要脚本剔除 prefill，仅分析之后八步 decode。
CPU range 是含同步等待的包围时间，可能和 GPU 时间重叠，不能相加当端到端成本。
Profiler 有明显开销，以下仅用于定位，不替代上表的无 profiler 测量。

1K/batch=4 图路径的 CPU 包围时间每步：metadata 5.58 ms、KV stage 20.11 ms、
图 launch range 9.26 ms、KV commit 69.88 ms。commit 包含等待在先的 GPU 工作，
不应把全部 69.88 ms 归因于分页写回自身。
GPU kernel/memcpy/memset 时间之和为 eager 47.46 ms、graph 67.67 ms/步；
图模式两个最热 copy/cast kernel 合计 25.57 ms，占 GPU 活动时间约 37.8%。
FP32 batched small GEMM、缩放、softmax 和历史拼接也在热点中。
这些数据支持优先消除 math attention 的中间转换、GQA 实体重复和完整历史 staging，
而不是继续只优化 Python launch。

[T06 接口与许可证决策](../../../t06-kernel-decision.md)：prefill/decode 均选择仓库内
Triton；decode 优先直接读两组物理页，按请求批处理并融合 mask/softmax，随后替换
prefill 平方 mask。官方 wheel 的 SM120 检测不兼容、分页接口缺失已记录。
T06 算子尚未实现，预期收益不是实测收益；5090 没有实机验证。
