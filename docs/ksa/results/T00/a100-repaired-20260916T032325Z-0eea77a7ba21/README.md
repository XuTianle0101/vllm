# T00 A100 已冻结基线

状态：DONE。服务器启动器退出码 0；23 个 BF16 精度案例、3 个 FP32 校准案例、5 档性能预热及 25 次正式重复均通过。无 OOM、error 或不可比较项。

- 配置：`official-flex-cu128-repaired-v1`，显式 BlockMask／标量 mask 适配；不是未经修改的官方 wheel 路径。
- baseline ID：`T00-305bf8f9dc429f1af60e`。
- tolerance ID：`T00-tol-fbd0eb931ac1c7153403`，冻结值及证据见 [正确性](correctness.json) 和 [校准](calibration.json)。
- 服务器代码快照：`0eea77a7ba216dfc79bf5d145ccf3ce76dd927da`，`git_dirty=false`；本地分支 `ksa-t00-baseline-snapshot`，未推送远端。快照与当前工作区四个执行脚本的 SHA256 一致。
- GPU：A100-SXM4-80GB；PyTorch 2.7.1/cu128、Transformers 4.57.1、tokenizers 0.22.1。
- 原始执行目录：`/tmp/ksa-t00-complete-20260916`。
- 持久化完整结果：`/workspace/volume/h20-data/xutianle/KSA/results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw`，包含所有 `.pt`；小包为同目录下 `baseline.tar.gz`，源码增量 bundle 为 `source.bundle`。旧的失败与中断产物已清理，以本轮已验收结果为准。

BF16 全部案例重复 logits 最大差为 0，重复生成 IDs 一致。跨 prefill/decode 观测最大 logits 差为 13.125；该指标按独立 FP32 噪声包络验收，不宣称近零误差。top1 分歧最大 margin 为 0.25，低于冻结门限 0.5。三组检索生成均包含预期答案，但这只是固定样例观察，不代表通用任务准确率。

正式性能取五次平均，显存取五次最大 PyTorch allocated（不等同于 nvidia-smi 进程显存）：

| 输入 tokens | TTFT 均值 ms | 输出 token/s | 峰值 GiB |
| --- | --- | --- | --- |
| 4096 | 318.95 | 30.189 | 8.65 |
| 16384 | 1471.29 | 21.648 | 11.62 |
| 32768 | 3791.81 | 14.156 | 15.61 |
| 65536 | 11129.81 | 6.865 | 23.58 |
| 130944 | 36298.40 | 2.661 | 39.51 |

[性能原始行](performance.csv)、`timings/` 逐步延迟和 `logs/` 工作进程日志保留；冷启动预热不计入平均。性能未导出全量 logits。仅 batch=1、固定生成 128 token；并发、分页、图和 chunked prefill 均 N/A。

复现命令见 [运行说明](../../../t00-baseline.md)。后续 ticket 必须绑定本轮 baseline ID 与 tolerance ID，不能继续使用早期 BLOCKED 结果。
