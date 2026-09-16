# T03 压缩 KV：开发测量

状态：IN_PROGRESS。基线 `T00-305bf8f9dc429f1af60e`，冻结容差
`T00-tol-fbd0eb931ac1c7153403`；比较对象为已验收 T02 SHA
`4f2845da8a3565d8bbedec082fdbf2fc40319099`。硬件 A100-SXM4-80GB，BF16，
单请求 eager Python runner。

## 已实现

短窗口层缓存只保留窗口覆盖的文本 KV，所有已完成块的 summary KV 保留；长窗口层
在当前 8192 文本预算内保留全部文本。窗口跨块前移时，先把已完成块的 summary 纳入
远端可见集合，再裁剪旧文本；未跨块时直接沿用已有 KV，避免无效复制。每个窗口的
key 位置和 summary 标志与 KV 一起提交，forward 异常前不修改已提交缓存。
`KSACache.clear()` 同时回收所有层与布局元数据。缓存仍属于专用 Python runner，
**尚未接入 vLLM scheduler 的 KV cache spec、容量规划和分页分配**。

## 本地与 A100 检查

`tests/model_executor/test_ksa_prefill.py`：44 passed；Ruff check/format 通过。
测试覆盖淘汰边界、部分块、重复复用、非连续张量存储、释放与异常原子性。
A100 `benchmarks/ksa/compressed_kv.py` 对冻结 T00 的 17 个教师强制案例逐一比较，
四次开发运行均为 17/17 pass；结果在
`/workspace/volume/h20-data/xutianle/KSA/results/T03/` 的
`a100-dev-20260916T0736Z`、`0739Z`、`0743Z`、`0747Z` 目录。
这些是开发运行，脚本后来加入冻结身份校验；正式绑定 SHA 的验收尚未完成。
提交 `ee0c2b8e90d0e14c478d01cdaa19602c8d3a89cc` 后，以同一冻结基线和模型哈希
复跑于 `a100-ee0c2b8/`：17/17 pass；1K/4K 稳态 TPOT 为 42.071/42.129 ms，
4K 峰值 14.538 GiB。此结果仍只验证专用 Python runner，不构成完整 T03 验收。

五次正式重复、一次预热，128 token，关闭 EOS；稳态 TPOT 均值如下：

| 版本或改动 | 1K ms | 4K ms | 4K 峰值 GiB |
| --- | ---: | ---: | ---: |
| T02 全历史缓存 | 41.042 | 41.113 | 16.040 |
| 压缩 KV 初版 | 46.898 | 46.715 | 14.538 |
| 简化布局构造 | 46.693 | 47.177 | 14.538 |
| 缓存布局元数据 | 45.146 | 45.728 | 14.538 |
| 只在窗口前移时裁剪 | 40.805 | 41.064 | 14.538 |

最终 TPOT 与 T02 差异不超过测量波动，**不宣称性能提升**。4K 峰值显存下降
约 1.50 GiB；该指标包含模型、临时张量和预填充，不能作为有效 KV 占用。
4K prompt 后实际各层 KV 行数为 `[1536, 1536, 1536, 4608]` 循环，
全模型有效 KV 324 MiB；1K prompt 为 162 MiB。短窗口层符合
`O(1024 + N/8)`；36 层总量包含长窗口层的线性文本 KV。当前 runner
没有独立预分配 KV 池，因此“预分配总池”不适用。

## 尚待完成

1. 将文本组与 summary 组真正纳入 vLLM KV cache spec、页分配、容量估算、
   请求结束及取消的生命周期。现在仅有 request-owned tensor 缓存。
2. 使用实际非连续物理页进行端到端复用与回收测试；当前只有非连续 tensor stride
   回归，不能证明分页正确。
3. 完成 HF/T02/当前版自由生成、峰值及逐项独立验收，并以正式 SHA 再跑实验。
4. 长上下文仍被首次 prefill 4096 文本上限、8192 总预算及平方级显式掩码阻断。
   后续需先实现分块 prefill/分页 attention，再测 16K 以上与 128K。

复现实验（结果目录必须不存在）：

```bash
.venv/bin/python benchmarks/ksa/compressed_kv.py \
  --model /workspace/volume/h20-data/xutianle/KSA/models/KSA-4B-base \
  --baseline /workspace/volume/h20-data/xutianle/KSA/results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw \
  --output /workspace/volume/h20-data/xutianle/KSA/results/T03/a100-final
```
