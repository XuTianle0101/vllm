# T06：分页 Triton attention

prefill 与 decode 均使用 `ksa_attention.py`，不修改 C++/CUDA。
CUDA 模型默认启用；`model.triton_attention = False` 可重现 T05 Python/SDPA 对照，
`model.reference_attention = True` 使用 FP32 oracle。CPU 与无分页的参考 cache
保留原路径。生产批次要求共用一个页池，不混合参考 cache 与分页 cache。

## 数据与归一化

每层一次批量 attention launch；program 按请求、KV head、query tile 分配，
GQA query heads 共享 KV，读取物理槽位而非 gather 历史 KV。支持独立 runner 的
共享页池和 V1 的独立 text/summary 页池。新 KV 在所有层 attention 完成后提交，
防止 chunked prefill 覆盖早期 query 仍需读取的页。

prefill 使用 tile 内的绝对位置/summary 谓词，不分配平方级 mask，也不遍历未来
key tile。decode 用 8 路 split-K；每段保存 FP32 最大值 `m`、指数和 `l` 和
未归一化加权值 `a`，以全局最大值 `M` 合并为
`sum(exp(m-M)*a) / sum(exp(m-M)*l)`。空分区权重为零。
局部文本与远端 summary 在同一归一化下计算，summary query 包含自身。

额外存储为线性页元数据、新 KV、输出以及固定 8 路 decode 部分结果；没有完整历史
KV 复制或 GQA KV 重复。图缓冲区固定输入、页元数据和输出地址，历史数据始终留在
scheduler 页池中。页分配、输入校验和 KV 提交仍在图外，decode 图仍需显式开启。

## 复现实验

从仓库根目录执行；输出目录不能已存在。`T06_SHA` 包含完整验证脚本、V1 profiling 与槽位整理修复；Triton 内核数学计算未改。

```bash
T06_SHA=14a46ddc3af405ceac6647305d736cf59a22d229
git fetch origin releases/v0.26.0-ksa
git checkout "$T06_SHA"
test "$(git rev-parse HEAD)" = "$T06_SHA"
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python -e . --torch-backend=auto
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/model_executor/test_ksa_prefill.py -q
OMP_NUM_THREADS=1 .venv/bin/python benchmarks/kernels/benchmark_ksa.py \
  --output ../results/T06/kernel-final.json
OMP_NUM_THREADS=1 .venv/bin/python benchmarks/ksa/attention_kernels.py \
  --expected-sha "$T06_SHA" --model ../models/KSA-4B-base \
  --baseline ../results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw \
  --output ../results/T06/a100-final --skip-trace
OMP_NUM_THREADS=1 .venv/bin/python benchmarks/ksa/cudagraph_v1.py \
  --expected-sha "$T06_SHA" --model ../models/KSA-4B-base \
  --baseline ../results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw \
  --output ../results/T06/v1-final --compare-t05 --timing-repeats 5
```

完整模型测量复用 T05 的固定输入、教师强制、冻结 HF logits、阈值和 decode 计时：
1K/4K × batch 1/4/8，32 步，一次预热与五次正式重复。额外单列 batched prefill
加末行 logits/greedy 的墙钟耗时。decode 包含 metadata、KV 提交、logits 与 greedy，
不包含网络或 scheduler。首次 JIT/capture 不计入稳态。HF 环境与 T00 不变。

单算子测量另列参考 gather、attention、合计和 Triton metadata staging，
没有复用官方接口，因此不存在被遗漏的官方格式转换成本。

请保留完整模型目录的 `environment.json`、`correctness.json`、`timing.json`、
`startup.json`、`status.json`，以及 V1 `results.json`、logits 文件和运行日志。
单算子加速不能代替整模型验收；`status.json` 的 pass 表示精度实验执行通过，
各路径性能收益必须另据五次测量判断。A100 结果不能作为 SM120/RTX 5090 验收。

T06 的标准 V1 命令额外加 `--compare-t05`，会将首轮 eager 设置为 T05 SDPA，
随后两轮 graph 使用 Triton。这也覆盖真实调度器的 257-token chunked prefill、
独立 text/summary 页池与请求重排。生成轨迹分歧后只比较共同前缀，不能混比后续 logits。

完整矩阵结束后运行性能门禁汇总（不接受缺失形状、少于五次或混入捕获的结果）：

```bash
.venv/bin/python benchmarks/ksa/summarize_attention_kernels.py \
  --raw ../results/T06/a100-final --output ../results/T06/a100-summary.json
```

脚本要求每个形状的 prefill、eager decode、graph decode 均满足 Triton 最慢重复
仍快于 T05 最快重复，作为超过本次观测波动的保守门禁；同时核验冻结精度与页归还。
单算子的最终计时在 CUDA Graph 中用事件测量，另用墙钟报告 metadata 更新开销。

本轮用户确认以 A100 验收；5090 保留为后续独立验证，不阻塞 T06 关闭。
`--timing-repeats 5` 另运行 V1 的 1K/batch=4、32 输出 token 测量，移除精度阶段
的逐请求 logits 导出钩子。包括真实 scheduler、chunked prefill、采样与 decode，
不含 HTTP。保存 `timing.json`，首轮预热和任何重捕获均须单列。
