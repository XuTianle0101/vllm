# T05：CUDA Graph 与 profiling

当前实现为固定形状的 decode 图，prefill 和包含 prefill 的 batch 继续 eager。
`KSADecodeGraphs` 按 batch 大小和历史长度的二次幂桶缓存最多四张图。
每个请求固定两行（文本、summary）；普通 decode 的第二行只作填充，不提交 KV。
输入 ID、RoPE 位置、可见性 mask 和历史 KV 有固定地址，图槽位不拥有请求或分页表。
在图外验证输入、更新 metadata、gather KV 和提交新 KV；这部分 CPU/同步开销仍存在。
输出复制为调用方所有，避免下一次 replay 覆盖上一次返回值。
LM head 与采样留在图外；V1 在两行 padding 会超过 worker 预算时回退 eager。

标准 V1 入口可通过 `--additional-config '{"ksa_cudagraph":true}'` 选择此路径；
仍需 `--enforce-eager` 禁用通用 dense CUDA Graph/编译器。
这是独立的 KSA 图执行器；默认仍为原 eager 行为。
捕获为首次遇到桶时懒执行，启动成本单列。图缓存有界，但图私有内存不在 KV 池中，
实验须为图预留显存（例如 `--gpu-memory-utilization 0.5`），不能沿用接近满显存的页池。

## 验证契约

CPU 测试在真实小模型上运行相同固定缓冲区计算，检查混合相位、窗口淘汰、
重排、取消后槽复用、旧输出存活及分页 KV。CUDA 测试使用真实 4B 权重：
冻结教师强制案例分别与 eager 和 HF 比较，保持 T00 阈值；请求批内每步反转，
不同长度请求反复占用同一图槽位。CPU 测试不能替代 CUDA 捕获验收。

性能入口 `benchmarks/ksa/cudagraph.py` 默认运行 1K/4K × batch 1/4/8，
32 decode 步，一次预热和五次正式重复。多请求故意错开块相位。
性能区间包括模型、图外 metadata/KV、logits 和 greedy；不含网络、scheduler 和 prefill。
正式重复若产生新捕获会报错，禁止将捕获成本算成稳态。

从仓库根目录执行（`T05_SHA` 使用 ticket 中最终代码 SHA，输出目录不能存在）：

```bash
T05_SHA=bc74fbe2c2b8099c987ee3d38eeda35027ef8418
git fetch origin releases/v0.26.0-ksa
git checkout "$T05_SHA"
test "$(git rev-parse HEAD)" = "$T05_SHA"
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python -e . --torch-backend=auto
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/model_executor/test_ksa_prefill.py -q
OMP_NUM_THREADS=1 .venv/bin/python benchmarks/ksa/cudagraph.py \
  --expected-sha "$T05_SHA" --model ../models/KSA-4B-base \
  --baseline ../results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw \
  --output ../results/T05/a100-final
```

输出 `correctness.json`、`timing.json`、`startup.json`、`status.json`、
`environment.json`、两种模式的 Chrome trace、hotspots 和 ranges。
trace 包含初始化请求的 eager prefill，应按 `ksa.graph_replay` 等范围定位 decode；
不能把整个 trace 的 GPU 总时间当 decode attention 时间。
`ksa.projection_qkv_rope`、`ksa.attention`、`ksa.projection_output_mlp` 分解模型计算，
`ksa.metadata`、`ksa.kv_stage`、`ksa.kv_commit` 定位图外成本。
图 replay 没有 Python 内层 range，GPU kernel 热点和图外范围需分别阅读。

A100 验收已完成，见 [完整报告](results/T05/a100-d38983dd88/README.md) 和
[T06 路径决策](t06-kernel-decision.md)。核心矩阵实际执行于
`d38983dd881a9766f0349254d5b0b6fda95b3753`；上面最终代码提交保持图数学计算及
矩阵脚本不变，增加了 V1 padding 预算保护和以下补充验证。

```bash
OMP_NUM_THREADS=1 .venv/bin/python benchmarks/ksa/cudagraph_v1.py \
  --expected-sha "$T05_SHA" --model ../models/KSA-4B-base \
  --baseline ../results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw \
  --output ../results/T05/v1-final
mkdir -p ../results/T05/summaries
.venv/bin/python benchmarks/ksa/summarize_cudagraph.py \
  --raw ../results/T05/a100-final --output ../results/T05/summaries/a100.json
```

V1 验证使用真实调度器、页表和槽位，导出共同生成前缀 logits；轨迹分歧单列，
分歧后的 logits 不作错误的逐步比较。摘要脚本剔除 trace 中的 prefill，分别汇总
八个 decode 步的 CPU 范围与 GPU kernel 活动时间，避免 GPU annotation 重复计数。
请返回两个结果目录的 JSON 和运行日志；失败时保留 trace、logits 与异常，不覆盖原目录。

本轮 1K/batch=1 图路径为 2.043×，4K/batch=8 为 0.399×，因此图模式保持显式开启。
该性能矩阵不含服务 scheduler/网络，不能将它作为标准服务吞吐或 5090 收益。
