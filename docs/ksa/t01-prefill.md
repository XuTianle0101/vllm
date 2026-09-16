# T01 Python prefill

状态：DONE，限已验收 A100 环境和单请求完整 prefill。[完整结果](results/T01/a100-20260916T063504Z/README.md)。

## 实现契约

`use_summary_attention=true` 的 Qwen3 checkpoint 自动解析为 `KSAForCausalLM`；普通 Qwen3 保持原路径。直接使用 vLLM 的 `get_model` 和 `AutoWeightsLoader` 加载原始权重，检查缺失、重复及多余参数并报告 checkpoint 名称。当前 mix_coeff=0，不创建独立 summary 投影；不支持非零混合系数、量化或 TP/PP 大于 1。

`forward(input_ids, positions, inputs_embeds=...)` 接受一条从 0 开始的完整文本序列或其 embedding，在每个完整 8-token 块末加入一个 summary，内部 RoPE 位置等于该块最后文本位置。返回值只包含文本隐藏状态，`text_row_indices` 保留文本到内部行映射。`compute_logits` 沿用官方文本预测词表裁剪，summary token 不进入默认采样词表。

FP32 oracle 和 BF16 PyTorch SDPA 共用位置与可见性生成逻辑；按不同窗口缓存当前 forward 的布尔掩码。支持 1–4096 文本 token，超限在分配掩码前报错。`layer_observer` 可观察每层残差合并后的内部状态。普通 serving forward context 会显式报错：T01 不支持 decode，也未填充供 serving 使用的 KV cache。

summary 查询可见同块因果文本与自身 KV，文本查询读取局部文本及严格超出窗口的远端 summary；与 T00 冻结契约一致。

## 环境与检查

服务器已有 `.venv` 和 `.venv-ksa-hf`。完整版本记录在结果的 environment.json；HF 子进程会核验冻结依赖集合以及 baseline.py、compat.py、模型和输入的哈希。启动器主动把本 checkout 放到导入路径首位，避免误用 site-packages 中的模型实现。

新环境按 vLLM 开发安装流程准备 `.venv`，HF 环境按 `benchmarks/ksa/requirements.lock` 创建。已有环境的检查命令：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/model_executor/test_ksa_prefill.py -q
.venv/bin/python -m pytest benchmarks/ksa/test_baseline.py -q
.venv/bin/pre-commit run --files \
  vllm/model_executor/models/ksa.py \
  vllm/model_executor/models/ksa_prefill.py \
  vllm/model_executor/models/registry.py \
  tests/model_executor/test_ksa_prefill.py \
  benchmarks/ksa/prefill.py
```

## 服务器复现

当前提交尚未推送：服务器 GitHub 认证失败。恢复认证并推送后可执行下面的远端同步；本服务器已有完整提交，可直接使用本地 SHA。源码增量 bundle 保存于 `/workspace/volume/h20-data/xutianle/KSA/results/T01/source.bundle`，以前置提交 `39ccd67fb529dd2dac2d7086af031d3e76838a9b` 为基点。

在干净的 vLLM checkout 中同步实验提交，结果必须写入新的空目录；已有结果不会被覆盖。以下命令对应本服务器实际模型与 T00 路径：

```bash
git fetch origin releases/v0.26.0-ksa
KSA_SHA=a7a696efd57283d56098e3b91cb73aa9c47c3901
git switch --detach "$KSA_SHA"
test "$(git rev-parse HEAD)" = "$KSA_SHA"
KSA_RESULT="docs/ksa/results/T01/a100-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$KSA_RESULT"
set -o pipefail
.venv/bin/python benchmarks/ksa/prefill.py \
  --model /workspace/volume/h20-data/xutianle/KSA/models/KSA-4B-base \
  --baseline /workspace/volume/h20-data/xutianle/KSA/results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw \
  --hf-python .venv-ksa-hf/bin/python \
  --expected-sha "$KSA_SHA" \
  --output "$KSA_RESULT" 2>&1 | tee "$KSA_RESULT/run.log"
```

输出 environment.json、inputs.json、correctness.json、timing.json、summary.md、HF 逐层张量和日志。失败时保留 failure.json 与 traceback，退出码非零。归档时保留大张量及哈希；不将 `.pt` 提交到 Git。

当前启动器涵盖 17 个冻结短序列/4K/中英文/检索输入、17/1025 的 FP32 全层参考、1K/4K 预热与各五次计时，以及 16K 超限检查。阈值与 baseline ID 来自 T00；不支持的长序列不是通过项，不执行生成或 decode。
