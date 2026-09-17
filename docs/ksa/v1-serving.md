# 通用 vllm serve / V1 接入

2026-09-17：KSA 已接入标准 `LLM`、V1 scheduler、GPU worker 和 OpenAI 服务入口。
这次补齐 T04 的通用框架接入；T05 CUDA Graph 和 T06 attention 算子优化仍独立推进。
精度和服务实测记录见 [A100 验证报告](results/V1/a100-20260917/README.md)。

## 运行

从本仓库根目录执行，现有 CUDA 扩展须与此版本 vLLM/PyTorch 匹配：

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=1 .venv/bin/vllm serve ../models/KSA-4B-base \
  --served-model-name ksa --host 127.0.0.1 --port 18005 \
  --enforce-eager --no-enable-prefix-caching \
  --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 257 \
  --gpu-memory-utilization 0.5
```

启动日志必须显示 `Resolved architecture: KSAForCausalLM` 和异步调度关闭。
`PYTHONPATH` 显式选择当前源码；若依赖 editable 安装，使用
`uv pip install ... --editable . --config-settings editable_mode=compat`。
在启用了 system-site-packages 的环境中，默认 editable finder 可能被系统同名包遮蔽。
可用 `.venv/bin/python -I -c 'import vllm; print(vllm.__file__)'` 验证，
不要只在仓库当前目录执行普通 `python -c` 验证。
工作区上一级的 `install_vllm.sh` 已补上 compat 安装及隔离模式路径检查。

```bash
curl http://127.0.0.1:18005/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ksa","prompt":"Hello world","temperature":0,"max_tokens":32}'
```

离线使用相同的标准引擎：

```python
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(
        model="../models/KSA-4B-base",
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=8192,
        max_num_seqs=8,
        max_num_batched_tokens=257,
    )
    result = llm.generate("Hello world", SamplingParams(temperature=0, max_tokens=32))
    print(result[0].outputs[0].text)
```

## 实现与边界

- 使用标准 `Scheduler`；新增预算计算只对 `use_summary_attention` 模型生效。
  每次分配的计算行数为 `n + floor((start+n)/8) - floor(start/8)`，抢占退还相同预算。
  scheduler 对外 token 数、输出位置和 usage 仍为文本语义。
- GPU worker 显式选择 `GPUModelRunner` 的 KSA 子类，复用标准 InputBatch、
  请求状态、采样器和 prompt logprobs。默认选择 V1，显式选择 V2 会报错。
- 文本页和 summary 页通过标准 KVCacheSpec 上报，纳入 V1 的显存估算、分配、
  抢占和释放。worker 只使用 scheduler 下发的 block IDs，不再创建第二个分配器。
  文本页覆盖 8 个文本位置；summary 页覆盖 64 个文本位置、实际保存 8 个 summary。
- 短窗口保留完整的最早可见文本块，避免调度时释放块内仍需读取的 KV。
  所有请求当前 chunk 的 attention 完成后才写回新 KV。重排按请求 ID 关联，
  抢占恢复从文本历史重新生成 summary，取消/完成清理对应 worker 状态。
- 显存 profiling 包括 summary、历史 gather、mask 和待提交 KV；历史缓存本体由
  scheduler 的 KV 预算承担，不重复计入 activation。当前 attention 仍逐请求执行，
  使用显式 mask 和历史 gather，不能把本次接入宣称为长上下文性能优化。
- 内部 summary token 从输入校验中排除，原始词表中对应 logits 为负无穷。
  标准 sampler 的词表宽度保持不变，支持采样、seed、`n`、惩罚参数、stop strings、
  token logprobs、prompt logprobs、echo、SSE 和标准文本 usage。
- 验证范围为单张 A100、BF16、TP=PP=DP=1、eager，总文本上下文最多 8192，
  单次每请求 chunk 最多 4096。前缀缓存、量化、LoRA、推测解码、V2、异步调度、
  非 auto KV dtype、KV 传输/卸载、sleep、prompt embeddings 和 pooling 明确拒绝。
  CUDA Graph、128K 和新的高性能 attention 算子尚未交付。
- Chat 使用标准服务的模板处理；base checkpoint 没有模板时，需要用户提供合适的
  `--chat-template`。此次 HTTP 验证针对 `/v1/completions`，未替 base 模型编造模板。

## 复现验证

输出目录必须不存在。GPU 实验顺序执行，避免相互占用显存。

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/model_executor/test_ksa_prefill.py -q
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py tests/v1/core/test_async_scheduler.py \
  -q -k 'not pp and not pipeline'
```

```bash
BASELINE=../results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw
OMP_NUM_THREADS=1 .venv/bin/python benchmarks/ksa/v1.py \
  --model ../models/KSA-4B-base --baseline "$BASELINE" \
  --row-budget 257 --output ../results/V1/accuracy-257
OMP_NUM_THREADS=1 .venv/bin/python benchmarks/ksa/v1.py \
  --model ../models/KSA-4B-base --baseline "$BASELINE" \
  --row-budget 31 --output ../results/V1/accuracy-31
OMP_NUM_THREADS=1 .venv/bin/python benchmarks/ksa/v1_pressure.py \
  --model ../models/KSA-4B-base --baseline "$BASELINE" \
  --output ../results/V1/pressure
```

精度脚本通过真正的 V1 调度执行请求，只观察已执行的文本 hidden states，
导出冻结教师位置的 logits；不替换模型执行或采样。
诊断 callable 使用进程内 V1 引擎，避免启用不安全序列化；HTTP 测试使用默认多进程引擎。
压力脚本将 KV 池限制为 330 块，两个不同 prompt 各执行 80 个固定教师 token，
比较串行与抢占重算的原始 logits，同时检查累计输出、取消、物理页回收和请求复用。

启动本页给出的标准服务后，在另一终端运行：

```bash
.venv/bin/python benchmarks/ksa/v1_serving.py \
  --url http://127.0.0.1:18005 --model ksa --output ../results/V1/http
```

该脚本验证协议和生命周期。并发耗时仅是 smoke 数据，没有五次重复和完整长度矩阵，
不构成相对 HF 或 T04 的性能收益结论。
