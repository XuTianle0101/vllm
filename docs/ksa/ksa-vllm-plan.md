# KSA-4B-base 的 vLLM 高性能推理适配计划

状态：T00–T06 已完成 A100 验收；T07 最终验收发现生成精度超限，保持 NEEDS_FIX。
当前结果与硬件边界见 [T07 报告](results/T07/a100-final/README.md)。

## 1. 目标和约束

在当前 vLLM checkout 中适配 KSA-4B-base，面向 Linux、单卡 RTX 5090、BF16 权重与 KV cache、TP=1。以同一份权重的 Transformers/论文仓库推荐部署路径为基准，先实现正常精度，再实现优于该基准的性能。

最终范围：原生模型加载、离线生成、OpenAI 兼容基础服务、压缩 KV decode、连续批处理、chunked prefill 和 CUDA Graph decode。保留 eager 路径用于对照。

首版不包含量化、LoRA、多卡并行、前缀缓存、推测解码和分离式部署。不支持的组合应明确报错，不得静默回退到错误语义。

开发分三阶段：

1. 仅修改 Python，利用 PyTorch 现成算子，完成正确的模型加载、prefill 和 decode。
2. 优化压缩 KV、数据搬运、调度和图执行，每项优化都有前后比较。
3. 根据 profiling 评估官方 KSA 算子；不能满足要求的路径使用 Triton，不修改 C++/CUDA 源码。

中间 ticket 可临时限制功能或使用低性能参考实现，但不得以中间原型代替最终交付。

## 2. 已检查的环境与参考资料

- 模型：`models/KSA-4B-base/`，包含权重、配置、`modeling_qwen3.py` 和 `summary_context.py`。
- 论文：根目录 `KSA.pdf`。
- vLLM 远端：`https://github.com/XuTianle0101/vllm.git`。
- 当前开发分支：`releases/v0.26.0-ksa`；正式开发前重新核验状态。
- 本地 GPU 为 RTX 4060 Ti 8GB，完整 BF16 与性能验收在用户服务器 RTX 5090 上进行。
- 官方参考：[KSA 仓库](https://github.com/Kuaishou-OneRec/KSA)、[模型发布页](https://huggingface.co/OpenOneRec/KSA-4B-base)。

参考版本和模型哈希在 T00 冻结。官方依赖能否在目标 5090 上直接运行尚未经过实测。

## 3. 模型与 attention 语义契约

- 通过 `use_summary_attention` 识别 KSA，新增原生模型实现，兼容现有 `Qwen3ForCausalLM` 架构声明，无须修改原始 checkpoint。普通 Qwen3 保持原行为。
- 复用 Qwen3 的权重加载、投影、Q/K normalization、RoPE、MLP 等组件。当前配置 `mix_coeff=0`，共享文本与 summary Q/K/V 投影，不创建无效独立参数。
- 安全解析配置中的列表重复表达式，禁止直接 `eval`。首版覆盖当前 checkpoint：每块 8 个文本 token、1 个 summary，36 层，窗口 `128/128/128/16768` 循环。
- 文本位置保持原编号，summary 位置等于所属块最后一个文本 token 的位置。summary 隐状态经过所有 Transformer 层。
- 文本位置 `i` 所在块为 `c=floor(i/8)`，层窗口为 `W`：读取从块 `max(0,c-W)` 开始的因果文本，并读取块编号小于 `c-W` 的远端 summary。局部文本与 summary 不重复覆盖同一历史块。
- 本地 checkpoint 的 decode 中 summary 读取本块文本及自身 KV。论文文字对此较简化；T00 必须核对官方 prefill 与 decode，若存在分歧先记录并解决基准问题，不能静默选择方便实现的一方。
- 长窗口层的文本 query 在支持的上下文范围内读取全历史文本，summary query 仍为块内专用 attention。不能直接对混合序列使用普通全因果 attention。
- 内部 summary 由运行器按绝对文本位置插入。块末文本和 summary 在同次 forward 执行；采样对应文本行，summary 行不参与输出或 prompt logprobs。内部 token ID 不作为普通生成结果。
- 对外生成长度、停止条件和 usage 使用文本语义；实际计算预算、缓冲区和图捕获尺寸必须计入 summary 行数。

## 4. 实现方向

### Python 正确性阶段

先建立 FP32 小规模显式掩码参考及 BF16 模型对照。T01 可以使用短序列 dense mask；T02 可以暂存完整历史 KV。这些是可替换的正确性基线，不是最终长上下文方案。

### 缓存与框架阶段

短窗口层分别管理按块淘汰的近期文本页和按 8:1 增长的 summary 页，长窗口层保留文本页。缓存全部纳入 vLLM 的容量估算、分配和回收。

chunked prefill 允许任意文本切分。当前批次的 Q/K/V 与历史分页缓存共同参与 attention；必须避免缓存提交或复用覆盖本批次早期 query 仍需读取的数据。

请求取消、完成、重排和抢占重算同步处理所有缓存组；从文本历史重算时正确恢复 summary。图执行使用预分配 metadata，覆盖普通与块边界 decode 混合批次。

首版接入现有 GPUModelRunner，显式选择受支持路径，不静默进入未经适配的其他运行器。

### 算子阶段

prefill 和 decode 分别评估官方算子。只有许可证、5090 兼容性、掩码语义、缓存接口和图支持均满足要求，且端到端收益成立，才复用；否则为该路径实现 Triton。

最终生产路径不构造平方级 mask、不复制完整历史 KV、不逐请求在 Python 中执行 attention。局部文本和远端 summary 使用统一 softmax 归一化；split-K 或分区计算必须稳定合并。

## 5. 精度门禁

固定模型、tokenizer、输入 token IDs、精度、RoPE 和生成配置。比较 teacher-forced logits/logprobs、top-1 一致率、固定生成用例以及长距离检索。

数值容差在 T00 根据参考重复运行、数值误差与仓库现有测试规范校准，在 T01 开始前冻结。不得为优化后的错误放宽容差。接近并列 logits 导致的生成分歧需要逐项说明，不能仅以文本相似判定通过。

必要场景：长度 1/7/8/9/15/16/17，1024-token 窗口及淘汰边界，非连续页，部分块，多次跨块，整段/分块 prefill，单请求/混合批次，eager/graph，取消、停止、重排、回收与抢占重算。

每个 ticket 必须同时记录相对 Transformers 和前一已验收 ticket 的变化；不适用的比较明确标记，不伪造结果。

## 6. 性能门禁

- HF 使用官方推荐缓存和算子，不能人为替换为慢速 dense 实现；官方路径不能运行时先解决或报告基线阻塞。
- HF/vLLM 使用独立、固定版本环境，同卡顺序运行，不争抢 GPU。记录运行时差异。
- 固定输入和输出长度，性能实验避免提前 EOS 改变输出长度；预热后至少重复 5 次。
- 报告 TTFT、稳态 TPOT、块边界 TPOT、输出 tokens/s、峰值显存与实际 KV 页占用；加载和编译时间单独记录。
- 主矩阵为 4K、16K、32K、64K；128K 场景使用接近上限的 prompt，预留生成空间，总文本长度不超过 131072。
- 4K–64K 并发为 1/4/8；128K 仅测单请求，用户 2026-09-18 明确无需测试 128K×4/8。OOM 保留记录。HF 如果仅支持 batch=1，直接比较限于单请求，vLLM 并发结果单列。
- 最终 16K、32K、64K 单请求 decode 均需优于 HF，收益超过重复测量波动；短窗口层实际 KV 呈 `O(1024 + N/8)` 增长，并支持 128K 单请求推理无 OOM。
- 短上下文或 prefill 的回退必须披露和分析。不能用单内核加速代替端到端结论。

## 7. 分步交付

顺序：T00 基线 → T01 prefill → T02 decode → T03 缓存优化 → T04 批处理 → T05 图与 profiling → T06 算子 → T07 验收。

每次只推进一个 ticket：实现和本地检查 → 更新文档、commit、普通 push → 提供绑定 SHA 的服务器命令和结果格式 → 用户运行并反馈 → 分析并关闭或修复复测 → 下一个 ticket。

完整流程见 [workflow.md](workflow.md)，ticket 列表见 [tickets/README.md](tickets/README.md)。当前按上述开发流程处理 T07 验收失败。
