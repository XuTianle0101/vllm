# T06 算子路径选择（T05 输入）

选择：prefill 与 decode 均实现仓库内 Triton，保留现有 Python/SDPA 作为精度参考。
这项接口兼容性决策已落实为 T06 的批量分页内核。A100 完整模型矩阵已测得收益，
见 [实验报告](results/T06/a100-a455b6608b/README.md)。性能优先级来自 T05 trace。

## 官方 KSA 算子审查

核查日期 2026-09-17，官方仓库提交
`9998565d0aa2907cc99e3ab0717b8bf246e77876`。
审查对象是官方发布的 wheel，未修改或安装进冻结 HF 环境。

- [KSA 仓库](https://github.com/Kuaishou-OneRec/KSA)代码为 Apache-2.0。
  附带 `flash_attn_cute` 的许可证为 BSD-3-Clause；复用时须保留各自声明。
- [summary_attn 0.3.0](https://github.com/Kuaishou-OneRec/KSA/blob/9998565d0aa2907cc99e3ab0717b8bf246e77876/summary_attention_kernel/summary_attn-0.3.0-py3-none-any.whl)
  的接口为 `summary_attn_func(q, k, v, chunk_size, summary_num,
  sliding_chunk_num, summary_pos=None, skip_old_summary=False)`。
  输入是连续 Q/K/V，没有物理页表、每请求绝对文本位置、独立 text/summary 池或 slot 参数。
- 语义支持块对齐局部文本、远端 summary 和 summary 自身；不能直接把普通滑动窗口
  当作等价实现。`summary_pos` 分支通过索引和拼接复制 K/V，仍需计入转换成本。
- [附带 CuTe 0.1.0 wheel](https://github.com/Kuaishou-OneRec/KSA/blob/9998565d0aa2907cc99e3ab0717b8bf246e77876/summary_attention_kernel/flash_attn_cute-0.1.0-py3-none-any.whl)
  的 `flash_attn/cute/interface.py:247` 仅接受 major capability 9、10、11。
  上层 summary wrapper 使用 `major >= 9` 选择 CuTe，因此 SM120 会进入不支持的分支。
  **该发布组合不支持直接在 RTX 5090 上执行**，不能以笼统的 Blackwell 支持代替 SM120 核验。
  [NVIDIA GPU 表](https://developer.nvidia.com/cuda/gpus)确认 RTX 5090 为计算能力 12.0。
  本轮没有 5090 实机，未做运行测试。
- A100 走 FlexAttention fallback；mask/cache 构建中存在 host `.item()` 和首次编译。
  有固定形状预热后捕获的可能，但不能据此认定它支持动态请求长度、分页或图重放。
  本轮没有对官方算子做新增性能实验，冻结 T00 仍是完整 HF 路径结果。

审查文件 SHA256：

| 文件 | SHA256 |
| --- | --- |
| summary_attn wheel | `3a8e7893b17b4bcc2daf13d75ed30d75631a25da18b63a515279dcd4f18cc536` |
| flash_attn_cute wheel | `1a687930435ba56daa067819069c56c2b3370b7c2e75ca78f63d211ace414830` |

## 其他库与实施边界

[FlashInfer](https://github.com/flashinfer-ai/flashinfer)为 Apache-2.0，当前支持列表包含
SM120；[分页 prefill API](https://docs.flashinfer.ai/api/attention.html)提供 custom mask
和 CUDA Graph 静态缓冲区接口。但这不证明某个 backend 已覆盖 KSA 的两组页池、
summary 自身和动态块相位；密集 custom mask 也不能满足 T06 的长序列内存要求。
本轮不引入新的依赖或把它当成已经验证的 KSA 算子。

T06 先实现直接读取 text/summary 两组页表的 decode kernel：请求维度批处理、
不复制完整历史，不将 GQA 的 KV 实体重复四份。文本 query 将近端文本与远端 summary
放入同一 softmax；summary query 仅读取本块文本及自身。
如需 split-K，按每段最大值与 exp-sum/LSE 稳定合并。

prefill 使用按绝对文本位置和 summary 标志计算的分块谓词，覆盖短尾、跨 chunk、
已有历史和新 KV；禁止构造完整平方 mask。写回须保持当前 chunk 的早期 query
在全部 attention 完成前仍能读取旧页。无需修改 C++/CUDA 源码。

用户于本轮明确确认按 A100 验收 T06，5090 不作为关闭门禁。未来部署到 5090
仍需单独验证 Triton 编译、FP32 oracle/BF16 误差、分页边界和图 replay；
A100 结果不代表 5090 通过。本轮接入以 A100 五次稳态端到端测量和冻结精度门禁为准。

## T06 实施记录（2026-09-17）

在上述决策基础上，prefill/decode 都使用仓库内 Triton：按请求、KV head 和
query tile 分配 program，GQA query heads 共享 KV。直接读取 scheduler 的物理槽位，
支持独立 text/summary storage，也支持独立 runner 的共享 storage；新 KV 在本次
attention 后才提交。query tile 内用绝对位置和 summary 标志生成谓词，局部文本、
远端 summary 和 summary 自身在同一个在线 softmax 中归一化。

不复用官方接口，因此没有官方 gather/格式转换；单算子脚本分别报告现有参考路径
的 gather、attention、合计，以及 Triton metadata staging。CUDA Graph 固定的是
metadata 和新输入地址，不再复制历史 KV。Python/SDPA 和 FP32 oracle 保留为对照。
A100 18 项完整模型性能门禁与 68 项冻结精度比较通过。五次稳态中，prefill
为 1.77–2.83×，eager decode 为 1.15–2.56×，graph 对旧 graph 为 2.07–11.50×。
详细样本、较快旧路径对照和原始数据哈希见上述报告，不外推到 5090 或长上下文。
