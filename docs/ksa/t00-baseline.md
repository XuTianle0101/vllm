# T00 基线交付与服务器运行

状态：DONE。A100 完整矩阵已通过，见 [验收结果](results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/README.md)。性能与精度必须绑定本轮 baseline ID；此前不完善的结果已清理。

## 固定来源与兼容配置

- [KSA 源码](https://github.com/Kuaishou-OneRec/KSA/tree/9998565d0aa2907cc99e3ab0717b8bf246e77876)：`9998565d0aa2907cc99e3ab0717b8bf246e77876`。
- [HF 发布](https://huggingface.co/OpenOneRec/KSA-4B-base/tree/6b60859be46422fc5949a0e69d2a338c4a618c90)：`6b60859be46422fc5949a0e69d2a338c4a618c90`。模型、tokenizer、remote code 逐文件 SHA256 必须匹配 `model-manifest.json`。
- 官方 Docker 环境为 PyTorch 2.6.0/cu126、Transformers 4.57.1；checkpoint 的 `transformers_version=4.51.0` 不是推荐安装版本。
- 官方 `summary_attn-0.3.0` wheel SHA256：`3a8e7893b17b4bcc2daf13d75ed30d75631a25da18b63a515279dcd4f18cc536`。
- 本配置使用 PyTorch 2.7.1/cu128、Transformers 4.57.1、tokenizers 0.22.1 和官方 summary_attn 0.3.0；44 项 Linux 依赖由 `requirements.lock` 锁定版本与哈希。

本轮配置明确命名为 **`official-flex-cu128-repaired-v1`**，不是未经修改的原始官方基线。模型文件保持发布版本；`compat.py` 显式安装以下适配，源码哈希写入 baseline lock：

1. 兼容 remote code 使用的 `check_model_inputs()` 装饰器工厂调用方式。
2. 将 `summary_pos` 规范为一维，修复长度 1 时 `.squeeze()` 产生标量、布尔索引把四维 Q/K/V 变成五维的问题。
3. 修复官方 FlexAttention BlockMask：普通块与完整块分离，保留完整宽度的索引表，并通过 PyTorch API 重建反向索引。官方构造器把完整块同时计入普通块；其压缩索引表也不符合本配置所需的填充布局。适配保留官方可见性谓词和编译 FlexAttention，不使用 dense attention 作为性能替代。只在块级元数据上展开布尔表。

适配仅接受 SHA256 为 `e131def6596be5fe6888f1fca4728c8a220e3a3745c6dc9ca72d6b56ad899b10` 的官方 `summary_attn/interface.py`。源码文件没有被就地修改；运行时适配与原始算子分别记录哈希。

调用方另为 config 补充等价的 `rope_parameters`，并给 `generate()` 显式传入发布模块的 `Qwen3RingBufferCache`，避免 HF 自动创建不兼容的 `DynamicCache`。不修改权重或缓存类。

A100 使用官方 FlexAttention 路径。隔离环境不安装 flash-attn-cute；固定 CuTe wheel 内核仅声明支持计算能力 major 9、10、11，不能据此声称 sm120 通过。其他 GPU 必须独立测量。

## 语义与精度契约

summary query 可见同块、因果的文本及自身；文本 query 可见距离不大于 W 的局部文本，以及块距离严格大于 W 的 summary。长窗口层保留 summary 专用掩码。发布模型 decode 显式拼接本块文本与 summary 自身 KV。

每个工作进程先运行语义探针：零 Q/K、summary V=9 时，summary 输出应为 1，文本输出应为 0；另以 144／1153 个内部位置、GQA、远端 summary 和非整块尾部，对照独立 FP32 显式掩码。小规模 dense oracle 只用于正确性，未进入性能路径。解析探针 atol=0.01，随机 BF16 算子探针 atol=0.03125。

`inputs.json` 冻结真实 token IDs，覆盖 1/7/8/9/15/16/17、1023/1024/1025/1031/1032/1033、4K/16K/32K/64K/130944、中英文固定生成及三个检索案例。所有精度案例有 24 个 teacher tokens，保留 prompt 最后位置与随后 24 个位置的全词表 logits。只投影这些文本位置，避免导出全长 logits。每例 teacher-forced decode 重复三次，并与同序列整段 prefill 比较；固定 128 token 贪心生成重复两次，保存 IDs、文本和检索答案出现情况。检索结果是模型能力观察项，不预设模型一定答对。

FP32 校准使用相同 checkpoint、输入和模型代码，在 17／1025／4096 三例以短序列显式掩码 oracle 替代 prefill 算子，验证 prefill/decode 数值等价，再分别测量 BF16 decode、prefill 对 FP32 的偏差。该参考路径限制内部 query 长度不超过 4700，拒绝 BF16 或性能模式；它只用于精度校准。A100 上短形状 FP32 FlexAttention 编译曾产生约 8MB PTX、每个编译进程约 16GB 内存。改用显式参考避免编译瓶颈，BF16 基线和性能路径保持编译 FlexAttention。`fp32_calibration_backend` 与兼容源码哈希一起写入 lock。

`assess.py` 的预声明策略为：

- FP32 prefill/decode 最大绝对误差不超过 0.005；BF16 重复 logits 必须完全一致，重复生成 IDs 必须一致。
- BF16 logits 最大误差、RMSE、logprobs 最大误差分别以独立 FP32 校准噪声的 2 倍冻结；下限为 0.5／0.125／0.5，上限为 24／4／12。超过预声明上限则阻塞，不继续放宽。
- prefill/decode 的 top1 不同只允许参考 top1 margin 不超过 0.5；记录全部不一致位置中的最大 margin。该限制对应 logit 大小为 32 时的两个 BF16 ULP，防止只用平均误差掩盖高置信度排序反转。

`tests/models/utils.py::check_logprobs_close` 检查 top-logprob 集合及候选 token，不能直接给出 KSA 全词表 logits 数值阈值；因此这里保留 top1/margin 约束，同时单独进行数值校准。

上述范围是该模型跨 prefill/decode 的 BF16 数值包络，不代表 FP32 精度，也不允许省略算子语义与重复运行门禁。阈值由独立 FP32 校准噪声确定，不按待验收 BF16 路径差异的最大值调整。校准记录、张量哈希、策略、评估器源码哈希与 baseline ID 共同生成 `tolerance_id`。

## 服务器运行

在包含本轮代码的干净 checkout 中执行；SHA 必须为完整提交 SHA，输出放在仓库外：

```bash
SHA=$(git rev-parse HEAD)
MODEL=/absolute/path/to/KSA-4B-base
RESULT=/absolute/path/to/results/T00/$(date -u +%Y%m%dT%H%M%SZ)
bash benchmarks/ksa/run_server.sh "$SHA" "$MODEL" "$RESULT"
```

启动器以 uv 创建 `.venv-ksa-hf`，按哈希锁安装依赖，保存 `.packages.txt` 和 `.launcher.log`，不安装或导入 vLLM。它调用 `run_matrix.py`，为每个校准、性能或精度案例创建独立 CUDA 进程。CUDA 非法访存不会污染下一个案例；缺失的性能重复补记为 error，不能补为成功。所有子进程退出码和日志保留。

`run_matrix.py` 最后执行门禁，失败返回非零退出码。启动器在退出时收集小结果包 `.tar.gz`，全量 `.pt` 精度张量留服务器。单例排查可给 `baseline.py` 传入 `--model`、`--output`、`--mode correctness --case length-1025`，但单例不能替代完整矩阵验收。

## 性能与结果文件

每档长度预热一次（repetition=-1，不计统计），再正式重复五次，固定输出 128 token，无 EOS 提前停止。130944 输入保留 128 token 生成空间。性能计时不导出全词表 logits。

TTFT 从 GPU 上的输入开始，包括 prefill 和首次 argmax；TPOT 包括官方 prepare、forward、argmax 和同步。块边界按被消费文本 token 的绝对位置判断，逐步延迟保存在 timing JSON。吞吐为 128 /（TTFT + 127 个 decode 间隔）。加载单独计时；冷启动包含编译，不能独立分离编译时间时 `compile_capture_ms` 留空。

峰值显存为每次运行的 PyTorch peak allocated。HF 无分页 KV 指标，对应列留空。仅 batch=1；并发 4/8、图、分页、chunked prefill、前一 ticket 对比均 N/A。

结果根目录包含环境、锁、输入、正确性、性能 CSV、所有工作进程语义探针、校准、进程退出状态与总结；每例子目录包含原始输出和日志。baseline ID 绑定权重/tokenizer/remote code、全部依赖、原始算子源码、兼容适配、执行器及评估器源码、GPU、CUDA、输入与生成配置。

仅 130944 的明确 OOM 可作为容量限制通过门禁，但必须列入 `unavailable_for_comparison`，不可用于后续对比；非法访存等 error 一律阻塞。4K/16K/32K/64K 必须有预热及五次成功测量。所有精度案例均需明确结果，容差未冻结或校准失败不得将 ticket 标为 DONE。
