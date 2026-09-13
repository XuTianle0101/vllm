# T00 基线交付与服务器运行

状态：AWAITING_SERVER。脚本交付不等于官方精度、性能及容差已验收。

## 已核验的官方来源

- [KSA 源码](https://github.com/Kuaishou-OneRec/KSA/tree/9998565d0aa2907cc99e3ab0717b8bf246e77876)：固定 `9998565d0aa2907cc99e3ab0717b8bf246e77876`。
- [HF 发布](https://huggingface.co/OpenOneRec/KSA-4B-base/tree/6b60859be46422fc5949a0e69d2a338c4a618c90)：发布 revision `6b60859be46422fc5949a0e69d2a338c4a618c90`。实际使用服务器模型目录的逐文件 SHA256；此 revision 不替代本地文件哈希核验。
- 官方 Dockerfile 固定 PyTorch 2.6.0/cu126，requirements 固定 Transformers 4.57.1；checkpoint 的 `transformers_version=4.51.0` 不是推荐安装版本。
- 官方 `summary_attn-0.3.0` wheel SHA256：`3a8e7893b17b4bcc2daf13d75ed30d75631a25da18b63a515279dcd4f18cc536`。
- 官方 `flash_attn_cute-0.1.0` wheel SHA256：`1a687930435ba56daa067819069c56c2b3370b7c2e75ca78f63d211ace414830`。

## 5090 兼容性与语义审计

已检查上述 wheel 内源码。`summary_attn/interface.py::_check_cute_available` 在 CuTe 可导入且计算能力 major >=9 时选择 CuTe。然而随附 `flash_attn/cute/interface.py` 明确断言 major 只能是 9、10、11。因此原始 CuTe 组合在 sm120 上存在静态可确认的兼容性阻塞；未声称服务器实测通过。

本交付采用单独命名的 `official-flex-cu128` 兼容配置：PyTorch 2.7.1/cu128、官方 summary_attn 0.3.0，隔离环境不安装 flash-attn-cute，由官方函数自行选择其自带的编译 FlexAttention 路径。没有 monkeypatch、模型修改或 dense 性能替代。此配置与原始 Docker 环境不同，是否可作为最终官方性能对照仍须由 5090 实验验收。若其编译失败，保留异常，不自动换算子。

另外，Transformers 4.57.1 的 `GenerationMixin._prepare_cache_for_generation` 默认创建 `DynamicCache`，发布模型的 attention 则读取环形缓存的 `_reorganized` 属性。调用方因此显式把发布模块原有的 `Qwen3RingBufferCache` 传给 `generate()`，不修改类或覆盖方法；该调用差异写入 baseline lock。性能和 teacher-forced 路径直接调用官方 forward，由模型自行创建同一个环形缓存。

本地 `modeling_qwen3.py` 与 `summary_context.py` 已下载指定 HF revision 的文件并确认 SHA256 一致。完整本地模型/tokenizer 哈希已存为 `model-manifest.json`；服务器文件必须逐项匹配，否则留档并停止，不能无记录更换权重。内存统计使用 PyTorch 2.7.1 的设备模块接口（该版本尚无 accelerator 内存 API），同步使用 accelerator API。

prefill 的 `summary_attn_mask_func` 和 `summary_attn_contiguous_mask_func` 对 summary query 使用“同块且因果”的谓词，包含 summary 自己。FlexAttention 对应谓词一致。checkpoint decode 显式拼接本块文本 KV 与 summary 自身 KV。因此当前发布代码在自身可见性上没有静态歧义。文本 query 的远端 summary 条件是块距离严格大于 W，局部文本为距离不大于 W。长窗口层仍保留 summary 专用掩码。

服务器 `semantics.json` 使用零 Q/K、summary V=9 的解析案例，验证 summary 输出为 1、文本输出为 0。这里只用小张量的已知解析值检查算子，不是模型容差或性能基线。端到端整段 prefill 与逐 token decode 对照仍需实测。

## 服务器命令

在 Linux RTX 5090 服务器已有的此 fork checkout 中运行，模型目录需含完整权重、tokenizer 及 remote code。将 `SHA` 换成本轮交付的完整提交 SHA，将 `MODEL` 换成模型绝对路径。输出应放仓库外。

```bash
git fetch origin releases/v0.26.0-ksa
git switch releases/v0.26.0-ksa
git pull --ff-only origin releases/v0.26.0-ksa
SHA=<本轮交付的完整提交SHA>
MODEL=/absolute/path/to/KSA-4B-base
RESULT=/absolute/path/to/results/T00/$(date -u +%Y%m%dT%H%M%SZ)-${SHA:0:12}
bash benchmarks/ksa/run_server.sh "$SHA" "$MODEL" "$RESULT"
```

启动器验证 SHA 和干净工作区，以 uv 创建独立 `.venv-ksa-hf`，按带哈希 lock 安装。不会安装或导入 vLLM。`RESULT.launcher.log` 包含环境安装、命令错误与退出码，即使安装阶段失败也保留。成功后生成小结果包，`.pt` 精度张量留服务器。运行脚本本身要求 uv 已安装。

## 数据与测量契约

- `baseline-lock.json`、`environment.json`：冻结模型、tokenizer、remote code、算子源码哈希、依赖全版本、GPU、运行时、脚本哈希、输入哈希和生成配置；baseline ID 为锁定内容的哈希。系统 nvcc 与 PyTorch CUDA 版本分别记录。
- `inputs.json`：真实 token IDs，覆盖 1/7/8/9/15/16/17、1024 与 1032 窗口/淘汰边界、4K/16K/32K/64K/130944、中英文固定生成和三个长距离检索案例。
- 精度：每例固定 24 个 teacher tokens，记录 prompt 末尾及后续 24 个位置的全词表 logits；同序列重复三次，比较逐 token 缓存 decode 与整段 prefill。仅投影所选文本位置，避免全长 logits 导出造成 OOM。每例另做两次固定长度贪心生成，记录文本、IDs、重复一致性和检索答案出现情况。
- 性能：先于模型精度运行；每个长度 1 次冷启动预热（repetition=-1，不计统计）及 5 次正式重复，128 输出 token，无 EOS 提前停止。加载单独计时。冷启动包含编译，未能单独隔离编译时间时 `compile_capture_ms` 留空。
- TTFT 从已在 GPU 上的输入开始，包含 prefill 与首次 argmax；TPOT 包含官方 prepare、forward、argmax 及同步。每步墙钟同步，无服务排队、网络、tokenization、模型加载。块边界按被消费文本 token 的绝对位置判定，保留逐步原始延迟。吞吐是 128 /（TTFT + 127 个 decode 间隔）。
- `peak_memory_bytes` 是每次运行的 PyTorch 峰值 allocated；官方内核缓存常驻开销可能保留。HF 无分页 KV 指标，对应列留空。OOM/error 每次保留，不填成功测量值。
- HF 仅 batch=1；并发 4/8、图、分页、chunked prefill 与前一已验收 ticket 比较均 N/A。

## 验收与容差冻结

`correctness.json` 的 `thresholds`、`tolerance_id` 初始为 null。已采集但未经阈值判断的 case 使用 `status=not_run` 并附解释；不能把脚本结束或性能 pass 行视为精度验收。现有 `tests/models/utils.py::check_logprobs_close` 主要检查 top logprob 集合，不能直接提供 KSA 全词表 logits 数值阈值。

返回小结果包、launcher 日志、packages 文件。先排查算子语义探针、非有限数、重复生成分歧和 prefill/decode 误差，再结合重复误差、BF16 校准与 top1 margin 确定并记录模型容差。容差冻结须绑定 baseline ID 和误差证据，在 T01 前完成；不以本次最大误差自动调大阈值掩盖错误。长上下文 OOM 明确标记不可比较，不能通过省略失败行验收。

本地检查只能覆盖脚本、输入与结果契约；本机 8GB GPU 不用于完整 BF16 验收。当前无服务器 baseline ID、实测性能或已冻结模型容差。
