# 实验结果文件约定

本文件定义各 ticket 的结果契约；T00 已实现的输出与限制见 [运行说明](../t00-baseline.md)。示例字段中的 null 表示未采集，不能解释为零误差或通过。

## 目录

每轮保存为 `Txx/<run-id>/`，run-id 使用时间戳加提交短 SHA，包含 environment.json、correctness.json、performance.csv、run.log 和 summary.md。可增加 profiler/、outputs/，大文件不默认提交 Git。

## environment.json

```json
{
  "schema_version": 1,
  "ticket": "Txx",
  "run_id": null,
  "timestamp": null,
  "git_branch": null,
  "git_sha": null,
  "git_dirty": null,
  "baseline_id": null,
  "gpu_name": null,
  "gpu_memory_bytes": null,
  "driver_version": null,
  "os": null,
  "backends": {
    "transformers": {"python": null, "torch": null, "cuda_runtime": null, "transformers": null, "summary_kernel": null},
    "vllm": {"python": null, "torch": null, "cuda_runtime": null, "vllm": null, "triton": null, "summary_kernel": null}
  },
  "model_hashes": {},
  "tokenizer_hashes": {},
  "reference_code_revision": null,
  "dtype": "bfloat16",
  "kv_dtype": "bfloat16",
  "tensor_parallel_size": 1,
  "commands": []
}
```

采集配置、模型与 tokenizer 文件哈希并记录文件名；版本未知填 null 并说明原因。系统 CUDA 工具链与 PyTorch CUDA runtime 应分开记录，不混为同一个版本。

## correctness.json

顶层字段：schema_version、ticket、git_sha、baseline_id、tolerance_id、thresholds、cases。

每个 case 至少包含：case_id、input_tokens、output_tokens、backend_pair、mode、status、logits_max_abs_error、logits_rmse、logprobs_max_abs_error、top1_agreement、first_mismatch、reason。

- status 使用 pass、fail、not_run、not_applicable、error。
- first_mismatch 可记录文本位置、生成步、层、期望/实际 token 及 top-logit margin。
- thresholds 引用 T00 冻结值；本模板不虚构容差。
- 教师强制与自由生成分别标记；不能把后续输入已分歧的 logits 当成同输入误差。
- 可补充整段/分块、单请求/批次、eager/graph 对照标签。

## performance.csv

每行代表一个后端、场景的一次重复运行；汇总不替代原始行。建议固定列：

```csv
ticket,git_sha,baseline_id,case_id,backend,mode,input_tokens,output_tokens,concurrency,repetition,status,ttft_ms,tpot_steady_ms,tpot_boundary_ms,output_tokens_per_s,peak_memory_bytes,active_kv_bytes,allocated_kv_pool_bytes,load_ms,compile_capture_ms,reason
```

时间单位为 ms，内存单位为 bytes。不可用值留空并写 reason，OOM 使用 status=oom。没有分离测量块边界延迟时留空，不用平均 TPOT 代填。

默认性能计数排除内部 summary 输出，但计时包括其真实计算成本。明确 TTFT 起止点、输出长度、是否包括排队与网络；服务与离线指标分开。HF 缺少内部 KV 页指标时标记不可用，不把进程显存当作 active_kv_bytes。

## run.log 与 summary.md

run.log 保存命令、退出码及完整异常栈。summary.md 使用 [反馈模板](experiment-feedback.md)，包含未执行项目和限制。服务器结果反馈后由开发方补充判定，不由空模板自动判定通过。
