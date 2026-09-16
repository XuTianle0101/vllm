# T01 A100 prefill 验收

状态：DONE（实现与服务器验收）。GitHub 推送认证失败，本地提交与源码 bundle 已保留，详见 [推送日志](push.log)。专用启动器退出码 0。实现与实验 SHA：`a7a696efd57283d56098e3b91cb73aa9c47c3901`。

- baseline ID：`T00-305bf8f9dc429f1af60e`。
- tolerance ID：`T00-tol-fbd0eb931ac1c7153403`，阈值未修改。
- 环境：A100-SXM4-80GB，vLLM 0.26.0，PyTorch 2.11.0/cu130，Transformers 5.14.1；独立 HF 参考环境维持 T00 的 44 项锁定依赖、PyTorch 2.7.1/cu128、Transformers 4.57.1。
- [环境与依赖](environment.json)、[冻结输入](inputs.json)、[正确性](correctness.json)、[计时](timing.json)、[执行日志](run.log)、[HF 参考日志](hf-reference.log)。原始大张量目录与 SHA256 见 [归档清单](artifacts.json)。

## 正确性

17/17 BF16 案例通过，涵盖全部 13 个短块/窗口边界、4096、中英文和 4096 检索输入。每例分别执行原始 prompt 和带 teacher continuation 的完整 prefill；4096 案例受显式 mask 上限约束，只比较 prompt 最后文本位置。全部重复 logits 最大差为 0。

跨路径最大 logits 绝对误差 3.25、最大 logprobs 误差 1.124974；单例最低 top1 一致率 0.88。全部 top1 分歧的最大参考 margin 为 0.25，小于冻结的 0.5 门限。不宣称 BF16 逐值或 top1 完全相同。

长度 17/1025 的 FP32 最后文本位置 logits 最大误差分别为 0.000022889 和 0.000150681，均低于 T00 的 0.005。保留全部 36 层、两个长度、text/summary 两种角色，共 144 组内部状态诊断，包括循环层位置、窗口、内部行映射、最大绝对/相对误差和首个超过 0.005 的行。

T00 的 FP32 0.005 门限针对 logits；未归一化隐状态只作逐层定位并检查有限值。其绝对误差最大 0.73828125，按参考峰值归一化后最大约 0.0001043；部分 summary 状态幅度超过数万。`feature_argmax_agreement` 是隐藏特征维度的诊断，不是 token top1 一致率。相对误差分母下限为 1e-6。

summary query 可见同块因果文本及自身 KV，遵循 T00 冻结语义；文本 query 不读取同块 summary，未来 token 不可见。CPU 分析探针和完整小模型前缀扰动检查均通过。

## Prefill 性能

每档一次预热、五次正式重复；TTFT 包括完整 prefill、最后文本位置词表投影与 argmax，不含模型加载。吞吐为输入 tokens/s。显存为 PyTorch allocated 峰值，包含已加载模型和当前进程存活张量，不等同于 nvidia-smi。

| 文本 tokens | 平均 TTFT ms | 标准差 ms | 输入 tokens/s | 峰值 GiB |
| --- | --- | --- | --- | --- |
| 1024 | 151.87 | 0.12 | 6742.51 | 8.12 |
| 4096 | 1592.05 | 0.17 | 2572.78 | 13.93 |

4K TTFT 约为 T00 官方 HF 的 318.95 ms 的 4.99 倍，当前 Python 显式掩码路径存在明显性能回退，不宣称加速。两侧 PyTorch/CUDA 版本不同，性能差异不能全部归因于 attention 算子。

16K 超过显式上限 4096，原始 ValueError 与 traceback 保存在 timing.json，未伪造该档计时。仅支持单请求、完整无缓存 prefill；decode、分页 KV、chunked prefill、批处理和 serving runner 均不支持。

## 检查与修复记录

禁用 GPU 的 37 项 CPU 测试、7 项 T00 harness 回归检查、代码完整 pre-commit 钩子通过。CPU 测试包括真实小模型初始化和 AutoWeightsLoader 权重缺失/多余/重复错误。

两轮失败均保留：

- [首轮](../a100-20260916T062935Z/failure.json)：缺少官方预测词表裁剪，151936 与 151937 维不匹配；已修复并加入测试。
- [第二轮](../a100-20260916T063119Z/failure.json)：启动器错误地把 logits 门限用于未归一化隐状态。输出门限实际通过；已纠正验收作用对象，完整保留原始层误差，没有修改 T00 冻结阈值。

复现命令见 [T01 运行说明](../../../t01-prefill.md)。
