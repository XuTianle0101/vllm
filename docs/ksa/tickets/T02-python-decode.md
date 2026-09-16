# T02：Python 缓存 decode

- 状态：DONE（A100 单请求 Python 缓存 decode 验收通过）
- 依赖：T01 DONE
- 目标：通过 vLLM 完成单请求精度正确的生成。

## 实现范围

接入缓存 decode、块末文本与 summary 的同次 forward、正确采样行、EOS、生成长度和输出过滤。运行器按绝对文本位置插入 summary，缓存文本与内部行计数不可混淆。此阶段允许完整历史 KV，限制批量和高级功能并明确报错。

只修改 Python，不引入自定义高性能算子。首轮端到端比较以精度为门禁，不要求此原型已经快于 HF。

## 服务器实验

教师强制逐步比较 logits/logprobs；固定提示比较生成 token。覆盖 prompt 恰好在块末、部分块起始、多次跨块、窗口边界、EOS 和 max_tokens。运行同模型、同精度、同长度的 HF/vLLM 单请求性能对比。

## 门禁与反馈

精度满足 T00 阈值；内部 summary 不泄漏；停止、输出长度及 token 计数正确。输出第一份端到端精度与性能报告，保留性能差距，不掩盖原型局限。

## 实现与验收范围

- `KSAForCausalLM.forward(..., cache=KSACache())` 保留全历史每层 K/V；仅允许首次完整 prefill 和后续逐文本 token decode。按绝对文本位置插入 summary，缓存更新在完整 forward 成功后提交。
- `KSAPythonRunner.generate()` 提供单请求贪心生成，支持 `max_tokens`、多个 EOS ID 与 `ignore_eos`。输出 IDs 和 token 计数只包含文本词表 token，EOS 计入 completion，显示文本使用 `skip_special_tokens=True`。
- prefill 上限 4096，prompt 加生成预算不超过 8192 或模型位置上限。普通 serving、批量、随机采样、chunked prefill、prefix cache、图、推测解码不支持。
- `benchmarks/ksa/decode.py` 核验冻结 T00 身份、模型、输入、HF 依赖与参考代码，覆盖全部 17 个支持长度的教师强制案例，每例重复三次；5 组固定提示各生成 128 token 并重复。
- BF16 自由生成的精确 token 一致性单独报告；每条 vLLM 生成轨迹由 HF 在同前缀下重新计算 logits，执行 T00 全词表误差及 top1 margin 门禁，不直接比较分叉后的不同前缀 logits。
- EOS 注入为第一次实际采样的 ID，覆盖立即停止；另验证 0、1、128 token 预算。1K/4K 同模型、同精度、同生成长度，HF 与 vLLM 各预热一次、正式五次。

## 交付记录

- 分支：`releases/v0.26.0-ksa`；实测实现 SHA：`4f2845da8a3565d8bbedec082fdbf2fc40319099`。
- 基线：`T00-305bf8f9dc429f1af60e`；容差：`T00-tol-fbd0eb931ac1c7153403`，未调整。
- 本地检查：42 项 CPU 模型契约测试、7 项 T00 harness 回归、所有适用 pre-commit 钩子通过。
- 服务器：A100-SXM4-80GB 实际执行 `benchmarks/ksa/decode.py`，退出码 0；[完整复现命令](../t02-decode.md)。
- 结果：[验收报告](../results/T02/a100-20260916T071809Z/README.md)。17/17 教师强制案例通过，重复 logits 差为 0；5/5 生成轨迹同前缀精度、重复、停止、长度和输出过滤检查通过。
- 生成：2/5 用例全部 IDs 与 HF 精确一致。其余首个分叉 margin 分别为 0/0/0.125，全部轨迹满足冻结 margin 门限；报告保留所有分叉及逐位置数据。
- 性能：1K/4K 吞吐 23.824/18.747 token/s，HF 为 32.004/30.453；原型吞吐低约 25.6%/38.4%，不宣称加速。
- 原始张量与哈希：`/workspace/volume/h20-data/xutianle/KSA/results/T02/a100-run1/`；报告内含文件清单。
- 推送：GitHub 凭据失效，匿名写入被拒绝；实现与验收结果保留本地提交，`results/T02/source.bundle` 可转移。恢复认证后执行 `git push origin HEAD:releases/v0.26.0-ksa`。
- 结论：T02 DONE，限 BF16 A100、单请求专用 Python runner、4096 prefill 与 8192 总文本预算；不包含普通 serving 和高级功能。未启动 T03。
