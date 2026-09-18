# 02: 抽取最小 HF reference 与复现输入

**What to build:** 验证者可使用独立、最小的 HF reference 和小型输入复现现有正确性结果，无需依赖旧阶段实验体系；为执行侧迁移提供稳定对照。

**Blocked by:** 01 — 建立整理前验证与性能基线。

**Status:** completed (2026-09-18; reference extracted, known failures preserved)

- [x] 调查最终验证对旧 baseline 的实际函数和资产依赖，只抽取仍需要的 HF 对照、输入构造和容差语义；不按文件名直接删除依赖。
- [x] 保留原容差和必要的小型 fixtures，包含 summary/窗口边界、分块 prefill、多步 decode 及历史 15 个失败用例的复现输入。
- [x] reference 仅用于验证，不作为生产执行选项暴露，也不引入假设中的模型或策略适配层。
- [x] 先增加最小 reference，再迁移保留的对照调用；迁移期间旧实现仍可支持尚未迁移的调用方，使本票能独立验证。
- [x] 通过已有验证接口对比抽取前后的结果，证明输入、容差及 reference 行为未发生意外变化；保留必要的 layout、visibility 语义验证。
- [x] 记录相关测试与实际结果；既有失败保持可观察，不放宽容差、不删失败用例、不将未执行项计为通过。

## 交付与实测结论

- 独立验证模块：[hf_reference.py](../benchmarks/ksa/hf_reference.py)。
  随代码保留约 79 KiB fixtures，包含全部 23 个冻结输入、原始精确容差、
  模型/环境身份及 15 条历史失败的生成 token IDs。
- 保留的标准 V1/HF 对照已脱离旧 baseline/decode/prefill 导入；旧实现仍可用，
  执行侧独立 runner/页池及旧性能子进程的删除留给后续票。
- 契约测试 13 passed，KSA 测试 132 passed；新 HF 导出到标准 V1 分块
  prefill 17/17 passed，graph/eager 8/8 passed，页压力与普通 Qwen3 回归通过。
- 17 例 teacher logits 抽取前后逐元素相等；全部 15 条既有失败仍失败，
  四项误差与票 01 完全一致。未重跑完整 69 条 HF 轨迹、HTTP 或性能矩阵，
  不将这些未执行项计为通过。
- 详见 [验证契约、复跑方法与本票结果](../docs/ksa/hf-reference.md)，
  以及 [实测汇总](../docs/ksa/results/refactor-02/validation.json)。
  未修改生产代码，未提交、推送或重写历史。
