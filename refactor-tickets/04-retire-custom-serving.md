# 04: 移除专用服务入口与独立 runner

**What to build:** 使用者通过标准 `vllm serve` 运行 KSA，调度和采样统一由 vLLM 管理，既有标准服务行为保持兼容。

**Blocked by:** 03 — 将保留验证迁移到标准执行路径。

**Status:** ready-for-agent

- [ ] 确认保留的验证调用已迁移后，删除专用 HTTP 入口、`KSABatchedRunner` 独立 runner 及其他旧的专用生成入口，清理对应注册、导入和失去用途的调用。
- [ ] 保留生产 `KSAGPUModelRunner` 和必要的 scheduler 页映射；外层标准服务启动便利封装保持可用。
- [ ] 标准服务仍通过 checkpoint 配置识别 KSA，调度、采样和页生命周期交由标准路径管理；reference implementation 仅留在验证侧。
- [ ] 通过标准服务接口验证采样、seed、多样本、惩罚、stop、logprobs、SSE 及批处理行为；运行与变更相关的静态检查、功能测试和模型评估。
- [ ] 验证普通 Qwen3 功能未受影响，且不存在保留代码依赖已删除入口或 runner 的情况。
- [ ] 使用基线和原容差判断回退，明确记录既有失败及未执行项；不承担已知精度问题的根因修复。
