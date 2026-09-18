# 03: 将保留验证迁移到标准执行路径

**What to build:** 验证者能够以标准 vLLM 执行路径完成 KSA 的 HF 正确性对照、服务行为验证和代表性性能测量，验证结果能够与整理前基线比较。

**Blocked by:** 02 — 抽取最小 HF reference 与复现输入。

**Status:** completed (2026-09-18; standard validation migrated, known HF failures retained)

- [x] 将保留验证的执行侧迁移至标准 vLLM 路径，由 checkpoint 配置识别 KSA，使用最小 HF reference 对照；旧入口与 runner 暂不删除，直至其调用迁移完成。
- [x] 使用原输入和原容差覆盖 summary/窗口边界、分块 prefill、多步 decode，以及已知失败用例；不将历史报告当作新验证结果。
- [x] 标准执行路径覆盖 eager/graph、批处理、取消及缓存复用，并保留普通 Qwen3 功能回归。
- [x] 保留可验证标准采样、seed、多样本、惩罚、stop、logprobs 和 SSE 行为的有效检查，优先复用已有测试，不堆叠仅验证接线的测试。
- [x] 短、中、长上下文性能测量通过当前标准执行路径完成，沿用 01 固定的输入、环境与口径，不依赖旧阶段比较或普通 Qwen3 性能 baseline。
- [x] 保留 `ksa_cudagraph` 配置键，验证其控制 KSA 内部局部 decode graph 且可与 `enforce_eager=True` 共存；不迁移到标准 compilation/cudagraph 配置。
- [x] 记录验证命令、结果和迁移后的剩余旧接口调用；新增回退需解决，环境导致未执行的项目单列。

## 交付与实测结论

- 当前入口与全部复跑命令见 [标准验证](../docs/ksa/standard-validation.md)。
- 标准 teacher decode 46/46、分块 prefill 17/17、eager/graph 对照 46/46；
  HTTP eager/graph、页压力/取消/复用、普通 Qwen3 功能均通过。
- 完整 HF 54/69 通过、15/69 既有失败；全部 69 条的四项误差与票 01
  完全一致，原输入和门限未变，不将精度门禁重标为通过。
- CPU 契约 15 passed，KSA pytest 132 passed；相关 pre-commit 通过。
- 六组性能取得原协议下的通过记录。首轮 4096 graph TTFT regression、
  16384 eager retest 及后续复测均保留；最终每组采用完整五次记录，
  未跨轮次挑样本或放宽判据，初次波动根因未作修复声明。
- 剩余旧接口调用清单已记录；本票不删除旧实现、不修改生产代码、不提交或改写历史。
