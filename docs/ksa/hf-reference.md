# HF reference 与复现输入

验证入口为 `benchmarks/ksa/hf_reference.py`，仅用于验证，不注册生产模型、
runner 或服务选项。模型与 HF 环境沿用 [整理前基线](refactor-baseline.md)。

## 契约与冻结资产

- `make_inputs(tokenizer)` 保留原输入构造：23 个输入，每例 24 个 teacher
  tokens；覆盖 summary 7/8/9、窗口 1023/1024/1025、1031/1032/1033、
  长上下文与检索。实际验证读取冻结 IDs，不依赖重新分词。
- `teacher(torch, model, ids, continuation)` 在官方 HF cache 上先执行 prompt，
  再逐 token decode；返回 prompt 最后一行及各 continuation 的 CPU FP32
  logits。模型计算仍为 BF16，保留原兼容补丁、RoPE 和 TF32 设置。
- `compare(...)` 原样保留全词表 logits、logprobs、top-1 margin、finite、
  逐位置结果与 first anomaly 判定。最终生成入口还保留 chosen-token margin
  门禁，不以生成 token 相同代替数值门禁。
- `load_reference(directory)` 核验 lock、校准、输入、容差和冻结生成序列；
  `validate(args)` 另核验模型全部文件、compat 源码及预期 Git HEAD；
  `load_model(...)` 核验隔离 HF 环境的完整 packages 清单。

小型资产位于 `benchmarks/ksa/fixtures/hf_reference/`：

| 文件 | 内容 |
| --- | --- |
| `inputs.json.gz` | 无损压缩的全部原始 IDs，解压后 canonical digest 与旧 lock 一致 |
| `baseline-lock.json` | 原模型文件指纹、环境及输入身份；保留原文 |
| `calibration.json` | 原校准与容差身份；保留原文，不重新校准 |
| `correctness.json` | 精确阈值、全部 23 例冻结 HF greedy tokens 与原身份 |
| `known-failures.json` | 票 01 的 15 条失败：case/mode/repeat、128 个生成 tokens 和四项误差 |

`correctness.json` 的 `status=pass` 是原 T00 HF 自对照身份，不是当前 vLLM
通过声明。15 条失败涉及长度 1023、1024、1031、1032、1033，分别在 eager、
graph repeat=1、graph repeat=2 出现；仍为已知问题。

精确门限：logits max `20.33521270751953`、RMSE `3.7059452533721924`、
logprobs max `8.020130157470703`、mismatch margin `0.5`、repeat error `0.0`。
传入旧目录或新导出目录时均不得放宽。

## 独立复跑

从 vLLM 仓库根目录执行。沿用 uv 管理的 `.venv` 和 `.venv-ksa-hf`；
HF 依赖来自 `benchmarks/ksa/requirements.lock`，不向其安装 vLLM 依赖。
所有输出目录必须为新目录，GPU 命令顺序执行。

生成标准 V1 分块 prefill 所需的 17 例 HF teacher logits：

```bash
export PYTHONPATH=. OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
uv run --no-project .venv-ksa-hf/bin/python benchmarks/ksa/hf_reference.py \
  --model ../models/KSA-4B-base --expected-sha "$(git rev-parse HEAD)" \
  --output ../results/hf-reference-new \
  --cases length-1 length-7 length-8 length-9 length-15 length-16 length-17 \
  length-1023 length-1024 length-1025 length-1031 length-1032 length-1033 \
  length-4096 english chinese retrieval-4096
uv run --no-project .venv/bin/python benchmarks/ksa/v1.py \
  --model ../models/KSA-4B-base --baseline ../results/hf-reference-new \
  --row-budget 257 --output ../results/v1-reference-new
```

省略 `--cases` 可导出全部 23 例。导出格式只包含 V1 实际读取的
`decode_logits` 和 `text_positions`，不生成旧阶段性能与 FP32 层诊断资产。
`v1.py --all-cases` 需提前导出全部所选用例，缺失张量会报错。

完整 eager/graph 轨迹验证不需要旧 baseline 目录或预存 HF 大张量：

```bash
uv run --no-project .venv/bin/python benchmarks/ksa/cudagraph_v1.py \
  --model ../models/KSA-4B-base \
  --baseline benchmarks/ksa/fixtures/hf_reference \
  --expected-sha "$(git rev-parse HEAD)" --all-cases \
  --output ../results/graphs-reference-new
uv run --no-project .venv-ksa-hf/bin/python benchmarks/ksa/final_generation.py \
  --model ../models/KSA-4B-base --expected-sha "$(git rev-parse HEAD)" \
  --graphs ../results/graphs-reference-new --output ../results/hf-generation-new
```

最终 HF 门禁保留非零失败退出码；现有 15 个失败不能计为通过。
`known-failures.json` 也允许直接将保存的 prompt 与 `generated_ids[:-1]`
传给 `teacher`，独立重建这 15 条实际前缀的 HF logits。

## 迁移边界

以下记录票 02 完成时的边界。票 03 随后完成最终 decode/performance 的
标准路径迁移，当前入口与命令见 [标准验证](standard-validation.md)。

标准 V1、graph、最终 HF generation、压力验证、普通 Qwen3 功能验证以及
最终 decode/performance 脚本的共享读取与比较已改用新模块。
旧 `baseline.py`、`prefill.py`、`decode.py` 保持原样，供未迁移的阶段调用方
及抽取前后对比使用。最终 decode 的独立页池、最终 performance 的旧 HF
性能子进程仍待后续票迁移；本票没有将它们纳入 reference，也没有删除它们。

新 reference 不再要求旧 `baseline.py` 的源码 hash 匹配。原 lock 内该字段
仅用于保留历史身份；compat hash 仍强制匹配。预期 HEAD 必须匹配，允许对
未提交的重构进行验证，并在 `reference-source.json` 记录 dirty、源码 diff
指纹与新 reference hash；这与旧阶段拒绝 dirty 的入口明确区分。
生产代码、旧阶段入口及其原始身份校验未修改。

## 票 02 验证记录（2026-09-18）

本次修改限定在验证侧，基于 `81d2c510ef` 的未提交工作区；生产代码、
旧 baseline/decode/prefill/compat 与 layout/visibility 测试源码均无修改。
硬件与两个隔离环境沿用票 01，模型文件和 HF packages 核验通过。

验证前确定最低层级：输入和门禁的错误行为用现有 CPU 契约测试；teacher
缓存次序用小模型替身；真实 BF16 行为用同进程旧/新 `teacher` 数值对照；
分块和图执行用现有标准 V1 接口。未为本票增加新的生产测试接口。

已执行：

- 现有 `test_baseline.py` 扩展后 **13 passed**；覆盖输入/顺序、失败集合、
  token 篡改、放宽门限拒绝、finite 与边界判定、teacher 缓存顺序。
- `tests/model_executor/test_ksa_prefill.py`：**132 passed**（63.05 s），
  包括原 layout、visibility 与 GPU 测试；15 条 warnings 保留于日志。
- 9 个提取函数的 AST 与原函数一致；全部 23 例输入内容、顺序、原容差一致。
  8 个迁移模块在禁止导入旧 baseline/decode/prefill 时均可加载。
- 17 例短输入/检索、每例 25 行 HF teacher logits：旧/新函数逐元素相等，
  最大差 **0**。与历史 T00 保存张量的比较全部通过原门限；最大差 1.3125，
  因此不声称历史张量逐位相等。初次额外的历史 bitwise 断言在 length-8
  失败，记录保留；旧/新函数的 bitwise 断言始终保留并通过，原门限未改变。
- 新 CLI 从随代码 fixtures 独立导出 17 例，再经标准 V1 row budget=257
  分块 prefill：**17/17 passed**，max logits=2.75、RMSE=0.7344133257865906、
  max logprobs=1.2812080383300781、margin=0.25，最大内部行数 257。
  四项最大值与票 01 一致。
- 标准 V1 eager/graph 默认四例、三轮重排、每例 32 输出：**8/8** graph/eager
  数值对照通过；不是完整长上下文矩阵。
- 页压力：通过，1 次抢占、154 steps、每请求 80 输出，取消与复用断言通过，
  全部 329 可用页归还。
- 普通 Qwen3 确定性小权重功能回归：两例同前缀门禁通过；第二例生成 tokens
  仍不同，margin=0.001953125 在原 0.02 门限内，不声称逐 token 相等。

15 条历史失败轨迹全部完成复核：在保存的实际前缀上重新执行旧/新 HF
teacher，logits 逐元素相等；旧/新 `compare` 的完整结果相等，四项门禁误差
及失败状态与票 01 逐 case/mode/repeat **完全一致**。最大 logits 误差仍为
35.75，最大 logprobs 误差仍为 14.351476669311523；这 15 条仍判失败。
这是 reference 抽取等价性通过，不是模型精度门禁通过。

提取身份见 [extraction.json](results/refactor-02/extraction.json)，
实测汇总见 [validation.json](results/refactor-02/validation.json)。
所有修改文件 pre-commit 与 `git diff --check` 通过；未提交、未推送、未重写历史。

CPU 和语义测试命令：

```bash
uv run --no-project .venv/bin/python -m unittest discover \
  -s benchmarks/ksa -p test_baseline.py -v
uv run --no-project .venv/bin/python -m pytest \
  tests/model_executor/test_ksa_prefill.py -q
```

原始命令、日志与结果保留在工作区 `../results/refactor-02/`，
`standard.sh` 记录独立导出及标准验证，`standard-local.sh` 记录本机 I/O 重跑；
`equivalence.py` 对比旧/新 teacher 与 compare，原始结果为 `equivalence.json`。
共享存储上的首次 V1 核验和 HF 轨迹回读因 I/O 等待主动中止，不计通过；
随后在 `/tmp/ksa-refactor-02-standard` 及 `/tmp/ksa-refactor-02-graphs` 执行，
标准验证的 JSON 与日志已复制回 `../results/refactor-02/standard/`。
三份历史 graph 张量副本的 SHA-256 与票 01 清单完全一致，见原始
`tensor-copy.json`。这些运行不用于性能测量。

本票未重跑全量 69 条 HF 轨迹、HTTP、性能矩阵或旧独立 runner；这些项目
不计通过。票 01 的 HTTP seed 现象仍保留为已知观察，未作修复声明。
