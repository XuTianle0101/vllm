# 标准执行路径验证

票 03 将保留验证统一到标准 V1 / `vllm serve`；checkpoint 的
`use_summary_attention` 负责识别 KSA，不指定模型类或独立 runner。
冻结输入、原容差和 HF 环境见 [HF reference](hf-reference.md)，
比较协议见 [票 01 基线](refactor-baseline.md)。BF16 为本次实测范围。

## 验证契约

- `v1.py`：row budget=257 的分块 prefill，观察标准 runner 执行的文本
  hidden states，对照 HF teacher 的 25 个位置；覆盖 summary 与窗口边界。
- `final_decode.py`：标准 `LLM`、scheduler、KV 页与 sampler 执行全部 23 例。
  验证侧 `FrozenTeacher` logits processor 保存原始 logits 后注入下一
  teacher token；输出列表决定步数，批次移位/槽位复用由 V1 更新。
  每例 24 个 teacher tokens 加一次最终采样，比较 prompt 最后一个位置及
  24 个 decode 位置的完整文本词表，不把 summary 占位列计入文本 logits。
  分别新建 eager/graph engine，检查 checkpoint 选择的 runner、图开关、
  实际 replay、冻结输出前缀与 scheduler 页归还。这个 processor 仅用于
  teacher 验证，不替代生成或 HTTP 的标准采样验证。
- `cudagraph_v1.py --all-cases` 与 `final_generation.py`：全部 23 例、128
  输出，eager/graph/graph 三轮重排；在实际生成前缀上进行 69 条 HF
  对照。原 15 条失败与原数值门限仍保留，失败时返回非零。
- `v1_pressure.py`：真实 scheduler 的页压力、抢占、取消、请求复用。
  `plain_qwen3.py` 保留普通 Qwen3 的确定性小权重功能回归。
- `serve_validation.sh` / `v1_serving.py`：标准 HTTP 的并发 1/4/8、SSE、
  seed、n=2、惩罚、logprobs、echo、stop、断连与页回收。
  两次固定 seed 响应原文存入结果；异常明确标为 fail，避免遗留 running。

`enforce_eager=True` 关闭 vLLM 外层 compilation/cudagraph；
`additional_config={"ksa_cudagraph": true}` 独立控制 KSA 内部局部 decode
图，两者可以共存。此票不迁移到标准 compilation/cudagraph 配置。

最低有效测试层级：processor 的原始 logits、teacher 步进和批次移位用
CPU 契约测试；标准执行、真实图 replay、页归还和 checkpoint 识别通过
GPU 集成验证；服务行为通过 HTTP 验证。性能汇总的缺测、捕获污染、
页泄漏与均值/范围判据用现有 `test_baseline.py` 验证。

## 复跑命令

从 vLLM 根目录执行；GPU 命令依次运行。每次使用全新的输出目录。
大张量可写本机磁盘，避免共享存储回读等待；性能的模型、输入、环境和
计时口径必须保持票 01 的设置。以下 OUT 可改为本机目录：

```bash
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
MODEL=../models/KSA-4B-base
OUT=/tmp/ksa-standard-new
SHA=$(git rev-parse HEAD)
mkdir "$OUT"
uv run --no-project .venv-ksa-hf/bin/python benchmarks/ksa/hf_reference.py \
  --model "$MODEL" --expected-sha "$SHA" --output "$OUT/reference"
uv run --no-project .venv/bin/python benchmarks/ksa/v1.py \
  --model "$MODEL" --baseline "$OUT/reference" --row-budget 257 \
  --output "$OUT/prefill"
uv run --no-project .venv/bin/python benchmarks/ksa/final_decode.py \
  --model "$MODEL" --baseline "$OUT/reference" --expected-sha "$SHA" \
  --output "$OUT/teacher"
uv run --no-project .venv/bin/python benchmarks/ksa/cudagraph_v1.py \
  --model "$MODEL" --baseline benchmarks/ksa/fixtures/hf_reference \
  --expected-sha "$SHA" --all-cases --output "$OUT/graphs"
uv run --no-project .venv-ksa-hf/bin/python benchmarks/ksa/final_generation.py \
  --model "$MODEL" --expected-sha "$SHA" --graphs "$OUT/graphs" \
  --output "$OUT/generation"
uv run --no-project .venv/bin/python benchmarks/ksa/v1_pressure.py \
  --model "$MODEL" --baseline benchmarks/ksa/fixtures/hf_reference \
  --output "$OUT/pressure"
uv run --no-project .venv/bin/python benchmarks/ksa/plain_qwen3.py \
  --output "$OUT/plain-qwen3"
OUT="$OUT" MODEL="$MODEL" bash benchmarks/ksa/serve_validation.sh
uv run --no-project .venv/bin/python benchmarks/ksa/final_validation.py \
  --model "$MODEL" --expected-sha "$SHA" --output "$OUT/performance"
uv run --no-project .venv/bin/python -m unittest discover \
  -s benchmarks/ksa -p test_baseline.py -v
uv run --no-project .venv/bin/python -m pytest \
  tests/model_executor/test_ksa_prefill.py -q
```

性能入口只运行 4096/16384/65536 × eager/graph，batch=1，128 输出，
一次预热、五次正式重复。worker 的计时循环保留票 01 口径，模型/输入/SHA
核验发生在计时之前；不启动旧 HF 性能子进程，也不运行普通 Qwen3 性能。
启动、profiling、图捕获及 KV 探针单列。`--worker --length N --batch 1
--mode eager|graph` 可独立复测一项，同样要求 `--expected-sha`。

默认 `--before` 为 `docs/ksa/results/refactor-01/performance.csv`，同目录
`environment.json` 用于核对 GPU UUID/驱动、依赖与模型。汇总保留五次值、
均值比和判定；延迟均值增加 >5% / 吞吐下降 >5% 且范围不重叠为
regression，仅满足一项为 retest。显存增加 >5% 为 retest。
缺项、捕获污染、页泄漏不计通过。retest/regression 都返回非零，不将
波动自动改判通过。

## 剩余旧接口调用

保留的 HF、teacher、分块、generation、pressure、HTTP、普通 Qwen3
功能及当前性能入口已不依赖独立入口/runner/page pool 或旧 baseline。
以下尚未删除，供后续票清理：

- `batching.py` 仍调用旧 `KSABatchedRunner`、`entrypoints.ksa` 与 `new_cache`。
- `compressed_kv.py`、`cudagraph.py`、`attention_kernels.py` 仍调用
  `new_cache` / `KSAPagePool`，属于旧阶段实验。
- `baseline.py`、`prefill.py`、`decode.py`、`run_matrix.py`、`assess.py`、
  `serving.py`、`summarize_cudagraph.py` 及上述旧阶段脚本仍形成旧实验依赖。
- `test_ksa_prefill.py` 的独立缓存/runner/旧 HTTP 实现测试仍调用旧接口；
  后续删除实现时应按行为迁移或删除，不把这些结果代替标准路径验证。
- `test_baseline.py` 仍含旧实验契约及票 02 旧/新提取等价性测试；后续清理
  旧实验时同步处理。标准路径新增契约不依赖旧 runner。
- `cudagraph_v1.py` 的可选 `--compare-t05` / `--timing-repeats` 历史诊断
  开关暂留；本票保留验证命令均不使用，代表性性能只走 `final_validation.py`。

生产入口、独立 runner 和页池本票均未删除，Git 历史未改写。

## 票 03 实测记录（2026-09-18）

沿用票 01 的 A100-SXM4-80GB 单卡、两个 uv 隔离环境和模型文件。
原始命令与日志位于工作区 `../results/refactor-03/`；HF reference 与完整
轨迹大张量使用 `/tmp/ksa-refactor-03-*` 本机目录。GPU 任务顺序执行。

已验证的迁移边界：

- CPU 契约 **15 passed**；相关文件 pre-commit 通过。
- 九个保留模块在禁止导入旧 baseline/decode/prefill、专用入口和独立
  runner 时均可加载。
- 新 HF reference 独立导出全部 23 例、每例 25 行 teacher logits。
- 标准分块 prefill **17/17 passed**，最大内部行数 257；max logits=2.75、
  RMSE=0.7344133257865906、max logprobs=1.2812080383300781、margin=0.25。
  保存的 V1 logits 与票 01 和票 02 均逐元素相同，最大差 **0**。
  length-8/9/15 的逐例误差变化来自新 HF reference 与历史参考的差异；
  所有用例通过原门限，四项全矩阵最大值与票 01 一致。
- 页压力通过：1 次抢占、154 steps、每请求 80 输出，取消和复用断言通过，
  全部 329 可用页归还。
- 普通 Qwen3 两例同前缀门禁通过；仍不声称逐 token 完全相同或模型能力评测。

首次新 teacher 入口运行因标准 sampler 的完整词表含 summary 占位列，
与 HF 文本 logits 宽度不同而退出。修正为按模型的 `text_vocab_size`
导出文本列（与既有 V1/graph 验证相同的口径），不改变数值门限；
初次失败日志保留在 `teacher.log`，修正后使用独立输出目录复测。

完整 eager/graph 同前缀对照 **46/46 passed**；逐 case/repeat 的四项误差
与票 01 完全一致，三组检索三轮 **9/9 passed**。内部图累计 replay 1480
次、捕获 25 次，与基线相同；这些是正确性矩阵中的变长捕获，不属于
稳态性能测量。

完整 HF 门禁 **54/69 passed、15/69 failed（NEEDS_FIX）**，退出码仍为 1。
69 条轨迹逐 case/mode/repeat 的四项误差与票 01 **完全一致**，没有新增
失败或误差恶化。最大 logits=35.75、logprobs=14.351476669311523。
五个失败输入（1023/1024/1031/1032/1033）在三轮中继续判失败。

标准 HTTP eager/graph **均全项通过**，使用 8192/257/0.5 的原服务配置。
固定 seed/n=2/惩罚的两份响应正文保存在各自 `results.json`；本轮未复现
票 01 的偶发 seed 差异，不据此声明该既有现象已修复。

性能首次启动在环境核对阶段被拒绝，未进入计时。原因是环境枚举同时
看到了虚拟环境和系统的重复 vLLM metadata，后者覆盖了实际解析版本，
且 `pre_commit` 与基线的 `pre-commit` 名称不同。核对改用
`importlib.metadata.version(name)` 与票 01 的查询口径一致；实际解析的
七项依赖版本全部相同，GPU UUID/驱动与模型指纹也相同。没有安装或
更换依赖，原失败记录保留在 `performance/`，复测使用 `performance-fixed/`。

修正后的标准 teacher decode **46/46 passed**，覆盖全部 23 例与 eager/graph；
max logits=6.375、RMSE=1.2672728300094604、max logprobs=4.249999046325684、
margin=0.25，均通过原门限。两种 engine 都由 checkpoint 选择
`KSAGPUModelRunner`；eager 无内部图，graph 实际 replay **192** 次。
各 engine 的初始可用页分别为 106433/106795，所有批次结束均完整归还。

现有 KSA 语义/GPU pytest **132 passed**（62.88 s，15 条 warnings 保留）。
没有修改生产代码或新增模型支持范围。scheduler 的完整独立 pytest 未重跑，
历史阶段性能矩阵、普通 Qwen3 性能、其他 dtype/GPU/分布式能力均不计为
本票已验证项；本票使用实际标准 scheduler 的分块、页压力和 HTTP 集成验证。

## 性能实测与复测

六组均有五次有效正式测量，正式计时无新图捕获、OOM、截断或页泄漏，
峰值显存与票 01 相同。计时从 startup timer 起的 worker AST 与整理前
**完全一致**，只在计时前增加身份核验。测量期间 GPU 任务顺序执行。

首轮结果保留在 `performance-fixed/`：四组 pass；4096 graph 的 TTFT
均值 +5.355%、范围不重叠，判为 regression；16384 eager 的 TTFT
+0.838%、范围不重叠，判为 retest。整个矩阵按原规则返回退出码 1。

后续使用全新进程和目录、相同五次协议独立复测：

- 4096 graph：`performance-repeat/graph-4096-1` 全指标 pass，TTFT
  均值 374.593 ms（基线 374.823 ms）；首轮异常未持续复现。
- 16384 eager：第一次复测 TTFT pass，但 decode 均值 +2.233%、范围
  不重叠，仍标为 retest，不直接计为通过；再次独立确认
  `performance-confirm/eager-16384-1` 的所有指标 pass，均值变化小于 5%，
  相应新旧范围重叠。

最终采用下表所列的**每组完整五次记录**，不跨轮次挑选单次样本或指标。
前两轮的 fail/retest 文件和日志均保持原样。结果表表示独立复测后获得
符合原判据的记录，不表示所有运行均通过，也不声称已定位初次波动根因。

| 输入 | 模式 | TTFT 均值 ms | decode 均值 ms/token | 输出 token/s | 记录 |
| ---: | --- | ---: | ---: | ---: | --- |
| 4096 | eager | 398.594 | 33.870 | 27.249 | 首轮 |
| 4096 | graph | 374.593 | 15.349 | 55.092 | 独立复测 |
| 16384 | eager | 2312.840 | 35.311 | 18.834 | 独立确认 |
| 16384 | graph | 2303.886 | 19.128 | 27.043 | 首轮 |
| 65536 | eager | 20994.840 | 42.458 | 4.851 | 首轮 |
| 65536 | graph | 20986.640 | 34.354 | 5.049 | 首轮 |

[最终性能比较](results/refactor-03/performance-final.json) 保存每项五次值、
均值比、判定和来源；[逐次记录](results/refactor-03/performance.csv) 保留
预热、边界/非边界 TPOT、峰值显存与页数。
[首轮比较](results/refactor-03/performance-initial.json) 及
[复测记录](results/refactor-03/performance-retests.json) 保留所有异常轮次。

## 验收与证据

票 03 的执行侧迁移完成。当前保留验证不再需要旧入口、独立 runner、
独立页池或旧 HF 性能子进程。生产代码和计时循环未修改，未提交、未推送、
未重写历史。15 条既有 HF 失败仍是 NEEDS_FIX；票 01 的 seed 现象仍未
作根因修复声明；性能首轮波动保留为观察，不隐藏于最终通过记录中。
所有本票必需项目均已实际执行，没有因环境原因最终未执行的项目。

- [验证汇总](results/refactor-03/validation.json)：本轮精度/服务结果、
  相对票 01 的逐项误差变化与初次验证脚本错误。
- [源码核验](results/refactor-03/source-verification.json)：生产 diff 为空，
  计时循环 AST 未变，15 条已知失败的生成 tokens 与冻结 fixture 相同。
- 工作区 `../results/refactor-03/`：`run.sh`、`performance-followup.sh`、
  全部日志、标准 HTTP 两次 seed 响应、完整 JSON 和进程退出状态。
  `teacher/`、`graphs/`、`generation-hf/`、`reference/` 保存本机运行的 JSON
  副本；大张量仍位于对应 `/tmp/ksa-refactor-03-*`，不加入代码树。
