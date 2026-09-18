# 整理前基线（ticket 01，2026-09-18）

## 固定范围与测量协议

在任何生产代码变更前固定本协议。代码为 `releases/v0.26.0-ksa` 的
`81d2c510efbfdf9efdd801e6b6204abfaff97c23`，基点 `568afb3a13`，共 28 commits。
初始工作区干净，`git ls-remote origin refs/heads/releases/v0.26.0-ksa`
返回同一 HEAD。本任务不改生产代码、不重写历史、不推送。
原有 `ksa-t00-baseline-snapshot` 引用在任务开始前已存在，本任务不创建归档引用。

原始证据位于工作区 `results/refactor-01-20260918/`；命令以其中 `run.sh`
为准，所有 GPU 工作顺序执行，结果写入新目录。后续复测更换输出目录，
使用最终代码 SHA，不 checkout 旧提交覆盖重构。Python 使用 uv 管理的
`.venv/bin/python`，HF 使用 `.venv-ksa-hf/bin/python`；现有 pre-commit hook 已安装。

- A100-SXM4-80GB 单卡、驱动 575.57.08；启动前 14 MiB、0% GPU utilization。
  BF16 是本次实测范围，不代表验证了其他 dtype，也不新增 dtype 限制。
- `PYTHONPATH=$PWD OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1`，模型
  `../models/KSA-4B-base`。环境、源码和输入指纹另存紧凑证据。
- 原始输入/参考位于
  `../results/T00/a100-repaired-20260916T032325Z-0eea77a7ba21/raw`。
  baseline `T00-305bf8f9dc429f1af60e`，容差 `T00-tol-fbd0eb931ac1c7153403`。
  `inputs.json`、`baseline-lock.json`、`calibration.json`、`correctness.json`
  与逐例 HF logits 是后续 reference 提取的保留源，不可先删除。
- 原容差以 `correctness.json.thresholds` 的精确值为准：logits 最大误差
  20.33521、RMSE 3.70595、logprobs 最大误差 8.02013、top-1 分歧 margin 0.5
  （此处仅显示舍入值）。不可放宽门限或删除失败输入。
- 精度：现有 KSA pytest 全套；标准 V1 的默认短输入集、row budget 257，
  覆盖 7/8/9、15/16/17、1023/1024/1025、1031/1032/1033、4096；
  `cudagraph_v1 --all-cases` 保留全部 23 个输入和 128 输出，eager/graph/graph
  三轮重排；`final_generation` 在相同实际前缀上复核全部 69 条轨迹。
  这不是阶段性能矩阵；保留长输入与失败轨迹避免改变旧 harness 身份。
- 生命周期：`v1_pressure` 的真实 scheduler 页压力、抢占、取消、复用；
  标准 HTTP eager/graph 协议验证另记录。普通 Qwen3 用 `plain_qwen3` 的确定性
  小权重功能回归（logprob 0.05、margin 0.02），不做普通 Qwen3 性能 baseline。
- 性能只跑标准 V1 worker：4096 / 16384 / 65536 输入，128 输出，batch=1，
  eager 与 graph 六项；每项一次预热、五次正式重复。使用冻结 token IDs，
  temperature=0、ignore_eos=True、row budget 4096、GPU memory utilization 0.85。
  max_model_len=输入+128。计时包括 scheduler、采样、metadata、页提交、设备执行，
  不含 HTTP；加载/profiling、图捕获和 KV 探针单列，不混入稳态。
  不重跑历史长度/并发性能矩阵或 HF 性能比较。
- 报告每项 TTFT、decode TPOT、块边界/非边界 TPOT、吞吐、峰值显存与归还页数。
  正式重复出现新图捕获、OOM、页泄漏、少于五次则无效，不能判通过。
  重构后在同硬件/依赖/输入/配置下比较五次均值和范围：任一延迟均值增加超过
  5% 且新旧范围不重叠，或吞吐下降超过 5% 且范围不重叠，视为性能回退；
  只满足其中一个条件视为待复测，不能直接通过。显存增加超过 5% 需解释并复测。
  这些是本次前后整理判据，不是历史 HF 加速门禁。
- 正确性按 case/mode/repeat 对照：新增失败视为回退；既有失败继续保留。
  对既有失败，误差超过本轮范围需调查，不能因其已失败而忽略恶化。
  缺依赖、未执行或证据不足单列，不计通过。

## 保留模块契约与最低有效验证层级

| 模块/职责 | 输入输出、顺序与不变量 | 错误/性能约束 | 最低有效验证 |
| --- | --- | --- | --- |
| `config/ksa.py` 与模型注册 | 按 `use_summary_attention` 识别；配置标准 V1；普通 Qwen3 保持原模型/runner | 不支持的组合启动时 ValueError；不扩展支持矩阵 | 现有配置/注册 unit；Qwen3 小权重 HF 集成 |
| `ksa_prefill.py` summary/layout/visibility | 文本 IDs/位置→插入 summary 的行与文本映射；每 8 文本一 summary；summary 自身可见、禁止未来泄漏 | 非法窗口表达式/输入拒绝；文本 usage 不含 summary | 边界/独立 visibility predicate unit；分块 V1 HF 对照 |
| `ksa.py` / `ksa_attention.py` 模型与分页 attention | 各请求文本与 KV metadata→文本 hidden/logits；同一 softmax 合并 text/summary；过滤 summary logits | 共享投影契约；不在生产构建平方 mask/完整历史 gather | GPU kernel 对 FP32 oracle；完整 HF 同前缀轨迹 |
| `ksa_cache.KSASummarySpec` / `ksa_gpu_model_runner.KSASchedulerPagePool` 页映射 | scheduler block IDs→各层 KV 读写；文本页 8 位置，summary 页覆盖 64 文本；先完成当前 batch attention 再提交 KV | 不另分配页池；保留最早可见窗口块；非法页映射拒绝；取消/结束归还 | 现有 V1 eviction 集成；真实 pressure 生命周期 |
| `ksa_gpu_model_runner.py` / scheduler | scheduler 输出→标准采样输入；按 request ID 关联，抢占从文本历史重算 | 行预算 n+floor((start+n)/8)-floor(start/8)；单请求 chunk≤4096；禁止重复/漏 token | scheduler unit；V1 chunked prefill；HTTP 批处理/取消 |
| `ksa_graph.py` decode graph | 固定形状 buffer、页表/slot 更新→与 eager 一致的 hidden；重排/复用不泄漏历史 | `ksa_cudagraph` 在 enforce_eager=True 下独立控制局部 decode 图；正式计时不得重捕获 | GPU graph buffer 测试；V1 eager/graph 重排与性能 |
| 验证侧 HF reference（待提取） | 冻结模型/输入/teacher 或实际前缀→原始 logits、误差与原容差判断 | HF 独立锁定环境；先核验模型/输入/校准身份再执行；不暴露生产执行选项 | 现有 baseline 契约与完整轨迹脚本 |

独立 runner、旧入口、独立页池本票暂不删除；其测试结果不能替代标准 V1 验证。
后续迁移先保留可观察行为，再移除只服务于旧实现的测试。

## 历史结论（不是本轮结果）

T07 **NEEDS_FIX**：HF 完整生成轨迹 54/69 通过、15/69 失败，涉及输入长度
1023、1024、1031、1032、1033；最大 logits 误差 35.75，最大 logprobs 误差
14.35148。见 [历史报告](results/T07/a100-final/README.md)。本任务不诊断或修复
该精度问题，不以 token 一致替代全词表数值门禁。当前实测结果将在下方单独记录。

## 复跑命令

从 vLLM 仓库根目录执行；两个脚本依次运行，避免 GPU 工作重叠。
`run.sh` 新建输出目录，已有目录会拒绝；HF 既有精度失败仍会返回非零，
必须阅读结果，不能把进程退出码重标成成功。`http.sh` 使用相同目录。

```bash
OUT=../results/refactor-01-repeat SHA=$(git rev-parse HEAD) \
  bash docs/ksa/results/refactor-01/run.sh
OUT=../results/refactor-01-repeat bash docs/ksa/results/refactor-01/http.sh
```

这些是本轮命令的可移植副本；原始绝对路径命令与执行日志保留在工作区结果目录。
性能使用 `final_validation.py --worker` 跳过 HF 性能矩阵，因此必须由外部证据
记录 SHA、环境与模型核验（worker 本身不会核验 `--expected-sha`）。
后续迁移 reference 后应替换对应入口，保持输入、容差和计时范围不变。

## 本次实测环境与已完成检查

本轮 vLLM 包元数据为 `0.26.1.dev9+ga493692cf.d20260917.empty`，不同于历史
报告中的 `0.26.0` 元数据；隔离模式 import 确认为当前仓库源码，代码身份以
上述 Git SHA 为准。PyTorch 2.11.0+cu130、Transformers 5.14.1、Triton 3.6.0、
tokenizers 0.22.2；HF 完整 packages 与冻结 lock 完全一致，模型 12 个文件
逐一 SHA-256 核验通过。Python 3.12.13、uv 0.11.32。

- KSA pytest：132 passed（65.83 s）；包括真实 GPU 测试。
- baseline 契约：9 tests，OK。
- scheduler / async scheduler：140 passed、5 deselected（59.22 s）。
  排除表达式为 `not pp and not pipeline`，排除项不计通过。
- 分块 prefill：17/17 passed，最大 logits 误差 2.75、RMSE 0.7344133、
  logprobs 误差 1.2812080、top-1 分歧 margin 0.25；最大实际内部行数 257。
- 页压力：pass，1 次抢占、154 steps、每请求 80 输出；取消与请求复用成功，
  结束归还全部 329 可用物理页。
- 普通 Qwen3：两例同前缀数值比较 pass，logprob 最大误差分别
  0.007792 / 0.004803；第二例 token 序列有分歧，margin 0.001953125 在既定
  0.02 门限内，不声称 token 完全一致或预训练模型能力评测通过。

以上是本次重新执行的结果。pytest 的弃用 warnings 原样保留于日志。
运行期间工作区新增本票文档，部分脚本的 `dirty=true` 因此不表示生产代码改变；
最终另存限定生产/验证源码范围的 diff 核验。

## 本次性能结果

六组均为五次有效正式重复：无 OOM、无正式计时内图重捕获，所有请求结束后
KV 页全部归还。以下为算术平均（括号为五次范围），TTFT 单位 ms，decode
单位 ms/token。逐次数据含预热行，见 [performance.csv](results/refactor-01/performance.csv)；
比较时排除 repetition=-1。

| 输入 | 模式 | TTFT | decode | 输出 token/s |
| ---: | --- | ---: | ---: | ---: |
| 4096 | eager | 392.32 (386.82–412.93) | 33.40 (33.11–33.69) | 27.623 |
| 4096 | graph | 374.82 (374.23–375.50) | 15.23 (15.17–15.28) | 55.450 |
| 16384 | eager | 2304.18 (2303.73–2305.08) | 34.51 (34.25–34.71) | 19.142 |
| 16384 | graph | 2316.48 (2304.90–2356.37) | 19.15 (18.93–19.47) | 26.955 |
| 65536 | eager | 20982.92 (20978.60–20991.25) | 42.76 (42.36–43.04) | 4.846 |
| 65536 | graph | 20998.59 (20985.85–21041.71) | 34.80 (34.53–34.92) | 5.036 |

本表仅建立重构前参照，不是重构后无回退结论，也不沿用历史 HF 性能门禁。
启动/profiling、图捕获明细与实际 KV 探针仍保留在各 `perf-*/` 原始目录。

大张量 I/O 说明：六组性能测量完成后，完整轨迹验证在共享存储回读
`eager-0.pt` 时明显等待文件 I/O。后续三份轨迹张量通过同名软链接暂存于
本机 `/tmp/ksa-refactor-01-20260918-tensors`，验证结束后复制回原结果目录并
核验 SHA-256。未修改 harness 或张量内容；此操作不发生在性能计时中。
原始 `tensor-storage.json` 记录调整，复跑也可直接使用共享存储（耗时更长）。

## 本次完整生成与 HF 门禁

标准 V1 eager/graph 两轮同前缀比较 **46/46 passed**；三组检索在三轮中
**9/9 passed**。最大 logits 误差 14.125、RMSE 2.5453410、logprobs 误差
5.9062386、分歧 margin 0.25。累计 graph replay 1480 次、捕获 25 次；这是
变长/重排正确性矩阵中的捕获，不属于前述稳态性能计时。

本轮冻结 HF 同前缀门禁为 **54/69 passed、15/69 failed，NEEDS_FIX**。
脚本按原门禁返回退出码 1，日志中的 `Final same-prefix HF generation gate failed`
是实际数值失败，不是环境不可执行，也未改写成通过。五个失败输入在 eager、
graph repeat=1、graph repeat=2 均失败，集合与历史完全一致；逐 case/mode/repeat
比较，四个门禁指标均无正向误差增量，未发现新增失败。

| 输入长度 | 本轮最大 logits 误差 | 本轮最大 logprobs 误差 | 状态 |
| ---: | ---: | ---: | --- |
| 1023 | 35.750 | 5.328 | 既有失败，3/3 |
| 1024 | 28.750 | 5.000 | 既有失败，3/3 |
| 1031 | 28.375 | 4.938 | 既有失败，3/3 |
| 1032 | 33.625 | 14.351477 | 既有失败，3/3 |
| 1033 | 27.750 | 4.813 | 既有失败，3/3 |

本轮全矩阵最大 logits 误差 35.75、RMSE 3.3940461、logprobs 误差
14.351476669311523、top-1 分歧 margin 0.25。原始失败输入、完整生成 IDs、
同前缀 logits 和精确阈值继续保留，后续 reference 提取不得删除这些复现条件。

## HTTP 新观察与补测

首轮标准服务使用 max_model_len=8192、row budget=257、GPU memory utilization=0.5：

- eager：并发 1/4/8 与 SSE 通过，随后 `v1_serving.py:108` 的固定 seed
  多样本响应相等断言失败（退出码 1）。输入为 `[42] * 65`，max_tokens=16、
  ignore_eos=True、temperature=0.8、top_p=0.9、seed=17、n=2、
  presence_penalty=0.2、repetition_penalty=1.05、logprobs=3。
  后续 echo/stop、HTTP 断连取消未在这一次执行，不能计为通过。
- graph：整个脚本通过，包括上述 seed 检查、echo/prompt logprobs、stop、
  断连取消与页回收。

原脚本 `finally` 写出的首轮 eager `results.json.status` 仍是 `running`，
但进程已退出失败；汇总根据日志退出码明确标为 fail，原始文件不改写。
该脚本未保存断言失败的两份响应正文，因此目前复现入口是原 harness 和上述
输入参数，不能声称已定位根因。

补测保持生产代码、输入和断言不变，改用历史 T07 服务参数
max_model_len=131072、row budget=4096、GPU memory utilization=0.85：
**eager 和 graph 均全项通过**，记录在 `http-t07-config/`。
这不是将首轮失败改判通过；两种配置的结果分别保留。

可复跑补测（先创建全新子目录，前一服务退出后再运行）：

```bash
mkdir ../results/refactor-01-repeat/http-t07-config
OUT=../results/refactor-01-repeat/http-t07-config \
  MAX_MODEL_LEN=131072 ROW_BUDGET=4096 GPU_MEMORY_UTILIZATION=0.85 \
  bash docs/ksa/results/refactor-01/http.sh
```

原 8192/257/0.5 eager 配置在独立新服务、新输出目录中复测：**全项通过**，
见 `http-eager-repeat/`。故首次 seed 不一致属于本票新观察到的整理前现象，
复测未复现、原因未定；不能称为已解决，也不能称为重构引入（本票未改生产代码）。
后续整理需保留 seed/n/惩罚验证及首轮失败证据；若再次出现，另行统计复现条件，
不得放宽响应相等断言。初次失败后的未执行项在独立复测中已执行通过。

```bash
mkdir ../results/refactor-01-repeat/http-eager-repeat
OUT=../results/refactor-01-repeat/http-eager-repeat MODES=eager \
  bash docs/ksa/results/refactor-01/http.sh
```

服务退出时仍可见 `resource_tracker` semaphore 清理 warning；成功轮次的
KV 页回收断言通过，不将该 warning 自动认定为 KV 泄漏。

## 基线结论与证据索引

票 01 的代表性整理前参照已取得；**这不表示 KSA 精度验收通过**。
后续可据此比较新增失败、误差恶化、性能回退及未执行项：

- 已确认既有问题：HF 五个边界输入的 15 条轨迹失败，原容差不变。
- 本票新观察：HTTP eager 首轮固定 seed 响应不一致；同配置独立复测通过，
  根因未定，作为后续验证注意项保留。
- 尚未进行代码整理，所以本票不能宣称“整理后无回退”；仅确认本轮完整 HF
  轨迹与历史逐项比较没有新增失败或门禁指标恶化。
- 没有因环境缺失而无法执行的必需项目。scheduler 的 5 个 PP/pipeline 用例、
  历史完整性能矩阵、普通 Qwen3 性能、其他 GPU/dtype/分布式支持不在本票范围，
  不计为通过。未运行独立旧 runner 的完整阶段性能实验。

| 证据 | 用途 |
| --- | --- |
| [environment.json](results/refactor-01/environment.json) | SHA/分支/基点/远端、GPU 与运行时、精确容差、模型和输入指纹 |
| [source-verification.json](results/refactor-01/source-verification.json) | 生产代码、验证脚本和测试无变更；文档工作区状态 |
| [summary.json](results/refactor-01/summary.json) | 逐 case/mode/repeat 精度状态、历史差异、HTTP 首次失败与补测分别记录 |
| [performance.csv](results/refactor-01/performance.csv) | 六组逐次性能、块边界延迟、显存、捕获与物理页归还 |
| [run.sh](results/refactor-01/run.sh)、[http.sh](results/refactor-01/http.sh) | 可移植复跑命令；新结果目录，不覆盖失败记录 |
| 工作区 `results/refactor-01-20260918/` | 原始命令、日志、依赖清单、完整 JSON、logits 与恢复校验 |

小型证据清单 `raw-files.sha256` 的路径相对工作区根目录，可在工作区执行
`sha256sum -c vllm/docs/ksa/results/refactor-01/raw-files.sha256`。
清单覆盖本轮非张量原始文件、冻结输入/校准/参考身份文件及三份回存轨迹张量；
其他 HF/prefill 大张量仍保留于原目录，但未逐一加入该清单。
本轮检查为相关 pytest/unittest、上述模型验证、shell 语法及新增文件 pre-commit，
没有运行全仓库测试。未提交、未推送、未重写 Git 历史。
