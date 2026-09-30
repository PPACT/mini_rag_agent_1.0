# 评测（Evaluation）

用于量化检索与歧义处理的实际效果，避免"改了但不知道好坏"。

> ⚠️ **本目录下的脚本是"结论的凭据"**：每个都对应一条**可复现**的判断
> （如「HNSW 索引失效」「思考链会污染精排顺序」）。
> **证据脚本不放会被清理的 `logs/`** —— 日志一清，结论就只剩说法、没有来路。

## 目录组成（语料与数据集）

| 文件/目录 | 说明 |
|---|---|
| `corpus/` | **16 篇基础语料**（评测目标文档） |
| `corpus_synth/` | **合成语料**（gitignore，由脚本生成）：624 篇部门变体 + 86 篇不同主题文档 |
| `dataset_clear.jsonl` | **清晰题**（80 条）：答案在整个语料中唯一 |
| `dataset_clarify.jsonl` | **歧义题 + 对抗题**（37 条）：应触发澄清 |
| `dataset_exact.jsonl` | **精确词题**（8 条）：含缩写/编号（CVSS、RTO、WPA3、`#JIRA-1234`），测混合检索 |
| `dataset_paraphrase.jsonl` | **改述题**：同一问题的不同问法，测稳定性 |
| `dataset.jsonl` | 早期 20 条（现存作参考；其题目指向多版本文档，**本质是歧义题**） |

## 脚本索引

按用途分组；括注是各脚本 docstring 的自我说明。

### 建库

| 脚本 | 作用 |
|---|---|
| `ingest.py` | 把 `corpus/` 灌入向量库（`--exclude-dept` 可只灌无变体部分，用于隔离变量） |

### 生成语料与题目

| 脚本 | 作用 |
|---|---|
| `gen_synth.py` | 生成合成语料（干扰文档），用于测「规模对召回的影响」 |
| `gen_questions.py` | 从「无变体」文档自动生成清晰题（问题 + 逐字答案片段），并校验片段真实存在 |
| `gen_paraphrase.py` | 为清晰题生成改述变体（语义等价但表述不同），测「换个问法是否稳定」 |

### 核心指标评测

| 脚本 | 作用 |
|---|---|
| `run_eval.py` | **检索质量**：对比多种检索策略的 Recall@K / MRR |
| `run_clarify_eval.py` | **澄清行为**：测歧义识别的漏报率 / 误报率 |
| `run_paraphrase_eval.py` | **改述鲁棒性**：同一问题换问法，结果是否稳定 |
| `run_ann_eval.py` | **ANN 自身误差**：测量 HNSW 近似索引损失的召回（与语料难度无关） |
| `run_backend_matrix.py` | **§8 后端矩阵**：一次性钉死「思考链」与「精排」各自对延迟与质量的贡献 |

### 对照与诊断（回答"该不该做某件事"）

| 脚本 | 作用 |
|---|---|
| `run_scope_eval.py` | 验证「范围收敛」能否解决近重复带来的检索崩溃 |
| `run_split_eval.py` | 日常集 / 鲁棒集 对照评测（同一批题、两套语料） |
| `run_chunk_check.py` | 切块质量检测：区分「检索失败」与「切块把答案切断了」 |
| `run_crowding_check.py` | 诊断 MMR 是否值得做：失败主因是「近似块挤占」还是「检索盲区」 |

### 专项测量（门控 / 精排 / 长尾）

| 脚本 | 作用 |
|---|---|
| `analyze_gate_signals.py` | 测量「门控信号」的区分度：哪个信号能分开「歧义」与「不歧义」 |
| `analyze_rerank_headroom.py` | 预判「精排的可救空间」：粗排把正确答案排在了第几位 |
| `measure_rerank_value.py` | 量化「LLM 精排」值不值：改变了多少题的 top-1、改对多少、改错多少 |
| `measure_ambiguity_snippet.py` | 测量「缩短歧义判定 prompt」的收益与代价 |
| `measure_llm_tail.py` | 测量歧义判定 LLM 调用的**长尾**，并判断长尾是不是并发造成的 |
| `measure_slowcall_cause.py` | 查清「10 秒长尾」的成因：是**输出太长**还是**调用真的卡住** |
| `probe_llm_response.py` | 探针：歧义判定的 LLM 响应到底长什么样（查"空输出"的根因） |
| `analyze_snippet_result.py` | 解读 snippet 测量的逐题明细：区分「摇摆」与「稳定判错」 |

### 验证与证据脚本

| 脚本 | 作用 |
|---|---|
| `verify_hnsw_plan.py` | **向量检索走不走 HNSW 索引**（`ORDER BY` 形态与计划形态对照） |
| `verify_provider_unbind.py` | 换 `openai/` 适配器后，`reasoning_effort="none"`（关思考）是否仍生效 |
| `verify_d8_runtime.py` | 改名 + 解绑后，`complete()` 是否仍能正确关掉思考链 |
| `check_rerank.py` | 排查：思考开/关时，LLM 精排到底有没有真的改变顺序 |
| `e2e_two_kb.py` | 端到端验证两库分离后各入口是否正确路由 |
| `smoke_two_kb.py` | 冒烟测试：两库分离后检索是否各查各的 |
| `verify_live.py` | 端到端真实链路：通过 HTTP 调 `/demo/ask` 看真实体验（通过率 + 单次延迟） |
| `check_env_example.py` | 对拍 `.env.example` 与 `settings.py` 的默认值，找出不一致 |

## 为什么分三类题

| 题型 | 特征 | 测什么 | 判据 |
|---|---|---|---|
| **清晰题** | 答案唯一 | 检索质量 | Recall@K / MRR；**且不应触发澄清** |
| **歧义题** | 多个同样合理的答案 | 歧义识别 | 是否触发澄清（漏报率） |
| **对抗题** | *看似*明确、实则歧义（如"公司统一的…"） | 抗误导 | 是否触发澄清（漏报率） |
| **精确词题** | 含缩写/编号等字面词 | 混合检索 | Recall@K / MRR（向量会糊掉字面词，词法侧能中） |

> **关键**：歧义题的"标准答案"是**任意的**——用它们测召回率会失真（规模实验里 0.25 的读数就混入了这层度量假象）。
> 所以检索指标只在**清晰题**上测。

## 用法

```bash
# 前置：PostgreSQL + Ollama 已启动（本项目端口：真库 15432 / 压测库 15433 / Ollama 11451）
python eval/ingest.py                    # 灌全部语料（含部门变体，~1200 切片）
python eval/ingest.py --exclude-dept     # 只灌无变体部分（隔离"近重复"变量用）

python eval/gen_questions.py             # 生成清晰题（需 LLM）
python eval/run_eval.py                  # 检索质量（清晰题）；多策略对照
python eval/run_eval.py --dataset dataset_exact.jsonl   # 精确词题（测混合检索）
python eval/run_clarify_eval.py          # 澄清行为（漏报/误报）
python eval/run_ann_eval.py              # ANN 自身损失（走索引 vs 强制精确）
```

## 已知局限

- 合成语料的部门变体数字偏移较大（如"试用期 38 个月"），**不追求语义真实**，只为制造"同主题多版本"的检索难度。
- 清晰题由 LLM 生成，个别题目偏机械（如问文档标题）。
- 歧义题的"应澄清"标注为人工判断，存在主观性。
- 评测集规模仍有限（~117 条）；指标差异小于 0.05 时不宜下结论。
