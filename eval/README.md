# 检索质量评测（Retrieval Eval）

用于量化检索改动（切块、改写、rerank 等）的实际效果，避免"改了但不知道好坏"。

## 组成

| 文件 | 说明 |
|---|---|
| `corpus/` | 16 篇测试语料（多主题 + 刻意设置的相近干扰文档），UTF-8 Markdown |
| `dataset.jsonl` | 20 条标注：问题 → 期望来源文件 + 答案片段（`answer_span`）+ 提问类型 |
| `ingest.py` | 把语料灌入向量库（文件名派生确定性 UUID，可重复执行） |
| `run_eval.py` | 跑评测：Recall@1/3/5 + MRR，逐题对比、baseline vs 改写对照 |

## 用法

```bash
# 前置：PostgreSQL + Ollama 已启动
python eval/ingest.py          # 灌语料（会覆盖同名的评测文档）
python eval/run_eval.py        # 跑评测
python eval/run_eval.py --top 20   # 观察更完整的排序
```

> 注意：`ingest.py` 只覆盖**评测自己的文档**（UUID 由文件名派生）。但如果库里混有主题相近的
> 业务文档，会干扰评测——建议在干净的库上跑，或先清理旧数据。

## 指标口径

- **hit**：检索结果中存在切片的内容包含该题的 `answer_span`。
- **Recall@K**：前 K 条命中该题的比例。
- **MRR**：首个命中的排名倒数的均值（越接近 1 越好）。

## 已知局限

- `answer_span` 若被切块边界切断，会造成"假未命中"（标注时需保证片段落在单块内）。
- 语料规模较小（26 块），指标波动敏感；扩充语料后结论会更稳。
- 只评检索，不评最终答案质量（LLM 裁判见改进计划 D 类）。
