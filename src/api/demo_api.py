"""WebUI 演示：透视 RAG 链路（零新依赖，纯 FastAPI + 单页 HTML）。

用途：打破"黑盒感"——让你亲眼看每一次问答的：
  - 可见范围、各环节开关
  - 粗排候选池 vs 精排结果（**精排到底提权了什么、误杀了什么**）
  - 各环节耗时
  - 最终答案及其**真实引用**来源

访问：起服务后打开 http://127.0.0.1:8000/demo
"""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.agent.graph_builder import get_agent
from src.auth.deps import DEMO_USERS
from src.config.prompts import load_templates
from src.config.settings import get_settings
from src.rag.ambiguity import check_ambiguity
from src.rag.retriever import retrieve
from src.vector_store.base import Chunk

router = APIRouter(tags=["demo"])


def _brief(c: Chunk, rank: int) -> dict:
    return {
        "rank": rank,
        "source": c.source_file,
        "chunk_index": c.chunk_index,
        "score": round(c.score, 4),
        "snippet": c.content[:120].replace("\n", " "),
    }


class AskRequest(BaseModel):
    question: str
    token: str = "demo-it-token"
    use_hybrid: bool = True
    use_rerank: bool = True
    with_answer: bool = True


@router.post("/demo/ask")
async def demo_ask(req: AskRequest) -> dict:
    """跑一次完整链路，返回答案 + 中间结果 + 耗时（供 WebUI 透视）。"""
    settings = get_settings()
    user = DEMO_USERS.get(req.token)
    if user is None:
        raise HTTPException(status_code=401, detail="无效 token")
    departments = list({user.department, settings.company_scope})

    trace: dict = {}
    t0 = time.perf_counter()
    _, chunks = await retrieve(
        req.question, departments, user.secret_level,
        use_hybrid=req.use_hybrid, use_rerank=req.use_rerank, trace=trace,
    )
    retrieve_ms = int((time.perf_counter() - t0) * 1000)

    candidates = trace.get("candidates", [])
    final = trace.get("final", [])
    answer, cited, generate_ms = None, [], 0

    if req.with_answer and settings.deepseek_api_key and final:
        _, human_tpl = load_templates()
        ctx = "\n\n".join(
            f"[来源{i}] (文件:{c.source_file}, 块:{c.chunk_index})\n{c.content}"
            for i, c in enumerate(final, 1)
        )
        t1 = time.perf_counter()
        try:
            agent = get_agent()
            result = await agent.ainvoke({"messages": [HumanMessage(content=human_tpl.format(context=ctx, question=req.question))]})
            answer = str(result["messages"][-1].content)
        except Exception as e:  # noqa: BLE001
            answer = f"[生成失败] {e}"
        generate_ms = int((time.perf_counter() - t1) * 1000)
        # 引用校验（与线上一致）
        from src.api.chat_api import _cited_chunks
        cited = [_brief(c, i) for i, c in enumerate(_cited_chunks(final, answer), 1)]

    amb = await check_ambiguity(req.question, final) if final else None

    return {
        "question": req.question,
        "user": {"name": user.name, "department": user.department, "scope": departments},
        "mode": trace.get("mode", {}),
        # 当前精排后端（供页面显示：LLM 还是本地 cross-encoder）
        "rerank": {
            "backend": settings.rerank_backend,
            "model": (settings.rerank_local_model if settings.rerank_backend == "local"
                      else settings.deepseek_model),
        },
        "timings": {"retrieve_ms": retrieve_ms, "generate_ms": generate_ms, "total_ms": retrieve_ms + generate_ms},
        "answer": answer,
        "cited": cited,
        "ambiguous": bool(amb and amb.ambiguous),
        "ambiguity_reason": (amb.reason if amb else ""),
        "candidates": [_brief(c, i) for i, c in enumerate(candidates, 1)],
        "final": [_brief(c, i) for i, c in enumerate(final, 1)],
    }


@router.get("/demo", response_class=HTMLResponse)
async def demo_page() -> str:
    return DEMO_HTML


DEMO_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG 链路透视</title>
<style>
  :root { --bg:#0f1115; --card:#181b22; --line:#2a2f3a; --fg:#e6e8ec; --dim:#8b93a3;
          --ok:#3fb950; --warn:#d29922; --bad:#f85149; --acc:#58a6ff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif; }
  .wrap { max-width:1080px; margin:0 auto; padding:24px; }
  h1 { font-size:20px; margin:0 0 4px; }
  .sub { color:var(--dim); font-size:13px; margin-bottom:20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px; margin-bottom:16px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  input[type=text] { flex:1; min-width:260px; background:#0d1017; border:1px solid var(--line);
                     color:var(--fg); padding:10px 12px; border-radius:8px; font-size:14px; }
  select, button { background:#0d1017; border:1px solid var(--line); color:var(--fg);
                   padding:10px 12px; border-radius:8px; font-size:14px; cursor:pointer; }
  button.primary { background:var(--acc); border-color:var(--acc); color:#04121f; font-weight:600; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  label.chk { display:flex; gap:6px; align-items:center; color:var(--dim); user-select:none; }
  .meta { color:var(--dim); font-size:12px; margin-top:10px; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:820px){ .grid { grid-template-columns:1fr; } }
  .item { border:1px solid var(--line); border-radius:8px; padding:8px 10px; margin-bottom:8px;
          background:#0d1017; font-size:12.5px; }
  .item .h { display:flex; justify-content:space-between; gap:8px; color:var(--dim); }
  .item .s { margin-top:4px; color:#b9c0cc; }
  .pill { display:inline-block; padding:1px 7px; border-radius:999px; font-size:11px;
          border:1px solid var(--line); color:var(--dim); }
  .up { color:var(--ok); border-color:var(--ok); }
  .down { color:var(--bad); border-color:var(--bad); }
  .ans { white-space:pre-wrap; background:#0d1017; border:1px solid var(--line);
         border-radius:8px; padding:12px; }
  .tag { color:var(--dim); font-size:12px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px; }
  .err { color:var(--bad); }
</style>
</head>
<body>
<div class="wrap">
  <h1>RAG 链路透视</h1>
  <div class="sub">问一个问题，看它是<b>怎么被检索到的</b>——粗排候选池 vs 精排结果、耗时、真实引用来源。</div>

  <div class="card">
    <div class="row">
      <input id="q" type="text" placeholder="例：远程办公得提前几天在OA上申请？" value="远程办公得提前几天在OA上申请？">
      <select id="token">
        <option value="demo-it-token">IT 用户（张三）</option>
        <option value="demo-hr-token">HR 用户（李四）</option>
        <option value="demo-public-token">公开用户（访客）</option>
      </select>
      <button id="go" class="primary">提问</button>
    </div>
    <div class="row" style="margin-top:10px">
      <label class="chk"><input type="checkbox" id="hybrid" checked> 混合检索（向量+词法）</label>
      <label class="chk"><input type="checkbox" id="rerank" checked> 启用精排</label>
      <label class="chk"><input type="checkbox" id="answer" checked> 生成答案（关掉更快，只看检索）</label>
    </div>
    <div class="meta" id="meta">就绪</div>
  </div>

  <div id="out"></div>
</div>

<script>
const $ = s => document.querySelector(s);
const esc = s => (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function itemHTML(x, marks) {
  const tag = marks && marks.get(x.rank) ? `<span class="pill ${marks.get(x.rank)}">${marks.get(x.rank)==='up'?'精排提权':'精排下移'}</span>` : '';
  return `<div class="item">
    <div class="h"><span>#${x.rank} ${esc(x.source)} · 块${x.chunk_index}</span>
    <span>score ${x.score} ${tag}</span></div>
    <div class="s">${esc(x.snippet)}…</div></div>`;
}

$('#go').onclick = async () => {
  const btn = $('#go'); btn.disabled = true;
  $('#meta').textContent = '运行中…（开精排+生成约 5-10 秒）';
  $('#out').innerHTML = '';
  try {
    const r = await fetch('/demo/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        question: $('#q').value, token: $('#token').value,
        use_hybrid: $('#hybrid').checked, use_rerank: $('#rerank').checked,
        with_answer: $('#answer').checked
      })
    });
    const d = await r.json();
    if (!r.ok) { $('#meta').innerHTML = `<span class="err">失败：${esc(d.detail)}</span>`; return; }

    const t = d.timings;
    const rk = d.rerank || {};
    const rkLabel = rk.backend === 'local'
        ? `本地 cross-encoder <span style="color:#8b93a3">(${esc(rk.model||'')})</span>`
        : `LLM <span style="color:#8b93a3">(${esc(rk.model||'')})</span>`;
    $('#meta').innerHTML = `可见范围 <b>${esc(d.user.scope.join(' + '))}</b>`
      + ` ｜ 精排后端：<b>${rkLabel}</b>`
      + ` ｜ 耗时：检索 <b>${t.retrieve_ms}ms</b> / 生成 <b>${t.generate_ms}ms</b> / 合计 <b>${t.total_ms}ms</b>`;

    // 精排前后变化标记
    const before = new Map(d.candidates.map(x=>[x.source+'#'+x.chunk_index, x.rank]));
    const after  = new Map(d.final.map(x=>[x.source+'#'+x.chunk_index, x.rank]));
    const marksB = new Map(), marksA = new Map();
    d.final.forEach(x => {
      const k = x.source+'#'+x.chunk_index;
      if (before.has(k) && before.get(k) > x.rank) { marksA.set(x.rank,'up'); }
    });
    d.candidates.forEach(x => {
      const k = x.source+'#'+x.chunk_index;
      if (after.has(k) && after.get(k) > x.rank) { marksB.set(x.rank,'down'); }
    });

    let html = '';
    if (d.answer !== null && d.answer !== undefined) {
      const amb = d.ambiguous ? `<div class="pill down" style="margin-bottom:8px;display:inline-block">⚠ 判定为歧义：${esc(d.ambiguity_reason)}</div>` : '';
      html += `<div class="card"><div class="tag">答案</div>${amb}<div class="ans">${esc(d.answer)}</div>
        <div class="meta">引用校验后保留 ${d.cited.length} 条来源：${d.cited.map(c=>esc(c.source)+' · 块'+c.chunk_index).join('，')||'（无）'}</div></div>`;
    }

    const n = Math.min(8, Math.max(d.candidates.length, d.final.length));
    html += `<div class="card"><div class="tag">检索链路：粗排候选池(${d.candidates.length}条) → 精排(${d.final.length}条)</div>
      <div class="grid">
        <div><div class="meta" style="margin:0 0 8px">粗排前${n}（RRF 融合后）</div>${d.candidates.slice(0,n).map(x=>itemHTML(x,marksB)).join('')}</div>
        <div><div class="meta" style="margin:0 0 8px">精排后（送给 LLM）</div>${d.final.slice(0,n).map(x=>itemHTML(x,marksA)).join('')}</div>
      </div>
      <div class="meta">绿色=精排把它提上来了；红色=精排把它压下去了。若正确块在左边却不在右边 → <b>精排误杀</b>。</div>
    </div>`;

    $('#out').innerHTML = html;
  } catch(e) {
    $('#meta').innerHTML = `<span class="err">异常：${esc(String(e))}</span>`;
  } finally { btn.disabled = false; }
};
</script>
</body>
</html>
"""
