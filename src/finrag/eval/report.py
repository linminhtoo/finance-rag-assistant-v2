from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_html_report(run: dict[str, Any], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(run, ensure_ascii=False)

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>finrag eval report</title>
    <style>
      body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
      .meta {{ display: flex; gap: 16px; flex-wrap: wrap; }}
      .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px 14px; }}
      .q {{ margin-top: 18px; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; }}
      .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#f1f5f9; margin-right:6px; }}
      details {{ border: 1px solid #eee; border-radius: 10px; padding: 10px 12px; }}
      summary {{ cursor: pointer; font-weight: 600; }}
      .muted {{ color: #555; }}
    </style>
  </head>
  <body>
    <h1>finrag eval report</h1>
    <div class="meta">
      <div class="card"><div class="muted">Items</div><div id="n"></div></div>
      <div class="card"><div class="muted">Recall (rerank, chunk)</div><div id="recall"></div></div>
      <div class="card"><div class="muted">MRR (rerank, chunk)</div><div id="mrr"></div></div>
      <div class="card"><div class="muted">Numeric accuracy</div><div id="num"></div></div>
    </div>
    <div id="list"></div>
    <script>
      const run = {data};
      const s = run.summary || {{}};
      document.getElementById("n").textContent = s.n ?? "";
      document.getElementById("recall").textContent = (s.recall_rerank_chunk ?? "").toString();
      document.getElementById("mrr").textContent = (s.mrr_rerank_chunk ?? "").toString();
      document.getElementById("num").textContent = (s.numeric_accuracy ?? "n/a").toString();

      const el = document.getElementById("list");
      (run.results || []).forEach((r, idx) => {{
        const d = document.createElement("details");
        d.className = "q";
        const title = document.createElement("summary");
        const kind = r.kind || "";
        const hit = r.recall_rerank_doc === 1.0 ? "hit" : "miss";
        title.textContent = (idx + 1) + ". [" + kind + "] (" + hit + ") " + (r.question || "");
        d.appendChild(title);

        const tags = document.createElement("div");
        (r.tags || []).slice(0, 8).forEach(t => {{
          const span = document.createElement("span");
          span.className = "pill";
          span.textContent = t;
          tags.appendChild(span);
        }});
        d.appendChild(tags);

        const pre = document.createElement("div");
        pre.className = "mono";
        const parts = [];
        parts.push("evidence_doc_ids: " + (r.evidence_doc_ids || []).join(", "));
        parts.push("recall_rerank_doc: " + r.recall_rerank_doc);
        parts.push("recall_rerank_chunk: " + r.recall_rerank_chunk);
        if (r.numeric_matched !== undefined) {{
          parts.push("numeric_matched: " + r.numeric_matched);
          parts.push("numeric_best_rel_error: " + r.numeric_best_rel_error);
        }}
        if (r.citation_hit !== undefined) {{
          parts.push("citation_hit: " + r.citation_hit);
        }}
        if (r.final_answer) {{
          parts.push("\\nfinal_answer:\\n" + r.final_answer);
        }}
        pre.textContent = parts.join("\\n");
        d.appendChild(pre);
        el.appendChild(d);
      }});
    </script>
  </body>
</html>
"""
    p.write_text(html, encoding="utf-8")
