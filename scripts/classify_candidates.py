#!/usr/bin/env python3
"""Classify trends/arxiv_candidates.jsonl with DeepSeek: relevance 0-3, topic tNN, tldr_zh, why_relevant, evidence.
Writes trends/new_papers_2026Q3.jsonl (relevance>=2)."""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.deepseek.com/chat/completions"
KEY = os.environ["DEEPSEEK_API_KEY"]
TOPICS = "\n".join(f"{t.upper()}: {json.loads((ROOT/f'data/topic_manifest/{t}.json').read_text())['topic_name']}" for t in [f"t{i:02d}" for i in range(1, 31)])
cands = [json.loads(l) for l in (ROOT / "trends/arxiv_candidates.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
SYS = f"""你是「扩散/流生成模型 × 最优传输」知识库的文献筛选员。给定 arXiv 论文（标题/摘要/评论），对每篇判断：
relevance: 3=核心（OT/SB/Wasserstein/耦合/桥 是方法或理论的主轴且与扩散/流生成直接相关）；2=相关（OT 概念或工具在扩散/流生成中起实质作用）；1=边缘（只是提到或应用领域不同）；0=无关。
topic: 从下列 30 个课题中选最贴近的一个（写 tNN）：
{TOPICS}
tldr_zh: ≤40 字中文一句话贡献（先说结论，不用渲染词）。why_relevant: ≤30 字说明与扩散×OT 的关系。evidence: 若 comment 明确写了已接收的会议/期刊则写 "A:<venue>"，否则写 "R"。
只输出 JSON 数组，元素为 {{"arxiv_id","relevance","topic","tldr_zh","why_relevant","evidence"}}，顺序与输入一致。"""
out = []
for i in range(0, len(cands), 8):
    batch = cands[i:i + 8]
    user = "\n\n".join(f"[{c['arxiv_id']}] {c['title']}\ncomment: {c['comment']}\nabstract: {c['abstract'][:1500]}" for c in batch)
    body = json.dumps(dict(model="deepseek-v4-pro", messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                           temperature=0.1, max_tokens=4000, thinking={"type": "disabled"})).encode()
    req = urllib.request.Request(API, data=body, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
    txt = txt.strip().strip("`").replace("json\n", "", 1) if txt.strip().startswith("`") else txt
    try:
        arr = json.loads(txt[txt.find("["):txt.rfind("]") + 1])
    except Exception as e:  # noqa: BLE001
        print("parse fail batch", i, e); continue
    byid = {c["arxiv_id"]: c for c in batch}
    for a in arr:
        c = byid.get(a.get("arxiv_id"))
        if c:
            out.append({**a, "title": c["title"], "date": c["published"], "authors": c["authors"], "primary": c["primary"]})
    print(f"batch {i//8+1}: {len(arr)} classified")
keep = sorted([o for o in out if int(o.get("relevance", 0)) >= 2], key=lambda o: (-int(o["relevance"]), o["date"]), reverse=False)
(ROOT / "trends/new_papers_2026Q3.jsonl").write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in keep), encoding="utf-8")
(ROOT / "trends/classified_all.jsonl").write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
from collections import Counter
print("kept", len(keep), "relevance:", Counter(int(o["relevance"]) for o in out), "topics:", Counter(o["topic"] for o in keep).most_common(10))
