#!/usr/bin/env python3
"""Build data/topic_manifest/tNN.json (per-topic paper lists with text paths) from data/. Usage: build_manifests.py [tNN ...]"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in (ROOT / "data/papers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
meta = json.loads((ROOT / "data/arxiv_meta.json").read_text())
tidx = json.loads((ROOT / "data/text_index.json").read_text())
kb = {p.name[:3]: p.name for p in (ROOT / "source/kb").glob("t*.md")}
only = set(sys.argv[1:])
by_topic = {}
for r in rows:
    if r["dup_of"]:
        continue
    a = r["arxiv_id"]
    stem = a.replace("/", "_") if a else None
    pdf = None
    if stem and (ROOT / f"papers/{stem}.pdf").exists():
        pdf = f"papers/{stem}.pdf"
    elif r.get("local_pdf") and (ROOT / f"papers/{r['topic']}_{r['local_pdf']}").exists():
        pdf = f"papers/{r['topic']}_{r['local_pdf']}"
    tstem = Path(pdf).stem if pdf else None
    text = f"data/text/{tstem}.txt" if tstem and (ROOT / f"data/text/{tstem}.txt").exists() else None
    m = meta.get(a, {}) if a else {}
    item = dict(key=r["key"], report_id=(stem or re.sub(r"[^A-Za-z0-9]+", "_", r["title"])[:50].strip("_")),
                title=m.get("title") or r.get("kb_title") or r["title"], authors=m.get("authors") or r["authors"],
                year=r["year"], venue=r["venue"], evidence=r["evidence"], star=r["star"], arxiv_id=a,
                published=m.get("published"), abstract=m.get("abstract", ""), contribution_zh=r["contribution_zh"],
                links=r["links"], pdf=pdf, text=text,
                text_chars=(tidx.get(tstem, {}).get("chars") if tstem else None), pages=(tidx.get(tstem, {}).get("pages") if tstem else None))
    by_topic.setdefault(r["topic"], dict(topic=r["topic"], topic_name=r["topic_name"], section=r["section"],
                                         section_name=r["section_name"], kb_note=f"source/kb/{kb[r['topic']]}", papers=[]))["papers"].append(item)
for t, d in sorted(by_topic.items()):
    if only and t not in only:
        continue
    d["papers"].sort(key=lambda p: (not p["star"], -(p["year"] or 0)))
    (ROOT / f"data/topic_manifest/{t}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{t} papers={len(d['papers'])} with_text={sum(1 for p in d['papers'] if p['text'])}")
