#!/usr/bin/env python3
"""Normalize reports: rename slug-named reports/meta to arXiv ids when resolvable; validate meta cards;
write data/reports_index.json {report_id: {title, tldr_zh, tldr_en, topic, read_full_text}} and print coverage per topic."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in (ROOT / "data/papers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def slug(title):
    return re.sub(r"[^A-Za-z0-9]+", "_", title)[:50].strip("_")


slug2id = {}
for r in rows:
    if r["arxiv_id"]:
        slug2id[slug(r["title"])] = r["arxiv_id"].replace("/", "_")
        if r.get("kb_title"):
            slug2id[slug(r["kb_title"])] = r["arxiv_id"].replace("/", "_")
renamed = 0
for d, ext in (("reports", ".md"), ("data/meta", ".json")):
    for p in (ROOT / d).glob("*" + ext):
        if re.match(r"^\d{4}\.\d{4,5}$", p.stem) or p.stem in slug2id and False:
            continue
        new = slug2id.get(p.stem)
        if new and not (p.parent / (new + ext)).exists():
            p.rename(p.parent / (new + ext))
            renamed += 1
            if ext == ".json":
                j = json.loads((p.parent / (new + ext)).read_text(encoding="utf-8"))
                j["report_id"] = new
                (p.parent / (new + ext)).write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
# index + validation
idx, bad = {}, []
topic_of = {}
for r in rows:
    rid = r["arxiv_id"].replace("/", "_") if r["arxiv_id"] else slug(r["title"])
    topic_of.setdefault(rid, r["topic"])
for p in sorted((ROOT / "reports").glob("*.md")):
    rid = p.stem
    txt = p.read_text(encoding="utf-8")
    mp = ROOT / "data/meta" / (rid + ".json")
    card = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    if not mp.exists():
        bad.append(f"no-meta {rid}")
    if len(re.findall(r"^## ", txt, re.M)) < 8:
        bad.append(f"sections<8 {rid}")
    m = re.search(r"^# (.+)$", txt, re.M)
    idx[rid] = dict(title=(m.group(1).strip() if m else card.get("title", rid)), tldr_zh=card.get("tldr_zh", ""), tldr_en=card.get("tldr_en", ""),
                    topic=topic_of.get(rid, ""), read_full_text=card.get("read_full_text", "未读全文" not in txt), chars=len(txt))
(ROOT / "data/reports_index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
cov = {}
for r in rows:
    if r["dup_of"]:
        continue
    rid = r["arxiv_id"].replace("/", "_") if r["arxiv_id"] else slug(r["title"])
    c = cov.setdefault(r["topic"], [0, 0])
    c[1] += 1
    c[0] += (rid in idx)
print(f"renamed={renamed} reports={len(idx)} full_text={sum(1 for v in idx.values() if v['read_full_text'])} issues={len(bad)}")
print(" ".join(f"{t}:{a}/{b}" for t, (a, b) in sorted(cov.items())))
if bad:
    print("\n".join(bad[:20]))
