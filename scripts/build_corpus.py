#!/usr/bin/env python3
"""Build data/papers.jsonl from source/MASTER_BIBLIOGRAPHY.md (30 topic tables) + kb/*.md local-PDF tables.

Each entry: key, topic, section, star, title, authors, year, venue, evidence, contribution_zh, links{}, arxiv_id,
local_pdf (filename in the survey papers/tNN dir if matched), pdf_status.
Deterministic; no network.
"""
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "source" / "MASTER_BIBLIOGRAPHY.md"
KB = ROOT / "source" / "kb"
OUT = ROOT / "data" / "papers.jsonl"

SECTION_OF = {  # topic -> (section id, section name)
    **{f"t{i:02d}": ("A", "Theory Foundations") for i in range(1, 7)},
    **{f"t{i:02d}": ("B", "Flow Matching & Trajectory Straightening") for i in range(7, 13)},
    **{f"t{i:02d}": ("C", "Cross-Domain Generation & Translation") for i in range(13, 19)},
    **{f"t{i:02d}": ("D", "Modalities") for i in range(19, 25)},
    **{f"t{i:02d}": ("E", "OT Variants Frontier") for i in range(25, 29)},
    **{f"t{i:02d}": ("F", "Systems, Benchmarks & Trends") for i in range(29, 31)},
}

STOP = {"a", "an", "the", "of", "for", "and", "in", "on", "with", "to", "via", "by", "models", "model"}
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/((?:\d{4}\.\d{4,5})(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})", re.I)
LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower()
    s = re.sub(r"\(.*?\)", " ", s)  # drop parenthetical author lists
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_title_authors(cell: str):
    """'⭐ Title (Authors)' / 'Authors, *Title*' / '"Title"' -> (title, authors, star)."""
    star = "⭐" in cell
    c = cell.replace("⭐", "").strip()
    m = re.match(r"^(.*?),\s*\*(.+?)\*(.*)$", c)  # Authors, *Title* (books)
    if m:
        return m.group(2).strip(), m.group(1).strip(), star
    m = re.match(r'^(.*?),\s*"(.+?)"(.*)$', c)  # Authors, "Title"
    if m:
        return m.group(2).strip(), m.group(1).strip(), star
    m = re.match(r"^(.*?)\s*\(([^()]*(?:,|&| and )[^()]*|[A-Z][^()]*)\)\s*$", c)  # Title (Authors)
    if m and not re.search(r"\d{4}", m.group(2)):
        return m.group(1).strip().strip('"'), m.group(2).strip(), star
    return c.strip().strip('"'), "", star


def parse_source(src: str):
    m = re.match(r"^(\d{4})\s*[·•]\s*(.+)$", src.strip())
    if m:
        return int(m.group(1)), m.group(2).strip()
    m = re.search(r"(\d{4})", src)
    return (int(m.group(1)) if m else None), src.strip()


def parse_bib():
    rows = []
    topic = None
    for line in BIB.read_text(encoding="utf-8").split("\n"):
        h = re.match(r"^## (T\d{2})\s+(.*)$", line)
        if h:
            topic = h.group(1).lower()
            topic_name = h.group(2).strip()
            continue
        if not topic or not line.startswith("| ") or line.startswith("| 论文") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) < 5:
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if len(cells) < 5:
            continue
        title, authors, star = split_title_authors(cells[0])
        year, venue = parse_source(cells[1])
        ev = re.sub(r"[^PARB]", "", cells[2])
        links = {name: url for name, url in LINK_RE.findall(cells[4])}
        arx = None
        for url in links.values():
            m = ARXIV_RE.search(url)
            if m:
                arx = re.sub(r"v\d+$", "", m.group(1))
                break
        rows.append(dict(topic=topic, topic_name=topic_name, section=SECTION_OF[topic][0], section_name=SECTION_OF[topic][1],
                         star=star, title=title, authors=authors, year=year, venue=venue, evidence=ev,
                         contribution_zh=cells[3], links=links, arxiv_id=arx))
    return rows


def parse_kb_pdfs():
    """topic -> list of (filename, title, status) from '## 7. 本地 PDF 清单'."""
    out = {}
    for f in sorted(KB.glob("t*.md")):
        topic = f.name[:3]
        txt = f.read_text(encoding="utf-8")
        i = txt.find("## 7.")
        if i < 0:
            continue
        for line in txt[i:].split("\n"):
            if not line.startswith("| ") or line.startswith("| 文件名") or line.startswith("|---"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            fn, title, status = cells[0], cells[1], cells[2]
            out.setdefault(topic, []).append((fn, title, status))
    return out


def main():
    rows = parse_bib()
    kb_pdfs = parse_kb_pdfs()
    # attach local pdf by fuzzy title match within topic
    for topic, items in kb_pdfs.items():
        cands = [r for r in rows if r["topic"] == topic]
        for fn, title, status in items:
            nt = norm_title(title)
            arx_in_title = ARXIV_RE.search(title.replace(" ", "")) or re.search(r"(\d{4}\.\d{4,5})", title)
            fm = re.match(r"^(\d{4})_([A-Za-z]+)", fn)
            f_year, f_auth = (int(fm.group(1)), fm.group(2).lower()) if fm else (None, "")
            best, best_s = None, 0.0
            for r in cands:
                rt = norm_title(r["title"])
                s = 0.4 * SequenceMatcher(None, nt, rt).ratio()
                a, b = set(nt.split()) - STOP, set(rt.split()) - STOP
                s += 0.6 * (len(a & b) / max(1, min(len(a), len(b))))  # overlap coefficient: abbreviated bib titles
                raw = unicodedata.normalize("NFKD", (r["title"] + " " + r["authors"]).lower()).encode("ascii", "ignore").decode()
                if f_auth and len(f_auth) >= 3 and re.search(r"\b" + re.escape(f_auth), raw) and r["year"] == f_year:
                    s += 0.3
                if f_year and r["year"] and abs(r["year"] - f_year) > 1:
                    s -= 0.35
                if arx_in_title and r["arxiv_id"] and arx_in_title.group(1) == r["arxiv_id"]:
                    s = 1.5
                if s > best_s:
                    best, best_s = r, s
            ok = fn.endswith(".pdf") and "成功" in status
            if best is not None and best_s >= 0.55 and best.get("local_pdf") in (None, fn):
                best["kb_title"] = re.sub(r"\s*[（(](?:arXiv|ETH|作者|FnT|\d{4}).*$", "", title).strip()
                if ok:
                    best["local_pdf"] = fn
                    best["pdf_status"] = "local"
                else:
                    best.setdefault("pdf_status", "failed:" + status[:40])
    # dedupe key + defaults
    seen = {}
    for r in rows:
        r.setdefault("local_pdf", None)
        r.setdefault("pdf_status", "missing")
        base = r["arxiv_id"] or norm_title(r["title"])[:60].replace(" ", "_")
        key = f"{r['topic']}__{base}"
        n = seen.get(key, 0)
        seen[key] = n + 1
        r["key"] = key if n == 0 else f"{key}__{n}"
        r["dup_of"] = None
    # cross-topic duplicates (same arxiv id or same normalized title)
    first = {}
    for r in rows:
        k = r["arxiv_id"] or norm_title(r["title"])
        if k in first:
            r["dup_of"] = first[k]["key"]
        else:
            first[k] = r
    if OUT.exists():  # carry over network-resolved fields
        prev = {json.loads(l)["key"]: json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()}
        for r in rows:
            p = prev.get(r["key"])
            if p:
                for k in ("arxiv_searched", "arxiv_title_ratio"):
                    if k in p:
                        r[k] = p[k]
                if p.get("arxiv_id") and not r.get("arxiv_id"):
                    r["arxiv_id"] = p["arxiv_id"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    uniq = [r for r in rows if not r["dup_of"]]
    stats = dict(rows=len(rows), unique=len(uniq), with_arxiv=sum(1 for r in uniq if r["arxiv_id"]),
                 local_pdf=sum(1 for r in uniq if r["local_pdf"]), starred=sum(1 for r in uniq if r["star"]),
                 evidence={e: sum(1 for r in uniq if r["evidence"] == e) for e in "PARB"},
                 kb_pdf_rows=sum(len(v) for v in kb_pdfs.values()),
                 kb_pdf_unmatched=sum(1 for t, v in kb_pdfs.items() for fn, ti, st in v
                                      if fn.endswith(".pdf") and "成功" in st and not any(r.get("local_pdf") == fn for r in rows)))
    (ROOT / "data" / "corpus_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
