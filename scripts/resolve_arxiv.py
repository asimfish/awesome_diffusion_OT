#!/usr/bin/env python3
"""Resolve arXiv ids / metadata / licenses for data/papers.jsonl (network; polite 3s spacing).

Outputs data/arxiv_meta.json {arxiv_id: {title, authors, abstract, published, updated, primary_category, doi, journal_ref, license}}
and rewrites data/papers.jsonl adding arxiv_id (title search) + arxiv_title_ratio.
Resumable: skips ids already in arxiv_meta.json. Never more than one request per 3 s.
"""
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "data" / "papers.jsonl"
META = ROOT / "data" / "arxiv_meta.json"
LOG = ROOT / "logs" / "resolve_arxiv.log"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "arx": "http://arxiv.org/OAI/arXiv/"}
UA = "awesome_diffusion_OT/0.1 (mailto:liyufeng854@gmail.com)"
_last = [0.0]


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def get(url, timeout=40):
    wait = 3.1 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            log(f"  retry {attempt} {type(e).__name__}: {str(e)[:80]}")
            time.sleep(10 * (attempt + 1))
    return None


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_query_title(t):
    t = re.sub(r"\(.*?\)", " ", t)          # (authors) / (acronym)
    t = re.sub(r"[（].*?[）]", " ", t)
    t = t.split(",")[0] if re.search(r", [A-Z][a-z]+ et al", t) else t
    t = re.sub(r"^[A-Za-z0-9\-\+]+:\s+", "", t) if len(t.split(":")[0]) < 12 and ":" in t else t  # drop 'ACRONYM: '
    return norm(t)


def parse_entries(xml_bytes):
    root = ET.fromstring(xml_bytes)
    out = []
    for e in root.findall("a:entry", NS):
        aid = e.findtext("a:id", default="", namespaces=NS)
        m = re.search(r"arxiv\.org/abs/(.+?)(v\d+)?$", aid)
        if not m:
            continue
        out.append(dict(
            arxiv_id=m.group(1), version=(m.group(2) or ""),
            title=re.sub(r"\s+", " ", e.findtext("a:title", default="", namespaces=NS)).strip(),
            authors=[x.findtext("a:name", default="", namespaces=NS) for x in e.findall("a:author", NS)],
            abstract=re.sub(r"\s+", " ", e.findtext("a:summary", default="", namespaces=NS)).strip(),
            published=e.findtext("a:published", default="", namespaces=NS)[:10],
            updated=e.findtext("a:updated", default="", namespaces=NS)[:10],
            primary_category=(e.find("arxiv:primary_category", NS).get("term") if e.find("arxiv:primary_category", NS) is not None else ""),
            doi=e.findtext("arxiv:doi", default="", namespaces=NS),
            journal_ref=e.findtext("arxiv:journal_ref", default="", namespaces=NS),
            comment=e.findtext("arxiv:comment", default="", namespaces=NS),
        ))
    return out


def fetch_license(aid):
    url = f"http://export.arxiv.org/oai2?verb=GetRecord&identifier=oai:arXiv.org:{aid}&metadataPrefix=arXiv"
    data = get(url)
    if not data:
        return ""
    try:
        root = ET.fromstring(data)
        lic = root.find(".//arx:license", OAI_NS)
        return lic.text.strip() if lic is not None and lic.text else ""
    except ET.ParseError:
        return ""


def main():
    rows = [json.loads(l) for l in PAPERS.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = json.loads(META.read_text()) if META.exists() else {}
    do_license = "--no-license" not in sys.argv
    # 1) title search for unique rows without arxiv id
    retry = "--retry-kb" in sys.argv
    todo = [r for r in rows if not r["arxiv_id"] and not r["dup_of"] and r["evidence"] != "B"
            and (not r.get("arxiv_searched") or (retry and r.get("kb_title") and r.get("arxiv_searched") != "kb"))]
    log(f"title-search todo={len(todo)}")
    for i, r in enumerate(todo):
        q = clean_query_title(r["kb_title"] if (retry and r.get("kb_title")) else r["title"])
        if len(q) < 8:
            r["arxiv_searched"] = "skip-short"
            continue
        words = [w for w in q.split() if len(w) > 2 and w not in {"the", "and", "for", "with", "via", "from", "into", "under"}][:8]
        if len(words) < 2:
            r["arxiv_searched"] = "skip-short"
            continue
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": " AND ".join(f"ti:{w}" for w in words), "max_results": 8})
        data = get(url)
        r["arxiv_searched"] = "kb" if (retry and r.get("kb_title")) else "done"
        if not data:
            continue
        best, best_s = None, 0.0
        for e in parse_entries(data):
            s = SequenceMatcher(None, q, norm(e["title"])).ratio()
            if s > best_s:
                best, best_s = e, s
        if best and best_s >= 0.82:
            r["arxiv_id"] = best["arxiv_id"]
            r["arxiv_title_ratio"] = round(best_s, 3)
            meta.setdefault(best["arxiv_id"], best)
            log(f"[{i+1}/{len(todo)}] {r['topic']} -> {best['arxiv_id']} ({best_s:.2f}) {r['title'][:60]}")
        else:
            log(f"[{i+1}/{len(todo)}] {r['topic']} NO MATCH ({best_s:.2f}) {r['title'][:60]}")
        if (i + 1) % 10 == 0:
            PAPERS.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
            META.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    # propagate ids to duplicates
    by_key = {r["key"]: r for r in rows}
    for r in rows:
        if r["dup_of"] and not r["arxiv_id"]:
            r["arxiv_id"] = by_key[r["dup_of"]]["arxiv_id"]
    PAPERS.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    # 2) batch metadata for all ids
    ids = sorted({r["arxiv_id"] for r in rows if r["arxiv_id"]})
    need = [a for a in ids if a not in meta or not meta[a].get("abstract")]
    log(f"metadata: ids={len(ids)} need={len(need)}")
    for i in range(0, len(need), 40):
        chunk = need[i:i + 40]
        data = get("http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": ",".join(chunk), "max_results": 40}))
        if not data:
            continue
        for e in parse_entries(data):
            meta[e["arxiv_id"]] = {**meta.get(e["arxiv_id"], {}), **e}
        META.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        log(f"  meta batch {i//40+1}/{(len(need)+39)//40}")
    # 3) licenses (OAI-PMH), one request per id
    if do_license:
        need = [a for a in ids if "license" not in meta.get(a, {})]
        log(f"license: need={len(need)}")
        for i, a in enumerate(need):
            meta.setdefault(a, {})["license"] = fetch_license(a)
            if (i + 1) % 20 == 0:
                META.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
                log(f"  license {i+1}/{len(need)}")
        META.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    n_id = sum(1 for r in rows if r["arxiv_id"] and not r["dup_of"])
    log(f"DONE unique_with_arxiv={n_id}/{sum(1 for r in rows if not r['dup_of'])}")


if __name__ == "__main__":
    main()
