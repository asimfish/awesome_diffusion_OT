#!/usr/bin/env python3
"""Scan arXiv for recent diffusion x OT papers. Usage: scan_arxiv_recent.py 20260801 20260901 -> trends/arxiv_candidates.jsonl"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
start, end = sys.argv[1], sys.argv[2]
QUERIES = [
    '(abs:"optimal transport" OR abs:Wasserstein OR abs:Sinkhorn OR abs:"Schrodinger bridge" OR abs:"Schrödinger bridge") AND (abs:diffusion OR abs:"flow matching" OR abs:"rectified flow" OR abs:"generative model" OR abs:"consistency model")',
    '(abs:"flow matching" OR abs:"rectified flow" OR abs:"stochastic interpolant") AND (abs:coupling OR abs:"optimal transport" OR abs:straight OR abs:"minibatch")',
    '(abs:"bridge matching" OR abs:"diffusion bridge" OR abs:"Schrodinger bridge" OR abs:"Schrödinger bridge")',
    '(abs:"noise" AND (abs:"initial noise" OR abs:"noise selection" OR abs:"inference-time scaling") AND abs:diffusion)',
    '(abs:"Gromov-Wasserstein" OR abs:"unbalanced optimal transport" OR abs:"semi-discrete optimal transport" OR abs:"Wasserstein gradient flow") AND (abs:generative OR abs:diffusion OR abs:flow)',
    '(abs:"mean flow" OR abs:"flow map" OR abs:"one-step generation") AND (abs:transport OR abs:Wasserstein OR abs:coupling)',
]
seen = {}
for q in QUERIES:
    for st in (0, 100):
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({
            "search_query": f"({q}) AND submittedDate:[{start}0000 TO {end}2359]", "start": st, "max_results": 100,
            "sortBy": "submittedDate", "sortOrder": "descending"})
        time.sleep(3.2)
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "awesome_diffusion_OT/0.1"}), timeout=60).read()
        except Exception as e:  # noqa: BLE001
            print("ERR", type(e).__name__, str(e)[:80]); continue
        root = ET.fromstring(data)
        entries = root.findall("a:entry", NS)
        for e in entries:
            aid = re.search(r"abs/(.+?)(v\d+)?$", e.findtext("a:id", default="", namespaces=NS))
            if not aid:
                continue
            k = aid.group(1)
            if k in seen:
                continue
            seen[k] = dict(arxiv_id=k, title=re.sub(r"\s+", " ", e.findtext("a:title", default="", namespaces=NS)).strip(),
                           abstract=re.sub(r"\s+", " ", e.findtext("a:summary", default="", namespaces=NS)).strip(),
                           authors=[x.findtext("a:name", default="", namespaces=NS) for x in e.findall("a:author", NS)][:6],
                           published=e.findtext("a:published", default="", namespaces=NS)[:10],
                           primary=(e.find("arxiv:primary_category", NS).get("term") if e.find("arxiv:primary_category", NS) is not None else ""),
                           comment=e.findtext("arxiv:comment", default="", namespaces=NS))
        print(f"q{QUERIES.index(q)+1} start={st}: {len(entries)} entries, total unique={len(seen)}")
        if len(entries) < 100:
            break
out = ROOT / "trends" / "arxiv_candidates.jsonl"
out.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in seen.values()), encoding="utf-8")
print("saved", out, len(seen))
