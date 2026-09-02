#!/usr/bin/env python3
"""Data figures for the synthesis report (English labels; PNG for pandoc/xelatex)."""
import json
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/figures"
OUT.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in (ROOT / "data/papers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
uniq = [r for r in rows if not r["dup_of"] and r["year"]]
SEC = {"A": "A Theory", "B": "B Flow matching / straightening", "C": "C Cross-domain translation", "D": "D Modalities", "E": "E OT variants", "F": "F Systems / benchmarks"}
COL = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "D": "#9467bd", "E": "#ff7f0e", "F": "#7f7f7f"}
years = list(range(2013, 2027))

# 1. papers per year stacked by section
fig, ax = plt.subplots(figsize=(10, 4.2))
bottom = np.zeros(len(years))
for s in "ABCDEF":
    c = Counter(r["year"] for r in uniq if r["section"] == s)
    v = np.array([c.get(y, 0) for y in years])
    ax.bar(years, v, bottom=bottom, color=COL[s], label=SEC[s], edgecolor="white", lw=0.5)
    bottom += v
for x, y in zip(years, bottom):
    if y:
        ax.text(x, y + 1, str(int(y)), ha="center", fontsize=8)
ax.set_xticks(years)
ax.set_xticklabels([str(y)[2:] if y < 2020 else str(y) for y in years], fontsize=8)
ax.set_ylabel("papers in the corpus")
ax.set_title("Corpus by year and section (446 unique entries; 2026 partial, cutoff 2026-08)")
ax.legend(fontsize=8, ncol=3, loc="upper left")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "corpus_by_year.png", dpi=170)

# 2. topic x year heatmap
topics = [f"t{i:02d}" for i in range(1, 31)]
M = np.zeros((30, len(years)))
for r in uniq:
    if r["year"] in years:
        M[topics.index(r["topic"]), years.index(r["year"])] += 1
fig, ax = plt.subplots(figsize=(10, 8.5))
im = ax.imshow(M, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(years)))
ax.set_xticklabels([str(y) for y in years], rotation=60, fontsize=8)
ax.set_yticks(range(30))
ax.set_yticklabels([t.upper() for t in topics], fontsize=8)
for i in range(30):
    for j in range(len(years)):
        if M[i, j]:
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7, color="white" if M[i, j] > M.max() * 0.55 else "black")
for k, s in enumerate("ABCDEF"):
    pass
ax.axhline(5.5, color="k", lw=0.6); ax.axhline(11.5, color="k", lw=0.6); ax.axhline(17.5, color="k", lw=0.6); ax.axhline(23.5, color="k", lw=0.6); ax.axhline(27.5, color="k", lw=0.6)
ax.set_title("Papers per topic and year (T01-T06 theory | T07-T12 FM | T13-T18 translation | T19-T24 modalities | T25-T28 OT variants | T29-T30 systems)", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.025)
fig.tight_layout()
fig.savefig(OUT / "topic_year_heatmap.png", dpi=170)

# 3. venue mix (normalized)
def venue_key(v):
    v = v or ""
    for k in ["NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV", "AAAI", "AISTATS", "TMLR", "JMLR", "MICCAI", "Interspeech", "ICASSP", "Nature", "SIAM", "TPAMI", "ACL", "EMNLP", "Medical Image Analysis", "TMI"]:
        if k.lower() in v.lower():
            return k
    if "arxiv" in v.lower() or "预印" in v:
        return "arXiv only"
    return "other journal/venue"
c = Counter(venue_key(r["venue"]) for r in uniq)
items = c.most_common(14)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#1f4e79")
for i, (k, v) in enumerate(items[::-1]):
    ax.text(v + 1, i, str(v), va="center", fontsize=8)
ax.set_title("Venue mix of the corpus")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(OUT / "venue_mix.png", dpi=170)

# 4. evidence level by section
fig, ax = plt.subplots(figsize=(7, 3.6))
evs = "PARB"
bottom = np.zeros(6)
for e, col in zip(evs, ["#1f4e79", "#2e86c1", "#f39c12", "#95a5a6"]):
    v = np.array([sum(1 for r in uniq if r["section"] == s and r["evidence"] == e) for s in "ABCDEF"])
    ax.bar(list("ABCDEF"), v, bottom=bottom, color=col, label={"P": "[P] proceedings", "A": "[A] accepted", "R": "[R] preprint", "B": "[B] book/survey"}[e])
    bottom += v
ax.set_title("Evidence level by section")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "evidence_by_section.png", dpi=170)
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
