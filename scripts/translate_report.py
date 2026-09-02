#!/usr/bin/env python3
"""Translate report/AWESOME_DIFFUSION_OT_REPORT_zh.md -> report/AWESOME_DIFFUSION_OT_REPORT_en.md with DeepSeek, section by section.
Preserves markdown structure, links, tables, math, numbers and report ids. Usage: DEEPSEEK_API_KEY=... python3 scripts/translate_report.py"""
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.deepseek.com/chat/completions"
KEY = os.environ["DEEPSEEK_API_KEY"]
SRC = ROOT / "report/AWESOME_DIFFUSION_OT_REPORT_zh.md"
DST = ROOT / "report/AWESOME_DIFFUSION_OT_REPORT_en.md"
SYS = """You translate a Chinese technical research report on diffusion models x optimal transport into precise academic English.
Rules: keep ALL markdown structure (headings, tables, lists, links, images, code blocks, LaTeX math) exactly; keep every number, arXiv id, report id, file path and URL unchanged; keep evidence tags [P]/[A]/[R]/[B]; translate Chinese punctuation to English; use standard terminology (optimal transport, coupling, marginal, Schrödinger bridge, flow matching, rectified flow, entropic regularization, Sinkhorn, Wasserstein gradient flow, semi-discrete OT, Laguerre cell, minimax rate, NFE). Style: claim-forward, no hedging filler, no added commentary. Output only the translated markdown."""
HEAD_EN = """---
title: "Diffusion Models × Optimal Transport: Problems, Theory, Classics, Frontier, and What We Can Do"
subtitle: "awesome_diffusion_OT synthesis report (446 papers · 30 topics · per-paper deep reads)"
author: "Yufeng Li (asimfish) · knowledge base and pipeline at github.com/asimfish/awesome_diffusion_OT"
date: "2026-09-01 · v1.0"
---
"""


def call(text):
    body = json.dumps(dict(model="deepseek-v4-pro", messages=[{"role": "system", "content": SYS}, {"role": "user", "content": text}],
                           temperature=0.1, max_tokens=8000, thinking={"type": "disabled"})).encode()
    req = urllib.request.Request(API, data=body, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, type(e).__name__)
    raise RuntimeError("translate failed")


def main():
    md = SRC.read_text(encoding="utf-8")
    body = md.split("---", 2)[2] if md.startswith("---") else md
    parts = re.split(r"(?=^## )", body, flags=re.M)
    chunks = []
    for p in parts:
        if not p.strip():
            continue
        if len(p) > 9000:  # split long sections at ### boundaries
            subs = re.split(r"(?=^### )", p, flags=re.M)
            chunks.extend(s for s in subs if s.strip())
        else:
            chunks.append(p)
    print("chunks", len(chunks))
    with ThreadPoolExecutor(max_workers=4) as ex:
        out = list(ex.map(call, chunks))
    text = HEAD_EN + "\n" + "\n\n".join(o.strip("`").replace("```markdown\n", "", 1).rstrip("`").strip() if o.startswith("```") else o for o in out) + "\n"
    DST.write_text(text, encoding="utf-8")
    print("wrote", DST, len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
