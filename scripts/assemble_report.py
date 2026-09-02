#!/usr/bin/env python3
"""Concatenate report/_part*_zh.md in order into report/AWESOME_DIFFUSION_OT_REPORT_zh.md; fix dangling reports/<id>.md links."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "report"
order = ["_part0_intro_zh.md", "_part1_corpus_zh.md", "_part2_problems_zh.md", "_part3_theory_zh.md", "_part4_canon_zh.md",
         "_part5_frontier_zh.md", "_part6_findings_zh.md", "_part7_insights_zh.md", "_part8_action_zh.md", "_part9_appendix_zh.md"]
text = "\n\n".join((R / p).read_text(encoding="utf-8").strip() for p in order) + "\n"
# report links: `reports/X.md` (inline code) -> markdown link to repo path if exists, else plain code
missing = set()
def fix(m):
    rid = m.group(1)
    if rid.startswith("<"):
        return m.group(0)
    if (ROOT / "reports" / f"{rid}.md").exists():
        return f"[`reports/{rid}.md`](../reports/{rid}.md)"
    missing.add(rid)
    return f"`{rid}`（深读见 README 对应条目）"
text = re.sub(r"`reports/([^`]+?)\.md`", fix, text)
# bare ids in 6.x evidence parentheses are fine. Write.
(R / "AWESOME_DIFFUSION_OT_REPORT_zh.md").write_text(text, encoding="utf-8")
print("lines", len(text.splitlines()), "chars", len(text), "missing report links:", sorted(missing))
