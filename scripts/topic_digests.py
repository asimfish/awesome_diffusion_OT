#!/usr/bin/env python3
"""Per-topic cross-paper digests from deep-dive reports (DeepSeek). Writes data/topic_digests/tNN.json and topics/tNN.md,
plus report/_part6_findings_zh.md. Only (re)generates a topic when >= --min-cov of its papers have reports (default 0.8)
or --force. Usage: DEEPSEEK_API_KEY=... python3 scripts/topic_digests.py [--min-cov 0.8] [--force] [--topics t01,t02]"""
import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.deepseek.com/chat/completions"
KEY = os.environ.get("DEEPSEEK_API_KEY", "")
REL = "https://github.com/asimfish/awesome_diffusion_OT/releases/download"
SECTIONS = {"A": ("理论基础", ["t01", "t02", "t03", "t04", "t05", "t06"]), "B": ("流匹配与轨迹拉直", ["t07", "t08", "t09", "t10", "t11", "t12"]),
            "C": ("跨域生成与翻译", ["t13", "t14", "t15", "t16", "t17", "t18"]), "D": ("模态扩展", ["t19", "t20", "t21", "t22", "t23", "t24"]),
            "E": ("OT 变体前沿", ["t25", "t26", "t27", "t28"]), "F": ("系统、评测与趋势", ["t29", "t30"])}
SYS = """你是「扩散/流生成模型 × 最优传输」知识库的综合分析员。给定一个子课题内若干篇论文的深读摘录（每篇：一句话、§5 地图位置、§6 局限、§7 启发、关键数字），写课题级综合。硬约束：先说结论；每条观察必须引用 ≥2 个 report_id 作证据；不编造摘录之外的数字；不用渲染词；中文，术语保留英文。
只输出 JSON：{"digest_zh": "250–400 字课题综合（这个课题解决什么、方法谱系怎么演进、当前共识与分歧）", "observations": [{"claim": "一句话跨论文观察", "evidence": ["report_id", ...], "detail": "1–2 句具体说明（含出处里的数字或定理）"} ×3–4], "open_problems": [{"problem": "...", "why_open": "...", "evidence": ["report_id"]} ×2–3], "reading_order": [{"report_id": "...", "why": "≤20 字"} ×5]}"""


def sec(txt, n):
    m = re.search(rf"^## {n}\..*?$(.*?)(?=^## |\Z)", txt, re.S | re.M)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def excerpt(rid):
    p = ROOT / "reports" / f"{rid}.md"
    txt = p.read_text(encoding="utf-8")
    one = re.search(r"\*\*一句话\*\*[：:]\s*(.+)", txt)
    card_p = ROOT / "data/meta" / f"{rid}.json"
    card = json.loads(card_p.read_text(encoding="utf-8")) if card_p.exists() else {}
    kn = card.get("key_numbers", [])
    kn = "; ".join(k if isinstance(k, str) else json.dumps(k, ensure_ascii=False) for k in kn[:5])
    return (f"[{rid}] {re.search(r'^# (.+)$', txt, re.M).group(1) if re.search(r'^# (.+)$', txt, re.M) else rid}\n"
            f"一句话: {one.group(1).strip() if one else card.get('tldr_zh','')}\n§5: {sec(txt,5)[:900]}\n§6: {sec(txt,6)[:700]}\n§7: {sec(txt,7)[:500]}\nkey_numbers: {kn[:500]}")


def call(user):
    body = json.dumps(dict(model="deepseek-v4-pro", messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                           temperature=0.2, max_tokens=5000, thinking={"type": "disabled"})).encode()
    req = urllib.request.Request(API, data=body, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=600).read())["choices"][0]["message"]["content"]
    return json.loads(txt[txt.find("{"):txt.rfind("}") + 1])


def render_topic(t, man, dig):
    L = [f"# {t.upper()} · {man['topic_name']}", "", f"> 板块 {man['section']} · {len(man['papers'])} 篇 · 课题背景笔记 [`{man['kb_note']}`](../{man['kb_note']})", ""]
    if dig:
        L += ["## 课题综合", "", dig["digest_zh"], "", "## 跨论文观察", ""]
        for o in dig["observations"]:
            ev = " ".join(f"[`{e}`](../reports/{e}.md)" for e in o["evidence"])
            L.append(f"- **{o['claim']}** {o.get('detail','')} 证据：{ev}")
        L += ["", "## 开放问题", ""]
        for o in dig["open_problems"]:
            L.append(f"- **{o['problem']}** {o.get('why_open','')} ({', '.join(o.get('evidence', []))})")
        L += ["", "## 推荐阅读顺序", ""]
        for i, r in enumerate(dig["reading_order"], 1):
            L.append(f"{i}. [`{r['report_id']}`](../reports/{r['report_id']}.md) — {r['why']}")
        L.append("")
    L += ["## 论文清单", "", "| # | 论文 | venue · year | 证据 | 深读 | PDF | 译文 |", "|---|---|---|---|---|---|---|"]
    for i, p in enumerate(man["papers"], 1):
        rid = p["report_id"]
        rep = f"[report](../reports/{rid}.md)" if (ROOT / f"reports/{rid}.md").exists() else "—"
        stem = p["arxiv_id"].replace("/", "_") if p["arxiv_id"] else None
        pdf = f"[PDF]({REL}/pdf-en-v1/{stem}.pdf)" if stem and (ROOT / f"papers/{stem}.pdf").exists() else "—"
        zh = f"[zh]({REL}/pdf-zh-v1/{stem}.zh.pdf)" if stem and (ROOT / f"papers_zh/{stem}.zh.pdf").exists() else "—"
        link = f"[{p['title'].rstrip('.')}](https://arxiv.org/abs/{p['arxiv_id']})" if p["arxiv_id"] else p["title"]
        L.append(f"| {i} | {'⭐ ' if p['star'] else ''}{link} | {p['venue']} {p['year'] or ''} | [{p['evidence']}] | {rep} | {pdf} | {zh} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cov", type=float, default=0.8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--topics", default="")
    a = ap.parse_args()
    topics = a.topics.split(",") if a.topics else [f"t{i:02d}" for i in range(1, 31)]
    for t in topics:
        man = json.loads((ROOT / f"data/topic_manifest/{t}.json").read_text(encoding="utf-8"))
        have = [p["report_id"] for p in man["papers"] if (ROOT / f"reports/{p['report_id']}.md").exists()]
        cov = len(have) / max(1, len(man["papers"]))
        dp = ROOT / f"data/topic_digests/{t}.json"
        dig = json.loads(dp.read_text(encoding="utf-8")) if dp.exists() else None
        need = (dig is None or dig.get("n_reports", 0) < len(have)) and (cov >= a.min_cov or a.force) and KEY
        if need:
            user = f"课题 {t.upper()} {man['topic_name']}（{len(have)}/{len(man['papers'])} 篇有深读）\n\n" + "\n\n".join(excerpt(r) for r in have)
            try:
                dig = call(user[:120000])
                dig["n_reports"] = len(have)
                dp.write_text(json.dumps(dig, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{t}: digest generated ({len(have)} reports)")
            except Exception as e:  # noqa: BLE001
                print(f"{t}: digest FAILED {type(e).__name__} {str(e)[:80]}")
        else:
            print(f"{t}: cov={cov:.2f} {'digest cached' if dig else 'skipped (low coverage)'}")
        (ROOT / f"topics/{t}.md").write_text(render_topic(t, man, dig), encoding="utf-8")
    # report part 6
    L = ["## 6. 逐篇深读的跨论文发现（按板块）", "", "本节由 30 个课题的深读综合（`topics/tNN.md`）汇总而成：每条观察至少引用两篇深读报告作证据。", ""]
    for s, (name, ts) in SECTIONS.items():
        L += [f"### 6.{'ABCDEF'.index(s)+1} 板块 {s}：{name}", ""]
        for t in ts:
            dp = ROOT / f"data/topic_digests/{t}.json"
            man = json.loads((ROOT / f"data/topic_manifest/{t}.json").read_text(encoding="utf-8"))
            if not dp.exists():
                L += [f"**{t.upper()} {man['topic_name']}**：深读综合待生成（见 `topics/{t}.md` 的论文清单）。", ""]
                continue
            dig = json.loads(dp.read_text(encoding="utf-8"))
            L += [f"**{t.upper()} {man['topic_name']}** — {dig['digest_zh']}", ""]
            for o in dig["observations"]:
                L.append(f"- {o['claim']} {o.get('detail','')}（证据：{', '.join(o['evidence'])}）")
            L.append("")
    (ROOT / "report/_part6_findings_zh.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote report/_part6_findings_zh.md")


if __name__ == "__main__":
    main()
