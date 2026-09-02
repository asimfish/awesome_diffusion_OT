#!/usr/bin/env python3
"""Generate README.md (English) and README_zh.md (Chinese) from data/ (awesome-ml4co style).

Sources: data/papers.jsonl, data/arxiv_meta.json, data/meta/*.json (agent cards), reports/*.md, papers/, papers_zh/,
trends/new_papers_2026Q3.jsonl (optional). Deterministic; run: python3 src/generator.py
"""
import json
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(l) for l in (ROOT / "data/papers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
meta = json.loads((ROOT / "data/arxiv_meta.json").read_text()) if (ROOT / "data/arxiv_meta.json").exists() else {}
cards = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "data/meta").glob("*.json")}
manifests = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "data/topic_manifest").glob("t*.json")}
trends_p = ROOT / "trends/new_papers_2026Q3.jsonl"
new_papers = [json.loads(l) for l in trends_p.read_text(encoding="utf-8").splitlines() if l.strip()] if trends_p.exists() else []

SECTIONS = OrderedDict([
    ("A", ("Theory Foundations", "理论基础", ["t01", "t02", "t03", "t04", "t05", "t06"])),
    ("B", ("Flow Matching and Trajectory Straightening", "流匹配与轨迹拉直", ["t07", "t08", "t09", "t10", "t11", "t12"])),
    ("C", ("Cross-Domain Generation and Translation", "跨域生成与翻译", ["t13", "t14", "t15", "t16", "t17", "t18"])),
    ("D", ("Modalities", "模态扩展", ["t19", "t20", "t21", "t22", "t23", "t24"])),
    ("E", ("OT Variants Frontier", "OT 变体前沿", ["t25", "t26", "t27", "t28"])),
    ("F", ("Systems, Benchmarks and Trends", "系统、评测与趋势", ["t29", "t30"])),
])
TOPIC_EN = {
    "t01": "OT Mathematical Foundations", "t02": "Diffusion-OT Theory", "t03": "Schrodinger Bridges for Generation",
    "t04": "Entropic OT and Sinkhorn in Generative Modeling", "t05": "Wasserstein Gradient Flows and JKO Schemes",
    "t06": "Convergence and Statistical Theory", "t07": "Flow Matching Foundations", "t08": "OT-CFM and Minibatch OT Coupling",
    "t09": "Rectified Flow and Trajectory Straightening", "t10": "Consistency Models and Few-Step Distillation",
    "t11": "Training-Free Samplers and ODE Solvers", "t12": "Inference-Time OT Alignment and Noise Coupling",
    "t13": "Neural OT Maps and Unpaired Translation", "t14": "Diffusion / Schrodinger Bridges for I2I",
    "t15": "Medical Modality Transfer", "t16": "OT-Guided Semantic Correspondence", "t17": "Style Transfer and Domain Adaptation",
    "t18": "Conditional Generation and Guidance as OT", "t19": "Video Generation and Temporal Consistency",
    "t20": "3D / Point Cloud / Geometry", "t21": "Molecules and Scientific Computing", "t22": "Discrete Data and Text",
    "t23": "Speech and Audio", "t24": "Single-Cell and Biological Trajectories", "t25": "Unbalanced and Partial OT",
    "t26": "Gromov-Wasserstein and Cross-Space Alignment", "t27": "Multi-Marginal OT and Wasserstein Barycenters",
    "t28": "Riemannian Flow Matching", "t29": "High-Performance OT Solvers and Infrastructure",
    "t30": "Edge Deployment, Benchmarks and Venue Trends",
}
REL = "https://github.com/asimfish/awesome_diffusion_OT/releases/download"
EV = {"P": "proceedings", "A": "accepted", "R": "preprint", "B": "book/survey"}
EV_ZH = {"P": "论文集", "A": "已接收", "R": "预印本", "B": "教材/综述"}


def venue_label(r):
    v = r["venue"] or ""
    v = re.sub(r"\s*\(PMLR[^)]*\)|,?\s*PMLR\s*\d+[:\d\-–]*", "", v).strip(" ,·")
    y = r["year"] or ""
    if r["evidence"] == "R" and not re.search(r"arxiv", v, re.I):
        v = (v + " (preprint)").strip()
    return f"{v}, {y}." if v else f"{y}."


def item_links(r, rid):
    links = []
    a = r["arxiv_id"]
    if a:
        links.append(f"[paper](https://arxiv.org/abs/{a})")
    else:
        for name, url in list(r["links"].items())[:1]:
            links.append(f"[paper]({url})")
    for name, url in r["links"].items():
        if re.search(r"code|github", name, re.I) or "github.com" in url:
            links.append(f"[code]({url})")
            break
    if (ROOT / f"reports/{rid}.md").exists():
        links.append(f"[report](reports/{rid}.md)")
    stem = a.replace("/", "_") if a else None
    if stem and (ROOT / f"papers/{stem}.pdf").exists():
        links.append(f"[PDF]({REL}/pdf-en-v1/{stem}.pdf)")
    if stem and (ROOT / f"papers_zh/{stem}.zh.pdf").exists():
        links.append(f"[zh-PDF]({REL}/pdf-zh-v1/{stem}.zh.pdf)")
    return " ".join(links)


def report_id_of(r):
    a = r["arxiv_id"]
    return a.replace("/", "_") if a else re.sub(r"[^A-Za-z0-9]+", "_", r["title"])[:50].strip("_")


def title_of(r):
    a = r["arxiv_id"]
    t = (meta.get(a, {}).get("title") if a else None) or r.get("kb_title") or r["title"]
    return t.rstrip(".")


def authors_of(r):
    a = r["arxiv_id"]
    au = meta.get(a, {}).get("authors") if a else None
    if au:
        return ", ".join(au[:6]) + (" et al." if len(au) > 6 else "")
    return r["authors"]


def render(lang):
    zh = lang == "zh"
    uniq = [r for r in rows if not r["dup_of"]]
    n_reports = len(list((ROOT / "reports").glob("*.md")))
    n_pdf = len(list((ROOT / "papers").glob("*.pdf")))
    n_zh = len(list((ROOT / "papers_zh").glob("*.zh.pdf")))
    n_star = sum(1 for r in uniq if r["star"])
    L = []
    if zh:
        L += ["# Awesome Diffusion × Optimal Transport", "",
              "[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) "
              f"[![Papers](https://img.shields.io/badge/papers-{len(uniq)}-orange.svg)](#content) [![Reports](https://img.shields.io/badge/deep--dive%20reports-{n_reports}-green.svg)](#reports) "
              f"[![zh-PDF](https://img.shields.io/badge/translated%20PDFs-{n_zh}-red.svg)](#reports)", "",
              "[English](README.md) | [中文](README_zh.md)", "",
              "**扩散/流生成模型 × 最优传输**的证据优先阅读清单：30 个子课题、六大板块，覆盖理论（扩散≟OT、Schrödinger 桥、收敛率）、",
              "流匹配与轨迹拉直、跨域翻译、多模态、OT 变体与系统基建。每篇论文附：",
              "", "- 中文**深读报告**（`reports/`，8 节模板：问题 / 方法 / 理论 / 实验数字 / 地图位置 / 局限 / 启发 / 资源，数字带出处）；",
              "- **原文 PDF**（`papers/`）与 [SuperTranslate](https://github.com/asimfish/super_translate) **保版式中文译文**（`papers_zh/`，附对象级 QA `*.inspect.json`）；",
              "- 机器可读元数据（`data/`），本 README 由 `src/generator.py` 生成。", "",
              f"总量：{len(uniq)} 篇（⭐ 核心 {n_star}）| 深读报告 {n_reports} | 原文 PDF {n_pdf} | 中文译文 {n_zh}。证据级：[P] 论文集 / [A] 官方已接收 / [R] 预印本 / [B] 教材综述——主会、期刊、workshop、预印本永远分开标。",
              "", "综合分析见 [`report/`](report/)（问题→理论→经典→前沿→我们能做什么，中英 PDF）与 [`slides/`](slides/)（HTML PPT + Beamer PDF）；2026 Q3 增量趋势见 [`trends/`](trends/)。",
              "", "*维护：[asimfish](https://github.com/asimfish)。结构参考 [awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co)。*", ""]
    else:
        L += ["# Awesome Diffusion × Optimal Transport", "",
              "[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) "
              f"[![Papers](https://img.shields.io/badge/papers-{len(uniq)}-orange.svg)](#content) [![Reports](https://img.shields.io/badge/deep--dive%20reports-{n_reports}-green.svg)](#reports) "
              f"[![zh-PDF](https://img.shields.io/badge/translated%20PDFs-{n_zh}-red.svg)](#reports)", "",
              "[English](README.md) | [中文](README_zh.md)", "",
              "An evidence-first reading list on **diffusion / flow generative models × optimal transport (OT)**: 30 sub-topics in six sections,",
              "from theory (is diffusion secretly OT? Schrodinger bridges, convergence rates) through flow matching and trajectory straightening,",
              "cross-domain translation and modalities, to OT variants and solver infrastructure. Every paper comes with:", "",
              "- a **deep-dive report** in Chinese (`reports/`, 8 fixed sections: problem / method / theory / numbers with sources / position on the map / limitations / takeaways / resources);",
              "- the **original PDF** (`papers/`) and a **layout-preserving Chinese translation** by [SuperTranslate](https://github.com/asimfish/super_translate) with object-level QA (`papers_zh/`, `*.inspect.json`);",
              "- machine-readable metadata (`data/`); this README is generated by `src/generator.py`.", "",
              f"Totals: {len(uniq)} papers (⭐ core {n_star}) | {n_reports} deep-dive reports | {n_pdf} original PDFs | {n_zh} Chinese translations. "
              "Evidence levels: [P] proceedings / [A] officially accepted / [R] preprint / [B] book or survey. Main conference, journal, workshop and preprint are always labelled separately.",
              "", "Synthesis: [`report/`](report/) (problem → theory → classics → frontier → what we can do; zh/en PDF) and [`slides/`](slides/) (HTML deck + Beamer PDF). Q3-2026 trend scan: [`trends/`](trends/).",
              "", "*Maintained by [asimfish](https://github.com/asimfish). Structure follows [awesome-ml4co](https://github.com/Thinklab-SJTU/awesome-ml4co).*", ""]
    # TOC
    L += ["## Content" if not zh else "## 目录", ""]
    for sid, (en, zhn, topics) in SECTIONS.items():
        L.append(f"{sid}. [{zhn if zh else en}](#sec-{sid.lower()})")
        for t in topics:
            name = manifests[t]["topic_name"] if zh else TOPIC_EN[t]
            L.append(f"&emsp;{t.upper()} [{name}](#{t})")
    L += [f"G. [{'2026 Q3 增量与趋势' if zh else 'Q3-2026 Additions and Trends'}](#trends)",
          f"H. [{'深读报告、译文与综合报告' if zh else 'Reports, Translations and Synthesis'}](#reports)",
          f"I. [{'贡献与引用' if zh else 'Contributing and Citation'}](#contributing)", ""]
    # sections
    for sid, (en, zhn, topics) in SECTIONS.items():
        L += [f'<a id="sec-{sid.lower()}"></a>', f"## {sid}. {zhn if zh else en}", ""]
        for t in topics:
            man = manifests[t]
            L += [f'<a id="{t}"></a>', f"### {t.upper()}. {man['topic_name'] if zh else TOPIC_EN[t]}", ""]
            kb = f"source/kb/{Path(man['kb_note']).name}"
            L += [(f"课题综合：[`topics/{t}.md`](topics/{t}.md)（跨论文观察 / 开放问题 / 阅读顺序）· 课题笔记：[`{kb}`]({kb})" if zh
                   else f"Topic digest: [`topics/{t}.md`](topics/{t}.md) (cross-paper observations / open problems / reading order) · topic note: [`{kb}`]({kb})"), ""]
            trs = sorted([r for r in rows if r["topic"] == t and not r["dup_of"]], key=lambda r: (not r["star"], -(r["year"] or 0), r["title"]))
            for i, r in enumerate(trs, 1):
                rid = report_id_of(r)
                star = "⭐ " if r["star"] else ""
                L.append(f"{i}. {star}**{title_of(r)}.** {venue_label(r)} [{r['evidence']}] {item_links(r, rid)}")
                au = authors_of(r)
                if au:
                    L.append(f"\n    *{au}*")
                if zh:
                    L.append(f"\n    {r['contribution_zh']}")
                else:
                    c = cards.get(rid, {})
                    if c.get("tldr_en"):
                        L.append(f"\n    {c['tldr_en']}")
                L.append("")
            xl = [r for r in rows if r["topic"] == t and r["dup_of"]]
            if xl:
                L.append(("另见（跨课题重复）：" if zh else "See also (cross-listed): ") + "; ".join(f"{title_of(r)} → {r['dup_of'].split('__')[0].upper()}" for r in xl))
                L.append("")
    # trends
    L += ['<a id="trends"></a>', f"## G. {'2026 Q3 增量与趋势' if zh else 'Q3-2026 Additions and Trends'}", ""]
    if new_papers:
        L.append(("扫描日期 2026-09-01；完整分析见 [`trends/TRENDS_2026Q3.md`](trends/TRENDS_2026Q3.md)。" if zh else "Scanned 2026-09-01; full analysis in [`trends/TRENDS_2026Q3.md`](trends/TRENDS_2026Q3.md)."))
        L.append("")
        for i, p in enumerate(sorted(new_papers, key=lambda p: p.get("date", ""), reverse=True), 1):
            L.append(f"{i}. **{p['title'].rstrip('.')}.** arXiv {p.get('date','')[:7]} [{p.get('evidence','R')}] → {p.get('topic','').upper()}. [paper](https://arxiv.org/abs/{p['arxiv_id']})")
            L.append(f"\n    {p.get('tldr_zh','') if zh else p.get('why_relevant', p.get('tldr_zh',''))}")
            L.append("")
    else:
        L += [("趋势扫描进行中，见 `trends/`。" if zh else "Trend scan in progress; see `trends/`."), ""]
    # reports / deliverables
    L += ['<a id="reports"></a>', f"## H. {'深读报告、译文与综合报告' if zh else 'Reports, Translations and Synthesis'}", ""]
    if zh:
        L += [f"- `reports/`：{n_reports} 份逐篇深读（文件名 = arXiv id）；`data/meta/`：每篇的 TL;DR / 关键数字 / 关系卡。",
              f"- `papers/`：{n_pdf} 份原文；`papers_zh/`：{n_zh} 份保版式中文译文 + `*.inspect.json` QA 报告。缺失的译文在持续补齐（`scripts/translate_batch.sh`）。",
              "- `report/`：综合分析报告（`AWESOME_DIFFUSION_OT_REPORT_zh.md` / `_en.md` 及 PDF）。",
              "- `slides/`：[HTML PPT](slides/awesome_diffusion_OT_deck.html)（浏览器打开，方向键翻页）与 [Beamer PDF](slides/beamer/awesome_diffusion_OT_slides.pdf)；报告 PDF：[中文](report/pdf/awesome_diffusion_ot_report_zh.pdf) / [English](report/pdf/awesome_diffusion_ot_report_en.pdf)。",
              "- 复现整条流水线：`scripts/build_corpus.py → resolve_arxiv.py → fetch_papers.py → extract_text.py → translate_batch.sh → src/generator.py`。", ""]
    else:
        L += [f"- `reports/`: {n_reports} per-paper deep dives (file name = arXiv id); `data/meta/`: TL;DR / key numbers / relation cards.",
              f"- `papers/`: {n_pdf} originals; `papers_zh/`: {n_zh} layout-preserving Chinese translations with `*.inspect.json` QA. Missing translations are being filled (`scripts/translate_batch.sh`).",
              "- `report/`: synthesis report (`AWESOME_DIFFUSION_OT_REPORT_zh.md` / `_en.md` and PDFs).",
              "- `slides/`: [HTML deck](slides/awesome_diffusion_OT_deck.html) (open in a browser, arrow keys) and [Beamer PDF](slides/beamer/awesome_diffusion_OT_slides.pdf); report PDFs: [中文](report/pdf/awesome_diffusion_ot_report_zh.pdf) / [English](report/pdf/awesome_diffusion_ot_report_en.pdf).",
              "- Reproduce the pipeline: `scripts/build_corpus.py → resolve_arxiv.py → fetch_papers.py → extract_text.py → translate_batch.sh → src/generator.py`.", ""]
    L += ['<a id="contributing"></a>', f"## I. {'贡献与引用' if zh else 'Contributing and Citation'}", ""]
    if zh:
        L += ["欢迎 PR：在 `data/papers.jsonl` 或对应课题笔记加入条目（标题 / venue / 证据级 / 链接），运行 `python3 src/generator.py` 重新生成 README。PDF 与译文仅供个人学习研究，版权归原作者与出版方。", "",
              "```bibtex", "@misc{awesome_diffusion_ot_2026,", "  title  = {Awesome Diffusion x Optimal Transport},", "  author = {Li, Yufeng},", "  year   = {2026},",
              "  url    = {https://github.com/asimfish/awesome_diffusion_OT}", "}", "```", ""]
    else:
        L += ["PRs welcome: add an entry to `data/papers.jsonl` (title / venue / evidence level / links) and run `python3 src/generator.py`. PDFs and translations are provided for personal study; copyright remains with the authors and publishers.", "",
              "```bibtex", "@misc{awesome_diffusion_ot_2026,", "  title  = {Awesome Diffusion x Optimal Transport},", "  author = {Li, Yufeng},", "  year   = {2026},",
              "  url    = {https://github.com/asimfish/awesome_diffusion_OT}", "}", "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    (ROOT / "README.md").write_text(render("en"), encoding="utf-8")
    (ROOT / "README_zh.md").write_text(render("zh"), encoding="utf-8")
    print("README.md", len(render("en").splitlines()), "lines; README_zh.md", len(render("zh").splitlines()), "lines")
