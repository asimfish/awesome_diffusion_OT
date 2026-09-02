## 9. 写作红线与附录

### 9.1 写作红线（从 446 篇的反例与失败案例总结）

1. 不说「我们的方法实现了最优传输」——说「以 OT 为设计目标并度量偏差」（I1）。
2. 少步比较必须固定 NFE 口径并报告多样性指标（FID 对少步失真；DDBM 的 N=40 实为 NFE 118）。
3. 会议归属与「已接收」表述必须官方可核验；预印本结论一律 [R] 限定。
4. 任何实例级耦合修改必须报告边缘漂移/多样性（改耦合 vs 保边缘张力）——本方法（MPNA）的核心卖点恰是这一项为零，要拿数据证明而不是引 Lemma。
5. 条件生成中的 OT 声明必须过 C²OT 检查；语义对应线须消融证明「耦合结构本身带来增益」而非 attention 重命名。
6. 桥模型横向比较前先统一评测协议（E→H/DIODE 指标是否在训练集上算、NFE 口径）。

### 9.2 证据底座

- 446 篇（30 课题、6 板块；⭐ 核心 139）；345 篇 arXiv id；原文 PDF 352 份、全文文本 352 份；逐篇深读报告见 `reports/`（文件名 = arXiv id），元数据卡 `data/meta/`。
- 审计链：`source/audit/ARIS_AUDIT_20260814.md`（三层审计）→ `INCREMENTAL_REVIEW_20260825.md`（触发点复审）→ `trends/TRENDS_2026Q3.md`（Q3 增量，62 篇）。
- 沙盒：`~/Code/mpna/sandbox/results/results.json`（5 seeds × 4096，与闭式理论逐点吻合）。
- 翻译 QA：`papers_zh/*.inspect.json`（对象级审计：漏翻、保护区改动、重叠、图片丢失）。

### 9.3 文档谱系与复现

`source/REPORT_DIFFUSION_OT_20260814.md`（调研收口）→ `source/SYNTHESIS_DIFFUSION_OT_20260825.md`（综合 v0）→ **本报告**（综合 v1：+逐篇深读 +Q3 增量 +MPNA）→ `slides/`（HTML PPT 与 Beamer PDF）。

复现整条流水线：

```bash
python3 scripts/build_corpus.py          # 引用库 → data/papers.jsonl
python3 scripts/resolve_arxiv.py         # arXiv id / 元数据 / 许可证
python3 scripts/fetch_papers.py          # 原文 PDF → papers/
python3 scripts/extract_text.py          # 全文文本 → data/text/
python3 scripts/build_manifests.py       # 课题清单
DEEPSEEK_API_KEY=... python3 scripts/gen_reports_llm.py   # 逐篇深读 → reports/, data/meta/
bash scripts/translate_batch.sh --loop   # 保版式中文译文 → papers_zh/
python3 scripts/scan_arxiv_recent.py 20260801 20260901 && python3 scripts/classify_candidates.py   # 趋势扫描
python3 src/generator.py                 # README.md / README_zh.md
bash scripts/build_pdf.sh                # 本报告 PDF（zh/en）
```

*报告完。2026-09-01。*
