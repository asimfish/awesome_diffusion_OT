# 逐篇深读报告 · 子代理作业规范（awesome_diffusion_OT）

## 你的任务
给定一个课题清单 `data/topic_manifest/tNN.json`，为其中每篇论文写一份中文深读报告到 `reports/<report_id>.md`，并写一张元数据卡到 `data/meta/<report_id>.json`。
- `text` 字段非空 → **深读报告**（读 `data/text/<stem>.txt` 全文；先读前 30k 字符，需要数字/定理时再按 offset 读后文）。
- `text` 为空 → **简报卡**（只依据 `abstract` 与 `contribution_zh`），报告顶部标注「⚠ 未读全文，依据摘要」，不得编造实验数字。
- 若 `reports/<report_id>.md` 已存在（跨课题重复），跳过。

## 报告模板（严格按此 8 节；中文；术语/模型名/数据集名保留英文；公式用 `$...$`）
```
# <英文标题>

> <作者（前 3 位 + et al.）> · <venue year> · [arXiv](https://arxiv.org/abs/<id>) · 证据级 [P/A/R/B] · 课题 <TNN 名称>
> **一句话**：<这篇论文做了什么、结果是什么，≤ 40 字，先说结论>

## 1. 问题
它解决什么问题；此前方法为什么不够（1–2 段）。
## 2. 方法
核心思想；关键公式 ≤ 3 个（标注原文编号如 Eq.(4)）；算法步骤或训练/采样流程。
## 3. 理论结果
定理/引理/保证，写清假设与结论；没有则写「无理论结果」。
## 4. 实验与数字
数据集、基线、关键数值（写成表），每个数字标注来源（Table N / Sec. N / p.N）。全文被截断读不到的写「原文截断，未见」。
## 5. 在 OT×扩散地图中的位置
与本课题及其他课题哪些工作的关系（继承/竞争/被取代）；对应综合报告的哪个张力或推理管线的哪个环节。
## 6. 局限与批评
作者承认的 + 你读出来的（各 1–3 条，具体到设置或假设）。
## 7. 对我们的启发
1–3 条可操作建议（可接 Top-10 切入点：#1 保边缘噪声指派 MPNA、#2 OT-aware 调度、#3 保耦合蒸馏、#7 医学 SB 等）。
## 8. 资源
代码链接（有则给，无则写「未公开」）；相关论文 report_id 互链。
```

## 元数据卡 `data/meta/<report_id>.json`
```json
{"report_id": "...", "arxiv_id": "...", "title": "...", "tldr_zh": "≤40字", "tldr_en": "<= 25 words",
 "tags": ["schrodinger-bridge", "theory"], "code_url": "" , "key_numbers": ["FID 1.98 on CIFAR-10 (Table 2)"],
 "relations": [{"report_id": "...", "type": "extends|competes|uses|superseded_by"}], "read_full_text": true}
```

## 写作纪律（硬约束）
1. **证据**：所有数字、定理、结论必须来自原文；标注出处；读不到就写读不到。禁止凭记忆补数字。
2. **说人话**：先说结论；删开场套话与价值拔高（「显著」「有效」「赋能」）；数字和它修饰的对象一起保留；一个对象全篇一个叫法。
3. **反防御写作**：段首直接给论断；限制只写在 §6，不散落在各节。
4. **不改事实**：范围、条件、否定、情态都算事实；摘要说「潜力」不能写成「实现了」。
5. **venue 纪律**：主会/期刊/workshop/预印本分开写；`[R]` 预印本的结论用「作者报告」限定。

## 文件写入（硬约束）
- **含中文的文件一律用 Shell heredoc 写入**：`cat > reports/x.md <<'EOF' ... EOF`。**禁止**用 Write / StrReplace 工具写含中文的文件（会把中文写坏）。
- 每写完一批，运行 `python3 scripts/check_utf8.py reports/<...>.md data/meta/<...>.json` 校验；WARN/BAD 必须重写。
- JSON 用 `python3 - <<'EOF'` + `json.dump(..., ensure_ascii=False)` 写。

## 完成汇报
最后输出：写了几份深读 / 几份简报、跳过几篇、校验结果、以及本课题 3 条最重要的跨论文观察（一句话各）。
