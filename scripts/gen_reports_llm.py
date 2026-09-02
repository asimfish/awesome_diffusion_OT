#!/usr/bin/env python3
"""Generate per-paper deep-dive reports (Chinese) + meta cards with the DeepSeek API, same template/discipline as
source/AGENT_BRIEF_REPORTS.md. Resumable; skips existing reports/<rid>.md.
Usage: DEEPSEEK_API_KEY=... python3 scripts/gen_reports_llm.py [--topics t16,t17] [--limit N] [--workers 4] [--max-chars 45000]
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com") + "/chat/completions"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LOG = ROOT / "logs" / "gen_reports.log"

SYSTEM = """你是一位严谨的机器学习研究员，负责为「扩散/流生成模型 × 最优传输」知识库写逐篇中文深读报告。硬约束：
1. 证据：所有数字、定理、结论必须来自给定的论文原文；每个数字标注出处（Table N / Eq.(N) / Sec. N / p.N）；原文里读不到的写「原文截断，未见」或「原文未读，未见」，禁止凭记忆补数字。
2. 说人话：先说结论；删开场套话与价值拔高词（「显著」「有效」「赋能」「深远意义」）；数字和它修饰的对象一起保留；同一对象全篇一个叫法；术语、模型名、数据集名保留英文；公式用 $...$。
3. 反防御写作：段首直接给论断；限制只写在 §6，不散落。
4. 不改事实：范围、条件、否定、情态都算事实；摘要说「潜力」不能写成「实现了」。
5. venue 纪律：主会/期刊/workshop/预印本分开写；预印本 [R] 的结论用「作者报告」限定。
6. 输出格式：先输出 Markdown 报告（严格 8 节，标题固定），然后一行 <<<META>>>，然后一个 JSON 对象，然后一行 <<<END>>>。JSON 字段：report_id, arxiv_id, title, tldr_zh(≤40字), tldr_en(≤25 words), tags(list), code_url, key_numbers(list, 每条带出处), relations(list of {report_id,type}), read_full_text(bool)。不要输出其他内容。"""

TEMPLATE = """# <英文标题>

> <作者（前 3 位 + et al.）> · <venue year> · [arXiv](https://arxiv.org/abs/<id>) · 证据级 [<P/A/R/B>] · 课题 <TNN 名称>
> **一句话**：<这篇论文做了什么、结果是什么，≤40 字，先说结论>

## 1. 问题
它解决什么问题；此前方法为什么不够（1–2 段）。
## 2. 方法
核心思想；关键公式 ≤3 个（标注原文编号）；算法步骤或训练/采样流程。
## 3. 理论结果
定理/引理/保证，写清假设与结论；没有则写「无理论结果」。
## 4. 实验与数字
数据集、基线、关键数值（写成表），每个数字标注来源。
## 5. 在 OT×扩散地图中的位置
与本课题及其他课题哪些工作的关系（继承/竞争/被取代）；对应哪个理论张力或推理管线环节。
## 6. 局限与批评
作者承认的 + 你读出来的（各 1–3 条，具体到设置或假设）。
## 7. 对我们的启发
1–3 条可操作建议（可接切入点：#1 免训练 batch 级保边缘噪声指派 MPNA、#2 OT-aware 采样调度、#3 保耦合蒸馏、#7 医学 SB 刷 SynthRAD 等）。
## 8. 资源
代码链接（有则给，无则写「未公开」）；相关论文 arXiv id 互链。"""


def log(m):
    line = time.strftime("%H:%M:%S ") + m
    print(line, flush=True)
    LOG.open("a").write(line + "\n")


def call(messages, max_tokens=8000, temperature=0.2):
    body = json.dumps(dict(model=MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature,
                           thinking={"type": os.environ.get("DEEPSEEK_THINKING", "disabled")})).encode()
    req = urllib.request.Request(API, data=body, headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                j = json.loads(r.read())
                return j["choices"][0]["message"]["content"], j.get("usage", {})
        except Exception as e:  # noqa: BLE001
            log(f"  api retry {attempt}: {type(e).__name__} {str(e)[:100]}")
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("api failed")


def topic_context(kb_path, limit=6000):
    txt = (ROOT / kb_path).read_text(encoding="utf-8")
    keep = []
    for sec in ("## 1.", "## 3.", "## 5."):
        i = txt.find(sec)
        if i >= 0:
            j = txt.find("\n## ", i + 5)
            keep.append(txt[i:j if j > 0 else None])
    return "\n".join(keep)[:limit]


def build_user(p, man, ctx, max_chars):
    meta = (f"report_id: {p['report_id']}\narxiv_id: {p['arxiv_id']}\ntitle: {p['title']}\nauthors: {p['authors']}\n"
            f"venue/year: {p['venue']} {p['year']}\nevidence: [{p['evidence']}]\ntopic: {man['topic'].upper()} {man['topic_name']}\n"
            f"links: {json.dumps(p['links'], ensure_ascii=False)}\n调研 agent 给的一句话贡献: {p['contribution_zh']}\nabstract: {p['abstract']}")
    if p["text"] and (ROOT / p["text"]).exists():
        body = (ROOT / p["text"]).read_text(encoding="utf-8")[:max_chars]
        mode = "有全文（PyMuPDF 抽取，可能截断）。请写完整深读报告。"
    else:
        body = "（无全文）"
        mode = "无全文：只依据 abstract 与一句话贡献写「简报卡」，报告标题下第一行加「⚠ 未读全文，依据摘要」，§3/§4 只能写摘要里有的内容，不得编造数字。"
    return f"""【任务模式】{mode}

【论文元数据】
{meta}

【课题背景（用于 §5 与 §7）】
{ctx}

【模板（严格遵守 8 节标题）】
{TEMPLATE}

【论文全文】
{body}
"""


def parse(out, p):
    if "<<<META>>>" not in out:
        raise ValueError("no META block")
    md, rest = out.split("<<<META>>>", 1)
    js = rest.split("<<<END>>>")[0].strip()
    js = js.strip("`").replace("```json", "").replace("```", "").strip()
    card = json.loads(js)
    card["report_id"] = p["report_id"]
    card.setdefault("arxiv_id", p["arxiv_id"])
    md = md.strip() + "\n"
    if len(re.findall(r"^## ", md, re.M)) < 8:
        raise ValueError("sections<8")
    return md, card


def work(p, man, ctx, max_chars):
    rid = p["report_id"]
    out_md = ROOT / "reports" / f"{rid}.md"
    if out_md.exists():
        return rid, "skip", {}
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_user(p, man, ctx, max_chars)}]
    for attempt in range(2):
        out, usage = call(msgs)
        try:
            md, card = parse(out, p)
            break
        except Exception as e:  # noqa: BLE001
            log(f"  parse fail {rid}: {e}; retry")
            msgs.append({"role": "assistant", "content": out})
            msgs.append({"role": "user", "content": "输出格式不合规：必须先给严格 8 节 Markdown，再一行 <<<META>>>，再 JSON，再 <<<END>>>。请重新完整输出。"})
    else:
        return rid, "fail", {}
    out_md.write_text(md, encoding="utf-8")
    (ROOT / "data/meta" / f"{rid}.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    return rid, "ok", usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=45000)
    a = ap.parse_args()
    if not KEY:
        sys.exit("DEEPSEEK_API_KEY missing")
    topics = a.topics.split(",") if a.topics else [f"t{i:02d}" for i in range(1, 31)]
    jobs = []
    for t in topics:
        man = json.loads((ROOT / f"data/topic_manifest/{t}.json").read_text(encoding="utf-8"))
        ctx = topic_context(man["kb_note"])
        for p in man["papers"]:
            if not (ROOT / "reports" / f"{p['report_id']}.md").exists():
                jobs.append((p, man, ctx))
    if a.limit:
        jobs = jobs[:a.limit]
    log(f"jobs={len(jobs)} workers={a.workers} model={MODEL}")
    tin = tout = 0
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, p, man, ctx, a.max_chars): p for p, man, ctx in jobs}
        for f in as_completed(futs):
            p = futs[f]
            try:
                rid, st, usage = f.result()
            except Exception as e:  # noqa: BLE001
                rid, st, usage = p["report_id"], f"error {type(e).__name__} {str(e)[:60]}", {}
            tin += usage.get("prompt_tokens", 0)
            tout += usage.get("completion_tokens", 0)
            n_ok += st == "ok"
            n_fail += st not in ("ok", "skip")
            log(f"{st} {rid} [{p['title'][:50]}] tokens in={tin} out={tout}")
    log(f"DONE ok={n_ok} fail={n_fail} tokens_in={tin} tokens_out={tout}")


if __name__ == "__main__":
    main()
