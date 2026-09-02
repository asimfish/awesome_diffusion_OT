#!/usr/bin/env python3
"""Download original PDFs into papers/<arxiv_id>.pdf (arXiv, 3s spacing) and copy non-arXiv PDFs from the survey dir.

Writes data/papers_manifest.json {file: {sha256, bytes, source}}. Resumable. Skips iCloud-evicted (dataless) sources.
"""
import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SURVEY = Path.home() / "Desktop/research/diffusion_ot_survey/papers"
PAPERS = ROOT / "data" / "papers.jsonl"
OUT = ROOT / "papers"
MAN = ROOT / "data" / "papers_manifest.json"
LOG = ROOT / "logs" / "fetch_papers.log"
UA = "awesome_diffusion_OT/0.1 (mailto:liyufeng854@gmail.com)"


def log(m):
    line = time.strftime("%H:%M:%S ") + m
    print(line, flush=True)
    LOG.open("a").write(line + "\n")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dataless(p):
    st = os.stat(p)
    return st.st_size > 0 and st.st_blocks * 512 < st.st_size * 0.5


def main():
    rows = [json.loads(l) for l in PAPERS.read_text(encoding="utf-8").splitlines() if l.strip()]
    man = json.loads(MAN.read_text()) if MAN.exists() else {}
    OUT.mkdir(exist_ok=True)
    uniq = [r for r in rows if not r["dup_of"]]
    # priority: starred first, then the rest
    uniq.sort(key=lambda r: (not r["star"], r["topic"]))
    n_ok = n_skip = n_fail = 0
    for r in uniq:
        if r["arxiv_id"]:
            dst = OUT / f"{r['arxiv_id'].replace('/', '_')}.pdf"
            if dst.exists() and dst.stat().st_size > 10_000:
                n_skip += 1
                continue
            url = f"https://arxiv.org/pdf/{r['arxiv_id']}"
            try:
                time.sleep(3.1)
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=90) as resp, open(dst, "wb") as f:
                    shutil.copyfileobj(resp, f)
                if dst.stat().st_size < 10_000 or open(dst, "rb").read(5) != b"%PDF-":
                    dst.unlink(missing_ok=True)
                    raise RuntimeError("not a pdf")
                man[dst.name] = dict(sha256=sha(dst), bytes=dst.stat().st_size, source=url, key=r["key"])
                n_ok += 1
                log(f"OK arxiv {r['arxiv_id']} {dst.stat().st_size//1024}KB {r['title'][:50]}")
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                log(f"FAIL arxiv {r['arxiv_id']} {type(e).__name__} {str(e)[:60]}")
        elif r.get("local_pdf"):
            src = SURVEY / r["topic"] / r["local_pdf"]
            dst = OUT / f"{r['topic']}_{r['local_pdf']}"
            if dst.exists():
                n_skip += 1
                continue
            if not src.exists():
                n_fail += 1
                log(f"MISSING local {src.name}")
                continue
            if dataless(src):
                log(f"EVICTED local {src.name} (iCloud dataless; retry later)")
                n_fail += 1
                continue
            shutil.copy2(src, dst)
            man[dst.name] = dict(sha256=sha(dst), bytes=dst.stat().st_size, source=f"survey:{r['topic']}/{r['local_pdf']}", key=r["key"])
            n_ok += 1
            log(f"OK copy {dst.name} {dst.stat().st_size//1024}KB")
        if (n_ok + n_fail) % 10 == 0:
            MAN.write_text(json.dumps(man, indent=1))
    MAN.write_text(json.dumps(man, indent=1))
    log(f"DONE ok={n_ok} skip={n_skip} fail={n_fail} total_files={len(list(OUT.glob('*.pdf')))}")


if __name__ == "__main__":
    main()
