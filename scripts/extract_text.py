#!/usr/bin/env python3
"""Extract plain text from papers/*.pdf into data/text/<stem>.txt (PyMuPDF), capped at --max-chars (default 90k).
Also writes data/text_index.json {stem: {pages, chars, truncated}}. Idempotent."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "papers", ROOT / "data" / "text"
MAXC = int(sys.argv[1]) if len(sys.argv) > 1 else 90_000


def main():
    DST.mkdir(parents=True, exist_ok=True)
    idx_p = ROOT / "data" / "text_index.json"
    idx = json.loads(idx_p.read_text()) if idx_p.exists() else {}
    for pdf in sorted(SRC.glob("*.pdf")):
        out = DST / (pdf.stem + ".txt")
        if out.exists() and pdf.stem in idx:
            continue
        try:
            doc = fitz.open(pdf)
            parts, n = [], 0
            for page in doc:
                t = page.get_text("text")
                parts.append(t)
                n += len(t)
                if n > MAXC:
                    break
            txt = "\n".join(parts)
            trunc = len(txt) > MAXC
            out.write_text(txt[:MAXC], encoding="utf-8")
            idx[pdf.stem] = dict(pages=doc.page_count, chars=min(len(txt), MAXC), truncated=trunc)
        except Exception as e:  # noqa: BLE001
            idx[pdf.stem] = dict(error=f"{type(e).__name__}: {str(e)[:80]}")
    idx_p.write_text(json.dumps(idx, indent=1))
    ok = sum(1 for v in idx.values() if "error" not in v)
    print(f"extracted {ok}/{len(idx)}; errors={len(idx)-ok}; truncated={sum(1 for v in idx.values() if v.get('truncated'))}")


if __name__ == "__main__":
    main()
