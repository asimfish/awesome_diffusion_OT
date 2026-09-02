#!/usr/bin/env bash
# Second pass: wait for pass-1 license step, then kb-title retry -> fetch -> licenses -> text extraction.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; export OPENBLAS_NUM_THREADS=1
until rg -q "license pass done" logs/pipeline.log; do sleep 30; done
echo "[$(date +%H:%M:%S)] pass2 start" >> logs/pipeline.log
nice -n 10 python3 scripts/build_corpus.py >> logs/pipeline.log 2>&1
nice -n 10 python3 scripts/resolve_arxiv.py --retry-kb --no-license >> logs/pipeline.log 2>&1
nice -n 10 python3 scripts/fetch_papers.py >> logs/pipeline.log 2>&1
nice -n 10 python3 scripts/resolve_arxiv.py >> logs/pipeline.log 2>&1
nice -n 10 python3 scripts/extract_text.py >> logs/pipeline.log 2>&1
echo "[$(date +%H:%M:%S)] pass2 done" >> logs/pipeline.log
