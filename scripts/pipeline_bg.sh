#!/usr/bin/env bash
# Background data pipeline: resolve arXiv ids/metadata/licenses -> download PDFs. Logs in logs/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export OPENBLAS_NUM_THREADS=1
echo "[$(date +%H:%M:%S)] pipeline start" >> logs/pipeline.log
nice -n 10 python3 scripts/resolve_arxiv.py --no-license >> logs/pipeline.log 2>&1
echo "[$(date +%H:%M:%S)] resolve done rc=$?" >> logs/pipeline.log
nice -n 10 python3 scripts/fetch_papers.py >> logs/pipeline.log 2>&1
echo "[$(date +%H:%M:%S)] fetch done rc=$?" >> logs/pipeline.log
nice -n 10 python3 scripts/resolve_arxiv.py >> logs/pipeline.log 2>&1   # second pass adds licenses
echo "[$(date +%H:%M:%S)] license pass done rc=$?" >> logs/pipeline.log
