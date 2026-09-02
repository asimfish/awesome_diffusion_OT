#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; export OPENBLAS_NUM_THREADS=1
echo "[$(date +%H:%M:%S)] pass3 start" >> logs/pipeline.log
nice -n 10 python3 scripts/resolve_arxiv.py --retry-kb --force-retry --no-license >> logs/pipeline.log 2>&1
nice -n 10 python3 scripts/fetch_papers.py >> logs/pipeline.log 2>&1
nice -n 10 python3 scripts/resolve_arxiv.py >> logs/pipeline.log 2>&1
nice -n 10 python3 scripts/extract_text.py >> logs/pipeline.log 2>&1
nice -n 10 python3 scripts/build_manifests.py t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 >> logs/pipeline.log 2>&1
echo "[$(date +%H:%M:%S)] pass3 done" >> logs/pipeline.log
