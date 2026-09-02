#!/usr/bin/env bash
# Hourly maintenance: upload new translations to the release, regenerate README/topics with fresh links, commit and push.
# Usage: bash scripts/refresh_and_push.sh [--loop HOURS]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; export OPENBLAS_NUM_THREADS=1
once() {
  bash scripts/publish_release.sh zh >> logs/release_zh.log 2>&1
  python3 scripts/sync_reports.py > /dev/null 2>&1
  python3 scripts/topic_digests.py --min-cov 2 > /dev/null 2>&1   # min-cov>1: never regenerate digests, only re-render topic pages/links
  python3 src/generator.py > /dev/null 2>&1
  if ! git diff --quiet; then
    n=$(ls papers_zh/*.zh.pdf 2>/dev/null | wc -l | tr -d ' ')
    git add -A > /dev/null 2>&1
    git -c user.name=asimplefish -c user.email=liyufeng854@gmail.com commit -q -m "[repo/chore]: refresh links (translations: $n zh-PDFs)" && git push -q origin main
    echo "[$(date +%H:%M:%S)] pushed refresh (zh-PDFs=$n)" | tee -a logs/refresh.log
  else
    echo "[$(date +%H:%M:%S)] nothing to refresh" | tee -a logs/refresh.log
  fi
}
if [ "${1:-}" = "--loop" ]; then
  end=$(( $(date +%s) + ${2:-12}*3600 ))
  while [ $(date +%s) -lt $end ]; do once; sleep 3600; done
else once; fi
