#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export DEEPSEEK_API_KEY="$(python3 -c "import yaml;print(yaml.safe_load(open('$HOME/.dsh/.credentials.yaml'))['refs']['DEEPSEEK_API_KEY'])")"
export OPENBLAS_NUM_THREADS=1
python3 scripts/gen_reports_llm.py --workers 4 >> logs/gen_reports_run.log 2>&1
python3 scripts/gen_reports_llm.py --workers 2 >> logs/gen_reports_run.log 2>&1   # second pass for failures
echo "[$(date +%H:%M:%S)] GEN_DONE" >> logs/gen_reports_run.log
