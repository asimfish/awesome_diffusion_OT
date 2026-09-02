#!/usr/bin/env bash
# Layout-preserving Chinese translation of papers/*.pdf -> papers_zh/<id>.zh.pdf with SuperTranslate + object-level QA.
# Usage: bash scripts/translate_batch.sh [--loop] [--max-mb 40] [id ...]
#   --loop : keep rescanning every 5 min (picks up newly downloaded PDFs) until logs/STOP_TRANSLATE exists
# Safe to run 2 instances in parallel (per-paper mkdir locks). Starred (core) papers first.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
ST_HOME="${SUPER_TRANSLATE_HOME:-$HOME/Code/super_translate}"; PY="$ST_HOME/.venv/bin/python"
OUT="$ROOT/papers_zh"; LOG="$ROOT/logs"; LOCKS="$LOG/locks"; mkdir -p "$OUT" "$LOG" "$LOCKS"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
LOOP=0; MAXMB=25; IDS=()
while [ $# -gt 0 ]; do case "$1" in --loop) LOOP=1;; --max-mb) MAXMB=$2; shift;; *) IDS+=("$1");; esac; shift; done
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  export DEEPSEEK_API_KEY="$(python3 -c "import yaml;print(yaml.safe_load(open('$HOME/.dsh/.credentials.yaml'))['refs']['DEEPSEEK_API_KEY'])")"
fi
[ -n "$DEEPSEEK_API_KEY" ] || { echo "no DEEPSEEK_API_KEY"; exit 2; }
priority() {  # arxiv ids, starred first, then by topic; only unique rows
  python3 - <<'PY'
import json
rows=[json.loads(l) for l in open('data/papers.jsonl') if l.strip()]
seen=set()
for r in sorted([r for r in rows if r['arxiv_id'] and not r['dup_of']], key=lambda r:(r['evidence']=='B', not r['star'], r['topic'])):
    a=r['arxiv_id'].replace('/','_')
    if a not in seen: seen.add(a); print(a)
PY
}
run_one() {
  local id=$1 src="$ROOT/papers/$1.pdf" dst="$OUT/$1.zh.pdf"
  [ -f "$src" ] || return 3
  if [ -f "$dst" ] && [ -f "$OUT/$id.inspect.json" ]; then return 4; fi
  local mb=$(( $(stat -f %z "$src") / 1048576 ))
  if [ "$mb" -gt "$MAXMB" ]; then echo "[$(date +%H:%M:%S)] DEFER $id (${mb}MB > ${MAXMB}MB)" | tee -a "$LOG/translate.log"; return 5; fi
  mkdir "$LOCKS/$id" 2>/dev/null || return 6
  echo "[$(date +%H:%M:%S)] START $id (${mb}MB)" | tee -a "$LOG/translate.log"
  local t0=$(date +%s)
  ( cd "$ST_HOME" && nice -n 10 "$PY" -m pdf_zh_translator translate "$src" "$dst" \
      --api-mode deepseek --api-key-env DEEPSEEK_API_KEY \
      --preserve-graphics-text --skip-overflow --quiet \
      --cache-file "$OUT/$id.translation-cache.jsonl" ) > "$LOG/$id.translate.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] translate rc=$rc ($(( $(date +%s)-t0 ))s) $id" | tee -a "$LOG/translate.log"
  if [ $rc -eq 0 ] && [ -f "$dst" ]; then
    ( cd "$ST_HOME" && nice -n 10 "$PY" -m pdf_zh_translator inspect "$src" "$dst" --json-out "$OUT/$id.inspect.json" ) > "$LOG/$id.inspect.log" 2>&1
    echo "[$(date +%H:%M:%S)] inspect rc=$? issues=$(python3 -c "import json;print(json.load(open('$OUT/$id.inspect.json')).get('issue_count','?'))" 2>/dev/null) $id" | tee -a "$LOG/translate.log"
  else
    rm -f "$dst"
  fi
  rmdir "$LOCKS/$id" 2>/dev/null
  return 0
}
pass() {
  local n=0
  local list; if [ ${#IDS[@]} -gt 0 ]; then list=("${IDS[@]}"); else IFS=$'\n' read -r -d '' -a list < <(priority && printf '\0'); fi
  for id in "${list[@]}"; do run_one "$id"; [ $? -eq 0 ] && n=$((n+1)); done
  echo "[$(date +%H:%M:%S)] PASS done translated_this_pass=$n done_total=$(ls "$OUT"/*.zh.pdf 2>/dev/null | wc -l | tr -d ' ')" | tee -a "$LOG/translate.log"
}
if [ $LOOP -eq 1 ]; then
  end=$(( $(date +%s) + 12*3600 ))
  while [ ! -f "$LOG/STOP_TRANSLATE" ] && [ $(date +%s) -lt $end ]; do pass; sleep 300; done
else pass; fi
