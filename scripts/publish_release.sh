#!/usr/bin/env bash
# Upload PDFs as GitHub Release assets (keeps git lean). Usage: bash scripts/publish_release.sh [zh|en|all]
# zh: papers_zh/*.zh.pdf -> release pdf-zh-v1 ; en: papers/*.pdf -> release pdf-en-v1 . Idempotent (--clobber), resumable.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
REPO="asimfish/awesome_diffusion_OT"; WHAT="${1:-zh}"
ensure_release() { gh release view "$1" -R "$REPO" >/dev/null 2>&1 || gh release create "$1" -R "$REPO" --title "$2" --notes "$3" --latest=false; }
upload_dir() {  # tag dir glob
  local tag=$1 dir=$2 glob=$3 n=0
  existing="$(gh release view "$tag" -R "$REPO" --json assets --jq '.assets[].name' 2>/dev/null)"
  for f in "$dir"/$glob; do
    [ -f "$f" ] || continue
    b=$(basename "$f")
    if echo "$existing" | rg -qx "$(printf '%s' "$b" | sed 's/[.[\*^$]/\\&/g')"; then continue; fi
    gh release upload "$tag" "$f" -R "$REPO" --clobber >/dev/null 2>&1 && n=$((n+1)) && echo "[$(date +%H:%M:%S)] uploaded $tag/$b" || echo "[$(date +%H:%M:%S)] FAILED $b"
  done
  echo "[$(date +%H:%M:%S)] $tag: uploaded $n new assets"
  gh release view "$tag" -R "$REPO" --json assets --jq '[.assets[].name]' > "data/release_assets_${tag}.json" 2>/dev/null || true
}
if [ "$WHAT" = zh ] || [ "$WHAT" = all ]; then
  ensure_release pdf-zh-v1 "Chinese translations (layout-preserving, SuperTranslate + object-level QA)" "Translated PDFs: <arxiv_id>.zh.pdf. QA reports are tracked in git under papers_zh/*.inspect.json. For personal study only; copyright remains with the original authors/publishers."
  upload_dir pdf-zh-v1 papers_zh "*.zh.pdf"
fi
if [ "$WHAT" = en ] || [ "$WHAT" = all ]; then
  ensure_release pdf-en-v1 "Original PDFs (mirror)" "Original paper PDFs mirrored from arXiv / open-access sources for offline reading: <arxiv_id>.pdf. Licenses recorded in data/arxiv_meta.json."
  upload_dir pdf-en-v1 papers "*.pdf"
fi
