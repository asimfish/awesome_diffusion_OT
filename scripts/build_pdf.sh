#!/usr/bin/env bash
# Build synthesis report PDFs (zh/en): pandoc -> standalone HTML (print CSS, MathML) -> headless Chrome -> PDF.
# Usage: bash scripts/build_pdf.sh [zh|en|all]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT/report"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
build() {
  lang=$1; src=$2; html="pdf/awesome_diffusion_ot_report_${lang}.html"; out="pdf/awesome_diffusion_ot_report_${lang}.pdf"
  pandoc "$src" -f markdown+tex_math_dollars+pipe_tables+implicit_figures -t html5 -s --mathml --toc --toc-depth=2 --shift-heading-level-by=-1 \
    --css print.css --embed-resources --resource-path=.:pdf -o "$html" --metadata lang=$([ "$lang" = zh ] && echo zh-CN || echo en)
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --virtual-time-budget=8000 \
    --print-to-pdf="$out" "file://$ROOT/report/$html" >/dev/null 2>&1
  ls -la "$out"
}
case "${1:-all}" in
  zh) build zh AWESOME_DIFFUSION_OT_REPORT_zh.md ;;
  en) build en AWESOME_DIFFUSION_OT_REPORT_en.md ;;
  all) build zh AWESOME_DIFFUSION_OT_REPORT_zh.md; build en AWESOME_DIFFUSION_OT_REPORT_en.md ;;
esac
