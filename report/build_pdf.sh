#!/usr/bin/env bash
# Build the PDF report from report.md (no LaTeX needed).
# Requires: pandoc, Microsoft Edge or Google Chrome.
# Usage: bash report/build_pdf.sh   (from the repository root)
set -euo pipefail

cd "$(dirname "$0")"

OUT="agent02_05_LyHaiQuan.pdf"

pandoc report.md \
    --standalone --mathml --embed-resources \
    --resource-path=.:.. \
    --css=style.css \
    --metadata title="AGENT02 Report" \
    -o report.html

BROWSER=""
for candidate in \
    "/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
    "/c/Program Files/Google/Chrome/Application/chrome.exe" \
    "$(command -v google-chrome || true)" \
    "$(command -v chromium || true)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        BROWSER="$candidate"
        break
    fi
done

if [ -z "$BROWSER" ]; then
    echo "No Edge/Chrome found; open report/report.html and print to PDF manually." >&2
    exit 1
fi

HTML_PATH="$(pwd -W 2>/dev/null || pwd)/report.html"
PDF_PATH="$(pwd -W 2>/dev/null || pwd)/$OUT"

"$BROWSER" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$PDF_PATH" "file:///$HTML_PATH"

echo "Wrote report/$OUT"
