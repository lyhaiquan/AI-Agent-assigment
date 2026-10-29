#!/usr/bin/env bash
# Build the PDF reports from Markdown (no LaTeX needed).
#   report.md     -> agent02_05_LyHaiQuan.pdf     (English)
#   report_vi.md  -> agent02_05_LyHaiQuan_vi.pdf  (Vietnamese)
# Requires: pandoc, Microsoft Edge or Google Chrome.
# Usage: bash report/build_pdf.sh   (from the repository root)
set -euo pipefail

cd "$(dirname "$0")"

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
    echo "No Edge/Chrome found; open the .html files and print to PDF manually." >&2
    exit 1
fi

DIR="$(pwd -W 2>/dev/null || pwd)"

build() {
    local src="$1" html="$2" out="$3"

    pandoc "$src" \
        --standalone --mathml --embed-resources \
        --resource-path=.:.. \
        --css=style.css \
        --metadata title="AGENT02 Report" \
        -o "$html"

    "$BROWSER" --headless --disable-gpu --no-pdf-header-footer \
        --print-to-pdf="$DIR/$out" "file:///$DIR/$html"

    echo "Wrote report/$out"
}

build report.md report.html agent02_05_LyHaiQuan.pdf
build report_vi.md report_vi.html agent02_05_LyHaiQuan_vi.pdf
