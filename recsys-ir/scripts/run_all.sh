#!/usr/bin/env bash
# run_all.sh — Orchestrate the full pipeline end-to-end.
# Usage: bash scripts/run_all.sh
set -euo pipefail

echo "  recsys-ir — Full Pipeline Run"

make data
make bm25
make embed
make eval
make submit

echo ""
echo "✅ All pipeline stages complete."
