#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPORTS_DIR="$ROOT_DIR/reports/static"
mkdir -p "$REPORTS_DIR"


echo "Running ruff (style/lint)"
# Use 'ruff check' which is the correct subcommand
if command -v ruff >/dev/null 2>&1; then
	ruff check src --format json > "$REPORTS_DIR/ruff.json" || true
else
	echo "ruff not installed" > "$REPORTS_DIR/ruff.txt"
fi

echo "Running radon (cyclomatic complexity)"
if command -v radon >/dev/null 2>&1; then
	radon cc -s -j src > "$REPORTS_DIR/radon_cc.json" || true
else
	echo "radon not installed" > "$REPORTS_DIR/radon_cc.txt"
fi

echo "Running vulture (dead code)"
if command -v vulture >/dev/null 2>&1; then
	vulture src --min-confidence 50 > "$REPORTS_DIR/vulture.txt" || true
else
	echo "vulture not installed" > "$REPORTS_DIR/vulture.txt"
fi

echo "Running jscpd (duplication)"
if command -v jscpd >/dev/null 2>&1; then
	jscpd --reporters json --output "$REPORTS_DIR/jscpd" src || true
else
	echo "jscpd not found; skipping duplication scan" > "$REPORTS_DIR/jscpd-notfound.txt"
fi

echo "Reports written to $REPORTS_DIR"

exit 0
