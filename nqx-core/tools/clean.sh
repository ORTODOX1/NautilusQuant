#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "This will remove:"
echo "  - audits/results/"
echo "  - audits/logs/"
echo "  - __pycache__ dirs"
echo "  - .pytest_cache"
echo ""
read -r -p "Proceed? [y/N] " reply
case "$reply" in
    [yY]|[yY][eE][sS])
        rm -rf "$DIR/audits/results/" "$DIR/audits/logs/"
        find "$DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
        find "$DIR" -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
        echo "Cleaned."
        ;;
    *)
        echo "Aborted."
        ;;
esac
