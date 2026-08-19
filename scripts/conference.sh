#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WLK_BIN="$PROJECT_DIR/.venv/bin/wlk"

if [ ! -x "$WLK_BIN" ]; then
    echo "Conference environment is not installed." >&2
    echo "Run:" >&2
    echo "  uv sync --python 3.12 --extra funasr --extra translation --no-sources" >&2
    exit 1
fi

exec "$WLK_BIN" \
    --backend funasr \
    --backend-policy localagreement \
    --language auto \
    --conference-mode \
    --pause-segmentation-seconds 0.8 \
    "$@"
