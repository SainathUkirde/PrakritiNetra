#!/usr/bin/env bash
# build.sh — manual build helper (NOT used by Render automatically).
#
# Render handles system packages via packages.txt (libeccodes-dev) and
# Python packages via the buildCommand in render.yaml.
# Run this script only for local Docker builds or CI pipelines that need
# full control.
#
# Usage:
#   ./build.sh          # installs both API and dashboard dependencies
#   ./build.sh api       # API only
#   ./build.sh dashboard # Streamlit dashboard only

set -e
TARGET=${1:-all}

echo "==> PrakritiNetra build script"

# ── System dependency (only needed outside Render / without packages.txt) ──
if [[ "$TARGET" == "all" || "$TARGET" == "api" || "$TARGET" == "dashboard" ]]; then
    if command -v apt-get &>/dev/null && [[ "$(id -u)" == "0" ]]; then
        echo "==> Installing system dependency: libeccodes-dev (GRIB2 support)"
        apt-get update -qq && apt-get install -y -qq libeccodes-dev
    else
        echo "==> Skipping apt-get (not root or apt not available)."
        echo "    Ensure libeccodes-dev (eccodes) is installed on this system."
        echo "    macOS: brew install eccodes"
        echo "    Ubuntu/Debian: sudo apt-get install libeccodes-dev"
    fi
fi

# ── Python dependencies ────────────────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "api" ]]; then
    echo "==> Installing API dependencies (requirements-api.txt)"
    pip install -r requirements-api.txt
fi

if [[ "$TARGET" == "all" || "$TARGET" == "dashboard" ]]; then
    echo "==> Installing dashboard dependencies (requirements.txt)"
    pip install -r requirements.txt
fi

echo "==> Build complete."
