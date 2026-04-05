#!/usr/bin/env bash
# =============================================================================
# setup_llamacpp.sh — Clone and build llama.cpp for head-to-head benchmarking
# =============================================================================
#
# Clones llama.cpp at a pinned commit and builds it for reproducible benchmarks.
# On macOS, Metal is enabled automatically.
#
# Usage:
#   bash bench/head_to_head/setup_llamacpp.sh [install_dir]
#
# Output:
#   Prints the path to the built llama.cpp binary directory on the last line.
#   Binaries: llama-cli, llama-perplexity
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------------------------------------------------------------------------
# Pinned llama.cpp version for reproducibility
# ---------------------------------------------------------------------------
LLAMA_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMA_COMMIT="f472633e0e62b7a96bdb3b0e68b2a5a85e5db332"  # b5200 (2025-03-30)
LLAMA_TAG="b5200"

INSTALL_DIR="${1:-$PROJECT_DIR/deps/llama.cpp}"
BIN_DIR="$INSTALL_DIR/build/bin"

echo "============================================================"
echo "  llama.cpp Setup for Head-to-Head Benchmark"
echo "============================================================"
echo ""
echo "  Repo:    $LLAMA_REPO"
echo "  Commit:  $LLAMA_COMMIT ($LLAMA_TAG)"
echo "  Install: $INSTALL_DIR"
echo ""

# ---------------------------------------------------------------------------
# Skip if already built at the correct commit
# ---------------------------------------------------------------------------
if [ -f "$BIN_DIR/llama-cli" ] && [ -f "$BIN_DIR/llama-perplexity" ]; then
    CURRENT_COMMIT=""
    if [ -d "$INSTALL_DIR/.git" ]; then
        CURRENT_COMMIT=$(git -C "$INSTALL_DIR" rev-parse HEAD 2>/dev/null || true)
    fi
    if [ "$CURRENT_COMMIT" = "$LLAMA_COMMIT" ]; then
        echo "  llama.cpp already built at $LLAMA_TAG. Skipping."
        echo ""
        echo "LLAMA_BIN_DIR=$BIN_DIR"
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# Clone
# ---------------------------------------------------------------------------
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "  Updating existing clone..."
    git -C "$INSTALL_DIR" fetch origin
else
    echo "  Cloning llama.cpp..."
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone "$LLAMA_REPO" "$INSTALL_DIR"
fi

echo "  Checking out $LLAMA_TAG ($LLAMA_COMMIT)..."
git -C "$INSTALL_DIR" checkout "$LLAMA_COMMIT"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo ""
echo "  Building llama.cpp..."

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"

# Enable Metal on macOS
if [ "$(uname -s)" = "Darwin" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=ON"
    echo "  Platform: macOS (Metal enabled)"
else
    echo "  Platform: $(uname -s)"
fi

NPROC=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

cmake -B "$INSTALL_DIR/build" -S "$INSTALL_DIR" $CMAKE_ARGS
cmake --build "$INSTALL_DIR/build" -j"$NPROC"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo ""
if [ ! -f "$BIN_DIR/llama-cli" ]; then
    echo "ERROR: llama-cli not found at $BIN_DIR/llama-cli"
    echo "Build may have failed. Check output above."
    exit 1
fi
if [ ! -f "$BIN_DIR/llama-perplexity" ]; then
    echo "ERROR: llama-perplexity not found at $BIN_DIR/llama-perplexity"
    exit 1
fi

echo "  Build successful."
echo ""
echo "  Binaries:"
echo "    llama-cli:        $BIN_DIR/llama-cli"
echo "    llama-perplexity: $BIN_DIR/llama-perplexity"
echo ""
echo "LLAMA_BIN_DIR=$BIN_DIR"
