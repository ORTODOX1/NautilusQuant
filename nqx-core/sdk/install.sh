#!/usr/bin/env bash
# NQX SDK installer — copies SDK components and links CLI tools.
set -euo pipefail

SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SDK_DIR/.." && pwd)"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/share/nqx-sdk}"
BIN_DIR="${BIN_DIR:-$HOME/.local/bin}"

echo "NQX SDK Installer"
echo "  source: $PROJECT_DIR"
echo "  target: $INSTALL_DIR"
echo "  bins:   $BIN_DIR"
echo

# ---- Dependencies ----
echo "[1/5] Checking dependencies..."
MISSING=()
python3 -c "import numpy" 2>/dev/null || MISSING+=("numpy")
python3 -c "import fastapi" 2>/dev/null || MISSING+=("fastapi")
python3 -c "import uvicorn" 2>/dev/null || MISSING+=("uvicorn")

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  Missing: ${MISSING[*]}"
    echo "  Installing with pip..."
    python3 -m pip install "${MISSING[@]}"
else
    echo "  All dependencies found."
fi

# ---- Pip install (if pyproject.toml exists) ----
echo "[2/5] Editable install..."
if [ -f "$PROJECT_DIR/pyproject.toml" ]; then
    python3 -m pip install -e "$PROJECT_DIR"
    echo "  done."
else
    echo "  No pyproject.toml found — skipping editable install."
    echo "  The SDK will work via PYTHONPATH instead."
fi

# ---- Copy SDK ----
echo "[3/5] Copying SDK to $INSTALL_DIR..."
rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# Copy core modules
cp -r "$PROJECT_DIR/nqx" "$INSTALL_DIR/nqx"
# Copy CLI tools
cp -r "$PROJECT_DIR/tools" "$INSTALL_DIR/tools"
# Copy C ABI library
cp -r "$PROJECT_DIR/sdk/libnqx" "$INSTALL_DIR/libnqx"
# Copy programs and docs
cp -r "$PROJECT_DIR/programs" "$INSTALL_DIR/programs"
cp -r "$PROJECT_DIR/docs" "$INSTALL_DIR/docs"
# Copy boot firmware
cp -r "$PROJECT_DIR/firmware/boot" "$INSTALL_DIR/firmware" 2>/dev/null || mkdir -p "$INSTALL_DIR/firmware"

# Create version file
echo "nqx-sdk v1.0.0" > "$INSTALL_DIR/VERSION"
echo "  done."

# ---- Link binaries ----
echo "[4/5] Linking CLI binaries to $BIN_DIR..."
mkdir -p "$BIN_DIR"

link_bin() {
    local name="$1"
    local src="$2"
    if [ -f "$src" ]; then
        chmod +x "$src"
        ln -sf "$src" "$BIN_DIR/$name"
        echo "  linked: $BIN_DIR/$name"
    else
        echo "  WARNING: $src not found, skipping $name"
    fi
}

link_bin nqx-asm     "$PROJECT_DIR/tools/cli/nqx-asm"
link_bin nqx-disasm  "$PROJECT_DIR/tools/cli/nqx-disasm"
link_bin nqx-sim     "$PROJECT_DIR/tools/cli/nqx-sim"
link_bin nqx-rig     "$PROJECT_DIR/tools/cli/nqx-rig"
link_bin nqx-debug   "$PROJECT_DIR/tools/cli/nqx-debug"

# Also link all existing nqx-* CLI tools
for f in "$PROJECT_DIR/tools/cli"/nqx-*; do
    name=$(basename "$f")
    [[ "$name" == *.desktop ]] && continue
    # Skip ones already linked above
    case "$name" in
        nqx-asm|nqx-disasm|nqx-sim|nqx-rig|nqx-debug) continue ;;
    esac
    chmod +x "$f"
    ln -sf "$f" "$BIN_DIR/$name"
    echo "  linked: $BIN_DIR/$name"
done

# ---- Setup PYTHONPATH wrapper ----
echo "[5/5] Setting up SDK wrapper..."
SDK_WRAPPER="$BIN_DIR/nqx-sdk-env"
cat > "$SDK_WRAPPER" <<WRAPEOF
#!/usr/bin/env bash
# Activate NQX SDK environment
export NQX_SDK_DIR="$INSTALL_DIR"
if [[ ":\$PYTHONPATH:" != *":$INSTALL_DIR:"* ]]; then
    export PYTHONPATH="\$PYTHONPATH:$INSTALL_DIR"
fi
echo "NQX SDK v1.0.0 — PYTHONPATH updated"
WRAPEOF
chmod +x "$SDK_WRAPPER"
echo "  wrapper: $SDK_WRAPPER"

# ---- Summary ----
echo
echo "┌──────────────────────────────────────────────┐"
echo "│  NQX SDK installed successfully             │"
echo "│  Add to ~/.bashrc / ~/.zshrc:               │"
echo "│    export PATH=\"\$PATH:$BIN_DIR\"              │"
echo "│    export NQX_SDK_DIR=\"$INSTALL_DIR\"          │"
echo "│    export PYTHONPATH=\"\$PYTHONPATH:$INSTALL_DIR\" │"
echo "└──────────────────────────────────────────────┘"
echo
echo "Available tools:"
ls -la "$BIN_DIR"/nqx-* 2>/dev/null | awk '{print "  " $9 " -> " $11}'
echo
echo "SDK contents:"
ls -la "$INSTALL_DIR"
echo
echo "Quick test: nqx-asm --help  |  nqx-sim --help"
