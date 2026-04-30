#!/usr/bin/env bash
# Install NQX-Core CLI launchers as ~/.local/bin/nqx-* symlinks
# + KDE/GNOME desktop shortcut on ~/Desktop.
set -e
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$HOME/.local/bin"
mkdir -p "$DST"

for f in "$SRC"/nqx-*; do
    [[ "$f" == *.desktop ]] && continue
    name=$(basename "$f")
    chmod +x "$f"
    ln -sf "$f" "$DST/$name"
    echo "linked: $DST/$name -> $f"
done

# Desktop shortcut
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESKTOP_DIR" ] && [ -f "$SRC/nqx-workbench.desktop" ]; then
    cp "$SRC/nqx-workbench.desktop" "$DESKTOP_DIR/NQX-Core.desktop"
    chmod +x "$DESKTOP_DIR/NQX-Core.desktop"
    # KDE Plasma 6 mark as "trusted"
    if command -v gio >/dev/null 2>&1; then
        gio set "$DESKTOP_DIR/NQX-Core.desktop" "metadata::trusted" true 2>/dev/null || true
    fi
    echo "desktop shortcut: $DESKTOP_DIR/NQX-Core.desktop"
fi

echo
echo "Installed launchers:"
ls -la "$DST"/nqx-* | awk '{print "  "$9" -> "$11}'
echo
case ":$PATH:" in
    *":$DST:"*) echo "PATH already includes $DST" ;;
    *) echo "WARNING: $DST is not in your PATH. Add to ~/.bashrc or ~/.zshrc:"
       echo "    export PATH=\"$DST:\$PATH\"" ;;
esac
echo
echo "Usage:"
echo "  nqx-claude            # interactive Claude in the project"
echo "  nqx-deepseek          # interactive DeepSeek V4 Pro"
echo "  nqx-flash             # interactive DeepSeek V4 Flash (cheap)"
echo "  nqx-codex             # interactive Codex"
echo "  nqx-audit architecture        # one prompt → all CLIs in parallel"
echo "  nqx-audit --all               # every prompt × every CLI (burn tokens)"
echo "  nqx-trio              # tmux 3-pane: Claude + DeepSeek + Codex"
