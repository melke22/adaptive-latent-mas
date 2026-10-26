
#!/usr/bin/env bash
set -Eeuo pipefail

REPO="https://github.com/melke22/adaptive-latent-mas.git"
INSTALL_DIR="${ALMAS_INSTALL_DIR:-$HOME/.local/share/almas}"
BIN_DIR="$HOME/.local/bin"

# Accept the same flag style as the ADA installer.
while (($#)); do
    case "$1" in
        --yes|-y) shift ;;
        --help|-h)
            echo "Usage: install.sh [--yes]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 2 ;;
    esac
done

echo "==> Installing ALMAS"

command -v git >/dev/null || {
    echo "Git is required. Install it with: sudo dnf install git"
    exit 1
}

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
    if [[ -e "$INSTALL_DIR" ]]; then
        echo "Install directory exists but is not a Git repository: $INSTALL_DIR"
        echo "Move it or set ALMAS_INSTALL_DIR to another directory."
        exit 1
    fi
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone "$REPO" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

if command -v python3.13 >/dev/null 2>&1; then
    PYTHON="$(command -v python3.13)"
else
    PYTHON="$(command -v python3 || true)"
fi

if [[ -z "$PYTHON" ]]; then
    echo "Python 3 is required. On Fedora, try: sudo dnf install python3.13"
    exit 1
fi

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11) and sys.version_info < (3, 14), "Use Python 3.11, 3.12, or 3.13"' || {
    echo "Please install Python 3.13 and rerun this installer."
    exit 1
}

echo "==> Creating virtual environment"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# Create a launcher that finds the project's Python UI entry point.
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/almas" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${ALMAS_INSTALL_DIR:-$HOME/.local/share/almas}"
cd "$APP_DIR"

for candidate in ui/app.py ui/Home.py ui/home.py app.py almas.py; do
    if [[ -f "$candidate" ]]; then
        exec "$APP_DIR/.venv/bin/python" -m streamlit run "$candidate" "$@"
    fi
done

echo "ALMAS is installed, but its Streamlit entry point wasn't detected."
echo "Check README.md and files under: $APP_DIR/ui"
exit 1
LAUNCHER

chmod +x "$BIN_DIR/almas"

echo
echo "ALMAS dependencies installed."
echo "Launch with: $BIN_DIR/almas"
echo "If needed, add $BIN_DIR to your PATH."

