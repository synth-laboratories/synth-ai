#!/bin/bash
# Install sqld binary for Synth AI

set -e

SQLD_VERSION="v0.26.2"
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

# Map architecture names
case "$ARCH" in
    x86_64) ARCH="x86_64" ;;
    aarch64|arm64) ARCH="aarch64" ;;
    *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

# Construct download URL
URL="https://github.com/tursodatabase/libsql/releases/download/libsql-server-${SQLD_VERSION}/sqld-${OS}-${ARCH}.tar.xz"

echo "📥 Downloading sqld ${SQLD_VERSION} for ${OS}-${ARCH}..."

# Download and extract
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"
curl -L -o sqld.tar.xz "$URL"
tar -xf sqld.tar.xz

# Install to user's local bin
mkdir -p ~/.local/bin
mv sqld ~/.local/bin/
chmod +x ~/.local/bin/sqld

# Clean up
cd -
rm -rf "$TMP_DIR"

echo "✅ sqld installed to ~/.local/bin/sqld"
echo ""
echo "🔧 Add ~/.local/bin to your PATH if needed:"
echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""