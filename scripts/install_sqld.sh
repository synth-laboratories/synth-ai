#!/bin/bash
# Install sqld (libsql-server) using the official installer
# This ensures we get the correct binary for the current OS/architecture

set -e

echo "🔧 Installing sqld (libsql-server)..."

# Check if sqld exists and is valid
if command -v sqld >/dev/null 2>&1; then
    if sqld --version >/dev/null 2>&1; then
        echo "✅ sqld is already installed and working:"
        sqld --version
        exit 0
    else
        echo "⚠️  Found invalid sqld binary at $(which sqld)"
        echo "Please run: sudo rm -f $(which sqld)"
        echo "Then run this script again."
        exit 1
    fi
fi

# Check if we have a bad stub at /usr/local/bin/sqld
if [ -f "/usr/local/bin/sqld" ]; then
    if ! /usr/local/bin/sqld --version >/dev/null 2>&1; then
        echo "⚠️  Found invalid sqld at /usr/local/bin/sqld"
        echo "Please run: sudo rm -f /usr/local/bin/sqld"
        echo "Then run this script again."
        exit 1
    fi
fi

# Determine installation directory
INSTALL_DIR="/usr/local/bin"
if [ ! -w "$INSTALL_DIR" ]; then
    echo "⚠️  Cannot write to $INSTALL_DIR without sudo"
    INSTALL_DIR="$HOME/.local/bin"
    echo "📁 Will install to $INSTALL_DIR instead"
    mkdir -p "$INSTALL_DIR"
    
    # Add to PATH if not already there
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        echo "⚠️  $INSTALL_DIR is not in PATH"
        echo "Add this to your shell profile: export PATH=\"\$PATH:$INSTALL_DIR\""
    fi
fi

# Install using the official installer
echo "📦 Downloading and installing sqld..."
# Set CARGO_HOME to control where binaries are installed
export CARGO_HOME="$HOME/.cargo"
export PATH="$CARGO_HOME/bin:$PATH"

curl --proto '=https' --tlsv1.2 -LsSf \
    https://github.com/tursodatabase/libsql/releases/latest/download/libsql-server-installer.sh \
    | sh

# Verify installation - check multiple locations
if command -v sqld >/dev/null 2>&1; then
    echo "✅ sqld installed successfully:"
    sqld --version
    echo "📍 Location: $(which sqld)"
elif [ -f "$HOME/.cargo/bin/sqld" ]; then
    echo "✅ sqld installed successfully:"
    "$HOME/.cargo/bin/sqld" --version
    echo "📍 Location: $HOME/.cargo/bin/sqld"
    echo "⚠️  Add to PATH: export PATH=\"\$PATH:\$HOME/.cargo/bin\""
else
    echo "❌ Failed to install sqld"
    exit 1
fi