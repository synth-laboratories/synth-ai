#!/bin/bash
# Helper script to set up a tunnel for local task apps
# Usage: ./setup_tunnel.sh <local_port> [tunnel_service]

set -e

LOCAL_PORT=${1:-8115}
TUNNEL_SERVICE=${2:-ngrok}

if [ -z "$1" ]; then
    echo "Usage: $0 <local_port> [tunnel_service]"
    echo "Example: $0 8115 ngrok"
    echo ""
    echo "Tunnel services: ngrok, localtunnel, cloudflared"
    exit 1
fi

echo "Setting up tunnel for localhost:$LOCAL_PORT using $TUNNEL_SERVICE..."

case $TUNNEL_SERVICE in
    ngrok)
        if ! command -v ngrok &> /dev/null; then
            echo "❌ ngrok not found. Install from https://ngrok.com/download"
            exit 1
        fi
        echo "Starting ngrok tunnel..."
        ngrok http $LOCAL_PORT --log=stdout &
        NGROK_PID=$!
        sleep 3
        TUNNEL_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o 'https://[^"]*\.ngrok[^"]*' | head -1)
        if [ -z "$TUNNEL_URL" ]; then
            echo "❌ Failed to get ngrok URL. Check ngrok is running."
            kill $NGROK_PID 2>/dev/null || true
            exit 1
        fi
        echo "✅ Tunnel URL: $TUNNEL_URL"
        echo "export TUNNEL_URL=$TUNNEL_URL"
        echo ""
        echo "To stop: kill $NGROK_PID"
        ;;
    localtunnel)
        if ! command -v lt &> /dev/null; then
            echo "❌ localtunnel not found. Install: npm install -g localtunnel"
            exit 1
        fi
        echo "Starting localtunnel..."
        TUNNEL_URL=$(lt --port $LOCAL_PORT --print-request 2>&1 | grep -o 'https://[^ ]*' | head -1)
        if [ -z "$TUNNEL_URL" ]; then
            echo "❌ Failed to start localtunnel"
            exit 1
        fi
        echo "✅ Tunnel URL: $TUNNEL_URL"
        echo "export TUNNEL_URL=$TUNNEL_URL"
        ;;
    cloudflared)
        if ! command -v cloudflared &> /dev/null; then
            echo "❌ cloudflared not found. Install from https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
            exit 1
        fi
        echo "Starting cloudflared tunnel..."
        # Start cloudflared and capture output to find URL
        # cloudflared prints URL to stdout, so we need to read it line by line
        TUNNEL_URL=""
        TUNNEL_LOG="/tmp/cloudflared_${LOCAL_PORT}.log"
        
        # Start cloudflared in background, redirecting both stdout and stderr to log file
        cloudflared tunnel --url http://127.0.0.1:$LOCAL_PORT > "$TUNNEL_LOG" 2>&1 &
        CLOUDFLARED_PID=$!
        
        # Wait up to 10 seconds for URL to appear in log
        for i in {1..20}; do
            sleep 0.5
            if [ -f "$TUNNEL_LOG" ]; then
                TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" | head -1)
                if [ -n "$TUNNEL_URL" ]; then
                    break
                fi
            fi
        done
        
        if [ -z "$TUNNEL_URL" ]; then
            echo "⚠️  Could not auto-detect cloudflared URL after 10 seconds."
            echo "   Check log: $TUNNEL_LOG"
            echo "   Look for a URL like: https://xxxx-xx-xx-xx-xx.trycloudflare.com"
            echo "   Process PID: $CLOUDFLARED_PID"
            echo ""
            echo "   You can manually check the log:"
            echo "   tail -f $TUNNEL_LOG"
            exit 1
        else
            echo "✅ Tunnel URL: $TUNNEL_URL"
            echo "export TUNNEL_URL=$TUNNEL_URL"
            echo ""
            echo "Tunnel process PID: $CLOUDFLARED_PID"
            echo "Log file: $TUNNEL_LOG"
            echo "To stop: kill $CLOUDFLARED_PID"
        fi
        ;;
    *)
        echo "❌ Unknown tunnel service: $TUNNEL_SERVICE"
        echo "Supported: ngrok, localtunnel, cloudflared"
        exit 1
        ;;
esac

