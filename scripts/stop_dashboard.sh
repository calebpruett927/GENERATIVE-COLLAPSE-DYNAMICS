#!/bin/bash
# UMCP Dashboard Server Stop Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_ROOT/logs"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🛑 Stopping UMCP Dashboard Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Stop dashboard
if [ -f "$LOG_DIR/dashboard.pid" ]; then
    PID=$(cat "$LOG_DIR/dashboard.pid")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "✅ Dashboard stopped (PID: $PID)"
    fi
    rm -f "$LOG_DIR/dashboard.pid"
fi

# Stop API
if [ -f "$LOG_DIR/api.pid" ]; then
    PID=$(cat "$LOG_DIR/api.pid")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "✅ API stopped (PID: $PID)"
    fi
    rm -f "$LOG_DIR/api.pid"
fi

# Also kill by process name as backup
pkill -f "streamlit run src/umcp/dashboard" 2>/dev/null && echo "✅ Killed remaining Streamlit processes" || true
pkill -f "uvicorn umcp.api_umcp:app" 2>/dev/null && echo "✅ Killed remaining API processes" || true

echo ""
echo "All UMCP services stopped."
