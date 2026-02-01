#!/bin/bash
# UMCP Dashboard Server Status Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$REPO_ROOT/logs"
PORT="${DASHBOARD_PORT:-8501}"
API_PORT="${API_PORT:-8000}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📊 UMCP Dashboard Server Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Dashboard
DASHBOARD_STATUS="🔴 Stopped"
DASHBOARD_PID=""
if [ -f "$LOG_DIR/dashboard.pid" ]; then
    DASHBOARD_PID=$(cat "$LOG_DIR/dashboard.pid")
    if kill -0 "$DASHBOARD_PID" 2>/dev/null; then
        if curl -s "http://localhost:$PORT" > /dev/null 2>&1; then
            DASHBOARD_STATUS="🟢 Running"
        else
            DASHBOARD_STATUS="🟡 Starting"
        fi
    fi
fi

# Check API
API_STATUS="🔴 Stopped"
API_PID=""
if [ -f "$LOG_DIR/api.pid" ]; then
    API_PID=$(cat "$LOG_DIR/api.pid")
    if kill -0 "$API_PID" 2>/dev/null; then
        if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
            API_STATUS="🟢 Running"
        else
            API_STATUS="🟡 Starting"
        fi
    fi
fi

echo "  Dashboard:"
echo "    Status:  $DASHBOARD_STATUS"
echo "    URL:     http://localhost:$PORT"
[ -n "$DASHBOARD_PID" ] && echo "    PID:     $DASHBOARD_PID"
echo ""
echo "  API:"
echo "    Status:  $API_STATUS"
echo "    URL:     http://localhost:$API_PORT"
[ -n "$API_PID" ] && echo "    PID:     $API_PID"
echo ""

# Show recent log entries if running
if [ "$DASHBOARD_STATUS" = "🟢 Running" ] && [ -f "$LOG_DIR/dashboard.log" ]; then
    echo "  Recent Dashboard Logs:"
    tail -3 "$LOG_DIR/dashboard.log" 2>/dev/null | sed 's/^/    /'
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
