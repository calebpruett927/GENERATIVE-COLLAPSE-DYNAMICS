#!/bin/bash
# UMCP Dashboard Server Startup Script
# Usage: ./scripts/start_dashboard.sh [--background] [--port PORT]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PORT="${DASHBOARD_PORT:-8501}"
API_PORT="${API_PORT:-8000}"
BACKGROUND=false
LOG_DIR="$REPO_ROOT/logs"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--background)
            BACKGROUND=true
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "UMCP Dashboard Server"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -b, --background    Run in background as daemon"
            echo "  -p, --port PORT     Dashboard port (default: 8501)"
            echo "      --api-port PORT API port (default: 8000)"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Environment Variables:"
            echo "  DASHBOARD_PORT      Dashboard port (default: 8501)"
            echo "  API_PORT            API port (default: 8000)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create log directory
mkdir -p "$LOG_DIR"

# Activate virtual environment if exists
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

cd "$REPO_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🔬 UMCP Dashboard Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Dashboard: http://localhost:$PORT"
echo "  API:       http://localhost:$API_PORT"
echo "  Logs:      $LOG_DIR/"
echo ""

if [ "$BACKGROUND" = true ]; then
    echo "  Mode:      Background (daemon)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Kill existing processes
    pkill -f "streamlit run src/umcp/dashboard.py" 2>/dev/null || true
    pkill -f "uvicorn umcp.api_umcp:app" 2>/dev/null || true
    sleep 1
    
    # Start API server in background
    echo "Starting API server..."
    nohup python -m uvicorn umcp.api_umcp:app --host 0.0.0.0 --port "$API_PORT" \
        > "$LOG_DIR/api.log" 2>&1 &
    echo $! > "$LOG_DIR/api.pid"
    
    # Start dashboard in background
    echo "Starting Dashboard..."
    nohup streamlit run src/umcp/dashboard.py \
        --server.port "$PORT" \
        --server.headless true \
        --server.address 0.0.0.0 \
        --server.enableCORS false \
        --server.enableXsrfProtection false \
        > "$LOG_DIR/dashboard.log" 2>&1 &
    echo $! > "$LOG_DIR/dashboard.pid"
    
    sleep 3
    
    # Verify servers started
    if curl -s "http://localhost:$PORT" > /dev/null 2>&1; then
        echo "✅ Dashboard started successfully (PID: $(cat $LOG_DIR/dashboard.pid))"
    else
        echo "⚠️  Dashboard may still be starting..."
    fi
    
    if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        echo "✅ API started successfully (PID: $(cat $LOG_DIR/api.pid))"
    else
        echo "⚠️  API may still be starting..."
    fi
    
    echo ""
    echo "To stop: ./scripts/stop_dashboard.sh"
    echo "To view logs: tail -f $LOG_DIR/dashboard.log"
else
    echo "  Mode:      Foreground (Ctrl+C to stop)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Start API in background
    python -m uvicorn umcp.api_umcp:app --host 0.0.0.0 --port "$API_PORT" &
    API_PID=$!
    
    # Trap to cleanup on exit
    trap "kill $API_PID 2>/dev/null; exit" INT TERM EXIT
    
    # Start dashboard in foreground
    streamlit run src/umcp/dashboard.py \
        --server.port "$PORT" \
        --server.headless true \
        --server.address 0.0.0.0
fi
