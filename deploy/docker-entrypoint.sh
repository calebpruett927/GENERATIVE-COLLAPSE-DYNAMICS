#!/bin/bash
# Docker entrypoint for UMCP Dashboard

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🔬 UMCP Dashboard Container"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Dashboard: http://localhost:${DASHBOARD_PORT:-8501}"
echo "  API:       http://localhost:${API_PORT:-8000}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start API server in background
echo "Starting API server..."
python -m uvicorn umcp.api_umcp:app \
    --host 0.0.0.0 \
    --port "${API_PORT:-8000}" &

# Start dashboard in foreground
echo "Starting Dashboard..."
exec streamlit run src/umcp/dashboard.py \
    --server.port "${DASHBOARD_PORT:-8501}" \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false
