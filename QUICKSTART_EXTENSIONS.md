# UMCP Extensions Quick Start

Get up and running with UMCP extensions in 5 minutes.

## 🚀 Installation

```bash
# Install UMCP with all extensions
pip install -e ".[extensions]"

# Or install individually
pip install -e ".[viz]"  # Visualization only
pip install -e ".[api]"  # API only
```

## ✅ 1. Continuous Ledger (Zero Config)

The continuous ledger is **automatically enabled** - no configuration needed!

```bash
# Run validation
umcp validate --out receipt.json

# Ledger is automatically created/updated at ledger/return_log.csv
cat ledger/return_log.csv
```

**Output**:
```csv
timestamp,run_status,delta_kappa,stiffness,omega,curvature
2026-01-19T00:17:31Z,CONFORMANT,,0.000000,0.000000,0.000000
2026-01-19T00:17:49Z,CONFORMANT,,0.000000,0.000000,0.000000
```

## 📊 2. Visualization Dashboard

Interactive Streamlit dashboard for exploring validation history.

```bash
# Start dashboard
streamlit run visualize_umcp.py

# Opens browser at http://localhost:8501
```

**Features**:
- 🎯 Phase space plot (ω vs C with regime colors)
- 📈 Time series of invariants
- 📋 Latest receipt summary
- 💾 Download ledger data

**Screenshot**: See phase space transitions and stability trends over time.

## 🔌 3. Public Audit API

RESTful API for programmatic access to UMCP data.

```bash
# Start API server
uvicorn api_umcp:app --reload

# API runs at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Quick Test**:
```bash
# Health check
curl http://localhost:8000/health

# Current regime
curl http://localhost:8000/regime

# Statistics
curl http://localhost:8000/stats

# Full ledger
curl http://localhost:8000/ledger
```

**Response Example**:
```json
{
  "regime": "Stable",
  "omega": 0.000000,
  "F": 1.000000,
  "S": 0.000000,
  "C": 0.000000,
  "timestamp": "2026-01-19T00:17:49Z"
}
```

## 🔄 Complete Workflow

```bash
# 1. Run validation (creates ledger entry)
umcp validate --out receipt.json

# 2. View in dashboard
streamlit run visualize_umcp.py &

# 3. Start API
uvicorn api_umcp:app --reload &

# 4. Query API
curl http://localhost:8000/stats | jq
```

## 📦 Production Deployment

### Docker Compose (Recommended)

```yaml
version: '3.8'
services:
  umcp-api:
    build: .
    ports:
      - "8000:8000"
    command: uvicorn api_umcp:app --host 0.0.0.0
    volumes:
      - ./ledger:/app/ledger
  
  umcp-dashboard:
    build: .
    ports:
      - "8501:8501"
    command: streamlit run visualize_umcp.py --server.port 8501
    volumes:
      - ./ledger:/app/ledger
```

### Systemd Service

```ini
[Unit]
Description=UMCP Audit API
After=network.target

[Service]
Type=simple
User=umcp
WorkingDirectory=/opt/umcp
ExecStart=/opt/umcp/.venv/bin/uvicorn api_umcp:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## 🔧 Configuration

### Ledger Location

Default: `ledger/return_log.csv`

To customize, set environment variable:
```bash
export UMCP_LEDGER_PATH=/path/to/custom/ledger.csv
```

### API Authentication (Production)

Add JWT authentication:
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/regime")
async def get_regime(token: str = Depends(security)):
    # Verify token
    pass
```

### Dashboard Theme

Edit `visualize_umcp.py`:
```python
st.set_page_config(
    page_title="UMCP Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    theme={
        "primaryColor": "#FF4B4B",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#262730",
        "font": "sans serif"
    }
)
```

## 📈 Monitoring

### Prometheus Metrics

Add to `api_umcp.py`:
```python
from prometheus_client import Counter, Histogram, generate_latest

validation_counter = Counter('umcp_validations_total', 'Total validations')
validation_duration = Histogram('umcp_validation_duration_seconds', 'Validation duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Grafana Dashboard

Import `grafana-dashboard.json` for pre-built visualization:
- Validation success rate over time
- Regime distribution pie chart
- Invariant trends (ω, C, S)
- API request rate

## 🐛 Troubleshooting

### Ledger Not Created

```bash
# Check permissions
ls -la ledger/

# Manually create if needed
mkdir -p ledger
touch ledger/return_log.csv
```

### Dashboard Won't Start

```bash
# Check Streamlit installation
streamlit --version

# Reinstall if needed
pip install --force-reinstall streamlit pandas plotly
```

### API 404 Errors

```bash
# Ensure you're in repo root
cd /path/to/UMCP-Metadata-Runnable-Code

# Check files exist
ls receipt.json
ls ledger/return_log.csv
ls outputs/invariants.csv
```

## 🎯 Next Steps

1. **Adaptive Thresholds**: Auto-adjust regime boundaries based on rolling statistics
2. **Distributed Validation**: Parallelize validation across multiple casepacks
3. **ML Integration**: Train models to predict regime transitions
4. **Epistemic Viz**: Animate return cycles as orbital plots

See [EXTENSIONS.md](EXTENSIONS.md) for implementation details.

## 📚 Resources

- [UMCP Documentation](README.md)
- [System Architecture](SYSTEM_ARCHITECTURE.md)
- [Extension Details](EXTENSIONS.md)
- [API Reference](http://localhost:8000/docs)

---

*Built with ❤️ for reproducible science*
