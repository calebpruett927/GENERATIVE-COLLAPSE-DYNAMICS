#!/usr/bin/env bash
# download_cern_data.sh — Reproducible download of CERN Open Data archive
#
# Downloads all datasets described in data/cern_open_data/MANIFEST.yaml
# License: CC0-1.0 for all datasets
# Total: ~225 MB, 1,178,244 events
#
# Usage:
#   bash scripts/download_cern_data.sh          # Download all
#   bash scripts/download_cern_data.sh --check  # Verify checksums only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/cern_open_data"

# ── CMS Record 545 (Derived Run2011A datasets) ──────────────────────
CMS_DIR="$DATA/cms_dimuon"
CMS_BASE="http://opendata.cern.ch/record/545/files"
CMS_FILES=(
    "Dimuon_DoubleMu.csv"
    "Jpsimumu.csv"
    "Ymumu.csv"
    "Zmumu.csv"
    "Zee.csv"
    "Wmunu.csv"
    "Wenu.csv"
)

# ── ATLAS Higgs ML Challenge (Record 328) ────────────────────────────
HIGGS_DIR="$DATA/atlas_higgs_ml"
HIGGS_URL="http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz"
HIGGS_FILE="atlas-higgs-challenge-2014-v2.csv"

# ── ATLAS Higgs Couplings (HEPData ins2104706) ──────────────────────
ATLAS_DIR="$DATA/atlas_higgs_couplings"
ATLAS_URL="https://www.hepdata.net/download/submission/ins2104706/1/original"

# ── Checksums (from MANIFEST.yaml) ──────────────────────────────────
declare -A CHECKSUMS=(
    ["Dimuon_DoubleMu.csv"]="58caa45c580f8bcf4b457281c8525a56de0e0b8eb0e0b29e86a9b655a8e64d35"
    ["Jpsimumu.csv"]="95bf87f7c14758b445d9cb76bc0111affc1ce70831453dde34f687310685e59f"
    ["Wenu.csv"]="03d32eaf82fc6821b9f45caec4394ce8de6007a38cf1b09c9eb57f87f8fcaf92"
    ["Wmunu.csv"]="8031d7f62935b75e02f066caf8d20e7661f98effe18dcc25b63c604bd158b04d"
    ["Ymumu.csv"]="b2757444c7339f8e43bda9b2f0066b821b20ab6038b9f188375b717f0fd4ea88"
    ["Zee.csv"]="ae6fbad67d78ffa71ac9044bcac69d1118a54d6e6c312e1ae234ac542c10c349"
    ["Zmumu.csv"]="7782778f8417d2c732f4a64efcbfceb6192c97c3bcfd21c0cf1322d38ed965d1"
    ["atlas-higgs-challenge-2014-v2.csv"]="948bc6a393495d7f807627e6f2ddc358e325d2db8e03644d98011c7dc640d135"
)

verify_checksum() {
    local file="$1" expected="$2"
    local actual
    actual=$(sha256sum "$file" | awk '{print $1}')
    if [[ "$actual" == "$expected" ]]; then
        echo "  ✓ $(basename "$file")"
        return 0
    else
        echo "  ✗ $(basename "$file"): expected ${expected:0:12}... got ${actual:0:12}..."
        return 1
    fi
}

if [[ "${1:-}" == "--check" ]]; then
    echo "Verifying CERN data checksums..."
    fail=0
    for f in "${CMS_FILES[@]}"; do
        if [[ -f "$CMS_DIR/$f" ]]; then
            verify_checksum "$CMS_DIR/$f" "${CHECKSUMS[$f]}" || fail=1
        else
            echo "  ? $f not found"
            fail=1
        fi
    done
    if [[ -f "$HIGGS_DIR/$HIGGS_FILE" ]]; then
        verify_checksum "$HIGGS_DIR/$HIGGS_FILE" "${CHECKSUMS[$HIGGS_FILE]}" || fail=1
    else
        echo "  ? $HIGGS_FILE not found"
        fail=1
    fi
    exit $fail
fi

echo "╔══════════════════════════════════════════════════════╗"
echo "║  CERN Open Data Download — 3 datasets, ~225 MB      ║"
echo "║  License: CC0-1.0 (all datasets)                    ║"
echo "╚══════════════════════════════════════════════════════╝"

# ── Download CMS Record 545 ─────────────────────────────────────────
echo ""
echo "── CMS Record 545: Derived Run2011A (7 files, ~38 MB) ──"
mkdir -p "$CMS_DIR"
for f in "${CMS_FILES[@]}"; do
    if [[ -f "$CMS_DIR/$f" ]]; then
        echo "  ✓ $f already exists"
    else
        echo "  ↓ Downloading $f..."
        curl -sL -o "$CMS_DIR/$f" "$CMS_BASE/$f"
    fi
done

# ── Download ATLAS Higgs ML Challenge ────────────────────────────────
echo ""
echo "── ATLAS Higgs ML Challenge: Record 328 (818k events, ~187 MB) ──"
mkdir -p "$HIGGS_DIR"
if [[ -f "$HIGGS_DIR/$HIGGS_FILE" ]]; then
    echo "  ✓ $HIGGS_FILE already exists"
else
    echo "  ↓ Downloading atlas-higgs-challenge-2014-v2.csv.gz..."
    curl -sL -o "$HIGGS_DIR/${HIGGS_FILE}.gz" "$HIGGS_URL"
    echo "  ↓ Decompressing..."
    gunzip "$HIGGS_DIR/${HIGGS_FILE}.gz"
fi

# ── Download ATLAS Higgs Couplings ───────────────────────────────────
echo ""
echo "── ATLAS Higgs Couplings: HEPData ins2104706 (58 YAML tables) ──"
mkdir -p "$ATLAS_DIR"
if [[ -f "$ATLAS_DIR/submission.yaml" ]]; then
    echo "  ✓ ATLAS Higgs coupling data already exists"
else
    echo "  ↓ Downloading HEPData submission..."
    curl -sL -o "$ATLAS_DIR/_download.zip" "$ATLAS_URL"
    cd "$ATLAS_DIR"
    unzip -o _download.zip
    rm -f _download.zip *.png  # Remove thumbnails, keep YAML only
    cd "$ROOT"
fi

# ── Verify checksums ────────────────────────────────────────────────
echo ""
echo "── Verifying checksums ──"
fail=0
for f in "${CMS_FILES[@]}"; do
    verify_checksum "$CMS_DIR/$f" "${CHECKSUMS[$f]}" || fail=1
done
verify_checksum "$HIGGS_DIR/$HIGGS_FILE" "${CHECKSUMS[$HIGGS_FILE]}" || fail=1

echo ""
if [[ $fail -eq 0 ]]; then
    total_events=$(( 100000 + 20000 + 20000 + 10000 + 10000 + 100000 + 100000 + 818238 ))
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  Download complete: $total_events events            ║"
    echo "║  All checksums verified ✓                           ║"
    echo "╚══════════════════════════════════════════════════════╝"
else
    echo "⚠  Some checksums failed — re-download affected files"
    exit 1
fi
