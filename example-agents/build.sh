#!/usr/bin/env bash
set -e

# build.sh — Build all four 100% Pure Kaze standalone AI agents
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFUU="${SCRIPT_DIR}/../sofuu"

if [ ! -f "$SOFUU" ]; then
    echo "Error: sofuu binary not found at $SOFUU" >&2
    exit 1
fi

echo "Building Pure Kaze Example Agents..."

# 1. Fintech: Quant Sentinel
echo "==> [1/4] Compiling Quant Sentinel (Fintech)..."
"$SOFUU" compile "${SCRIPT_DIR}/quant_sentinel/agent.kaze" -o "${SCRIPT_DIR}/quant_sentinel/quant-sentinel"
chmod +x "${SCRIPT_DIR}/quant_sentinel/quant-sentinel"

# 2. Healthcare: Med Protocol
echo "==> [2/4] Compiling Med Protocol (Healthcare)..."
"$SOFUU" compile "${SCRIPT_DIR}/med_protocol/agent.kaze" -o "${SCRIPT_DIR}/med_protocol/med-protocol"
chmod +x "${SCRIPT_DIR}/med_protocol/med-protocol"

# 3. Agri-Tech: Agri Sentinel
echo "==> [3/4] Compiling Agri Sentinel (Agri-Tech)..."
"$SOFUU" compile "${SCRIPT_DIR}/agri_sentinel/agent.kaze" -o "${SCRIPT_DIR}/agri_sentinel/agri-sentinel"
chmod +x "${SCRIPT_DIR}/agri_sentinel/agri-sentinel"

# 4. Simulated Trading: Paper Trader
echo "==> [4/4] Compiling Paper Trader (Simulated Market Analysis & Paper Execution)..."
"$SOFUU" compile "${SCRIPT_DIR}/paper_trader/agent.kaze" -o "${SCRIPT_DIR}/paper_trader/paper-trader"
chmod +x "${SCRIPT_DIR}/paper_trader/paper-trader"

echo "All 4 agents compiled successfully into standalone native binaries!"
ls -lh "${SCRIPT_DIR}/quant_sentinel/quant-sentinel" "${SCRIPT_DIR}/med_protocol/med-protocol" "${SCRIPT_DIR}/agri_sentinel/agri-sentinel" "${SCRIPT_DIR}/paper_trader/paper-trader"
