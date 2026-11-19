#!/usr/bin/env bash
set -e

# build.sh — Build the pure Kaze example agents into standalone executables.
#
# Compiling an agent packages the routine. It does not connect a model, an
# MCP service, a policy or any hardware: those remain runtime dependencies
# that each example's README states.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFUU="${SOFUU_BIN:-${SCRIPT_DIR}/../sofuu}"

if [ ! -f "$SOFUU" ]; then
    echo "Error: sofuu binary not found at $SOFUU (set SOFUU_BIN to override)" >&2
    exit 1
fi

echo "Building Pure Kaze Example Agents..."

# 1. Read-only real workflow: local file through an MCP tool, answered by a model.
echo "==> [1/4] Compiling Local Brief (read-only, real MCP + model)..."
"$SOFUU" compile "${SCRIPT_DIR}/local_brief/agent.kaze" -o "${SCRIPT_DIR}/local_brief/local-brief"
chmod +x "${SCRIPT_DIR}/local_brief/local-brief"

# 2. Fintech language example: Quant Sentinel
echo "==> [2/4] Compiling Quant Sentinel (Fintech language example)..."
"$SOFUU" compile "${SCRIPT_DIR}/quant_sentinel/agent.kaze" -o "${SCRIPT_DIR}/quant_sentinel/quant-sentinel"
chmod +x "${SCRIPT_DIR}/quant_sentinel/quant-sentinel"

# 3. Healthcare language example: Med Protocol
echo "==> [3/4] Compiling Med Protocol (Healthcare language example)..."
"$SOFUU" compile "${SCRIPT_DIR}/med_protocol/agent.kaze" -o "${SCRIPT_DIR}/med_protocol/med-protocol"
chmod +x "${SCRIPT_DIR}/med_protocol/med-protocol"

# 4. Agri-Tech language example: Agri Sentinel
echo "==> [4/4] Compiling Agri Sentinel (Agri-Tech language example)..."
"$SOFUU" compile "${SCRIPT_DIR}/agri_sentinel/agent.kaze" -o "${SCRIPT_DIR}/agri_sentinel/agri-sentinel"
chmod +x "${SCRIPT_DIR}/agri_sentinel/agri-sentinel"

echo "All 4 agents compiled successfully into standalone native binaries!"
ls -lh "${SCRIPT_DIR}/local_brief/local-brief" "${SCRIPT_DIR}/quant_sentinel/quant-sentinel" "${SCRIPT_DIR}/med_protocol/med-protocol" "${SCRIPT_DIR}/agri_sentinel/agri-sentinel"
