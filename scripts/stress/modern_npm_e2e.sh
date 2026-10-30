#!/usr/bin/env bash
# scripts/stress/modern_npm_e2e.sh — PLAN-MODERN-NPM acceptance gates.
#
# Builds a throwaway project with the three shapes the plan names, then runs
# scripts/stress/modern_npm_probe.js inside it against a real chunked HTTP
# server (so "streaming" is proven by arrival time, not by a mocked promise):
#
#   A. package.json conditional exports — exact subpath, wildcard, import vs
#      require conditions, deep-import fallback
#   B. WHATWG streams — ReadableStream/WritableStream/TransformStream,
#      getReader() chunk delivery, pipeThrough(TextDecoderStream), tee
#   C. diagnostics — .node addon, absent Node built-ins, require(esm),
#      missing package/subpath, native-package recommendation
#
# Requires python3 for the SSE server (same dependency the contract suite has).
#
# Usage: bash scripts/stress/modern_npm_e2e.sh [path-to-sofuu]
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOFUU="${1:-${SOFUU:-$ROOT/sofuu}}"
case "$SOFUU" in
    /*) ;;
    *) SOFUU="$(cd "$(dirname "$SOFUU")" 2>/dev/null && pwd)/$(basename "$SOFUU")" ;;
esac
if [ ! -x "$SOFUU" ]; then
    echo "✗ sofuu binary not found: $SOFUU (run make, or pass a path)" >&2
    exit 1
fi
command -v python3 > /dev/null 2>&1 || { echo "✗ python3 is required for the SSE server" >&2; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sofuu-modern-npm.XXXXXX")"
SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT INT TERM

# ── fixtures ───────────────────────────────────────────────────────
mkdir -p "$WORK/node_modules/native-thing/build" \
         "$WORK/node_modules/modern-pkg/dist" \
         "$WORK/node_modules/modern-pkg/locales" \
         "$WORK/node_modules/dual-pkg/cjs" \
         "$WORK/node_modules/dual-pkg/esm"

cat > "$WORK/node_modules/native-thing/package.json" <<'JSON'
{"name":"native-thing","version":"1.2.3","main":"build/x.node"}
JSON
printf '\xcf\xfa\xed\xfe not javascript' > "$WORK/node_modules/native-thing/build/x.node"

cat > "$WORK/node_modules/modern-pkg/package.json" <<'JSON'
{"name":"modern-pkg","version":"0.9.0","main":"dist/legacy.js",
 "exports":{".":{"import":"./dist/index.mjs","default":"./dist/index.js"},
            "./feature":{"import":"./dist/feature.mjs"},
            "./locales/*":"./locales/*.js"}}
JSON
echo 'export default { where: "index.mjs" };' > "$WORK/node_modules/modern-pkg/dist/index.mjs"
echo 'module.exports = "root-cjs";' > "$WORK/node_modules/modern-pkg/dist/legacy.js"
echo 'export const feature = "feature-mjs";' > "$WORK/node_modules/modern-pkg/dist/feature.mjs"
echo 'export const fr = "bonjour";' > "$WORK/node_modules/modern-pkg/locales/fr.js"
echo 'module.exports = "deep-import";' > "$WORK/node_modules/modern-pkg/extra.js"

cat > "$WORK/node_modules/dual-pkg/package.json" <<'JSON'
{"name":"dual-pkg","version":"3.0.0","exports":{".":{"require":"./cjs/index.cjs","import":"./esm/index.mjs"}}}
JSON
echo 'module.exports = { flavor: "cjs-build" };' > "$WORK/node_modules/dual-pkg/cjs/index.cjs"
echo 'export const flavor = "esm-build"; export default { flavor };' > "$WORK/node_modules/dual-pkg/esm/index.mjs"

# ── chunked SSE server ─────────────────────────────────────────────
cat > "$WORK/sse.py" <<'PY'
import sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for i in range(5):
            payload = ('data: {"i":%d}\n\n' % i).encode()
            self.wfile.write(b"%x\r\n" % len(payload) + payload + b"\r\n")
            self.wfile.flush()
            time.sleep(0.05)          # forces five distinct arrivals
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
    def log_message(self, *a):
        pass

srv = HTTPServer(("127.0.0.1", 0), H)
print(srv.server_address[1], flush=True)
srv.serve_forever()
PY

python3 "$WORK/sse.py" > "$WORK/port.txt" &
SERVER_PID=$!
for _ in $(seq 1 50); do
    [ -s "$WORK/port.txt" ] && break
    sleep 0.1
done
PORT="$(head -1 "$WORK/port.txt" 2>/dev/null)"
if [ -z "$PORT" ]; then
    echo "✗ the SSE server did not start" >&2
    exit 1
fi

# ── run the probe from inside the fixture tree ─────────────────────
cp "$ROOT/scripts/stress/modern_npm_probe.js" "$WORK/probe.js"
cd "$WORK"
SOFUU_SSE_URL="http://127.0.0.1:$PORT/stream" \
SOFUU_NO_SHELL=1 \
    "$SOFUU" run probe.js
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "modern-npm e2e failed (exit $RC)" >&2
    exit 1
fi
exit 0
