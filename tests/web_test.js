// tests/web_test.js — self-contained engine-level battery for sofuu.web
// (search + open), over local mock endpoints. No network, no API keys.
// (P2-28, AUDIT-2026-09-01: the examples/ fork of this file is deleted;
// tests/ holds the real battery.)
//
// Covers the follow-up flagged in TASKS.md beyond agent_test.js's tool-path
// check (which uses SOFUU_WEB_ENDPOINT with a DDG-shaped mock):
//   duckduckgo  HTML parsing (result__a/result__snippet, uddg redirect
//               decoding, dedup, count cap) · HTTP-error guidance ·
//               no-parsable-results error
//   brave       X-Subscription-Token header auth · q/count query params ·
//               web.results JSON shape · 401 invalid-key · missing-key error
//   tavily      POST body shape (api_key/query/max_results/search_depth) ·
//               results JSON shape · 403 invalid-key · missing-key error
//   open        HTML→text extraction (scripts/styles/entities), text/plain
//               passthrough, maxChars truncation, non-http rejection
//   misc        unknown-engine error
//
// Run:  ./sofuu run tests/web_test.js
// Every target here is a local loopback mock, so the SSRF guard's internal
// opt-out is armed for this process only (P2-3 ships the guard itself).

process.env.SOFUU_WEB_ALLOW_INTERNAL = "1";

let failures = 0;
let skips = 0;
function check(name, cond) {
  if (cond) { console.log("PASS " + name); }
  else { failures++; console.log("FAIL " + name); }
}
function skip(name, why) { skips++; console.log("SKIP " + name + " — " + why); }

/* ── Captured request state (probes assert on these) ─────────────── */

const CAP = {
  braveHeaders: null, braveQuery: null,
  tavilyBody: null,
  ddgQuery: null,
};

/* ── Mock servers ─────────────────────────────────────────────────── */

function ddgHtml() {
  return (
    '<div class="result">' +
    '<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Frust&amp;rut=abc">Rust guide title</a>' +
    '<a class="result__snippet" href="//x">Rust snippet &amp; details</a>' +
    '</div>' +
    '<div class="result">' +
    '<a class="result__a" href="https://example.com/second">Second title</a>' +
    '<a class="result__snippet">Second snippet</a>' +
    '</div>' +
    /* duplicate URL — must be deduped */
    '<a class="result__a" href="https://example.com/second">Second title again</a>' +
    /* ad/junk links — dropped */
    '<a class="result__a" href="https://duckduckgo.com/ad">Ad</a>'
  );
}

function ddgHandler(req, res) {
  const q = String(req.url || "");
  const m = q.match(/[?&]q=([^&]*)/);
  CAP.ddgQuery = m ? decodeURIComponent(m[1]) : "";
  if (CAP.ddgQuery === "ratelimited") {
    res.writeHead(503, { "Content-Type": "text/html" });
    res.send("<html>slow down</html>");
    return;
  }
  if (CAP.ddgQuery === "markupchange") {
    res.writeHead(200, { "Content-Type": "text/html" });
    res.send("<html><body>totally different layout</body></html>");
    return;
  }
  res.writeHead(200, { "Content-Type": "text/html" });
  res.send(ddgHtml());
}

/* The runtime's server req carries {method, url, body} only (no header
 * surface), so header auth is captured on the CLIENT side: the test wraps
 * globalThis.fetch and records the outgoing opts for the next request. */
let FETCH_CAPTURE = null;
const REAL_FETCH = globalThis.fetch;
globalThis.fetch = function (url, opts) {
  if (FETCH_CAPTURE) {
    const cap = FETCH_CAPTURE;
    FETCH_CAPTURE = null;
    try { cap({ url: String(url), opts: opts || {} }); } catch (e) {}
  }
  return REAL_FETCH(url, opts);
};

function braveHandler(req, res) {
  CAP.braveQuery = String(req.url || "");
  /* Header-based rejection isn't observable server-side (no req headers),
   * so 401 on demand via the query — it maps the status to the error. */
  if (CAP.braveQuery.indexOf("q=unauthorized") >= 0) {
    res.writeHead(401, { "Content-Type": "application/json" });
    res.send("{}");
    return;
  }
  res.writeHead(200, { "Content-Type": "application/json" });
  res.send(JSON.stringify({
    web: {
      results: [
        { title: "Brave One", url: "https://brave.example/1", description: "first brave result" },
        { title: "Brave Two", url: "https://brave.example/2", description: "second brave result" },
      ],
    },
  }));
}

function tavilyHandler(req, res) {
  try { CAP.tavilyBody = JSON.parse(req.body || "{}"); } catch (e) { CAP.tavilyBody = null; }
  if (CAP.tavilyBody && CAP.tavilyBody.api_key === "bad-key") {
    res.writeHead(403, { "Content-Type": "application/json" });
    res.send("{}");
    return;
  }
  res.writeHead(200, { "Content-Type": "application/json" });
  res.send(JSON.stringify({
    results: [
      { title: "Tavily One", url: "https://tavily.example/1", content: "first tavily result" },
    ],
  }));
}

function pageHandler(req, res) {
  const u = String(req.url || "");
  if (u.indexOf("/plain") >= 0) {
    res.writeHead(200, { "Content-Type": "text/plain" });
    res.send("just plain text");
    return;
  }
  if (u.indexOf("/noct") >= 0) {
    res.writeHead(200, { "Content-Type": "application/octet-stream" });
    res.send("<html><body>looks like html but no content-type</body></html>");
    return;
  }
  res.writeHead(200, { "Content-Type": "text/html" });
  res.send(
    "<html><head><title>T</title><style>.x{color:red}</style>" +
    "<script>var a = 1;</script></head><body>" +
    "<h1>Hello &amp; welcome</h1><p>Line one</p><p>Line two</p>" +
    "<noscript>no script content</noscript><!-- a comment -->" +
    "</body></html>"
  );
}

/* ── Main ─────────────────────────────────────────────────────────── */

async function main() {
  check("sofuu.web.search exists", !!(sofuu.web && typeof sofuu.web.search === "function"));
  check("sofuu.web.open exists", !!(sofuu.web && typeof sofuu.web.open === "function"));

  /* Bind four mock servers. */
  const servers = {};
  const specs = [["ddg", ddgHandler], ["brave", braveHandler], ["tavily", tavilyHandler], ["page", pageHandler]];
  const base = 20500 + (Date.now() % 4000);
  let ok = true;
  for (const [name, handler] of specs) {
    let bound = false;
    for (let i = 0; i < 8 && !bound; i++) {
      try {
        const s = sofuu.createServer(handler);
        s.listen(base + i * 37 + (name.length * 7), "127.0.0.1");
        servers[name] = { server: s, port: base + i * 37 + (name.length * 7) };
        bound = true;
      } catch (e) {}
    }
    if (!bound) ok = false;
  }
  if (!ok) { console.log("FAIL could not bind mock ports"); process.exit(1); }
  const DDG = "http://127.0.0.1:" + servers.ddg.port + "/html/";
  const BRAVE = "http://127.0.0.1:" + servers.brave.port + "/res/v1/web/search";
  const TAVILY = "http://127.0.0.1:" + servers.tavily.port + "/search";
  const PAGE = "http://127.0.0.1:" + servers.page.port + "/";
  console.log("mocks: ddg=" + DDG + " brave=" + BRAVE + " tavily=" + TAVILY);

  /* Keys must not leak in from the environment for the missing-key checks. */
  delete process.env.BRAVE_API_KEY;
  delete process.env.TAVILY_API_KEY;
  delete process.env.SOFUU_WEB_ENGINE;
  process.env.SOFUU_WEB_ENDPOINT = DDG;

  /* ── duckduckgo ─────────────────────────────────────────────── */
  {
    const r = await sofuu.web.search("rust vs zig", { count: 5 });
    check("ddg engine selected by default", r.engine === "duckduckgo");
    check("ddg query echoed", r.query === "rust vs zig");
    check("ddg uddg redirect decoded",
      r.results.length >= 2 && r.results[0].url === "https://example.com/rust");
    check("ddg titles stripped + entities decoded", r.results[0].title === "Rust guide title");
    check("ddg snippet attached + entity decoded", (r.results[0].snippet || "").indexOf("Rust snippet & details") >= 0);
    check("ddg dedup by url (2 unique, not 3)", r.results.filter(x => x.url === "https://example.com/second").length === 1);
    check("ddg junk duckduckgo.com links dropped", !r.results.some(x => x.url.indexOf("duckduckgo.com") >= 0));
    check("ddg count cap respected", r.results.length <= 5);
    check("ddg CAP saw the query", CAP.ddgQuery === "rust vs zig");
    check("ddg ms recorded", typeof r.ms === "number" && r.ms >= 0);
  }
  {
    const r = await sofuu.web.search("rust vs zig", { count: 1 });
    check("ddg count=1 truncates", r.results.length === 1);
  }
  {
    let err = "";
    try { await sofuu.web.search("ratelimited"); } catch (e) { err = String(e && e.message || e); }
    check("ddg HTTP error names the fallback engines", err.indexOf("503") >= 0 && err.indexOf("SOFUU_WEB_ENGINE") >= 0);
  }
  {
    let err = "";
    try { await sofuu.web.search("markupchange"); } catch (e) { err = String(e && e.message || e); }
    check("ddg unparsable markup errors loudly", err.indexOf("no parsable results") >= 0);
  }

  /* ── brave ──────────────────────────────────────────────────── */
  {
    let cap = null;
    FETCH_CAPTURE = (c) => { cap = c; };
    const r = await sofuu.web.search("brave query", { engine: "brave", count: 2, api_key: "tok-123", endpoint: BRAVE });
    check("brave engine selected", r.engine === "brave");
    check("brave auth header sent (client-side capture)", !!cap &&
      String((cap.opts.headers && cap.opts.headers["X-Subscription-Token"]) || "") === "tok-123");
    check("brave GET to the endpoint", !!cap && cap.opts.method === "GET" && cap.url.indexOf(BRAVE) === 0);
    check("brave query param carried", CAP.braveQuery.indexOf("q=brave%20query") >= 0);
    check("brave count param carried", CAP.braveQuery.indexOf("count=2") >= 0);
    check("brave results mapped", r.results.length === 2 &&
      r.results[0].title === "Brave One" && r.results[0].url === "https://brave.example/1" &&
      r.results[0].snippet === "first brave result");
  }
  {
    let err = "";
    try { await sofuu.web.search("unauthorized", { engine: "brave", api_key: "k", endpoint: BRAVE }); } catch (e) { err = String(e && e.message || e); }
    check("brave 401 → invalid API key", err.indexOf("invalid API key") >= 0);
  }
  {
    let err = "";
    try { await sofuu.web.search("x", { engine: "brave", endpoint: BRAVE }); } catch (e) { err = String(e && e.message || e); }
    check("brave missing key is an actionable error", err.indexOf("brave") >= 0 && err.indexOf("BRAVE_API_KEY") >= 0);
  }

  /* ── tavily ─────────────────────────────────────────────────── */
  {
    const r = await sofuu.web.search("tavily query", { engine: "tavily", count: 3, api_key: "tv-1", depth: "advanced", endpoint: TAVILY });
    check("tavily engine selected", r.engine === "tavily");
    check("tavily POST body shape", !!CAP.tavilyBody &&
      CAP.tavilyBody.api_key === "tv-1" &&
      CAP.tavilyBody.query === "tavily query" &&
      CAP.tavilyBody.max_results === 3 &&
      CAP.tavilyBody.search_depth === "advanced");
    check("tavily results mapped", r.results.length === 1 &&
      r.results[0].title === "Tavily One" && r.results[0].snippet === "first tavily result");
  }
  {
    let err = "";
    try { await sofuu.web.search("x", { engine: "tavily", endpoint: TAVILY }); } catch (e) { err = String(e && e.message || e); }
    check("tavily missing key is an actionable error", err.indexOf("tavily") >= 0 && err.indexOf("TAVILY_API_KEY") >= 0);
  }
  {
    let err = "";
    try { await sofuu.web.search("x", { engine: "tavily", api_key: "bad-key", endpoint: TAVILY }); } catch (e) { err = String(e && e.message || e); }
    check("tavily 403 → invalid API key", err.indexOf("invalid API key") >= 0);
  }

  /* ── open ───────────────────────────────────────────────────── */
  {
    const p = await sofuu.web.open(PAGE + "article");
    check("open: html stripped to text", p.text.indexOf("<script") < 0 && p.text.indexOf("<style") < 0 && p.text.indexOf("<") < 0);
    check("open: script/style/noscript/comment content removed",
      p.text.indexOf("var a") < 0 && p.text.indexOf("color:red") < 0 &&
      p.text.indexOf("no script content") < 0 && p.text.indexOf("a comment") < 0);
    check("open: entity decoded", p.text.indexOf("Hello & welcome") >= 0);
    check("open: block tags become newlines", p.text.indexOf("Line one") >= 0 && p.text.indexOf("Line two") >= 0 &&
      p.text.indexOf("Line one\n") >= 0);
    check("open: status + url echoed", p.status === 200 && p.url === PAGE + "article");
  }
  {
    const p = await sofuu.web.open(PAGE + "plain");
    check("open: text/plain passes through verbatim", p.text === "just plain text");
  }
  {
    const p = await sofuu.web.open(PAGE + "noct", { maxChars: 10000 });
    check("open: html-shaped body without content-type still stripped", p.text.indexOf("<html") < 0);
  }
  {
    const p = await sofuu.web.open(PAGE + "article", { maxChars: 10 });
    check("open: maxChars truncation marker", p.text.length < 60 && p.text.indexOf("truncated at 10") >= 0);
  }
  {
    let err = "";
    try { await sofuu.web.open("ftp://example.com/x"); } catch (e) { err = String(e && e.message || e); }
    check("open: non-http rejected", err.indexOf("http(s)") >= 0);
  }

  /* ── misc ───────────────────────────────────────────────────── */
  {
    let err = "";
    try { await sofuu.web.search("x", { engine: "nope" }); } catch (e) { err = String(e && e.message || e); }
    check("unknown engine error lists the choices", err.indexOf("nope") >= 0 && err.indexOf("duckduckgo") >= 0);
  }
  {
    let err = "";
    try { await sofuu.web.search("   "); } catch (e) { err = String(e && e.message || e); }
    check("empty query rejected", err.indexOf("query required") >= 0);
  }

  console.log(failures === 0
    ? "\nWEB TEST: ALL PASSED" + (skips ? " (" + skips + " skipped)" : "")
    : "\nWEB TEST: " + failures + " FAILURE(S)");
  process.exit(failures === 0 ? 0 : 1);
}

main().catch(e => {
  console.error("FAIL exception: " + (e && e.stack || e));
  process.exit(1);
});
