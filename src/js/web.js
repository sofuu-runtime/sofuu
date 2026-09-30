/* src/js/web.js — sofuu.web: provider-neutral web search + page reading.
 *
 * Shipped JS, eval'd once per engine context by the Rust seam
 * (crates/sofuu-core/src/shipped.rs, called from
 * sofuu_rust_register_engine_js). Pure JS over the runtime's own `fetch`
 * — no new host functions, no new C.
 *
 * Engines:
 *   duckduckgo (default) — NO API KEY. Parses the public HTML endpoint
 *     (html.duckduckgo.com/html/). Free and keyless, but it is best-effort
 *     scraping: if it rate-limits or changes markup, set SOFUU_WEB_ENGINE
 *     (or pass opts.engine) to an API engine.
 *   brave — needs an API key (opts.api_key or BRAVE_API_KEY).
 *   tavily — needs an API key (opts.api_key or TAVILY_API_KEY); designed
 *     for LLM consumers.
 *
 * API:
 *   const r = await sofuu.web.search("rust vs zig", { count: 8 });
 *   // → { query, engine, results: [{title, url, snippet}], ms, note? }
 *   const page = await sofuu.web.open("https://example.com/x");
 *   // → { url, status, text }  (HTML stripped to readable text, capped)
 *
 * Both are async and safe to call from any script (`sofuu run`), the chat
 * agent loop, and agent definitions (the built-in web_search / web_open
 * tools in agent.js wrap them).
 */
(function () {
  'use strict';
  var g = globalThis;
  var VERSION = '0.1.0';

  if (typeof g.sofuu === 'undefined' || g.sofuu === null) g.sofuu = {};
  var sofuu = g.sofuu;
  if (typeof sofuu.web !== 'undefined' && sofuu.web && sofuu.web._loaded) return;
  if (!sofuu.web) sofuu.web = {};

  function env(name) {
    try { return (typeof process !== 'undefined' && process.env && process.env[name]) || ''; }
    catch (e) { return ''; }
  }

  /* ── fetch with timeout (same pattern as the chat driver) ─────── */
  /* skipGuard: true ONLY for operator-supplied endpoints — the
   * SOFUU_WEB_ENDPOINT env var or a caller's opts.endpoint (proxy /
   * self-host / test mocks). Model-chosen URLs (web.open arguments,
   * engine result links) always go through the SSRF guard. */
  function fetchWithTimeout(url, opts, ms, skipGuard) {
    guardInternalUrl(url, skipGuard);
    var limit = ms || 10000;
    /* js-8 (AUDIT-2026-09-07): Promise.race alone left the losing fetch
     * running — it used to outlive the deadline under the global 120s
     * curl cap. Hand curl the same deadline via fetch's timeoutMs option
     * (becomes CURLOPT_TIMEOUT_MS), +750ms slack so the JS race still
     * rejects first with its message and curl reaps the zombie after. */
    var o = {};
    if (opts) { for (var k in opts) o[k] = opts[k]; }
    if (!o.timeoutMs) o.timeoutMs = limit + 750;
    var timer = null;
    var to = new Promise(function (_, rej) {
      timer = setTimeout(function () { rej(new Error('timeout after ' + limit + 'ms')); }, limit);
    });
    return Promise.race([fetch(url, o), to]).finally(function () {
      if (timer) { try { clearTimeout(timer); } catch (e) {} }
    });
  }

  /* ── SSRF guard (P2-3, AUDIT-2026-09-01) ─────────────────────────
   * web.open/web.search results flow back into MODEL CONTEXT, so a
   * prompt-injected "fetch http://169.254.169.254/latest/meta-data/"
   * reads cloud credentials straight into the conversation. Refuse the
   * internal/private targets: loopback names + IPs, RFC1918, link-local
   * (incl. the cloud metadata endpoint), unique-local IPv6, and
   * non-http(s) schemes. DNS-rebinding is out of scope for a JS-level
   * guard — this closes the named surfaces (localhost / RFC1918 /
   * 169.254.169.254). Opt-out for deliberate internal use:
   * SOFUU_WEB_ALLOW_INTERNAL=1. */
  var IPV4_BLOCK_RE = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/;
  /* js-7 (AUDIT-2026-09-07): classify a decoded 32-bit IPv4 VALUE against
   * the private/internal blocks — spelling-independent. */
  function isInternalIpv4(v) {
    var a = (v >>> 24) & 255, b = (v >>> 16) & 255;
    if (a === 10) return true;                     /* 10/8 */
    if (a === 172 && b >= 16 && b <= 31) return true; /* 172.16/12 */
    if (a === 192 && b === 168) return true;       /* 192.168/16 */
    if (a === 169 && b === 254) return true;       /* link-local + metadata */
    if (a === 127) return true;                    /* loopback */
    if (a === 0) return true;                      /* 0.0.0.0/8 */
    return false;
  }
  /* inet_aton semantics: parts may be decimal, hex (0x..) or octal (0..);
   * fewer than 4 parts = the last part fills the remaining bytes
   * (2130706433 = 127.0.0.1, 127.1 = 127.0.0.1). Returns the 32-bit value,
   * or -1 when the spelling doesn't decode (let fetch fail on it). */
  function parseInetAtonParts(parts) {
    var nums = [];
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i], n;
      if (p.length > 1 && p[0] === '0') {
        if (p[1] === 'x' || p[1] === 'X') n = parseInt(p, 16);
        else n = parseInt(p, 8);
      } else {
        n = parseInt(p, 10);
      }
      if (isNaN(n) || n < 0) return -1;
      nums.push(n);
    }
    if (parts.length === 1) { if (nums[0] > 0xFFFFFFFF) return -1; return nums[0] >>> 0; }
    if (parts.length === 2) { if (nums[0] > 255 || nums[1] > 0xFFFFFF) return -1; return ((nums[0] << 24) | nums[1]) >>> 0; }
    if (parts.length === 3) {
      if (nums[0] > 255 || nums[1] > 255 || nums[2] > 0xFFFF) return -1;
      return ((nums[0] << 24) | (nums[1] << 16) | nums[2]) >>> 0;
    }
    if (nums[0] > 255 || nums[1] > 255 || nums[2] > 255 || nums[3] > 255) return -1;
    return ((nums[0] << 24) | (nums[1] << 16) | (nums[2] << 8) | nums[3]) >>> 0;
  }
  function isInternalHost(host) {
    var h = String(host || '').toLowerCase();
    if (!h) return true;
    if (h === 'localhost' || h === '127.0.0.1' || h === '0.0.0.0' || h === '::1' || h === '[::1]') return true;
    if (h === 'metadata.google.internal' || h === '169.254.169.254') return true;
    var m4 = h.match(IPV4_BLOCK_RE);
    /* A part like "0177" is octal, not decimal 177 — such quads decode
     * through inet_aton below instead of the plain decimal branch. */
    if (m4 && !/(^|\.)0\d/.test(h)) {
      var a = (+m4[1]), b = (+m4[2]), c = (+m4[3]), d = (+m4[4]);
      if (a > 255 || b > 255 || c > 255 || d > 255) return false; /* weird literal — let it fail at fetch */
      return isInternalIpv4(((a << 24) | (b << 16) | (c << 8) | d) >>> 0);
    }
    /* js-7: alternate IPv4 spellings that the dotted-quad regex above
     * misses (or mishandles) — 0x7f000001, 2130706433, 127.1, 0177.0.0.1.
     * Decode with inet_aton rules and classify the VALUE, not the spelling. */
    var parts = h.split('.');
    if (parts.length >= 1 && parts.length <= 4) {
      var allNumeric = true;
      for (var i = 0; i < parts.length; i++) {
        if (!/^(0[xX][0-9a-fA-F]+|[0-9]+)$/.test(parts[i])) { allNumeric = false; break; }
      }
      if (allNumeric) {
        var v = parseInetAtonParts(parts);
        if (v >= 0) return isInternalIpv4(v);
        /* undecodable numeric soup — let it fail at fetch */
      }
    }
    if (h.indexOf(':') >= 0) {
      /* IPv6 literal: refuse loopback, link-local fe80::/10, unique-local fc00::/7. */
      var h6 = h.replace(/^\[|\]$/g, '');
      if (h6 === '::1' || h6 === '::') return true;
      /* js-7: IPv4-mapped tails — ::ffff:169.254.169.254 is the metadata
       * endpoint in mapped spelling. Decode the embedded v4 (dotted or
       * two hex groups) and classify it. */
      var mapped = h6.match(/^::ffff:(.+)$/i);
      if (mapped) {
        var t = mapped[1];
        var tm = t.match(IPV4_BLOCK_RE);
        if (tm) {
          var mv = parseInetAtonParts([tm[1], tm[2], tm[3], tm[4]]);
          if (mv >= 0) return isInternalIpv4(mv);
        } else {
          var hm = t.match(/^([0-9a-fA-F]{1,4}):([0-9a-fA-F]{1,4})$/);
          if (hm) return isInternalIpv4(((parseInt(hm[1], 16) << 16) | parseInt(hm[2], 16)) >>> 0);
        }
      }
      if (/^f[cd][0-9a-f]{2}:/i.test(h6)) return true;
      if (/^fe[89ab][0-9a-f]:/i.test(h6)) return true;
      return false;
    }
    return false;
  }
  function guardInternalUrl(url, skipGuard) {
    if (skipGuard) return;
    if (env('SOFUU_WEB_ALLOW_INTERNAL') === '1') return;
    var u = String(url == null ? '' : url);
    var schemeM = u.match(/^([a-zA-Z][a-zA-Z0-9+.-]*):\/\//);
    if (schemeM && !/^https?$/i.test(schemeM[1])) {
      throw new Error('web: refused non-http(s) URL: ' + u.slice(0, 120));
    }
    var hostM = u.match(/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\/([^\/?#]+)/);
    if (!hostM) return; /* relative — resolves against nothing out there; let fetch fail */
    /* P1-15 (AUDIT-2026-09-07): the HOST is the authority MINUS userinfo and
     * port. The old code stripped only the port, so `http://x@169.254.169.254/`
     * classified the "host" as `x@169.254.169.254` and sailed past this
     * guard. Userinfo may contain ':' but not an unencoded '@', so split on
     * the LAST '@'; bracket-wrap IPv6 before the port strip. */
    var auth = hostM[1];
    var at = auth.lastIndexOf('@');
    var hostport = at >= 0 ? auth.slice(at + 1) : auth;
    var host;
    if (hostport.charAt(0) === '[') {
      var close = hostport.indexOf(']');
      host = close < 0 ? hostport : hostport.slice(0, close + 1);
    } else {
      host = hostport.replace(/:\d+$/, '');
    }
    if (isInternalHost(host)) {
      throw new Error('web: refused internal/private address (SSRF guard): ' + host +
        ' — set SOFUU_WEB_ALLOW_INTERNAL=1 to override');
    }
  }

  /* ── HTML helpers ──────────────────────────────────────────────── */
  var ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', '#39': "'", '#x27': "'" };
  function decodeEntities(s) {
    return String(s).replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g, function (m, name) {
      var k = name.toLowerCase();
      if (ENTITIES[k] !== undefined) return ENTITIES[k];
      if (k[0] === '#' && k[1] === 'x') {
        var hx = parseInt(k.slice(2), 16);
        return isNaN(hx) ? m : String.fromCharCode(hx);
      }
      if (k[0] === '#') {
        var dec = parseInt(k.slice(1), 10);
        return isNaN(dec) ? m : String.fromCharCode(dec);
      }
      return m;
    });
  }
  function stripTags(s) {
    return decodeEntities(String(s).replace(/<[^>]*>/g, ''));
  }
  function clean(s) {
    return String(s).replace(/\s+/g, ' ').trim();
  }

  /* DDG redirect links look like:
   *   //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs&rut=...
   * Extract + decode the uddg param; otherwise return the href as-is. */
  function ddgTargetUrl(href) {
    var h = decodeEntities(href);
    if (h.indexOf('//duckduckgo.com/l/') >= 0 || h.indexOf('uddg=') >= 0) {
      var m = h.match(/[?&]uddg=([^&]+)/);
      if (m) {
        try { return decodeURIComponent(m[1]); } catch (e) { return h; }
      }
    }
    if (h.indexOf('//') === 0) return 'https:' + h;
    return h;
  }

  /* Collect class-anchored <a> pairs from DDG HTML without a DOM:
   * result__a for titles/urls, result__snippet for snippets. Markup puts
   * class and href in either order depending on skin — try both. */
  function ddgParse(html) {
    var results = [];
    var seen = {};
    var patterns = [
      /<a[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/g,
      /<a[^>]*href="([^"]*)"[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*>([\s\S]*?)<\/a>/g,
    ];
    for (var pi = 0; pi < patterns.length; pi++) {
      var re = patterns[pi];
      var m;
      while ((m = re.exec(html)) !== null) {
        var url = ddgTargetUrl(m[1]);
        var title = clean(stripTags(m[2]));
        if (!url || !title || url.indexOf('http') !== 0) continue;
        if (url.indexOf('duckduckgo.com') >= 0) continue;
        if (seen[url]) continue;
        seen[url] = 1;
        results.push({ title: title, url: url, snippet: '' });
      }
    }
    var sn = /<(?:a|div)[^>]*class="[^"]*\bresult__snippet\b[^"]*"[^>]*>([\s\S]*?)<\/(?:a|div)>/g;
    var i = 0;
    while ((m = sn.exec(html)) !== null) {
      if (i < results.length) { results[i].snippet = clean(stripTags(m[1])); i++; }
      else break;
    }
    return results;
  }

  /* ── Engines ───────────────────────────────────────────────────── */

  var UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 sofuu-runtime/' + VERSION;

  async function engineDuckDuckGo(query, opts) {
    /* SOFUU_WEB_ENDPOINT overrides the endpoint (proxy / self-host / tests). */
    var base = opts.endpoint || env('SOFUU_WEB_ENDPOINT') || 'https://html.duckduckgo.com/html/';
    var opEndpoint = !!(opts.endpoint || env('SOFUU_WEB_ENDPOINT'));
    var url = base + (base.indexOf('?') >= 0 ? '&' : '?') + 'q=' + encodeURIComponent(query) +
              '&kl=' + encodeURIComponent(opts.region || 'wt-wt');
    var r = await fetchWithTimeout(url, {
      method: 'GET',
      headers: { 'User-Agent': UA, 'Accept': 'text/html' },
    }, opts.timeoutMs, opEndpoint);
    if (r.status !== 200) {
      throw new Error('duckduckgo returned HTTP ' + r.status +
        ' — the keyless endpoint may be rate-limiting; set SOFUU_WEB_ENGINE=brave|tavily with an API key');
    }
    var html = await r.text();
    var results = ddgParse(html).slice(0, opts.count);
    if (results.length === 0) {
      throw new Error('duckduckgo returned no parsable results (markup change or block) — try SOFUU_WEB_ENGINE=brave|tavily');
    }
    return { results: results, note: 'keyless duckduckgo endpoint' };
  }

  async function engineBrave(query, opts) {
    var key = opts.api_key || env('BRAVE_API_KEY');
    if (!key) {
      throw new Error("engine 'brave' needs an API key — pass opts.api_key or set BRAVE_API_KEY (the default engine 'duckduckgo' needs no key)");
    }
    var base = opts.endpoint || 'https://api.search.brave.com/res/v1/web/search';
    var url = base + (base.indexOf('?') >= 0 ? '&' : '?') +
      'q=' + encodeURIComponent(query) + '&count=' + opts.count;
    var r = await fetchWithTimeout(url, {
      method: 'GET',
      headers: { 'Accept': 'application/json', 'X-Subscription-Token': key },
    }, opts.timeoutMs, !!opts.endpoint);
    if (r.status === 401 || r.status === 403) throw new Error('brave: invalid API key (HTTP ' + r.status + ')');
    if (r.status !== 200) throw new Error('brave returned HTTP ' + r.status);
    var j = await r.json();
    var raw = (j && j.web && j.web.results) || [];
    var results = [];
    for (var i = 0; i < raw.length && results.length < opts.count; i++) {
      var it = raw[i];
      if (!it || !it.url) continue;
      results.push({ title: clean(it.title || it.url), url: it.url, snippet: clean(it.description || '') });
    }
    return { results: results };
  }

  async function engineTavily(query, opts) {
    var key = opts.api_key || env('TAVILY_API_KEY');
    if (!key) {
      throw new Error("engine 'tavily' needs an API key — pass opts.api_key or set TAVILY_API_KEY (the default engine 'duckduckgo' needs no key)");
    }
    var base = opts.endpoint || 'https://api.tavily.com/search';
    var r = await fetchWithTimeout(base, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: key,
        query: query,
        max_results: opts.count,
        search_depth: opts.depth || 'basic',
      }),
    }, opts.timeoutMs, !!opts.endpoint);
    if (r.status === 401 || r.status === 403) throw new Error('tavily: invalid API key (HTTP ' + r.status + ')');
    if (r.status !== 200) throw new Error('tavily returned HTTP ' + r.status);
    var j = await r.json();
    var raw = (j && j.results) || [];
    var results = [];
    for (var i = 0; i < raw.length && results.length < opts.count; i++) {
      var it = raw[i];
      if (!it || !it.url) continue;
      results.push({ title: clean(it.title || it.url), url: it.url, snippet: clean(it.content || '') });
    }
    return { results: results };
  }

  var ENGINES = {
    duckduckgo: engineDuckDuckGo,
    ddg: engineDuckDuckGo, /* alias */
    brave: engineBrave,
    tavily: engineTavily,
  };

  /**
   * sofuu.web.search(query, opts) → Promise<{query, engine, results, ms, note?}>
   * opts: { engine:'duckduckgo'|'brave'|'tavily', count:8, api_key,
   *         endpoint (test override), region, depth:'basic'|'advanced',
   *         timeoutMs:10000 }
   */
  async function search(query, opts) {
    opts = opts || {};
    var q = String(query == null ? '' : query).trim();
    if (!q) throw new Error('web.search: query required');
    var count = Math.max(1, Math.min(20, (opts.count | 0) || 8));
    var engineName = (opts.engine || env('SOFUU_WEB_ENGINE') || 'duckduckgo').toLowerCase();
    var fn = ENGINES[engineName];
    if (!fn) {
      throw new Error("web.search: unknown engine '" + engineName + "' (duckduckgo | brave | tavily)");
    }
    var o = {
      count: count,
      api_key: opts.api_key,
      endpoint: opts.endpoint,
      region: opts.region,
      depth: opts.depth,
      timeoutMs: opts.timeoutMs || 10000,
    };
    var started = Date.now();
    var out = await fn(q, o);
    return {
      query: q,
      engine: engineName,
      results: out.results,
      ms: Date.now() - started,
      note: out.note,
    };
  }

  /**
   * sofuu.web.open(url, opts) → Promise<{url, status, text}>
   * Fetches a page and reduces it to readable text (scripts/styles/tags
   * stripped, entities decoded, whitespace collapsed), capped at
   * opts.maxChars (default 20 000). http(s) only. The transfer itself is
   * capped at max(2 MB, maxChars*8) bytes — a giant page errors with a
   * maxBodyBytes message instead of buffering whole.
   */
  async function open(url, opts) {
    opts = opts || {};
    var u = String(url == null ? '' : url).trim();
    if (!/^https?:\/\//i.test(u)) {
      throw new Error('web.open: only http(s) URLs are supported');
    }
    var maxChars = (opts.maxChars | 0) || 20000;
    /* js-9 (AUDIT-2026-09-07): fetch buffers the WHOLE body before
     * text() resolves — no JS-side check can prevent that. Cap the
     * transfer itself (maxBodyBytes aborts mid-body): text out is
     * maxChars, but markup in can be far larger, so the cap keeps
     * generous headroom while bounding the worst case. */
    var maxBody = Math.max(2 * 1024 * 1024, maxChars * 8);
    var r = await fetchWithTimeout(u, {
      method: 'GET',
      headers: { 'User-Agent': UA, 'Accept': 'text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8' },
      maxBodyBytes: maxBody,
    }, opts.timeoutMs || 10000);
    var body = await r.text();
    var ct = '';
    try { ct = String((r.headers && r.headers.get('content-type')) || ''); } catch (e) {}
    var text;
    if (ct.indexOf('html') >= 0 || /^\s*<(!doctype|html)/i.test(body)) {
      text = body
        .replace(/<script[\s\S]*?<\/script>/gi, ' ')
        .replace(/<style[\s\S]*?<\/style>/gi, ' ')
        .replace(/<noscript[\s\S]*?<\/noscript>/gi, ' ')
        .replace(/<!--[\s\S]*?-->/g, ' ')
        .replace(/<\/(p|div|li|h[1-6]|tr|br)>/gi, '\n')
        .replace(/<br\s*\/?>/gi, '\n')
        .replace(/<[^>]*>/g, ' ');
      text = decodeEntities(text).replace(/[ \t]+/g, ' ')
        .replace(/\n\s*\n\s*\n+/g, '\n\n').trim();
    } else {
      text = body;
    }
    if (text.length > maxChars) text = text.slice(0, maxChars) + '\n…(truncated at ' + maxChars + ' chars)';
    return { url: u, status: r.status, text: text };
  }

  sofuu.web._loaded = true;
  sofuu.web.VERSION = VERSION;
  sofuu.web.ENGINES = ['duckduckgo', 'brave', 'tavily'];
  sofuu.web.search = search;
  sofuu.web.open = open;

  /* Built-in agent tool specs (consumed by src/js/agent.js and the chat
   * driver): MCP-shaped, exactly like inline tool definitions. */
  sofuu.web.TOOLS = {
    web_search: {
      name: 'web_search',
      description: 'Search the web. Returns a list of results with title, url and snippet. Use web_open to read a result page.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'The search query' },
          count: { type: 'number', description: 'Max results (1-20, default 8)' },
        },
        required: ['query'],
      },
      execute: async function (args) {
        var r = await search(args && args.query, { count: (args && args.count) || 8 });
        return r.results.map(function (x, i) {
          return (i + 1) + '. ' + x.title + '\n   ' + x.url + '\n   ' + (x.snippet || '');
        }).join('\n') || '(no results)';
      },
    },
    web_open: {
      name: 'web_open',
      description: 'Fetch a web page by URL and return its readable text (HTML stripped, ~20k chars). Use after web_search to read a result.',
      parameters: {
        type: 'object',
        properties: {
          url: { type: 'string', description: 'The http(s) URL to read' },
        },
        required: ['url'],
      },
      execute: async function (args) {
        var p = await open(args && args.url);
        return p.text || '(empty page)';
      },
    },
  };
})();
