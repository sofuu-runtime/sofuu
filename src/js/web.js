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
  function fetchWithTimeout(url, opts, ms) {
    var timer = null;
    var to = new Promise(function (_, rej) {
      timer = setTimeout(function () { rej(new Error('timeout after ' + (ms || 10000) + 'ms')); }, ms || 10000);
    });
    return Promise.race([fetch(url, opts), to]).finally(function () {
      if (timer) { try { clearTimeout(timer); } catch (e) {} }
    });
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
    var url = base + (base.indexOf('?') >= 0 ? '&' : '?') + 'q=' + encodeURIComponent(query) +
              '&kl=' + encodeURIComponent(opts.region || 'wt-wt');
    var r = await fetchWithTimeout(url, {
      method: 'GET',
      headers: { 'User-Agent': UA, 'Accept': 'text/html' },
    }, opts.timeoutMs);
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
    }, opts.timeoutMs);
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
    }, opts.timeoutMs);
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
   * opts.maxChars (default 20 000). http(s) only.
   */
  async function open(url, opts) {
    opts = opts || {};
    var u = String(url == null ? '' : url).trim();
    if (!/^https?:\/\//i.test(u)) {
      throw new Error('web.open: only http(s) URLs are supported');
    }
    var maxChars = (opts.maxChars | 0) || 20000;
    var r = await fetchWithTimeout(u, {
      method: 'GET',
      headers: { 'User-Agent': UA, 'Accept': 'text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8' },
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
