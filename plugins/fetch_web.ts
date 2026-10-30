// plugins/fetch_web.ts - fetch a web page as clean text (cookbook plugin).
//
// Demonstrates the whole 5-Shield tool contract in one file:
//   Shield 1  z.object schema for url / maxBytes / format
//   Shield 2  no permissions.net, so the sandboxed fetch keeps the private-IP
//             guard, plus an explicit transport-level URL screen below
//   Shield 3  circuit breaker left at its defaults (5 failures / 30 s)
//   Shield 4  timeoutMs 5000 plus an internal deadline that also covers the
//             body read (the shield timer cannot see a stalled socket once
//             the headers have arrived)
//   Shield 5  every failure path raises a typed Sofuu error, never a raw throw
//
// Guarantees: http(s) only, no credential URLs, 5 s hard deadline, 1 MB body
// cap with an explicit truncation marker, script/style stripped, common
// entities decoded, whitespace collapsed.
//
// Standalone: no import beyond the SDK primitives and no runtime helpers.
// Enable it by dropping it in ./plugins (see plugins/README.md).

import { defineTool, z } from 'sofuu';

// -- local types (this block is stripped before the runtime sees the file) --

interface FetchWebArgs {
  url: string;
  maxBytes?: number;
  format?: 'text' | 'markdown';
}

interface FetchWebResult {
  url: string;
  status: number;
  contentType: string;
  text: string;
  truncated: boolean;
}

interface CappedBody {
  text: string;
  truncated: boolean;
}

interface PluginError extends Error {
  code: string;
}

interface Deadline {
  signal?: AbortSignal;
  promise: Promise<never>;
  cancel(): void;
}

// -- constants ------------------------------------------------------------

var HARD_TIMEOUT_MS = 5000;
var MAX_BYTES = 1024 * 1024;
var TRUNCATION_MARKER = '\n\n[sofuu] body truncated at the 1 MB cap';

function fail(code: string, message: string): PluginError {
  var err = new Error(message) as PluginError;
  err.code = code;
  return err;
}

// -- URL screening --------------------------------------------------------

// Shield 2 starts at the sandboxed fetch, but a clear message here beats a
// generic E_DENIED from the guard, and a credential URL never leaves the
// process at all.
function screenUrl(raw: string): string {
  var s = String(raw || '').trim();
  if (!s) throw fail('E_SCHEMA', 'url is required');
  var schemeMatch = s.match(/^([a-zA-Z][a-zA-Z0-9+.-]*):/);
  if (!schemeMatch) throw fail('E_DENIED', 'refusing a URL without a scheme: ' + s);
  var scheme = schemeMatch[1].toLowerCase();
  if (scheme !== 'http' && scheme !== 'https') {
    throw fail('E_DENIED', 'only http(s) URLs are allowed, got ' + scheme + ':');
  }
  var sep = s.indexOf('//');
  if (sep < 0) throw fail('E_DENIED', 'refusing a URL without an authority: ' + s);
  var authority = s.slice(sep + 2);
  var cut = authority.search(/[\/?#]/);
  if (cut >= 0) authority = authority.slice(0, cut);
  if (!authority) throw fail('E_DENIED', 'refusing a URL without a host: ' + s);
  if (authority.indexOf('@') >= 0) {
    throw fail('E_DENIED', 'refusing a URL with embedded credentials (user:pass@host)');
  }
  return s;
}

// -- HTML to text ---------------------------------------------------------

function codePointToString(raw: string, radix: number): string {
  var n = parseInt(String(raw), radix);
  if (!isFinite(n) || n < 1 || n > 0x10ffff) return '';
  if (n <= 0xffff) return String.fromCharCode(n);
  var v = n - 0x10000;
  return String.fromCharCode(0xd800 + (v >> 10), 0xdc00 + (v & 0x3ff));
}

function decodeEntities(input: string): string {
  var s = String(input || '');
  s = s.replace(/&nbsp;/gi, ' ');
  s = s.replace(/&quot;/gi, String.fromCharCode(34));
  s = s.replace(/&apos;/gi, String.fromCharCode(39));
  s = s.replace(/&#39;/g, String.fromCharCode(39));
  s = s.replace(/&lt;/gi, '<');
  s = s.replace(/&gt;/gi, '>');
  s = s.replace(/&amp;/gi, '&');
  s = s.replace(/&#x([0-9a-f]+);/gi, function (all: string, hex: string) {
    return codePointToString(hex, 16);
  });
  s = s.replace(/&#([0-9]+);/g, function (all: string, dec: string) {
    return codePointToString(dec, 10);
  });
  return s;
}

function stripTags(input: string): string {
  return String(input || '').replace(/<\/?[a-zA-Z][^>]*>/g, ' ');
}

function repeatChar(ch: string, n: number): string {
  var out = '';
  for (var i = 0; i < n; i++) out += ch;
  return out;
}

function collapseWhitespace(input: string): string {
  var s = String(input || '');
  s = s.replace(/[ \t\u00a0]+/g, ' ');
  s = s.replace(/ *\n */g, '\n');
  s = s.replace(/\n{3,}/g, '\n\n');
  return s.trim();
}

function htmlToText(html: string, markdown: boolean): string {
  var s = String(html || '');
  s = s.replace(/<!--[\s\S]*?-->/g, ' ');
  s = s.replace(/<script\b[\s\S]*?<\/script>/gi, ' ');
  s = s.replace(/<style\b[\s\S]*?<\/style>/gi, ' ');
  s = s.replace(/<noscript\b[\s\S]*?<\/noscript>/gi, ' ');
  s = s.replace(/<svg\b[\s\S]*?<\/svg>/gi, ' ');
  if (markdown) {
    s = s.replace(/<h([1-6])[^>]*>([\s\S]*?)<\/h\1>/gi, function (all: string, level: string, body: string) {
      return '\n\n' + repeatChar('#', parseInt(level, 10)) + ' ' + stripTags(body).trim() + '\n\n';
    });
    s = s.replace(/<a\b[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>/gi, function (all: string, href: string, body: string) {
      var label = stripTags(body).trim();
      if (!label) return ' ';
      return '[' + label + '](' + href + ')';
    });
    s = s.replace(/<li\b[^>]*>/gi, '\n- ');
  }
  s = s.replace(/<br\s*\/?>/gi, '\n');
  s = s.replace(/<\/(p|div|section|article|li|tr|h[1-6]|blockquote|pre)>/gi, '\n');
  s = stripTags(s);
  s = decodeEntities(s);
  return collapseWhitespace(s);
}

function looksLikeHtml(contentType: string, body: string): boolean {
  var ct = String(contentType || '').toLowerCase();
  if (ct.indexOf('html') >= 0) return true;
  if (ct.indexOf('json') >= 0) return false;
  if (ct.indexOf('text/plain') >= 0) return false;
  var head = String(body || '').slice(0, 512).toLowerCase();
  if (head.indexOf('<!doctype html') >= 0) return true;
  if (head.indexOf('<html') >= 0) return true;
  if (head.indexOf('<body') >= 0) return true;
  return false;
}

// -- capped body read -----------------------------------------------------

function decodeBytes(bytes: Uint8Array): string {
  if (typeof TextDecoder === 'function') {
    try { return new TextDecoder('utf-8').decode(bytes); } catch (e) { /* latin-1 below */ }
  }
  var out = '';
  for (var i = 0; i < bytes.length; i++) out += String.fromCharCode(bytes[i]);
  return out;
}

function joinChunks(chunks: Array<Uint8Array | string>): string {
  if (!chunks.length) return '';
  if (typeof chunks[0] === 'string') {
    var parts: string[] = [];
    for (var i = 0; i < chunks.length; i++) parts.push(String(chunks[i]));
    return parts.join('');
  }
  var total = 0;
  for (var j = 0; j < chunks.length; j++) {
    total += chunks[j].length;
  }
  var buf = new Uint8Array(total);
  var off = 0;
  for (var k = 0; k < chunks.length; k++) {
    var item = chunks[k];
    if (typeof item === 'string') continue;
    buf.set(item, off);
    off += item.length;
  }
  return decodeBytes(buf);
}

function mineChunks(reader: ReadableStreamDefaultReader<Uint8Array>, cap: number, chunks: Array<Uint8Array | string>): Promise<CappedBody> {
  return reader.read().then(function (step) {
    if (!step || step.done) {
      var text = joinChunks(chunks);
      var truncated = false;
      return { text: text, truncated: truncated };
    }
    var piece = step.value;
    var size = piece && typeof piece.length === 'number' ? piece.length : 0;
    var used = 0;
    for (var i = 0; i < chunks.length; i++) used += chunks[i].length;
    if (used + size > cap) {
      var room = cap - used;
      if (room > 0 && piece) chunks.push(piece.slice(0, room));
      if (typeof reader.cancel === 'function') {
        try { reader.cancel(); } catch (e) { /* best effort */ }
      }
      var text = joinChunks(chunks);
      var truncated = true;
      return { text: text, truncated: truncated };
    }
    if (piece) chunks.push(piece);
    return mineChunks(reader, cap, chunks);
  });
}

function readCapped(res: Response, cap: number): Promise<CappedBody> {
  var body = res ? res.body : null;
  if (body && typeof body.getReader === 'function') {
    var chunks: Array<Uint8Array | string> = [];
    return mineChunks(body.getReader(), cap, chunks);
  }
  return Promise.resolve(res.text()).then(function (text) {
    var s = String(text || '');
    var truncated = s.length > cap;
    var body = truncated ? s.slice(0, cap) : s;
    return { text: body, truncated: truncated };
  });
}

function readContentType(res: Response): string {
  var headers = res ? res.headers : null;
  if (headers && typeof headers.get === 'function') {
    var value = headers.get('content-type');
    if (value) return String(value);
  }
  return '';
}

// -- the deadline ---------------------------------------------------------

function startDeadline(ms: number, tool: string, target: string): Deadline {
  var controller: AbortController | null = (typeof AbortController === 'function') ? new AbortController() : null;
  var timer: ReturnType<typeof setTimeout> | null = null;
  /* No type argument on the call itself: "new Promise<never>(...)" is not
     something the runtime stripper can rewrite, and plain JS reads it as two
     comparisons. The annotation on the variable carries the type instead. */
  var promise: Promise<never> = new Promise(function (resolve, reject) {
    timer = setTimeout(function () {
      if (controller) {
        try { controller.abort(); } catch (e) { /* best effort */ }
      }
      reject(fail('E_TIMEOUT', tool + ' exceeded the ' + ms + ' ms deadline for ' + target));
    }, ms);
  });
  return {
    signal: controller ? controller.signal : undefined,
    promise: promise,
    cancel: function () {
      if (timer !== null) clearTimeout(timer);
    }
  };
}

// -- operator escape hatch -------------------------------------------------
//
// By default the tool declares no network permissions, so Shield 2 refuses
// loopback and private hosts — an SSRF guard the MODEL cannot talk its way
// past, because it is not an argument. Local development still needs
// http://127.0.0.1, so the OPERATOR (not the model) can widen the allowlist
// from the environment:
//
//     SOFUU_FETCH_WEB_ALLOW=127.0.0.1,localhost sofuu plugin test fetch_web …
//
// An allowlist is authoritative and exact, so it also keeps refusing
// host-name tricks such as a trailing dot.
const ALLOW_ENV = (typeof process !== 'undefined' && process.env) ? process.env.SOFUU_FETCH_WEB_ALLOW : '';
const ALLOW_HOSTS = String(ALLOW_ENV || '')
  .split(',')
  .map(function (s) { return s.trim(); })
  .filter(function (s) { return s.length > 0; });
const NET_PERMISSIONS = ALLOW_HOSTS.length > 0 ? { net: ALLOW_HOSTS } : undefined;

// -- the tool -------------------------------------------------------------

export default defineTool({
  /* Leading property on purpose: the runtime stripper rewrites ", key: IDENT"
     into ", key", and the first property is never preceded by a comma. */
  timeoutMs: HARD_TIMEOUT_MS,
  name: 'fetch_web',
  description: 'Fetch an http(s) URL and return clean text (5 s deadline, 1 MB cap)',
  permissions: NET_PERMISSIONS,
  schema: z.object({
    url: z.string().describe('Absolute http(s) URL to fetch'),
    maxBytes: z.number().int().positive().optional().describe('Body cap in bytes (default and maximum 1048576)'),
    format: z.enum(['text', 'markdown']).optional().describe('text (default) or markdown (keeps headings, links, list items)')
  }),
  async run(args, ctx): Promise<FetchWebResult> {
    var target = screenUrl(args.url);
    var cap = (typeof args.maxBytes === 'number' && args.maxBytes > 0) ? Math.min(args.maxBytes, MAX_BYTES) : MAX_BYTES;
    var markdown = args.format === 'markdown';
    var deadline = startDeadline(HARD_TIMEOUT_MS, 'fetch_web', target);
    var work = (async function (): Promise<FetchWebResult> {
      var res = await ctx.fetch(target, {
        method: 'GET',
        headers: {
          'accept': 'text/html,text/plain;q=0.9,*/*;q=0.5',
          'user-agent': 'sofuu-fetch_web/0.3 (+https://sofuu.xyz)'
        },
        signal: deadline.signal
      });
      if (!res || typeof res.ok !== 'boolean') {
        throw fail('E_TOOL', 'fetch returned no usable response for ' + target);
      }
      if (!res.ok) {
        throw fail('E_TOOL', 'HTTP ' + res.status + ' for ' + target);
      }
      var contentType = readContentType(res);
      var body = await readCapped(res, cap);
      var raw = body.text;
      var text = looksLikeHtml(contentType, raw) ? htmlToText(raw, markdown) : collapseWhitespace(raw);
      if (body.truncated) text = text + TRUNCATION_MARKER;
      if (!text) ctx.log('empty body for ' + target);
      return {
        url: target,
        status: res.status,
        contentType: contentType,
        text: text,
        truncated: body.truncated
      };
    })();
    try {
      return await Promise.race([work, deadline.promise]);
    } finally {
      deadline.cancel();
    }
  }
});
