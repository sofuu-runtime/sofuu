// @sofuu-source-only — eval'd at registration; NOT a shipped .bc driver
// (scripts/gen-driver-bc.sh and scripts/check-driver-bc.sh skip this marker).
//
// compat.js — Web Standard APIs & Node.js Compatibility Shim
// Eliminates Problem B: The "Compatibility Wall" (Why Devs Leave in 60 Seconds)
//
// Exposes:
//   - Web APIs: TextEncoder, TextDecoder, atob, btoa, URL, URLSearchParams,
//               crypto (randomUUID, getRandomValues, subtle), Headers, Request, Response
//   - Node globals: Buffer, EventEmitter, process.nextTick, process.versions, process.platform
//   - Node built-in virtual module catalog (__sofuu_node_modules):
//     path, events, crypto, buffer, fs, os, util, stream, assert, url, http, https

(function () {
  "use strict";

  // ── 1. TextEncoder & TextDecoder ───────────────────────────────────
  if (typeof globalThis.TextEncoder === "undefined") {
    class TextEncoder {
      constructor() {
        this.encoding = "utf-8";
      }
      encode(input) {
        const str = input === undefined ? "" : String(input);
        const bin = [];
        for (let i = 0; i < str.length; i++) {
          let code = str.charCodeAt(i);
          if (code >= 0xd800 && code <= 0xdbff) {
            if (i + 1 < str.length) {
              const next = str.charCodeAt(i + 1);
              if (next >= 0xdc00 && next <= 0xdfff) {
                code = ((code - 0xd800) << 10) + (next - 0xdc00) + 0x10000;
                i++;
              }
            }
          }
          if (code < 0x80) {
            bin.push(code);
          } else if (code < 0x800) {
            bin.push(0xc0 | (code >> 6), 0x80 | (code & 0x3f));
          } else if (code < 0x10000) {
            bin.push(0xe0 | (code >> 12), 0x80 | ((code >> 6) & 0x3f), 0x80 | (code & 0x3f));
          } else {
            bin.push(
              0xf0 | (code >> 18),
              0x80 | ((code >> 12) & 0x3f),
              0x80 | ((code >> 6) & 0x3f),
              0x80 | (code & 0x3f)
            );
          }
        }
        return new Uint8Array(bin);
      }
      encodeInto(source, destination) {
        const u8 = this.encode(source);
        const len = Math.min(u8.length, destination.length);
        destination.set(u8.subarray(0, len));
        return { read: source.length, written: len };
      }
    }
    globalThis.TextEncoder = TextEncoder;
  }

  if (typeof globalThis.TextDecoder === "undefined") {
    class TextDecoder {
      constructor(label = "utf-8", options = {}) {
        this.encoding = "utf-8";
        this.fatal = Boolean(options.fatal);
        this.ignoreBOM = Boolean(options.ignoreBOM);
      }
      decode(input, options = {}) {
        if (!input) return "";
        let bytes;
        if (input instanceof Uint8Array) {
          bytes = input;
        } else if (input instanceof ArrayBuffer) {
          bytes = new Uint8Array(input);
        } else if (ArrayBuffer.isView(input)) {
          bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
        } else {
          bytes = new Uint8Array(input);
        }
        let out = "";
        let i = 0;
        const len = bytes.length;
        while (i < len) {
          let b1 = bytes[i++];
          if (b1 < 0x80) {
            out += String.fromCharCode(b1);
          } else if (b1 > 0xbf && b1 < 0xe0) {
            let b2 = bytes[i++];
            out += String.fromCharCode(((b1 & 0x1f) << 6) | (b2 & 0x3f));
          } else if (b1 > 0xdf && b1 < 0xf0) {
            let b2 = bytes[i++];
            let b3 = bytes[i++];
            out += String.fromCharCode(((b1 & 0x0f) << 12) | ((b2 & 0x3f) << 6) | (b3 & 0x3f));
          } else {
            let b2 = bytes[i++];
            let b3 = bytes[i++];
            let b4 = bytes[i++];
            let cp = (((b1 & 0x07) << 18) | ((b2 & 0x3f) << 12) | ((b3 & 0x3f) << 6) | (b4 & 0x3f)) - 0x10000;
            out += String.fromCharCode(0xd800 + (cp >> 10), 0xdc00 + (cp & 0x3ff));
          }
        }
        return out;
      }
    }
    globalThis.TextDecoder = TextDecoder;
  }

  // ── 2. atob & btoa ────────────────────────────────────────────────
  const B64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
  if (typeof globalThis.btoa === "undefined") {
    globalThis.btoa = function (string) {
      let str = String(string);
      let output = "";
      for (
        let block = 0, charCode, i = 0, map = B64_CHARS;
        str.charAt(i | 0) || ((map = "="), i % 1);
        output += map.charAt(63 & (block >> (8 - (i % 1) * 8)))
      ) {
        charCode = str.charCodeAt((i += 3 / 4));
        if (charCode > 0xff) {
          throw new Error("'btoa' failed: The string contains characters outside Latin1 range.");
        }
        block = (block << 8) | charCode;
      }
      return output;
    };
  }

  if (typeof globalThis.atob === "undefined") {
    globalThis.atob = function (input) {
      let str = String(input).replace(/=+$/, "");
      if (str.length % 4 === 1) {
        throw new Error("'atob' failed: Invalid base64 length.");
      }
      let output = "";
      for (
        let bc = 0, bs = 0, buffer, i = 0;
        (buffer = str.charAt(i++));
        ~buffer && ((bs = bc % 4 ? bs * 64 + buffer : buffer), bc++ % 4)
          ? (output += String.fromCharCode(255 & (bs >> ((-2 * bc) & 6))))
          : 0
      ) {
        buffer = B64_CHARS.indexOf(buffer);
      }
      return output;
    };
  }

  // ── 3. URLSearchParams & URL ──────────────────────────────────────
  if (typeof globalThis.URLSearchParams === "undefined") {
    class URLSearchParams {
      constructor(init = "") {
        this._params = [];
        if (typeof init === "string") {
          let s = init.startsWith("?") ? init.slice(1) : init;
          if (s) {
            for (const pair of s.split("&")) {
              if (!pair) continue;
              const idx = pair.indexOf("=");
              if (idx !== -1) {
                const k = decodeURIComponent(pair.slice(0, idx).replace(/\+/g, " "));
                const v = decodeURIComponent(pair.slice(idx + 1).replace(/\+/g, " "));
                this._params.push([k, v]);
              } else {
                this._params.push([decodeURIComponent(pair.replace(/\+/g, " ")), ""]);
              }
            }
          }
        } else if (Array.isArray(init)) {
          for (const item of init) {
            if (Array.isArray(item) && item.length >= 2) {
              this._params.push([String(item[0]), String(item[1])]);
            }
          }
        } else if (init && typeof init === "object") {
          for (const [k, v] of Object.entries(init)) {
            this._params.push([String(k), String(v)]);
          }
        }
      }
      get(name) {
        const found = this._params.find(([k]) => k === name);
        return found ? found[1] : null;
      }
      getAll(name) {
        return this._params.filter(([k]) => k === name).map(([, v]) => v);
      }
      has(name) {
        return this._params.some(([k]) => k === name);
      }
      set(name, value) {
        let replaced = false;
        const next = [];
        for (const [k, v] of this._params) {
          if (k === name) {
            if (!replaced) {
              next.push([name, String(value)]);
              replaced = true;
            }
          } else {
            next.push([k, v]);
          }
        }
        if (!replaced) {
          next.push([name, String(value)]);
        }
        this._params = next;
      }
      append(name, value) {
        this._params.push([String(name), String(value)]);
      }
      delete(name) {
        this._params = this._params.filter(([k]) => k !== name);
      }
      forEach(callback, thisArg) {
        for (const [k, v] of this._params) {
          callback.call(thisArg, v, k, this);
        }
      }
      keys() {
        return this._params.map(([k]) => k)[Symbol.iterator]();
      }
      values() {
        return this._params.map(([, v]) => v)[Symbol.iterator]();
      }
      entries() {
        return this._params[Symbol.iterator]();
      }
      [Symbol.iterator]() {
        return this.entries();
      }
      get size() {
        return this._params.length;
      }
      toString() {
        return this._params
          .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
          .join("&");
      }
    }
    globalThis.URLSearchParams = URLSearchParams;
  }

  if (typeof globalThis.URL === "undefined") {
    class URL {
      constructor(url, base) {
        let full = String(url);
        if (base !== undefined) {
          let baseStr = String(base);
          if (!full.includes("://")) {
            if (!baseStr.includes("://")) {
              throw new TypeError(`Invalid base URL: ${base}`);
            }
            if (full.startsWith("/")) {
              const baseMatch = baseStr.match(/^([a-zA-Z][a-zA-Z0-9+-.]*:\/\/[^\/]+)/);
              if (baseMatch) {
                full = baseMatch[1] + full;
              } else {
                full = baseStr.replace(/\/+$/, "") + full;
              }
            } else {
              const lastSlash = baseStr.lastIndexOf("/");
              if (lastSlash > baseStr.indexOf("://") + 2) {
                full = baseStr.slice(0, lastSlash + 1) + full;
              } else {
                full = baseStr.replace(/\/+$/, "") + "/" + full;
              }
            }
          }
        }

        const match = full.match(/^([a-zA-Z][a-zA-Z0-9+-.]*:)\/\/([^\/?#]*)(.*)$/);
        if (!match) {
          // File or custom URI
          if (full.startsWith("file://")) {
            this.protocol = "file:";
            this.host = "";
            this.hostname = "";
            this.port = "";
            this.origin = "null";
            this.pathname = full.slice(7);
            this.search = "";
            this.hash = "";
            this._searchParams = new globalThis.URLSearchParams();
            return;
          }
          throw new TypeError(`Invalid URL: ${url}`);
        }

        this.protocol = match[1].toLowerCase();
        let authAndHost = match[2];
        let rest = match[3] || "";

        this.username = "";
        this.password = "";
        if (authAndHost.includes("@")) {
          const atIdx = authAndHost.indexOf("@");
          const auth = authAndHost.slice(0, atIdx);
          authAndHost = authAndHost.slice(atIdx + 1);
          if (auth.includes(":")) {
            const colon = auth.indexOf(":");
            this.username = auth.slice(0, colon);
            this.password = auth.slice(colon + 1);
          } else {
            this.username = auth;
          }
        }

        this.host = authAndHost;
        if (authAndHost.includes(":")) {
          const colon = authAndHost.lastIndexOf(":");
          this.hostname = authAndHost.slice(0, colon);
          this.port = authAndHost.slice(colon + 1);
        } else {
          this.hostname = authAndHost;
          this.port = "";
        }

        this.origin = `${this.protocol}//${this.host}`;

        let hashIdx = rest.indexOf("#");
        if (hashIdx !== -1) {
          this.hash = rest.slice(hashIdx);
          rest = rest.slice(0, hashIdx);
        } else {
          this.hash = "";
        }

        let searchIdx = rest.indexOf("?");
        if (searchIdx !== -1) {
          this.search = rest.slice(searchIdx);
          this.pathname = rest.slice(0, searchIdx) || "/";
        } else {
          this.search = "";
          this.pathname = rest || "/";
        }

        this._searchParams = new globalThis.URLSearchParams(this.search);
      }

      get searchParams() {
        return this._searchParams;
      }

      get href() {
        let q = this._searchParams.toString();
        let s = q ? `?${q}` : this.search;
        return `${this.protocol}//${this.host}${this.pathname}${s}${this.hash}`;
      }

      set href(val) {
        const u = new URL(val);
        Object.assign(this, u);
      }

      toString() {
        return this.href;
      }

      toJSON() {
        return this.href;
      }

      static createObjectURL() {
        return `blob:sofuu-${Math.random().toString(36).slice(2)}`;
      }

      static revokeObjectURL() {}
    }
    globalThis.URL = URL;
  }

  // ── 4. Web Crypto & Node Crypto ────────────────────────────────────
  if (typeof globalThis.crypto === "undefined") {
    globalThis.crypto = {};
  }

  const cryptoObj = globalThis.crypto;

  if (typeof cryptoObj.randomUUID === "undefined") {
    cryptoObj.randomUUID = function () {
      if (globalThis.sofuu && globalThis.sofuu.crypto && globalThis.sofuu.crypto.randomUUID) {
        return globalThis.sofuu.crypto.randomUUID();
      }
      return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
        const r = (Math.random() * 16) | 0;
        const v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
      });
    };
  }

  if (typeof cryptoObj.getRandomValues === "undefined") {
    cryptoObj.getRandomValues = function (typedArray) {
      if (!typedArray || !typedArray.buffer) {
        throw new TypeError("Argument must be a TypedArray");
      }
      if (globalThis.sofuu && globalThis.sofuu.crypto && globalThis.sofuu.crypto.randomFill) {
        return globalThis.sofuu.crypto.randomFill(typedArray);
      }
      const bytes = new Uint8Array(typedArray.buffer, typedArray.byteOffset, typedArray.byteLength);
      for (let i = 0; i < bytes.length; i++) {
        bytes[i] = (Math.random() * 256) | 0;
      }
      return typedArray;
    };
  }

  if (typeof cryptoObj.subtle === "undefined") {
    cryptoObj.subtle = {
      digest: async function (algorithm, data) {
        const algo = typeof algorithm === "string" ? algorithm.toUpperCase() : (algorithm?.name || "").toUpperCase();
        if (algo !== "SHA-256") {
          throw new Error(`SubtleCrypto.digest: unsupported algorithm ${algo}`);
        }
        let text = "";
        if (typeof data === "string") {
          text = data;
        } else {
          const u8 = data instanceof Uint8Array ? data : new Uint8Array(data.buffer || data);
          text = new TextDecoder().decode(u8);
        }
        let hex = "";
        if (globalThis.sofuu && globalThis.sofuu.crypto && globalThis.sofuu.crypto.sha256) {
          hex = globalThis.sofuu.crypto.sha256(text);
        } else {
          hex = "0".repeat(64);
        }
        const out = new Uint8Array(32);
        for (let i = 0; i < 32; i++) {
          out[i] = parseInt(hex.substr(i * 2, 2), 16);
        }
        return out.buffer;
      },
    };
  }

  // ── 5. Buffer (Subclassing Uint8Array) ──────────────────────────────
  if (typeof globalThis.Buffer === "undefined") {
    class Buffer extends Uint8Array {
      static isBuffer(b) {
        return Boolean(b && (b instanceof Buffer || b._isBuffer === true));
      }
      static alloc(size, fill = 0, encoding = "utf8") {
        const buf = new Buffer(size);
        if (fill !== 0) {
          buf.fill(fill);
        }
        return buf;
      }
      static allocUnsafe(size) {
        return new Buffer(size);
      }
      static from(value, encodingOrOffset, length) {
        if (typeof value === "string") {
          const enc = (encodingOrOffset || "utf8").toLowerCase();
          if (enc === "hex") {
            const clean = value.replace(/[^0-9a-fA-F]/g, "");
            const len = clean.length >> 1;
            const buf = new Buffer(len);
            for (let i = 0; i < len; i++) {
              buf[i] = parseInt(clean.substr(i * 2, 2), 16);
            }
            return buf;
          }
          if (enc === "base64") {
            const binary = globalThis.atob(value);
            const buf = new Buffer(binary.length);
            for (let i = 0; i < binary.length; i++) {
              buf[i] = binary.charCodeAt(i);
            }
            return buf;
          }
          const u8 = new TextEncoder().encode(value);
          const buf = new Buffer(u8.length);
          buf.set(u8);
          return buf;
        }
        if (value instanceof ArrayBuffer) {
          return new Buffer(value, encodingOrOffset, length);
        }
        if (ArrayBuffer.isView(value) || Array.isArray(value)) {
          const buf = new Buffer(value.length);
          buf.set(value);
          return buf;
        }
        return new Buffer(0);
      }
      static concat(list, totalLength) {
        if (!Array.isArray(list)) throw new TypeError("list must be an Array");
        if (list.length === 0) return new Buffer(0);
        const len = totalLength !== undefined ? totalLength : list.reduce((acc, c) => acc + c.length, 0);
        const out = new Buffer(len);
        let offset = 0;
        for (const b of list) {
          if (offset >= len) break;
          const copyLen = Math.min(b.length, len - offset);
          out.set(b.subarray(0, copyLen), offset);
          offset += copyLen;
        }
        return out;
      }
      static byteLength(string, encoding = "utf8") {
        return Buffer.from(string, encoding).length;
      }
      toString(encoding = "utf8", start = 0, end = this.length) {
        const enc = String(encoding).toLowerCase();
        const sub = this.subarray(start, end);
        if (enc === "hex") {
          let hex = "";
          for (let i = 0; i < sub.length; i++) {
            const h = sub[i].toString(16);
            hex += h.length === 1 ? "0" + h : h;
          }
          return hex;
        }
        if (enc === "base64") {
          let bin = "";
          for (let i = 0; i < sub.length; i++) {
            bin += String.fromCharCode(sub[i]);
          }
          return globalThis.btoa(bin);
        }
        return new TextDecoder().decode(sub);
      }
      slice(start, end) {
        return Buffer.from(this.subarray(start, end));
      }
      equals(other) {
        if (!other || this.length !== other.length) return false;
        for (let i = 0; i < this.length; i++) {
          if (this[i] !== other[i]) return false;
        }
        return true;
      }
      copy(target, targetStart = 0, sourceStart = 0, sourceEnd = this.length) {
        const slice = this.subarray(sourceStart, sourceEnd);
        target.set(slice, targetStart);
        return slice.length;
      }
    }
    Buffer.prototype._isBuffer = true;
    globalThis.Buffer = Buffer;
  }

  // ── 6. EventEmitter ───────────────────────────────────────────────
  if (typeof globalThis.EventEmitter === "undefined") {
    class EventEmitter {
      constructor() {
        this._events = Object.create(null);
        this._maxListeners = EventEmitter.defaultMaxListeners;
      }
      on(event, listener) {
        return this.addListener(event, listener);
      }
      addListener(event, listener) {
        if (typeof listener !== "function") throw new TypeError("listener must be a function");
        if (!this._events[event]) this._events[event] = [];
        this._events[event].push(listener);
        return this;
      }
      once(event, listener) {
        if (typeof listener !== "function") throw new TypeError("listener must be a function");
        const g = (...args) => {
          this.removeListener(event, g);
          listener.apply(this, args);
        };
        g.listener = listener;
        return this.addListener(event, g);
      }
      emit(event, ...args) {
        const list = this._events[event];
        if (!list || list.length === 0) return false;
        const copy = list.slice();
        for (const fn of copy) {
          fn.apply(this, args);
        }
        return true;
      }
      removeListener(event, listener) {
        const list = this._events[event];
        if (!list) return this;
        this._events[event] = list.filter((fn) => fn !== listener && fn.listener !== listener);
        return this;
      }
      off(event, listener) {
        return this.removeListener(event, listener);
      }
      removeAllListeners(event) {
        if (event) {
          delete this._events[event];
        } else {
          this._events = Object.create(null);
        }
        return this;
      }
      listeners(event) {
        return (this._events[event] || []).slice();
      }
      rawListeners(event) {
        return (this._events[event] || []).slice();
      }
      listenerCount(event) {
        return (this._events[event] || []).length;
      }
      eventNames() {
        return Object.keys(this._events);
      }
      setMaxListeners(n) {
        this._maxListeners = n;
        return this;
      }
      getMaxListeners() {
        return this._maxListeners;
      }
    }
    EventEmitter.defaultMaxListeners = 10;
    globalThis.EventEmitter = EventEmitter;
  }

  // ── 7. Headers, Request & Response ────────────────────────────────
  if (typeof globalThis.Headers === "undefined") {
    class Headers {
      constructor(init = {}) {
        this._map = new Map();
        if (init instanceof Headers) {
          init.forEach((v, k) => this.append(k, v));
        } else if (Array.isArray(init)) {
          for (const [k, v] of init) this.append(k, v);
        } else if (init && typeof init === "object") {
          for (const [k, v] of Object.entries(init)) this.append(k, v);
        }
      }
      append(name, value) {
        const key = String(name).toLowerCase();
        const existing = this._map.get(key);
        this._map.set(key, existing ? `${existing}, ${value}` : String(value));
      }
      delete(name) {
        this._map.delete(String(name).toLowerCase());
      }
      get(name) {
        return this._map.get(String(name).toLowerCase()) || null;
      }
      has(name) {
        return this._map.has(String(name).toLowerCase());
      }
      set(name, value) {
        this._map.set(String(name).toLowerCase(), String(value));
      }
      forEach(callback, thisArg) {
        for (const [k, v] of this._map.entries()) {
          callback.call(thisArg, v, k, this);
        }
      }
      keys() {
        return this._map.keys();
      }
      values() {
        return this._map.values();
      }
      entries() {
        return this._map.entries();
      }
      [Symbol.iterator]() {
        return this.entries();
      }
    }
    globalThis.Headers = Headers;
  }

  if (typeof globalThis.Response === "undefined") {
    class Response {
      constructor(body = null, init = {}) {
        this._body = body;
        this.status = init.status !== undefined ? init.status : 200;
        this.statusText = init.statusText || "OK";
        this.ok = this.status >= 200 && this.status < 300;
        this.headers = new globalThis.Headers(init.headers);
      }
      async text() {
        if (typeof this._body === "string") return this._body;
        if (this._body instanceof Uint8Array) return new TextDecoder().decode(this._body);
        return String(this._body || "");
      }
      async json() {
        const t = await this.text();
        return JSON.parse(t);
      }
      async arrayBuffer() {
        if (this._body instanceof ArrayBuffer) return this._body;
        if (this._body instanceof Uint8Array) return this._body.buffer;
        const u8 = new TextEncoder().encode(await this.text());
        return u8.buffer;
      }
      clone() {
        return new Response(this._body, {
          status: this.status,
          statusText: this.statusText,
          headers: this.headers,
        });
      }
      static json(data, init = {}) {
        const body = JSON.stringify(data);
        const headers = new globalThis.Headers(init.headers);
        headers.set("content-type", "application/json");
        return new Response(body, { ...init, headers });
      }
    }
    globalThis.Response = Response;
  }

  if (typeof globalThis.Request === "undefined") {
    class Request {
      constructor(input, init = {}) {
        this.url = typeof input === "string" ? input : input.url;
        this.method = (init.method || "GET").toUpperCase();
        this.headers = new globalThis.Headers(init.headers || input.headers);
        this.body = init.body !== undefined ? init.body : null;
      }
      async text() {
        return String(this.body || "");
      }
      async json() {
        return JSON.parse(await this.text());
      }
    }
    globalThis.Request = Request;
  }

  // ── 8. Process enhancements ───────────────────────────────────────
  if (typeof globalThis.process !== "undefined") {
    if (typeof globalThis.process.nextTick === "undefined") {
      globalThis.process.nextTick = function (fn, ...args) {
        if (typeof queueMicrotask === "function") {
          queueMicrotask(() => fn(...args));
        } else {
          setTimeout(() => fn(...args), 0);
        }
      };
    }
    if (!globalThis.process.versions) {
      globalThis.process.versions = {};
    }
    globalThis.process.versions.node = "22.0.0";
    globalThis.process.versions.sofuu = (globalThis.sofuu && globalThis.sofuu.version) || "0.3.0";
    globalThis.process.versions.quickjs = "2024-01-13";

    if (!globalThis.process.platform) {
      // Determine platform from environment or fallback to darwin
      globalThis.process.platform = (globalThis.sofuu && globalThis.sofuu.platform) || "darwin";
    }
    if (!globalThis.process.arch) {
      globalThis.process.arch = "arm64";
    }
  }

  // ── 9. Node Built-in Modules Virtual Catalog ───────────────────────
  const nodePath = {
    sep: "/",
    delimiter: ":",
    join(...parts) {
      return parts
        .filter(Boolean)
        .join("/")
        .replace(/\/+/g, "/");
    },
    resolve(...parts) {
      let resolved = "";
      for (let i = parts.length - 1; i >= 0; i--) {
        const p = parts[i];
        if (!p) continue;
        resolved = resolved ? `${p}/${resolved}` : p;
        if (resolved.startsWith("/")) break;
      }
      if (!resolved.startsWith("/")) {
        const cwd = (globalThis.process && globalThis.process.cwd && globalThis.process.cwd()) || "/";
        resolved = `${cwd}/${resolved}`;
      }
      return nodePath.normalize(resolved);
    },
    normalize(p) {
      if (!p) return ".";
      const isAbs = p.startsWith("/");
      const segments = p.split("/").filter(Boolean);
      const stack = [];
      for (const seg of segments) {
        if (seg === ".") continue;
        if (seg === "..") {
          if (stack.length > 0 && stack[stack.length - 1] !== "..") {
            stack.pop();
          } else if (!isAbs) {
            stack.push("..");
          }
        } else {
          stack.push(seg);
        }
      }
      let res = stack.join("/");
      if (isAbs) res = "/" + res;
      return res || (isAbs ? "/" : ".");
    },
    isAbsolute(p) {
      return typeof p === "string" && p.startsWith("/");
    },
    dirname(p) {
      if (!p) return ".";
      const norm = nodePath.normalize(p);
      const idx = norm.lastIndexOf("/");
      if (idx === -1) return ".";
      if (idx === 0) return "/";
      return norm.slice(0, idx);
    },
    basename(p, ext) {
      if (!p) return "";
      const norm = nodePath.normalize(p);
      const idx = norm.lastIndexOf("/");
      let base = idx === -1 ? norm : norm.slice(idx + 1);
      if (ext && base.endsWith(ext)) {
        base = base.slice(0, -ext.length);
      }
      return base;
    },
    extname(p) {
      const base = nodePath.basename(p);
      const idx = base.lastIndexOf(".");
      if (idx <= 0) return "";
      return base.slice(idx);
    },
    parse(p) {
      const dir = nodePath.dirname(p);
      const base = nodePath.basename(p);
      const ext = nodePath.extname(base);
      const name = base.slice(0, base.length - ext.length);
      const root = p.startsWith("/") ? "/" : "";
      return { root, dir, base, ext, name };
    },
    format(obj) {
      const dir = obj.dir || obj.root;
      const base = obj.base || `${obj.name || ""}${obj.ext || ""}`;
      if (!dir) return base;
      return dir === "/" ? `/${base}` : `${dir}/${base}`;
    },
  };
  nodePath.posix = nodePath;
  nodePath.win32 = nodePath;

  const nodeEvents = {
    EventEmitter: globalThis.EventEmitter,
    defaultMaxListeners: 10,
    getEventListeners(emitter, event) {
      return emitter.listeners(event);
    },
    once(emitter, event) {
      return new Promise((resolve) => {
        emitter.once(event, (...args) => resolve(args));
      });
    },
  };

  const nodeCrypto = {
    randomUUID: cryptoObj.randomUUID,
    getRandomValues: cryptoObj.getRandomValues,
    subtle: cryptoObj.subtle,
    randomBytes(size) {
      const buf = globalThis.Buffer.alloc(size);
      cryptoObj.getRandomValues(buf);
      return buf;
    },
    createHash(algorithm) {
      let data = "";
      return {
        update(chunk) {
          if (typeof chunk === "string") {
            data += chunk;
          } else if (chunk && chunk.toString) {
            data += chunk.toString("utf8");
          }
          return this;
        },
        digest(encoding = "hex") {
          let hex = "";
          if (globalThis.sofuu && globalThis.sofuu.crypto && globalThis.sofuu.crypto.sha256) {
            hex = globalThis.sofuu.crypto.sha256(data);
          } else {
            hex = "0".repeat(64);
          }
          if (encoding === "hex") return hex;
          const buf = globalThis.Buffer.from(hex, "hex");
          if (encoding === "base64") return buf.toString("base64");
          return buf;
        },
      };
    },
    createHmac(algorithm, key) {
      return nodeCrypto.createHash(algorithm);
    },
  };

  const nodeBuffer = {
    Buffer: globalThis.Buffer,
    SlowBuffer: globalThis.Buffer,
    kMaxLength: 2147483647,
  };

  const sfs = (globalThis.sofuu && globalThis.sofuu.fs) || {};
  const nodeFs = {
    readFileSync(p, enc) {
      if (sfs.readFileSync) {
        return sfs.readFileSync(p, enc);
      }
      throw new Error(`fs.readFileSync unavailable for ${p}`);
    },
    writeFileSync(p, data, enc) {
      if (sfs.writeFileSync) {
        return sfs.writeFileSync(p, data, enc);
      }
      throw new Error(`fs.writeFileSync unavailable for ${p}`);
    },
    existsSync(p) {
      if (sfs.existsSync) {
        return sfs.existsSync(p);
      }
      return false;
    },
    readdirSync(p) {
      if (sfs.readdirSync) {
        return sfs.readdirSync(p);
      }
      return [];
    },
    mkdirSync(p) {
      if (sfs.mkdirSync) {
        return sfs.mkdirSync(p);
      }
    },
    unlinkSync(p) {
      if (sfs.unlinkSync) {
        return sfs.unlinkSync(p);
      }
    },
    statSync(p) {
      if (sfs.statSync) {
        const s = sfs.statSync(p);
        return {
          size: s.size,
          mtimeMs: s.mtimeMs,
          isDirectory: () => Boolean(s.is_dir),
          isFile: () => Boolean(s.is_file),
        };
      }
      return { isDirectory: () => false, isFile: () => true, size: 0, mtimeMs: Date.now() };
    },
    promises: {
      async readFile(p, enc) {
        if (sfs.readFile) {
          return await sfs.readFile(p);
        }
        return nodeFs.readFileSync(p, enc);
      },
      async writeFile(p, data, enc) {
        if (sfs.writeFile) {
          return await sfs.writeFile(p, data);
        }
        return nodeFs.writeFileSync(p, data, enc);
      },
      async exists(p) {
        if (sfs.exists) {
          return await sfs.exists(p);
        }
        return nodeFs.existsSync(p);
      },
      async readdir(p) {
        if (sfs.readdir) {
          return await sfs.readdir(p);
        }
        return nodeFs.readdirSync(p);
      },
      async mkdir(p) {
        if (sfs.mkdir) {
          return await sfs.mkdir(p);
        }
        return nodeFs.mkdirSync(p);
      },
      async stat(p) {
        return nodeFs.statSync(p);
      },
    },
  };

  const nodeOs = {
    platform() {
      return (globalThis.process && globalThis.process.platform) || "darwin";
    },
    arch() {
      return (globalThis.process && globalThis.process.arch) || "arm64";
    },
    homedir() {
      return (globalThis.process && globalThis.process.env && globalThis.process.env.HOME) || "/tmp";
    },
    tmpdir() {
      return "/tmp";
    },
    hostname() {
      return "localhost";
    },
    release() {
      return "1.0.0";
    },
    type() {
      return "Darwin";
    },
    cpus() {
      return [{ model: "Apple M Series", speed: 3200 }];
    },
    totalmem() {
      return 16 * 1024 * 1024 * 1024;
    },
    freemem() {
      return 8 * 1024 * 1024 * 1024;
    },
    EOL: "\n",
  };

  const nodeUtil = {
    promisify(fn) {
      return function (...args) {
        return new Promise((resolve, reject) => {
          fn(...args, (err, res) => {
            if (err) reject(err);
            else resolve(res);
          });
        });
      };
    },
    inspect(obj) {
      try {
        return JSON.stringify(obj, null, 2);
      } catch (_) {
        return String(obj);
      }
    },
    format(f, ...args) {
      if (typeof f !== "string") return [f, ...args].join(" ");
      let i = 0;
      return f.replace(/%[sdj%]/g, (match) => {
        if (match === "%%") return "%";
        if (i >= args.length) return match;
        const arg = args[i++];
        if (match === "%s") return String(arg);
        if (match === "%d") return Number(arg);
        if (match === "%j") {
          try {
            return JSON.stringify(arg);
          } catch (_) {
            return "[Circular]";
          }
        }
        return match;
      });
    },
    types: {
      isUint8Array(val) {
        return val instanceof Uint8Array;
      },
      isPromise(val) {
        return Boolean(val && typeof val.then === "function");
      },
    },
    inherits(ctor, superCtor) {
      if (superCtor) {
        ctor.super_ = superCtor;
        Object.setPrototypeOf(ctor.prototype, superCtor.prototype);
      }
    },
    TextEncoder: globalThis.TextEncoder,
    TextDecoder: globalThis.TextDecoder,
  };

  class StreamReadable extends globalThis.EventEmitter {
    constructor() {
      super();
    }
    pipe(dest) {
      this.on("data", (chunk) => dest.write(chunk));
      this.on("end", () => dest.end());
      return dest;
    }
  }

  class StreamWritable extends globalThis.EventEmitter {
    constructor() {
      super();
    }
    write(chunk) {
      return true;
    }
    end() {
      this.emit("finish");
    }
  }

  const nodeStream = {
    Readable: StreamReadable,
    Writable: StreamWritable,
    Transform: StreamReadable,
    PassThrough: StreamReadable,
    pipeline(...streams) {
      const cb = typeof streams[streams.length - 1] === "function" ? streams.pop() : null;
      if (cb) cb(null);
    },
    finished(stream, cb) {
      stream.on("finish", () => cb(null));
      stream.on("end", () => cb(null));
      stream.on("error", (err) => cb(err));
    },
  };

  const nodeAssert = function (value, message) {
    if (!value) {
      throw new Error(message || "Assertion failed");
    }
  };
  nodeAssert.ok = nodeAssert;
  nodeAssert.equal = function (a, b, msg) {
    if (a != b) throw new Error(msg || `${a} != ${b}`);
  };
  nodeAssert.strictEqual = function (a, b, msg) {
    if (a !== b) throw new Error(msg || `${a} !== ${b}`);
  };
  nodeAssert.deepEqual = function (a, b, msg) {
    if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(msg || `deepEqual failed`);
  };
  nodeAssert.notEqual = function (a, b, msg) {
    if (a == b) throw new Error(msg || `${a} == ${b}`);
  };
  nodeAssert.notStrictEqual = function (a, b, msg) {
    if (a === b) throw new Error(msg || `${a} === ${b}`);
  };

  const nodeHttp = {
    request(url, options, callback) {
      const emitter = new globalThis.EventEmitter();
      return emitter;
    },
    get(url, options, callback) {
      return nodeHttp.request(url, options, callback);
    },
    createServer() {
      return new globalThis.EventEmitter();
    },
    Agent: class Agent {},
  };

  // Register on global catalog for CJS require and ESM loaders
  globalThis.__sofuu_node_modules = {
    path: nodePath,
    events: nodeEvents,
    crypto: nodeCrypto,
    buffer: nodeBuffer,
    fs: nodeFs,
    os: nodeOs,
    util: nodeUtil,
    stream: nodeStream,
    assert: nodeAssert,
    url: { URL: globalThis.URL, URLSearchParams: globalThis.URLSearchParams },
    http: nodeHttp,
    https: nodeHttp,
  };
})();

// ── WHATWG Streams (PLAN-MODERN-NPM pillar B): ReadableStream/WritableStream/TransformStream + TextDecoderStream over native chunks ──
if (typeof globalThis.ReadableStream === "undefined") {
  class ReadableStreamDefaultController {
    constructor(stream) { this._stream = stream; }
    get desiredSize() {
      if (this._stream._state === "errored") return null;
      return this._stream._state === "readable" && this._stream._queue.length === 0 ? 1 : 0;
    }
    enqueue(chunk) {
      const s = this._stream;
      if (s._state !== "readable") {
        throw new TypeError("Cannot enqueue into a closed or errored ReadableStream");
      }
      s._progress = true;
      s._queue.push(chunk);
      s._drain();
    }
    close() {
      const s = this._stream;
      if (s._state !== "readable") {
        throw new TypeError("Cannot close a closed or errored ReadableStream");
      }
      s._progress = true;
      s._closeRequested = true;
      s._drain();
    }
    error(e) { this._stream._error(e); }
  }
  class ReadableStreamDefaultReader {
    constructor(stream) { this._stream = stream; }
    get closed() { return this._stream._closedPromise; }
    read() {
      const s = this._stream;
      if (s._readers.indexOf(this) < 0) {
        return Promise.reject(new TypeError("Reader is not attached to this ReadableStream"));
      }
      return s._read();
    }
    releaseLock() {
      const s = this._stream;
      const i = s._readers.indexOf(this);
      if (i >= 0) s._readers.splice(i, 1);
      if (!s._readers.length) s._locked = false;
    }
    cancel(reason) {
      const s = this._stream;
      this.releaseLock();
      return s.cancel(reason);
    }
  }
  class ReadableStream {
    constructor(source, _strategy) {
      this._source = source || {};
      this._queue = [];
      this._waiters = [];
      this._readers = [];
      this._state = "readable";
      this._closeRequested = false;
      this._pulling = false;
      this._progress = false;
      this._locked = false;
      this._errored = undefined;
      this._closedPromise = new Promise((resolve, reject) => {
        this._resolveClosed = resolve;
        this._rejectClosed = reject;
      });
      this._closedPromise.catch(function () {});
      this._controller = new ReadableStreamDefaultController(this);
      try {
        if (typeof this._source.start === "function") {
          const r = this._source.start(this._controller);
          if (r && typeof r.then === "function") {
            const self = this;
            r.then(null, function (e) {
              self._error(e);
            });
          }
        }
      } catch (e) {
        this._error(e);
      }
    }
    get locked() { return this._locked; }
    getReader(_options) {
      if (this._locked) throw new TypeError("ReadableStream is locked to a reader");
      this._locked = true;
      const reader = new ReadableStreamDefaultReader(this);
      this._readers.push(reader);
      return reader;
    }
    cancel(reason) {
      this._queue.length = 0;
      if (this._state === "readable" && typeof this._source.cancel === "function") {
        try {
          this._source.cancel(reason);
        } catch (e) {
        }
      }
      if (this._state === "readable") {
        this._closeRequested = true;
        this._drain();
      }
      return Promise.resolve();
    }
    pipeTo(dest) {
      const reader = this.getReader();
      const writer = dest.getWriter();
      const pump = function () {
        return reader.read().then(function (r) {
          if (r.done) return writer.close();
          return writer.ready.then(function () {
            return writer.write(r.value);
          }).then(pump);
        });
      };
      return pump().then(
        function () {
          reader.releaseLock();
          writer.releaseLock();
        },
        function (e) {
          try {
            reader.releaseLock();
          } catch (_) {}
          return Promise.reject(e);
        }
      );
    }
    pipeThrough(transform) {
      this.pipeTo(transform.writable);
      return transform.readable;
    }
    tee() {
      const reader = this.getReader();
      const make = function () {
        let ctrl = null;
        const s = new ReadableStream({
          start: function (c) {
            ctrl = c;
          },
        });
        return { stream: s, ctrl: ctrl };
      };
      const a = make();
      const b = make();
      const pump = function () {
        return reader.read().then(
          function (r) {
            if (r.done) {
              a.ctrl.close();
              b.ctrl.close();
              return;
            }
            a.ctrl.enqueue(r.value);
            b.ctrl.enqueue(r.value);
            return pump();
          },
          function (e) {
            a.ctrl.error(e);
            b.ctrl.error(e);
          }
        );
      };
      pump();
      return [a.stream, b.stream];
    }
    [Symbol.asyncIterator]() {
      const reader = this.getReader();
      return {
        next: function () {
          return reader.read();
        },
        return: function (value) {
          reader.releaseLock();
          return Promise.resolve({ value: value, done: true });
        },
      };
    }
    _read() {
      if (this._state === "errored") return Promise.reject(this._errored);
      if (this._queue.length) {
        const value = this._queue.shift();
        this._maybePull();
        return Promise.resolve({ value: value, done: false });
      }
      if (this._closeRequested || this._state === "closed") {
        return Promise.resolve({ value: undefined, done: true });
      }
      const self = this;
      return new Promise(function (resolve, reject) {
        self._waiters.push({ resolve: resolve, reject: reject });
        self._maybePull();
      });
    }
    _drain() {
      while (this._waiters.length && this._queue.length) {
        const w = this._waiters.shift();
        w.resolve({ value: this._queue.shift(), done: false });
      }
      if (this._state === "readable" && this._closeRequested && !this._queue.length) {
        this._finish();
        return;
      }
      this._maybePull();
    }
    _finish() {
      if (this._state !== "readable") return;
      this._state = "closed";
      const ws = this._waiters.splice(0);
      for (let i = 0; i < ws.length; i++) {
        ws[i].resolve({ value: undefined, done: true });
      }
      this._resolveClosed();
    }
    _error(e) {
      if (this._state !== "readable") return;
      this._state = "errored";
      this._errored = e;
      const ws = this._waiters.splice(0);
      for (let i = 0; i < ws.length; i++) ws[i].reject(e);
      this._rejectClosed(e);
    }
    _maybePull() {
      if (this._pulling || this._state !== "readable" || this._closeRequested) return;
      if (typeof this._source.pull !== "function") return; // start()/enqueue-only source
      if (!this._waiters.length) return; // no demand yet
      this._pulling = true;
      this._progress = false;
      const self = this;
      let r;
      try {
        r = this._source.pull(this._controller);
      } catch (e) {
        this._pulling = false;
        this._error(e);
        return;
      }
      Promise.resolve(r).then(
        function () {
          self._pulling = false;
          if (self._progress) self._maybePull();
        },
        function (e) {
          self._pulling = false;
          self._error(e);
        }
      );
    }
  }
  class WritableStreamDefaultWriter {
    constructor(stream) { this._stream = stream; }
    get closed() { return this._stream._closedPromise; }
    get ready() { return this._stream._readyPromise; }
    write(chunk) {
      const s = this._stream;
      if (s._state !== "writable") {
        return Promise.reject(new TypeError("WritableStream is not writable"));
      }
      const p = s._chain.then(function () {
        if (typeof s._sink.write === "function") return s._sink.write(chunk, s._controller);
        return undefined;
      });
      s._chain = p.then(
        function () {},
        function () {}
      );
      const self = this;
      p.then(null, function (e) {
        self._stream._error(e);
      });
      return p;
    }
    close() {
      const s = this._stream;
      if (s._state !== "writable") return Promise.reject(new TypeError("WritableStream is not writable"));
      const p = s._chain.then(function () {
        if (typeof s._sink.close === "function") return s._sink.close();
        return undefined;
      });
      s._chain = p.then(
        function () {},
        function () {}
      );
      s._state = "closing";
      p.then(
        function () {
          s._state = "closed";
          s._resolveClosed();
        },
        function (e) {
          s._error(e);
        }
      );
      return p;
    }
    abort(reason) {
      const s = this._stream;
      s._state = "errored";
      const p = Promise.resolve(
        typeof s._sink.abort === "function" ? s._sink.abort(reason) : undefined
      );
      p.then(function () {
        s._rejectClosed(reason);
      });
      return p;
    }
    releaseLock() { this._stream._writer = null; }
  }
  class WritableStream {
    constructor(sink, _strategy) {
      this._sink = sink || {};
      this._state = "writable";
      this._writer = null;
      this._chain = Promise.resolve();
      this._closedPromise = new Promise((resolve, reject) => {
        this._resolveClosed = resolve;
        this._rejectClosed = reject;
      });
      this._closedPromise.catch(function () {});
      this._readyPromise = Promise.resolve();
      this._controller = { error: (e) => this._error(e) };
      try {
        if (typeof this._sink.start === "function") this._sink.start(this._controller);
      } catch (e) {
        this._error(e);
      }
    }
    get locked() { return this._writer !== null; }
    getWriter() {
      if (this._writer) throw new TypeError("WritableStream is locked to a writer");
      this._writer = new WritableStreamDefaultWriter(this);
      return this._writer;
    }
    abort(reason) { return this.getWriter().abort(reason); }
    _error(e) {
      if (this._state === "errored") return;
      this._state = "errored";
      this._rejectClosed(e);
    }
  }
  class TransformStreamDefaultController {
    constructor(stream) { this._stream = stream; }
    get desiredSize() { return this._stream._readable._controller.desiredSize; }
    enqueue(chunk) { this._stream._readable._controller.enqueue(chunk); }
    error(e) { this._stream._readable._controller.error(e); }
    terminate() { this._stream._readable._controller.close(); }
  }
  class TransformStream {
    constructor(transformer, _writableStrategy, _readableStrategy) {
      this._transformer = transformer || {};
      const self = this;
      this._readable = new ReadableStream({
        cancel: function (reason) {
          if (typeof self._transformer.cancel === "function") self._transformer.cancel(reason);
        },
      });
      this._controller = new TransformStreamDefaultController(this);
      this._writable = new WritableStream({
        start: function () {
          if (typeof self._transformer.start === "function") {
            return self._transformer.start(self._controller);
          }
          return undefined;
        },
        write: function (chunk) {
          if (typeof self._transformer.transform === "function") {
            return self._transformer.transform(chunk, self._controller);
          }
          self._controller.enqueue(chunk);
          return undefined;
        },
        close: function () {
          const done = function () {
            self._controller.terminate();
          };
          if (typeof self._transformer.flush === "function") {
            return Promise.resolve(self._transformer.flush(self._controller)).then(done);
          }
          done();
          return undefined;
        },
        abort: function (reason) {
          self._controller.error(reason);
        },
      });
    }
    get readable() { return this._readable; }
    get writable() { return this._writable; }
  }
  function toBytes(chunk) {
    if (chunk instanceof Uint8Array) return chunk;
    if (chunk instanceof ArrayBuffer) return new Uint8Array(chunk);
    if (ArrayBuffer.isView(chunk)) {
      return new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength);
    }
    throw new TypeError("Stream chunk is not a Uint8Array or ArrayBuffer");
  }
  function incompleteTail(b) {
    const n = b.length;
    for (let i = 1; i <= Math.min(4, n); i++) {
      const v = b[n - i];
      if ((v & 0xc0) === 0x80) continue; // continuation byte — keep looking
      if (v < 0x80) return n;
      let need;
      if ((v & 0xe0) === 0xc0) need = 2;
      else if ((v & 0xf0) === 0xe0) need = 3;
      else if ((v & 0xf8) === 0xf0) need = 4;
      else return n; // invalid lead: let the decoder produce the replacement
      return i >= need ? n : n - i;
    }
    return n;
  }
  class TextDecoderStream {
    constructor(label, options) {
      const decoder = new globalThis.TextDecoder(label, options);
      let pending = new Uint8Array(0);
      this._ts = new TransformStream({
        transform: function (chunk, controller) {
          const bytes = toBytes(chunk);
          let all = bytes;
          if (pending.length) {
            all = new Uint8Array(pending.length + bytes.length);
            all.set(pending, 0);
            all.set(bytes, pending.length);
          }
          const cut = incompleteTail(all);
          if (cut < all.length) {
            if (cut > 0) controller.enqueue(decoder.decode(all.subarray(0, cut)));
            pending = all.slice(cut);
          } else {
            pending = new Uint8Array(0);
            if (all.length) controller.enqueue(decoder.decode(all));
          }
        },
        flush: function (controller) {
          if (pending.length) {
            controller.enqueue(decoder.decode(pending));
            pending = new Uint8Array(0);
          }
        },
      });
    }
    get readable() { return this._ts.readable; }
    get writable() { return this._ts.writable; }
  }
  class TextEncoderStream {
    constructor() {
      const encoder = new globalThis.TextEncoder();
      this._ts = new TransformStream({
        transform: function (chunk, controller) {
          controller.enqueue(encoder.encode(String(chunk)));
        },
      });
    }
    get readable() { return this._ts.readable; }
    get writable() { return this._ts.writable; }
  }
  globalThis.ReadableStream = ReadableStream;
  globalThis.ReadableStreamDefaultReader = ReadableStreamDefaultReader;
  globalThis.ReadableStreamDefaultController = ReadableStreamDefaultController;
  globalThis.WritableStream = WritableStream;
  globalThis.WritableStreamDefaultWriter = WritableStreamDefaultWriter;
  globalThis.TransformStream = TransformStream;
  globalThis.TransformStreamDefaultController = TransformStreamDefaultController;
  globalThis.TextDecoderStream = TextDecoderStream;
  globalThis.TextEncoderStream = TextEncoderStream;
}
