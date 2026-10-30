// plugins.js — Sofuu Plugin & Extensibility Architecture (PLAN-PLUGIN-AND-EXTENSIBILITY-STANDARD).
//
// Exposes standard primitives for external extensions:
//   - defineTool(spec)         — Action plugins (LLM tools) with 5-Shield fault isolation
//   - defineProvider(spec)     — External model inference substrate adapters
//   - defineStorage(spec)      — External storage adapters
//   - defineVectorStore(spec)  — External semantic vector database adapters
//   - defineService(spec)      — Background lifecycle services with hook dispatch
//   - useOpenAPI(spec, opts)   — Zero-code OpenAPI 3.x / Swagger 2.0 REST auto-bridging
//   - ingestMcp(server, opts)  — MCP tool ingestion into 5-Shield plugins
//
// The 5-Shield Safety Boundary:
//   Shield 1: Input/output schema validation (E_SCHEMA, micro-Zod + JSON Schema)
//   Shield 2: Security & network jailing (E_DENIED per R7 policy and private IP / SSRF egress guard)
//   Shield 3: Automated circuit breakers (CLOSED / OPEN / HALF_OPEN, E_CIRCUIT_OPEN)
//   Shield 4: Hardware timer watchdog with signal abort (timeoutMs, E_TIMEOUT)
//   Shield 5: Isolated JS execution / panic guard (E_TOOL, host process never crashes)
//
// Conforms to normative Sofuu error taxonomy:
//   E_SCHEMA | E_TIMEOUT | E_DENIED | E_CIRCUIT_OPEN | E_TOOL | E_CAPABILITY
//
// 0.3.0 layout: the IIFE below is split into the eleven modules listed in
// P.modules (schema, security, breaker, watchdog, registry, provider, store,
// service, openapi, discovery, cli). The split is organisational only — every
// public name and result shape from 0.2.0 still exists.
//
// NOTE (services): the lifecycle hooks (emitTurnStart / emitToolCall /
// emitToolResult) are the AGENT-loop seam and are deliberately separate from
// the Shield 2 tool path. Callers own hook invocation so a plugin tool call is
// never double-intercepted.

(function () {
  'use strict';

  var g = (typeof globalThis !== 'undefined') ? globalThis :
          (typeof window !== 'undefined') ? window :
          (typeof global !== 'undefined') ? global : this;

  if (typeof g.sofuu === 'undefined' || !g.sofuu) g.sofuu = {};
  var sofuu = g.sofuu;
  if (sofuu.plugins && sofuu.plugins._loaded) return;

  var P = {};
  sofuu.plugins = P;
  P._loaded = true;
  P.VERSION = '0.3.0';
  P.version = '0.3.0';
  P.modules = ['schema', 'security', 'breaker', 'watchdog', 'registry',
               'provider', 'store', 'service', 'openapi', 'discovery', 'cli'];

  /* ══ module: core ════════════════════════════════════════════════════
   * Registries, error construction, environment helpers and the zero-
   * dependency URL parser shared by every other module. Nothing here
   * touches the host: a runtime without sofuu.fs / sofuu.mcp still loads. */

  P.TOOLS = {};
  P.PROVIDERS = {};
  P.STORES = {};
  P.SERVICES = {};
  P.CIRCUITS = {};
  P.MCP_CLIENTS = {};

  var NUL = String.fromCharCode(0);
  var BT = String.fromCharCode(96);

  function makeError(code, message, details) {
    var err = new Error(message || code);
    err.code = code;
    if (details) err.details = details;
    return err;
  }

  function noop() {}

  function envOf() {
    try {
      if (typeof process !== 'undefined' && process && process.env) return process.env;
    } catch (e) {}
    return {};
  }

  function cwdOf() {
    try {
      if (typeof process !== 'undefined' && typeof process.cwd === 'function') return process.cwd();
    } catch (e) {}
    return '.';
  }

  /* A Promise that is already "handled": the loser of every Promise.race
   * must not surface as an unhandled rejection (QuickJS reports those). */
  function handled(p) {
    if (p && typeof p.then === 'function') p.then(noop, noop);
    return p;
  }

  /* Built-in Robust URL Parser (zero-dependency for QuickJS) */

  function parseURL(urlStr) {
    if (!urlStr || typeof urlStr !== 'string') return null;
    var s = urlStr.trim();
    var match = s.match(/^([a-zA-Z][a-zA-Z0-9+.-]*):(?:\/\/([^\/?#]*))?([^?#]*)(?:\?([^#]*))?(?:#(.*))?$/);
    if (!match) return null;

    var protocol = match[1].toLowerCase();
    var authority = match[2] || '';
    var pathname = match[3] || '/';
    var search = match[4] ? ('?' + match[4]) : '';
    var hash = match[5] ? ('#' + match[5]) : '';

    var atIdx = authority.lastIndexOf('@');
    var userinfo = atIdx >= 0 ? authority.slice(0, atIdx) : '';
    var host = atIdx >= 0 ? authority.slice(atIdx + 1) : authority;

    var hostname = host;
    var port = '';
    if (host.charAt(0) === '[') {
      var closeBracket = host.indexOf(']');
      if (closeBracket > 0) {
        hostname = host.slice(1, closeBracket);
        if (host.charAt(closeBracket + 1) === ':') port = host.slice(closeBracket + 2);
      }
    } else {
      var colonIdx = host.indexOf(':');
      if (colonIdx >= 0) {
        hostname = host.slice(0, colonIdx);
        port = host.slice(colonIdx + 1);
      }
    }

    return {
      protocol: protocol + ':',
      authority: authority,
      userinfo: userinfo,
      hostname: hostname.toLowerCase(),
      port: port,
      pathname: pathname,
      search: search,
      hash: hash,
      href: s,
      origin: protocol + '://' + (port ? (hostname + ':' + port) : hostname)
    };
  }

  /* ── capabilities ─────────────────────────────────────────────────── *
   * mcp:true means the ingestion WRAPPER ships in this engine. The wrapper
   * still feature-detects sofuu.mcp at call time and returns E_CAPABILITY
   * when the host build has no MCP transport. */

  P.capabilities = function capabilities() {
    return {
      categories: {
        tool: true,
        provider: true,
        store: true,
        service: true,
        openapi: true,
        mcp: true
      },
      shields: ['schema', 'network', 'breaker', 'watchdog', 'panic'],
      zod: true,
      yaml: true
    };
  };

  /* ══ module: schema ══════════════════════════════════════════════════
   * Micro-Zod, JSON-Schema projection and the Shield 1 validators. The
   * defineTool entry point uses classifySchema() to reject a spec whose
   * schema is neither a micro-Zod object nor a JSON-Schema object instead
   * of silently degrading it to { type: 'object' }. */

  function isZodSchema(s) {
    return !!(s && (typeof s.parse === 'function' || typeof s.safeParse === 'function' || s._def));
  }

  /* 'zod' | 'json' — throws E_SCHEMA for anything that is not a schema.
   * Intentionally permissive about WHICH JSON-Schema keywords appear (the
   * validator only enforces a subset) but strict about the shape, because
   * a silently emptied schema is how bad tool arguments reach user code. */
  function classifySchema(schema, label) {
    if (schema === null || typeof schema !== 'object' || Array.isArray(schema)) {
      throw makeError('E_SCHEMA', label + ' must be a micro-Zod schema or a JSON-Schema object');
    }
    if (isZodSchema(schema)) return 'zod';
    if (typeof schema.type === 'string') return 'json';
    if (schema.properties && typeof schema.properties === 'object') return 'json';
    if (Array.isArray(schema.enum)) return 'json';
    if (schema.anyOf || schema.oneOf || schema.allOf || schema.$ref) return 'json';
    throw makeError('E_SCHEMA', label + ' is neither a micro-Zod schema nor a JSON-Schema object (expected {type:...} or {properties:...})');
  }

  /* A microZod node is an object that carries its builder methods (describe,
   * optional, default, int, safeParse …) as own properties, and returning it
   * verbatim used to leak functions into the "JSON schema": JSON.stringify()
   * then dropped the declared default (and every other function), so a model
   * never saw the default the author wrote, and `sofuu plugin list --json`
   * printed a schema whose `default` was a function. Emit a clean, JSON-safe
   * node instead — the live object keeps its methods for validation. */
  function zodNodeToJSON(node) {
    if (!node || typeof node !== 'object') return { type: 'string' };
    var out = {};
    if (node.type) out.type = node.type;
    if (Array.isArray(node.enum)) out.enum = node.enum.slice();
    if (node.description) out.description = node.description;
    if (node.minimum !== undefined && typeof node.minimum !== 'function') out.minimum = node.minimum;
    if (node.maximum !== undefined && typeof node.maximum !== 'function') out.maximum = node.maximum;
    if (node._default !== undefined) out.default = node._default;
    else if (node.default !== undefined && typeof node.default !== 'function') out.default = node.default;
    if (node.items) out.items = zodNodeToJSON(node.items);
    if (node.properties && typeof node.properties === 'object') {
      out.properties = {};
      for (var k in node.properties) {
        if (Object.prototype.hasOwnProperty.call(node.properties, k)) {
          out.properties[k] = zodNodeToJSON(node.properties[k]);
        }
      }
    }
    if (Array.isArray(node.required) && node.required.length) out.required = node.required.slice();
    if (!out.type) out.type = 'object';
    if (out.type === 'object' && !out.properties) out.properties = {};
    return out;
  }

  function zodToJSONSchema(zObj) {
    if (!zObj || typeof zObj !== 'object') return { type: 'object', properties: {} };
    if (zObj._isJSONSchema) return zodNodeToJSON(zObj);
    if (zObj.type && typeof zObj.type === 'string') return zObj;

    if (zObj._def) {
      var def = zObj._def;
      var typeName = def.typeName || (def.type && def.type.name) || '';
      if (typeName === 'ZodObject' || typeName === 'object' || def.shape) {
        var shape = (typeof def.shape === 'function') ? def.shape() : (def.shape || {});
        var props = {};
        var required = [];
        for (var k in shape) {
          if (Object.prototype.hasOwnProperty.call(shape, k)) {
            var field = shape[k];
            var fDef = field._def || {};
            var fType = fDef.typeName || '';
            var isOpt = fType === 'ZodOptional' || field.isOptional === true;
            props[k] = zodToJSONSchema(field);
            if (!isOpt) required.push(k);
          }
        }
        var res = { type: 'object', properties: props };
        if (required.length) res.required = required;
        if (zObj.description) res.description = zObj.description;
        return res;
      }
      if (typeName === 'ZodString' || typeName === 'string') return { type: 'string', description: zObj.description || '' };
      if (typeName === 'ZodNumber' || typeName === 'number') return { type: 'number', description: zObj.description || '' };
      if (typeName === 'ZodBoolean' || typeName === 'boolean') return { type: 'boolean', description: zObj.description || '' };
      if (typeName === 'ZodArray' || typeName === 'array') return { type: 'array', items: zodToJSONSchema(def.type), description: zObj.description || '' };
    }

    if (zObj.properties && typeof zObj.properties === 'object') return zObj;
    return { type: 'object', properties: {} };
  }

  function typeMismatch(schema, data, path) {
    var expectedType = schema.type;
    var actualType = Array.isArray(data) ? 'array' : (data === null ? 'null' : typeof data);
    if (expectedType === 'string' && actualType !== 'string') return { ok: false, path: path, error: 'Expected string, received ' + actualType };
    if (expectedType === 'number' && (actualType !== 'number' || isNaN(data))) return { ok: false, path: path, error: 'Expected number, received ' + actualType };
    if (expectedType === 'integer' && !Number.isInteger(data)) return { ok: false, path: path, error: 'Expected integer, received ' + actualType };
    if (expectedType === 'boolean' && actualType !== 'boolean') return { ok: false, path: path, error: 'Expected boolean, received ' + actualType };
    if (expectedType === 'array' && actualType !== 'array') return { ok: false, path: path, error: 'Expected array, received ' + actualType };
    if (expectedType === 'object' && (actualType !== 'object')) return { ok: false, path: path, error: 'Expected object, received ' + actualType };
    if (expectedType === 'null' && actualType !== 'null') return { ok: false, path: path, error: 'Expected null, received ' + actualType };
    if (Array.isArray(schema.enum) && schema.enum.indexOf(data) < 0) {
      return { ok: false, path: path, error: 'Value must be one of ' + JSON.stringify(schema.enum) };
    }
    return { ok: true };
  }

  function validateJSONSchema(schema, data, path) {
    path = path || '$';
    if (!schema || typeof schema !== 'object') return { ok: true };
    if (schema.type && schema.type !== 'any') {
      var tm = typeMismatch(schema, data, path);
      if (!tm.ok) return tm;
    }
    if (schema.type === 'object' && data && typeof data === 'object' && !Array.isArray(data)) {
      var required = schema.required || [];
      for (var i = 0; i < required.length; i++) {
        var req = required[i];
        if (!(req in data) || data[req] === undefined) {
          return { ok: false, path: path === '$' ? req : path + '.' + req, error: req + ' is required' };
        }
      }
      var props = schema.properties || {};
      for (var k in props) {
        if (Object.prototype.hasOwnProperty.call(props, k) && k in data && data[k] !== undefined) {
          var inner = validateJSONSchema(props[k], data[k], path === '$' ? k : path + '.' + k);
          if (!inner.ok) return inner;
        }
      }
    }
    if (schema.type === 'array' && Array.isArray(data) && schema.items) {
      for (var a = 0; a < data.length; a++) {
        var r = validateJSONSchema(schema.items, data[a], path + '[' + a + ']');
        if (!r.ok) return r;
      }
    }
    return { ok: true };
  }

  var microZod = {
    string: function () {
      return {
        _isJSONSchema: true,
        type: 'string',
        describe: function (d) { this.description = d; return this; },
        optional: function () { this.isOptional = true; return this; },
        /* PLAN §2 Cat 1 uses z.enum([...]).default('celsius'): a missing key
         * must fall back to the default instead of failing validation. */
        default: function (v) { this._default = v; this.isOptional = true; return this; },
        safeParse: function (val) {
          if (val === undefined && this.isOptional) return { success: true, data: val };
          if (typeof val !== 'string') return { success: false, error: { message: 'Expected string, received ' + typeof val } };
          return { success: true, data: val };
        }
      };
    },
    number: function () {
      return {
        _isJSONSchema: true,
        type: 'number',
        int: function () { this.type = 'integer'; this.isInt = true; return this; },
        positive: function () { this.minimum = 1; this.isPositive = true; return this; },
        describe: function (d) { this.description = d; return this; },
        optional: function () { this.isOptional = true; return this; },
        /* PLAN §2 Cat 1 uses z.enum([...]).default('celsius'): a missing key
         * must fall back to the default instead of failing validation. */
        default: function (v) { this._default = v; this.isOptional = true; return this; },
        safeParse: function (val) {
          if (val === undefined && this.isOptional) return { success: true, data: val };
          if (typeof val !== 'number' || isNaN(val)) return { success: false, error: { message: 'Expected number, received ' + typeof val } };
          if (this.isInt && !Number.isInteger(val)) return { success: false, error: { message: 'Expected integer, received float' } };
          if (this.isPositive && val <= 0) return { success: false, error: { message: 'Number must be positive' } };
          return { success: true, data: val };
        }
      };
    },
    boolean: function () {
      return {
        _isJSONSchema: true,
        type: 'boolean',
        describe: function (d) { this.description = d; return this; },
        optional: function () { this.isOptional = true; return this; },
        /* PLAN §2 Cat 1 uses z.enum([...]).default('celsius'): a missing key
         * must fall back to the default instead of failing validation. */
        default: function (v) { this._default = v; this.isOptional = true; return this; },
        safeParse: function (val) {
          if (val === undefined && this.isOptional) return { success: true, data: val };
          if (typeof val !== 'boolean') return { success: false, error: { message: 'Expected boolean, received ' + typeof val } };
          return { success: true, data: val };
        }
      };
    },
    array: function (item) {
      return {
        _isJSONSchema: true,
        type: 'array',
        items: zodToJSONSchema(item),
        itemSchema: item,
        describe: function (d) { this.description = d; return this; },
        optional: function () { this.isOptional = true; return this; },
        /* PLAN §2 Cat 1 uses z.enum([...]).default('celsius'): a missing key
         * must fall back to the default instead of failing validation. */
        default: function (v) { this._default = v; this.isOptional = true; return this; },
        safeParse: function (val) {
          if (val === undefined && this.isOptional) return { success: true, data: val };
          if (!Array.isArray(val)) return { success: false, error: { message: 'Expected array, received ' + typeof val } };
          if (this.itemSchema && typeof this.itemSchema.safeParse === 'function') {
            for (var i = 0; i < val.length; i++) {
              var r = this.itemSchema.safeParse(val[i]);
              if (!r.success) return { success: false, error: { message: r.error.message, path: [i].concat(r.error.path || []) } };
            }
          }
          return { success: true, data: val };
        }
      };
    },
    enum: function (values) {
      return {
        _isJSONSchema: true,
        type: 'string',
        enum: Array.isArray(values) ? values : [],
        describe: function (d) { this.description = d; return this; },
        optional: function () { this.isOptional = true; return this; },
        /* PLAN §2 Cat 1 uses z.enum([...]).default('celsius'): a missing key
         * must fall back to the default instead of failing validation. */
        default: function (v) { this._default = v; this.isOptional = true; return this; },
        safeParse: function (val) {
          if (val === undefined && this.isOptional) return { success: true, data: val };
          if (typeof val !== 'string' || this.enum.indexOf(val) < 0) {
            return { success: false, error: { message: 'Expected one of ' + JSON.stringify(this.enum) + ', received ' + JSON.stringify(val) } };
          }
          return { success: true, data: val };
        }
      };
    },
    any: function () {
      return {
        _isJSONSchema: true,
        type: 'any',
        describe: function (d) { this.description = d; return this; },
        optional: function () { this.isOptional = true; return this; },
        /* PLAN §2 Cat 1 uses z.enum([...]).default('celsius'): a missing key
         * must fall back to the default instead of failing validation. */
        default: function (v) { this._default = v; this.isOptional = true; return this; },
        safeParse: function (val) { return { success: true, data: val }; }
      };
    },
    record: function () {
      return {
        _isJSONSchema: true,
        type: 'object',
        describe: function (d) { this.description = d; return this; },
        optional: function () { this.isOptional = true; return this; },
        /* PLAN §2 Cat 1 uses z.enum([...]).default('celsius'): a missing key
         * must fall back to the default instead of failing validation. */
        default: function (v) { this._default = v; this.isOptional = true; return this; },
        safeParse: function (val) {
          if (val === undefined && this.isOptional) return { success: true, data: val };
          if (!val || typeof val !== 'object' || Array.isArray(val)) return { success: false, error: { message: 'Expected record object' } };
          return { success: true, data: val };
        }
      };
    },
    literal: function (expected) {
      return {
        _isJSONSchema: true,
        type: typeof expected,
        describe: function (d) { this.description = d; return this; },
        optional: function () { this.isOptional = true; return this; },
        /* PLAN §2 Cat 1 uses z.enum([...]).default('celsius'): a missing key
         * must fall back to the default instead of failing validation. */
        default: function (v) { this._default = v; this.isOptional = true; return this; },
        safeParse: function (val) {
          if (val === undefined && this.isOptional) return { success: true, data: val };
          if (val !== expected) return { success: false, error: { message: 'Expected literal ' + JSON.stringify(expected) } };
          return { success: true, data: val };
        }
      };
    },
    object: function (shape) {
      var props = {};
      var required = [];
      shape = shape || {};
      for (var k in shape) {
        if (Object.prototype.hasOwnProperty.call(shape, k)) {
          var item = shape[k];
          props[k] = zodToJSONSchema(item);
          if (!item.isOptional) required.push(k);
        }
      }
      return {
        _isJSONSchema: true,
        type: 'object',
        properties: props,
        required: required,
        shape: shape,
        describe: function (d) { this.description = d; return this; },
        safeParse: function (val) {
          if (!val || typeof val !== 'object' || Array.isArray(val)) {
            return { success: false, error: { message: 'Expected object, received ' + typeof val, path: [] } };
          }
          /* Defaults are applied to a shallow COPY before validation, so a
           * declared default satisfies a required-looking key and the caller's
           * argument object is never mutated. */
          var out = {};
          var ck;
          for (ck in val) { if (Object.prototype.hasOwnProperty.call(val, ck)) out[ck] = val[ck]; }
          for (ck in shape) {
            if (!Object.prototype.hasOwnProperty.call(shape, ck)) continue;
            var defSchema = shape[ck];
            if (out[ck] === undefined && defSchema && defSchema._default !== undefined) out[ck] = defSchema._default;
          }
          for (var r = 0; r < required.length; r++) {
            var reqKey = required[r];
            if (!(reqKey in out) || out[reqKey] === undefined) {
              return { success: false, error: { message: reqKey + ' is required', path: [reqKey] } };
            }
          }
          for (var propKey in shape) {
            if (Object.prototype.hasOwnProperty.call(shape, propKey)) {
              if (propKey in out && out[propKey] !== undefined) {
                var fieldValidator = shape[propKey];
                if (fieldValidator && typeof fieldValidator.safeParse === 'function') {
                  var res = fieldValidator.safeParse(out[propKey]);
                  if (!res.success) {
                    var innerPath = [propKey].concat(res.error.path || []);
                    return { success: false, error: { message: 'Invalid value for ' + propKey + ': ' + res.error.message, path: innerPath } };
                  }
                }
              }
            }
          }
          return { success: true, data: out };
        },
        parse: function (val) {
          var r = this.safeParse(val);
          if (!r.success) throw makeError('E_SCHEMA', r.error.message, r.error);
          return r.data;
        }
      };
    }
  };
  P.z = microZod;
  g.z = g.z || microZod;

  /* ── Shield 1 helpers (shared by tools and providers) ─────────────── */

  function validateArgs(schema, kind, args) {
    if (kind === 'zod' && typeof schema.safeParse === 'function') {
      var zRes = schema.safeParse(args);
      if (!zRes.success) {
        var msg = (zRes.error && zRes.error.message) || 'Argument validation failed';
        var p = (zRes.error && zRes.error.path && zRes.error.path.length) ? zRes.error.path.join('.') : '$';
        return { ok: false, message: msg, path: p };
      }
      return { ok: true, data: zRes.data === undefined ? args : zRes.data };
    }
    var jsonSchema = (kind === 'zod') ? zodToJSONSchema(schema) : schema;
    var v = validateJSONSchema(jsonSchema, args, '$');
    if (!v.ok) return { ok: false, message: v.error, path: v.path };
    return { ok: true, data: args };
  }

  /* ══ module: security ════════════════════════════════════════════════
   * Shield 2: URL scheme/credential policy, the SSRF guard, the
   * permissions.net allowlist (which is the ONLY way a plugin reaches
   * loopback) and the jailed filesystem view. */

  function isPrivateHost(host) {
    if (!host) return true;
    var h = String(host).toLowerCase().trim().replace(/^\[|\]$/g, '');
    /* A fully-qualified name ends with a dot: 'localhost.' and '127.0.0.1.'
     * are the same hosts to the resolver, and the trailing empty label used
     * to push the dotted-quad split to five parts, skipping the IPv4 range
     * test entirely — a one-character Shield 2 bypass. */
    while (h.length > 1 && h.charAt(h.length - 1) === '.') h = h.slice(0, -1);
    if (!h || h === 'localhost' || h === '0.0.0.0' || h === '::' || h === '::1') return true;
    if (h.endsWith('.localhost') || h.endsWith('.local') || h.endsWith('.internal')) return true;
    if (h === 'metadata.google.internal' || h.indexOf('169.254.') === 0) return true;
    if (h.indexOf('fe80:') === 0 || h.indexOf('fc00:') === 0 || h.indexOf('fd00:') === 0) return true;
    if (h.indexOf('::ffff:') === 0) h = h.slice(7);

    // Decimal or hex integer IP (e.g. 2130706433 or 0x7f000001)
    if (/^\d+$/.test(h) || /^0x[0-9a-f]+$/i.test(h)) {
      var n = parseInt(h, h.startsWith('0x') ? 16 : 10);
      if (!isNaN(n) && n >= 0 && n <= 0xffffffff) {
        h = [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255].join('.');
      }
    }

    // IPv4 dotted notation (including octal/hex segments)
    var parts = h.split('.');
    if (parts.length >= 1 && parts.length <= 4) {
      var nums = [];
      var valid = true;
      for (var i = 0; i < parts.length; i++) {
        var p = parts[i];
        var num;
        if (/^0x[0-9a-f]+$/i.test(p)) num = parseInt(p, 16);
        else if (/^0[0-7]+$/.test(p)) num = parseInt(p, 8);
        else if (/^\d+$/.test(p)) num = parseInt(p, 10);
        else { valid = false; break; }
        nums.push(num);
      }
      if (valid && nums.length > 0) {
        var a = nums[0];
        var b = nums[1] || 0;
        if (a === 127 || a === 10 || a === 0) return true;
        if (a === 172 && b >= 16 && b <= 31) return true;
        if (a === 192 && b === 168) return true;
        if (a === 169 && b === 254) return true;
        if (a === 100 && b >= 64 && b <= 127) return true;
      }
    }
    return false;
  }

  /* Link-local only: where cloud metadata endpoints live. A provider whose
   * baseUrl is loopback may reach other private hosts, but never these. */
  function isLinkLocalHost(host) {
    var h = String(host || '').toLowerCase().replace(/^\[|\]$/g, '');
    if (h === 'metadata.google.internal') return true;
    if (h.indexOf('169.254.') === 0) return true;
    if (h.indexOf('::ffff:169.254.') === 0) return true;
    if (h.indexOf('fe80:') === 0) return true;
    if (h === '0.0.0.0' || h === '::') return true;
    return false;
  }

  function netEntryMatches(host, entry) {
    var e = String(entry || '').toLowerCase().trim();
    if (!e) return false;
    if (e === '*') return true;
    if (e.indexOf('/') >= 0) e = e.slice(0, e.indexOf('/'));
    if (e.charAt(0) === '[' && e.indexOf(']') > 0) e = e.slice(1, e.indexOf(']'));
    var colon = e.lastIndexOf(':');
    if (colon > 0 && /^\d+$/.test(e.slice(colon + 1))) e = e.slice(0, colon);
    if (e.indexOf('*.') === 0) e = e.slice(2);
    return host === e || host.endsWith('.' + e);
  }

  /* Shared URL gate for both tool and provider fetches. Returns null when
   * the URL is acceptable, or an E_DENIED error. */
  function urlPolicy(url, label) {
    var parsed = parseURL(url);
    if (!parsed) return { error: makeError('E_DENIED', 'Invalid URL in ' + label + ': ' + url) };
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
      return { error: makeError('E_DENIED', 'URL scheme ' + parsed.protocol + ' is not allowed in ' + label + ' (http/https only)') };
    }
    if (parsed.userinfo) {
      return { error: makeError('E_DENIED', 'URLs with embedded credentials (user:pass@) are rejected in ' + label) };
    }
    return { parsed: parsed };
  }

  function createSandboxedFetch(toolName, permissions, defaultSignal) {
    var baseFetch = (typeof g.fetch === 'function') ? g.fetch : null;
    if (!baseFetch) return null;
    var allowedNets = (permissions && Array.isArray(permissions.net)) ? permissions.net : null;

    return function sandboxedFetch(url, init) {
      init = init || {};
      if (!init.signal && defaultSignal) init.signal = defaultSignal;
      var gate = urlPolicy(url, 'tool ' + toolName);
      if (gate.error) return Promise.reject(gate.error);
      var host = gate.parsed.hostname;
      if (allowedNets) {
        /* An explicit permissions.net allowlist is authoritative: only the
         * listed hosts (loopback included) are reachable. */
        var matched = false;
        for (var i = 0; i < allowedNets.length; i++) {
          if (netEntryMatches(host, allowedNets[i])) { matched = true; break; }
        }
        if (!matched) {
          return Promise.reject(makeError('E_DENIED', 'Host ' + host + ' is not allowed by network permissions for ' + toolName));
        }
      } else if (isPrivateHost(host)) {
        return Promise.reject(makeError('E_DENIED', 'Access to internal/private IP ' + host + ' is blocked for ' + toolName));
      }
      return baseFetch(url, init);
    };
  }

  /* Providers are the local-model path: when the adapter's own baseUrl is
   * loopback/private, private targets are allowed (127.0.0.1:11434 and
   * 127.0.0.1:8080 are the primary use case). Schemes and credentials are
   * still gated, and link-local metadata is still unreachable. */
  function createProviderFetch(providerName, spec, defaultSignal) {
    var baseFetch = (typeof g.fetch === 'function') ? g.fetch : null;
    if (!baseFetch) return null;
    var base = parseURL(spec && spec.baseUrl) || null;
    var basePrivate = base ? isPrivateHost(base.hostname) : false;
    var baseLinkLocal = base ? isLinkLocalHost(base.hostname) : false;

    return function providerFetch(url, init) {
      init = init || {};
      if (!init.signal && defaultSignal) init.signal = defaultSignal;
      var gate = urlPolicy(url, 'provider ' + providerName);
      if (gate.error) return Promise.reject(gate.error);
      var host = gate.parsed.hostname;
      if (basePrivate) {
        if (!baseLinkLocal && isLinkLocalHost(host)) {
          return Promise.reject(makeError('E_DENIED', 'Provider ' + providerName + ' may not reach link-local/metadata host ' + host));
        }
      } else if (isPrivateHost(host)) {
        return Promise.reject(makeError('E_DENIED', 'Access to internal/private IP ' + host + ' is blocked for provider ' + providerName));
      }
      return baseFetch(url, init);
    };
  }

  function createJailedFs(toolName, cwd) {
    if (!sofuu.fs) return null;
    cwd = (cwd || '.').replace(/\\/g, '/');

    function resolvePath(p) {
      if (!p || typeof p !== 'string') throw makeError('E_DENIED', 'Invalid path in tool ' + toolName);
      var norm = p.replace(/\\/g, '/');
      if (norm.indexOf('..') >= 0 || norm.indexOf('/.') >= 0) {
        throw makeError('E_DENIED', 'Path traversal outside project root is blocked for ' + toolName);
      }
      var lower = norm.toLowerCase();
      if (lower.indexOf('.env') >= 0 || lower.indexOf('.ssh') >= 0 || lower.indexOf('.aws') >= 0 || lower.indexOf('.git') >= 0) {
        throw makeError('E_DENIED', 'Access to sensitive file ' + p + ' is blocked for ' + toolName);
      }
      if (norm.charAt(0) === '/') {
        if (!norm.startsWith(cwd)) {
          throw makeError('E_DENIED', 'Access outside working directory is blocked for ' + toolName);
        }
        return norm;
      }
      return cwd.replace(/\/+$/, '') + '/' + norm.replace(/^\/+/, '');
    }

    return {
      readFile: function (p, opt) { return sofuu.fs.readFile(resolvePath(p), opt); },
      writeFile: function (p, data, opt) { return sofuu.fs.writeFile(resolvePath(p), data, opt); },
      exists: function (p) { return sofuu.fs.exists(resolvePath(p)); },
      readdir: function (p) { return (sofuu.fs.readdir || sofuu.fs.listDir)(resolvePath(p)); },
      listDir: function (p) { return (sofuu.fs.readdir || sofuu.fs.listDir)(resolvePath(p)); }
    };
  }

  /* R7 policy evaluator (host-provided). Returns a denial result or null. */
  function policyDenial(name) {
    try {
      if (sofuu.policy && typeof sofuu.policy.evaluate === 'function') {
        var polRaw = sofuu.policy.evaluate({ surface: 'js', op: 'tool.run', tool: name });
        var polDec = polRaw ? (typeof polRaw === 'string' ? JSON.parse(polRaw) : polRaw) : null;
        if (polDec && polDec.decision === 'deny') {
          return {
            ok: false,
            source: 'plugin',
            error: { code: 'E_DENIED', message: polDec.reason || ('Tool ' + name + ' denied by security policy') },
            text: 'tool error [E_DENIED]: denied by policy'
          };
        }
      }
    } catch (ePol) {}
    return null;
  }

/* ══ module: breaker ═════════════════════════════════════════════════
   * Shield 3. Counters are per circuit name ('tool' circuits use the bare
   * tool name; provider circuits are namespaced 'provider:<name>').
   * HALF_OPEN admits exactly ONE trial call via a trial token; concurrent
   * callers get E_CIRCUIT_OPEN until the trial records success/failure. */

  function getCircuit(name, opts) {
    if (!P.CIRCUITS[name]) {
      P.CIRCUITS[name] = {
        state: 'CLOSED',
        failures: 0,
        totalFailures: 0,
        opened: 0,
        lastFailure: 0,
        lastTripped: 0,
        trialInFlight: false,
        trialStartedAt: 0,
        failureThreshold: (opts && opts.failureThreshold) || 5,
        resetTimeoutMs: (opts && opts.resetTimeoutMs) || 30000
      };
    }
    return P.CIRCUITS[name];
  }

  function checkCircuit(name, opts) {
    var circuit = getCircuit(name, opts);
    var now = Date.now();
    if (circuit.state === 'OPEN') {
      if (now - circuit.lastTripped >= circuit.resetTimeoutMs) {
        circuit.state = 'HALF_OPEN';
        circuit.trialInFlight = true;
        circuit.trialStartedAt = now;
        return { ok: true, state: 'HALF_OPEN', trial: true };
      }
      return {
        ok: false,
        error: makeError(
          'E_CIRCUIT_OPEN',
          'Circuit breaker is OPEN for ' + name + ' (' + circuit.failures + ' failures; retry after ' +
            Math.ceil((circuit.resetTimeoutMs - (now - circuit.lastTripped)) / 1000) + 's)'
        )
      };
    }
    if (circuit.state === 'HALF_OPEN') {
      if (!circuit.trialInFlight) {
        circuit.trialInFlight = true;
        circuit.trialStartedAt = now;
        return { ok: true, state: 'HALF_OPEN', trial: true };
      }
      /* A trial that never recorded (a caller that died between the check
       * and the run) must not wedge the circuit forever. */
      if (now - circuit.trialStartedAt > circuit.resetTimeoutMs) {
        circuit.trialStartedAt = now;
        return { ok: true, state: 'HALF_OPEN', trial: true };
      }
      return {
        ok: false,
        error: makeError('E_CIRCUIT_OPEN', 'Circuit is HALF_OPEN for ' + name + ': a trial call is already in flight')
      };
    }
    return { ok: true, state: circuit.state, trial: false };
  }

  function recordCircuitSuccess(name) {
    var c = P.CIRCUITS[name];
    if (c) { c.state = 'CLOSED'; c.failures = 0; c.trialInFlight = false; }
  }

  function recordCircuitFailure(name, opts) {
    var c = getCircuit(name, opts);
    c.failures++;
    c.totalFailures++;
    c.lastFailure = Date.now();
    c.trialInFlight = false;
    if (c.state === 'HALF_OPEN' || c.failures >= c.failureThreshold) {
      if (c.state !== 'OPEN') c.opened++;
      c.state = 'OPEN';
      c.lastTripped = Date.now();
    }
  }

  P.resetCircuit = function resetCircuit(name) {
    if (name === undefined || name === null || name === '') {
      var keys = Object.keys(P.CIRCUITS);
      for (var i = 0; i < keys.length; i++) delete P.CIRCUITS[keys[i]];
      return true;
    }
    delete P.CIRCUITS[name];
    return true;
  };

  P.circuitState = function circuitState(name) {
    var c = P.CIRCUITS[name];
    if (!c) {
      return { name: name, state: 'CLOSED', known: false, failures: 0, totalFailures: 0, opened: 0, trialInFlight: false };
    }
    var now = Date.now();
    return {
      name: name,
      state: c.state,
      known: true,
      failures: c.failures,
      totalFailures: c.totalFailures,
      opened: c.opened,
      failureThreshold: c.failureThreshold,
      resetTimeoutMs: c.resetTimeoutMs,
      trialInFlight: c.trialInFlight,
      lastFailure: c.lastFailure,
      lastTripped: c.lastTripped,
      msUntilRetry: c.state === 'OPEN' ? Math.max(0, c.resetTimeoutMs - (now - c.lastTripped)) : 0
    };
  };

  /* ══ module: watchdog ════════════════════════════════════════════════
   * Shield 4 (deadline + AbortSignal) and Shield 5 (panic guard) in one
   * runner shared by tools and providers. It NEVER rejects.
   *
   * The 0.2.0 timeout promise leaked: when the tool won the race its
   * rejection was never handled, and QuickJS reports unhandled
   * rejections. Both race participants are now pre-handled and the timer
   * is cleared on every exit path. */

  var DEFAULT_TOOL_TIMEOUT_MS = 30000;
  var DEFAULT_PROVIDER_TIMEOUT_MS = 120000;

  function linkAbort(signal, cb) {
    if (!signal || typeof signal.addEventListener !== 'function') return function () {};
    try { signal.addEventListener('abort', cb); } catch (e) { return function () {}; }
    return function () {
      try { signal.removeEventListener('abort', cb); } catch (e) {}
    };
  }

  /* Shield 4 timers. One setTimeout per call costs one slot in the engine's
   * 1024-slot timer table, so 512 calls in flight wedged the table and every
   * later call failed with "too many timers (max 1024)". A single shared
   * ticker now serves every in-flight call with one slot, whatever the
   * concurrency. */
  var WD_TICK_MS = 15;
  var wdPending = [];
  var wdTimer = null;

  function wdPump() {
    wdTimer = null;
    if (wdPending.length === 0) return;
    var now = Date.now();
    var keep = [];
    for (var i = 0; i < wdPending.length; i++) {
      var e = wdPending[i];
      if (now >= e.deadline) { try { e.fire(); } catch (eF) {} }
      else keep.push(e);
    }
    wdPending = keep;
    if (wdPending.length > 0) wdArmTicker();
  }

  function wdArmTicker() {
    if (wdTimer !== null) return;
    wdTimer = setTimeout(wdPump, WD_TICK_MS);
    if (wdTimer && typeof wdTimer.unref === 'function') wdTimer.unref();
  }

  function wdAdd(entry) { wdPending.push(entry); wdArmTicker(); return entry; }

  function wdDrop(entry) {
    var i = wdPending.indexOf(entry);
    if (i >= 0) wdPending.splice(i, 1);
  }

  /* Shield 4's hard stop. The engine can interrupt the interpreter, but only
   * while a deadline is armed: without this, timeoutMs was enforced only if
   * the plugin happened to await, so a synchronous loop ran forever and a
   * synchronous 5s burn returned ok under a 300ms deadline.
   *
   * Exactly ONE arm is in force at a time and it is always the EARLIEST
   * deadline among the calls in flight: the host API only knows 'arm(ms)' and
   * 'disarm()', so leaving one arm per call would make a finished short call
   * disarm a longer one's deadline. Every start/settle recomputes the set and
   * re-arms the earliest. */
  var wdArms = [];
  var wdArmed = false;

  function hostHasDeadline() {
    return !!(sofuu.host && typeof sofuu.host.armDeadline === 'function' &&
              typeof sofuu.host.disarmDeadline === 'function');
  }

  /* An aborted turn can leave an entry behind (nothing runs in JS after an
   * uncatchable interrupt), so drop entries whose deadline passed long ago:
   * a stale arm would otherwise fire against unrelated work. */
  function hostPruneArms() {
    var t = Date.now();
    for (var i = wdArms.length - 1; i >= 0; i--) {
      if (wdArms[i].at + 5000 < t) wdArms.splice(i, 1);
    }
  }

  function hostRefreshArms() {
    if (!hostHasDeadline()) return;
    hostPruneArms();
    var earliest = null;
    var now = Date.now();
    for (var i = 0; i < wdArms.length; i++) {
      if (earliest === null || wdArms[i].at < earliest) earliest = wdArms[i].at;
    }
    try {
      if (wdArmed) { sofuu.host.disarmDeadline(); wdArmed = false; }
      if (earliest !== null) {
        sofuu.host.armDeadline(Math.max(1, earliest - Date.now()));
        wdArmed = true;
      }
    } catch (eR) { wdArmed = false; }
  }

  function hostArmFor(entry, ms) {
    if (!hostHasDeadline()) return;
    wdArms.push({ entry: entry, at: Date.now() + ms });
    hostRefreshArms();
  }

  function hostDisarmFor(entry) {
    for (var i = 0; i < wdArms.length; i++) {
      if (wdArms[i].entry === entry) { wdArms.splice(i, 1); break; }
    }
    hostRefreshArms();
  }

  function isInterrupt(err) {
    return /interrupt/i.test(String((err && err.message) || err || ''));
  }

  /* Grace window after a timeout: keep an arm alive briefly so a plugin that
   * ignores the abort signal and spins after its await is still cut off. */
  function wdArmGrace(entry, ms) {
    var grace = ms < 1000 ? 1000 : (ms > 5000 ? 5000 : ms);
    for (var i = 0; i < wdArms.length; i++) {
      if (wdArms[i].entry === entry) { wdArms[i].at = Date.now() + grace; break; }
    }
    hostRefreshArms();
    wdAdd({ deadline: Date.now() + grace, fire: function () { hostDisarmFor(entry); } });
  }

  /* fn(signal) may throw synchronously: that lands in the same catch as a
   * rejection, so execute()/completeVia() can never reject. */
  async function runWithWatchdog(label, timeoutMs, outerSignal, fn) {
    var startTs = Date.now();
    var ms = (typeof timeoutMs === 'number' && timeoutMs > 0) ? timeoutMs : DEFAULT_TOOL_TIMEOUT_MS;
    var ctrl = (typeof AbortController === 'function') ? new AbortController() : null;
    var signal = ctrl ? ctrl.signal : null;
    var entry = null;
    var timedOut = false;
    var aborted = false;

    if (outerSignal && outerSignal.aborted) {
      return { ok: false, code: 'E_TIMEOUT', message: label + ' aborted before start', durationMs: 0 };
    }

    var cancelReject = null;
    var cancelPromise = new Promise(function (_res, rej) { cancelReject = rej; });
    handled(cancelPromise);
    var unlink = linkAbort(outerSignal, function () {
      aborted = true;
      if (ctrl) { try { ctrl.abort(); } catch (eA) {} }
      if (cancelReject) cancelReject(makeError('E_TIMEOUT', label + ' aborted by caller'));
    });

    var timeoutPromise = new Promise(function (_res, rej) {
      entry = wdAdd({
        deadline: Date.now() + ms,
        fire: function () {
          timedOut = true;
          if (ctrl) { try { ctrl.abort(); } catch (eT) {} }
          rej(makeError('E_TIMEOUT', label + ' exceeded deadline of ' + ms + 'ms'));
        }
      });
    });
    handled(timeoutPromise);

    hostArmFor(entry, ms);
    var execPromise;
    try {
      execPromise = Promise.resolve(fn(signal));
    } catch (eSync) {
      /* The interpreter was interrupted at the deadline while the plugin
       * was running synchronously: that IS the timeout, not a tool fault. */
      if (isInterrupt(eSync)) {
        timedOut = true;
        execPromise = Promise.reject(makeError('E_TIMEOUT', label + ' exceeded deadline of ' + ms + 'ms (interrupted)'));
      } else {
        execPromise = Promise.reject(eSync);
      }
    }
    handled(execPromise);

    try {
      var value = await Promise.race([execPromise, timeoutPromise, cancelPromise]);
      wdDrop(entry);
      hostDisarmFor(entry);
      unlink();
      return { ok: true, value: value, durationMs: Date.now() - startTs };
    } catch (err) {
      wdDrop(entry);
      /* The arm must go with the call on EVERY exit path: a leftover entry
       * keeps a deadline in wdArms, and the next refresh then arms it as
       * "a few ms from now" and interrupts whatever unrelated work is
       * running at that moment. */
      hostDisarmFor(entry);
      unlink();
      if (ctrl) { try { ctrl.abort(); } catch (eB) {} }
      var code = (err && err.code) || 'E_TOOL';
      var message = String((err && err.message) || err);
      if (timedOut || aborted || isInterrupt(err)) {
        if (!timedOut && !aborted) message = label + ' exceeded deadline of ' + ms + 'ms (interrupted)';
        code = 'E_TIMEOUT';
        /* Hold the interrupt open briefly so work that ignored the abort
         * signal cannot spin past the deadline. */
        wdArmGrace(entry, ms);
      } else {
        hostDisarmFor(entry);
      }
      return { ok: false, code: code, message: message, durationMs: Date.now() - startTs };
    }
  }

  /* ══ module: registry ════════════════════════════════════════════════
   * Category 1 — action tools plus the inventory/lookup surface. */

  function buildToolCtx(name, spec, customCtx, signal) {
    var ctx = customCtx || {};
    var procCwd = cwdOf();
    var sandboxedFetch = createSandboxedFetch(name, spec.permissions, signal);
    var jailedFs = createJailedFs(name, procCwd);
    var env = envOf();
    return {
      name: name,
      fetch: sandboxedFetch || g.fetch,
      fs: jailedFs,
      signal: signal,
      abortSignal: signal,
      secrets: env,
      env: env,
      cwd: procCwd,
      model: ctx.model || '',
      state: ctx.state || {},
      log: function () {
        var msg = Array.prototype.slice.call(arguments).join(' ');
        if (ctx.log) ctx.log(msg);
        else console.log('[' + name + '] ' + msg);
      },
      error: function () {
        var msg = Array.prototype.slice.call(arguments).join(' ');
        console.error('[' + name + '] ' + msg);
      }
    };
  }

  function schemaFailure(message, path) {
    return {
      ok: false,
      source: 'plugin',
      error: { code: 'E_SCHEMA', message: message, path: path || '$' },
      text: 'tool error [E_SCHEMA]: ' + message + ' at ' + (path || '$')
    };
  }

  P.defineTool = function defineTool(spec) {
    if (!spec || typeof spec !== 'object') throw makeError('E_SCHEMA', 'defineTool requires a spec object');
    var name = String(spec.name || '').trim();
    if (!name) throw makeError('E_SCHEMA', 'defineTool requires a non-empty name');
    var runFn = spec.run || spec.execute;
    if (typeof runFn !== 'function') throw makeError('E_SCHEMA', 'defineTool requires run or execute function for ' + name);

    var timeoutMs = (typeof spec.timeoutMs === 'number' && spec.timeoutMs > 0) ? spec.timeoutMs : DEFAULT_TOOL_TIMEOUT_MS;
    var cbConfig = spec.circuitBreaker || { failureThreshold: 5, resetTimeoutMs: 30000 };

    /* Shield 1 at DEFINE time: a malformed schema used to be silently
     * replaced by {type:'object'}, which let every bad argument through. */
    var hadSchema = (spec.schema !== undefined) || (spec.parameters !== undefined);
    var rawSchema = (spec.schema !== undefined) ? spec.schema
                  : (spec.parameters !== undefined) ? spec.parameters
                  : { type: 'object', properties: {} };
    var schemaKind = classifySchema(rawSchema, 'Tool ' + name + ' schema');
    var jsonSchema = (schemaKind === 'zod') ? zodToJSONSchema(rawSchema) : rawSchema;

    var outputSchema = (spec.output !== undefined) ? spec.output : spec.resultSchema;
    var outputKind = (outputSchema !== undefined) ? classifySchema(outputSchema, 'Tool ' + name + ' output schema') : null;

    var wrappedTool = {
      name: name,
      description: String(spec.description || spec.summary || ''),
      schema: jsonSchema,
      parameters: jsonSchema,
      timeoutMs: timeoutMs,
      requiresApproval: spec.requiresApproval,
      permissions: spec.permissions || {},
      category: 'tool',
      /* Surfaced by `sofuu plugin validate`: a tool with no schema has an
       * inert Shield 1, which the author should hear about once. */
      warnings: hadSchema ? [] : ['no schema declared: arguments are not validated'],

      execute: async function execute(args, customCtx) {
        var startTs = Date.now();
        var cbState = null;
        try {
          args = (args && typeof args === 'object' && !Array.isArray(args)) ? args : {};

          var inCheck = validateArgs(rawSchema, schemaKind, args);
          if (!inCheck.ok) return schemaFailure(inCheck.message, inCheck.path);
          args = inCheck.data;

          var denied = policyDenial(name);
          if (denied) return denied;

          var cbCheck = checkCircuit(name, cbConfig);
          if (!cbCheck.ok) {
            return {
              ok: false,
              source: 'plugin',
              error: { code: 'E_CIRCUIT_OPEN', message: cbCheck.error.message },
              text: 'tool error [E_CIRCUIT_OPEN]: ' + cbCheck.error.message
            };
          }
          cbState = cbCheck;

          var outerSignal = (customCtx && customCtx.signal) || null;
          var outcome = await runWithWatchdog('Tool ' + name, timeoutMs, outerSignal, function (signal) {
            var toolCtx = buildToolCtx(name, spec, customCtx, signal);
            return runFn.call(spec, args, toolCtx);
          });

          if (!outcome.ok) {
            /* Shield 3 counts E_TIMEOUT like any other fault, but NOT a
             * policy denial: a denied call is Shield 2 doing its job, and
             * letting it open the breaker would turn eleven hostile targets
             * into eleven E_CIRCUIT_OPEN replies that hide the real code. */
            if (outcome.code !== 'E_DENIED') recordCircuitFailure(name, cbConfig);
            return {
              ok: false,
              source: 'plugin',
              error: { code: outcome.code, message: outcome.message },
              text: 'tool error [' + outcome.code + ']: ' + outcome.message,
              duration_ms: outcome.durationMs
            };
          }

          var rawRes = outcome.value;

          /* Shield 1 output invariant. A tool that reports a soft failure
           * ({ok:false}) is passed through untouched. */
          var softFailure = !!(rawRes && typeof rawRes === 'object' && rawRes.ok === false);
          if (outputKind && !softFailure) {
            var outCheck = validateArgs(outputSchema, outputKind, rawRes);
            if (!outCheck.ok) {
              recordCircuitFailure(name, cbConfig);
              return {
                ok: false,
                source: 'plugin',
                error: { code: 'E_SCHEMA', message: outCheck.message, path: '$output' },
                text: 'tool error [E_SCHEMA]: output mismatch: ' + outCheck.message + ' at $output',
                duration_ms: Date.now() - startTs
              };
            }
          }

          recordCircuitSuccess(name);

          if (rawRes && typeof rawRes === 'object' && typeof rawRes.ok === 'boolean') {
            rawRes.duration_ms = Date.now() - startTs;
            rawRes.source = rawRes.source || 'plugin';
            return rawRes;
          }

          var textResult = (typeof rawRes === 'string')
            ? rawRes
            : ((rawRes !== undefined) ? JSON.stringify(rawRes, null, 2) : 'ok');

          return {
            ok: true,
            source: 'plugin',
            data: rawRes,
            text: textResult,
            duration_ms: Date.now() - startTs
          };
        } catch (err) {
          /* Shield 5 outer guard: execute() never rejects, whatever a
           * service/policy/ctx helper throws. */
          if (cbState && cbState.ok) recordCircuitFailure(name, cbConfig);
          var eCode = (err && err.code) || 'E_TOOL';
          var eMsg = String((err && err.message) || err);
          return {
            ok: false,
            source: 'plugin',
            error: { code: eCode, message: eMsg },
            text: 'tool error [' + eCode + ']: ' + eMsg,
            duration_ms: Date.now() - startTs
          };
        }
      }
    };

    /* PLAN §2 Cat 1: Shield 1 rejects malformed arguments "with typed
     * JSON-RPC -32602 error data". The engine's own codes are the primary
     * taxonomy; this adds the JSON-RPC code beside them so an MCP/JSON-RPC
     * client can map a plugin failure without string-matching E_* names. */
    var RPC_CODES = {
      E_SCHEMA: -32602, E_CAPABILITY: -32601, E_TOOL: -32603,
      E_TIMEOUT: -32000, E_DENIED: -32001, E_CIRCUIT_OPEN: -32002, E_BACKPRESSURE: -32003
    };
    var rawExecute = wrappedTool.execute;
    wrappedTool.execute = async function execute(args, customCtx) {
      var res = await rawExecute.call(wrappedTool, args, customCtx);
      if (res && res.error) {
        if (typeof res.error.jsonrpc !== 'number') {
          res.error.jsonrpc = (RPC_CODES[res.error.code] !== undefined) ? RPC_CODES[res.error.code] : -32603;
        }
        if (!res.error.data) {
          res.error.data = { tool: name, code: res.error.code };
          if (res.error.path) res.error.data.path = res.error.path;
        }
      }
      return res;
    };
    wrappedTool.run = wrappedTool.execute;
    P.TOOLS[name] = wrappedTool;

    if (sofuu.tools) {
      if (!sofuu.tools.TOOLS) sofuu.tools.TOOLS = {};
      sofuu.tools.TOOLS[name] = {
        name: name,
        description: wrappedTool.description,
        parameters: wrappedTool.parameters,
        execute: wrappedTool.execute,
        source: 'plugin',
        selfTimed: true,
        requiresApproval: spec.requiresApproval
      };
      if (Array.isArray(sofuu.tools.GROUP) && sofuu.tools.GROUP.indexOf(name) < 0) {
        sofuu.tools.GROUP.push(name);
      }
    }

    return wrappedTool;
  };

  P.invoke = async function invoke(name, args, ctx) {
    var tool = P.TOOLS[name];
    if (!tool) {
      return {
        ok: false,
        source: 'plugin',
        error: { code: 'E_CAPABILITY', message: 'Plugin tool "' + name + '" not found' },
        text: 'tool error [E_CAPABILITY]: plugin tool not found'
      };
    }
    try {
      return await tool.execute(args, ctx);
    } catch (e) {
      return {
        ok: false,
        source: 'plugin',
        error: { code: (e && e.code) || 'E_TOOL', message: String((e && e.message) || e) },
        text: 'tool error [' + ((e && e.code) || 'E_TOOL') + ']: ' + String((e && e.message) || e)
      };
    }
  };

  P.list = function list() {
    var out = [];
    var tKeys = Object.keys(P.TOOLS);
    for (var i = 0; i < tKeys.length; i++) {
      var t = P.TOOLS[tKeys[i]];
      var circ = P.CIRCUITS[tKeys[i]];
      out.push({
        name: t.name,
        category: 'tool',
        circuit: (circ && circ.state) || 'CLOSED',
        timeoutMs: t.timeoutMs || DEFAULT_TOOL_TIMEOUT_MS,
        description: t.description || '',
        requiresApproval: t.requiresApproval || null,
        origin: t._origin || ''
      });
    }

    var pKeys = Object.keys(P.PROVIDERS);
    for (var pi = 0; pi < pKeys.length; pi++) {
      var p = P.PROVIDERS[pKeys[pi]];
      var pcirc = P.CIRCUITS['provider:' + pKeys[pi]];
      out.push({
        name: p.name,
        category: 'provider',
        circuit: (pcirc && pcirc.state) || 'CLOSED',
        timeoutMs: p.timeoutMs || DEFAULT_PROVIDER_TIMEOUT_MS,
        description: p.description || '',
        kind: p.kind || 'custom',
        origin: p._origin || ''
      });
    }

    var stKeys = Object.keys(P.STORES);
    for (var sti = 0; sti < stKeys.length; sti++) {
      var st = P.STORES[stKeys[sti]];
      out.push({
        name: st.name,
        category: 'store',
        circuit: 'CLOSED',
        timeoutMs: DEFAULT_TOOL_TIMEOUT_MS,
        description: st.description || '',
        kind: st.kind || (typeof st.upsert === 'function' ? 'vector' : 'kv'),
        origin: st._origin || ''
      });
    }

    var sKeys = Object.keys(P.SERVICES);
    for (var si = 0; si < sKeys.length; si++) {
      var s = P.SERVICES[sKeys[si]];
      out.push({ name: s.name, category: 'service', circuit: 'CLOSED', timeoutMs: DEFAULT_TOOL_TIMEOUT_MS, description: s.description || '', origin: s._origin || '' });
    }
    return out;
  };

/* ══ module: provider ════════════════════════════════════════════════
   * Category 2 — inference substrate. defineProvider() is the registration
   * surface used by plugin files; registerProvider() is the runtime entry
   * point used by src/js/providers.js (built-in adapters). Both store the
   * same spec shape:
   *
   *   { name, kind:'ollama'|'llamacpp'|'openai'|'custom', baseUrl, model,
   *     description, async complete(req, ctx), async stream(req, ctx, onEvent) }
   *   req = { messages:[{role,content}], model, tools, temperature, maxTokens }
   *
   * completeVia/streamVia wrap a call in the same Shield 3/4/5 machinery a
   * tool uses and normalise the result to one shape:
   *   { ok, source:'plugin', data:{ text, toolCalls?, usage:{promptTokens,
   *     completionTokens}, model }, text, duration_ms }
   *   { ok:false, error:{ code, message } }
   */

  var PROVIDER_KINDS = { ollama: 1, llamacpp: 1, openai: 1, custom: 1 };

  function assertProviderSpec(spec, label) {
    if (!spec || typeof spec !== 'object') throw makeError('E_SCHEMA', label + ' requires a spec object');
    var name = String(spec.name || '').trim();
    if (!name) throw makeError('E_SCHEMA', label + ' requires a name');
    if (typeof spec.complete !== 'function' && typeof spec.stream !== 'function') {
      throw makeError('E_SCHEMA', label + ' requires a complete() or stream() function');
    }
    if (spec.kind !== undefined && !PROVIDER_KINDS[String(spec.kind)]) {
      throw makeError('E_SCHEMA', label + ' kind must be one of ollama, llamacpp, openai, custom (got ' + spec.kind + ')');
    }
    if (spec.baseUrl !== undefined && typeof spec.baseUrl !== 'string') {
      throw makeError('E_SCHEMA', label + ' baseUrl must be a string');
    }
    if (spec.kind === 'ollama' && spec.baseUrl) {
      var kp = parseURL(spec.baseUrl);
      if (!kp) throw makeError('E_SCHEMA', label + ' baseUrl is not a valid URL: ' + spec.baseUrl);
    }
    return spec;
  }

  P.registerProvider = function registerProvider(spec) {
    assertProviderSpec(spec, 'registerProvider');
    spec.name = String(spec.name).trim();
    spec.kind = spec.kind || 'custom';
    spec.category = 'provider';
    P.PROVIDERS[spec.name] = spec;
    return spec;
  };

  P.getProvider = function getProvider(name) {
    return lookupProvider(name);
  };

  function lookupProvider(name) {
    if (name === undefined || name === null || name === '') return null;
    var n = String(name);
    if (P.PROVIDERS[n]) return P.PROVIDERS[n];
    var lower = n.toLowerCase();
    var keys = Object.keys(P.PROVIDERS);
    var i;
    for (i = 0; i < keys.length; i++) if (keys[i].toLowerCase() === lower) return P.PROVIDERS[keys[i]];
    var norm = lower.replace(/[^a-z0-9]/g, '');
    for (i = 0; i < keys.length; i++) {
      if (keys[i].toLowerCase().replace(/[^a-z0-9]/g, '') === norm) return P.PROVIDERS[keys[i]];
    }
    /* Sibling namespace: providers.js may publish adapters on sofuu.providers
     * before (or instead of) calling registerProvider. */
    var ext = sofuu.providers;
    if (ext && typeof ext === 'object') {
      if (ext.PROVIDERS && ext.PROVIDERS[n]) return ext.PROVIDERS[n];
      if (typeof ext.get === 'function') {
        try { var got = ext.get(n); if (got && typeof got === 'object') return got; } catch (e) {}
      }
    }
    return null;
  }

  P.resolveProvider = function resolveProvider(name) {
    return lookupProvider(name);
  };

  P.listProviders = function listProviders() {
    var keys = Object.keys(P.PROVIDERS);
    var out = [];
    for (var i = 0; i < keys.length; i++) {
      var p = P.PROVIDERS[keys[i]];
      var c = P.CIRCUITS['provider:' + keys[i]];
      out.push({
        name: p.name,
        kind: p.kind || 'custom',
        baseUrl: p.baseUrl || '',
        model: p.model || '',
        description: p.description || '',
        hasStream: typeof p.stream === 'function',
        circuit: (c && c.state) || 'CLOSED'
      });
    }
    return out;
  };

  function normalizeProviderRequest(spec, req) {
    req = req || {};
    var messages = Array.isArray(req.messages) ? req.messages : [];
    if (!messages.length && typeof req.prompt === 'string') messages = [{ role: 'user', content: req.prompt }];
    var clean = [];
    for (var i = 0; i < messages.length; i++) {
      var m = messages[i] || {};
      clean.push({ role: String(m.role || 'user'), content: (m.content === undefined || m.content === null) ? '' : m.content });
    }
    /* PLAN §2 Cat 2 documents complete({messages, model, stream, tools}) — so
     * the normalised request carries every field an adapter may branch on,
     * including the llama.cpp GBNF grammar and OpenAI response_format, which
     * used to exist only on the direct-adapter path. */
    var out = {
      messages: clean,
      model: req.model || spec.model || '',
      tools: req.tools || undefined,
      temperature: (typeof req.temperature === 'number') ? req.temperature : undefined,
      maxTokens: (typeof req.maxTokens === 'number') ? req.maxTokens : undefined
    };
    /* complete() is never a streaming call, but the field is documented, so
     * an adapter that branches on it sees a real boolean instead of undefined. */
    out.stream = (req.stream !== undefined) ? !!req.stream : false;
    if (req.grammar !== undefined) out.grammar = req.grammar;
    if (req.response_format !== undefined) out.response_format = req.response_format;
    if (req.responseFormat !== undefined && out.response_format === undefined) out.response_format = req.responseFormat;
    return out;
  }

  function buildProviderCtx(spec, customCtx, signal) {
    var ctx = customCtx || {};
    var env = envOf();
    /* A per-call override wins over the adapter's own endpoint/credential:
     * without this, ctx.baseUrl and ctx.apiKey were silently discarded and
     * only process.env could authenticate through completeVia. */
    return {
      name: spec.name,
      baseUrl: (typeof ctx.baseUrl === 'string' && ctx.baseUrl) ? ctx.baseUrl : (spec.baseUrl || ''),
      apiKey: (typeof ctx.apiKey === 'string' && ctx.apiKey) ? ctx.apiKey : (ctx.api_key || ''),
      model: (typeof ctx.model === 'string' && ctx.model) ? ctx.model : (spec.model || ''),
      fetch: createProviderFetch(spec.name, spec, signal) || g.fetch,
      signal: signal,
      abortSignal: signal,
      secrets: env,
      env: env,
      state: ctx.state || {},
      log: function () {
        var msg = Array.prototype.slice.call(arguments).join(' ');
        if (ctx.log) ctx.log(msg);
        else console.log('[provider:' + spec.name + '] ' + msg);
      }
    };
  }

  function firstDefined() {
    for (var i = 0; i < arguments.length; i++) {
      if (arguments[i] !== undefined && arguments[i] !== null) return arguments[i];
    }
    return undefined;
  }

  function providerText(raw) {
    if (raw === null || raw === undefined) return '';
    if (typeof raw === 'string') return raw;
    if (typeof raw === 'number' || typeof raw === 'boolean') return String(raw);
    if (typeof raw.text === 'string') return raw.text;
    if (raw.data && typeof raw.data === 'object' && typeof raw.data.text === 'string') return raw.data.text;
    if (raw.message && typeof raw.message === 'object' && typeof raw.message.content === 'string') return raw.message.content;
    if (Array.isArray(raw.choices) && raw.choices.length) {
      var c0 = raw.choices[0] || {};
      if (c0.message && typeof c0.message.content === 'string') return c0.message.content;
      if (typeof c0.text === 'string') return c0.text;
    }
    if (typeof raw.content === 'string') return raw.content;
    return '';
  }

  function providerToolCalls(raw) {
    if (!raw || typeof raw !== 'object') return undefined;
    if (Array.isArray(raw.toolCalls)) return raw.toolCalls;
    if (Array.isArray(raw.tool_calls)) return raw.tool_calls;
    if (raw.data && Array.isArray(raw.data.toolCalls)) return raw.data.toolCalls;
    if (raw.message && Array.isArray(raw.message.tool_calls)) return raw.message.tool_calls;
    if (Array.isArray(raw.choices) && raw.choices[0] && raw.choices[0].message && Array.isArray(raw.choices[0].message.tool_calls)) {
      return raw.choices[0].message.tool_calls;
    }
    return undefined;
  }

  function providerUsage(raw) {
    var u = (raw && typeof raw === 'object') ? (raw.usage || (raw.data && raw.data.usage) || null) : null;
    var pt = u ? firstDefined(u.promptTokens, u.prompt_tokens, u.input_tokens) : undefined;
    var ct = u ? firstDefined(u.completionTokens, u.completion_tokens, u.output_tokens) : undefined;
    if (pt === undefined && raw && typeof raw === 'object') pt = raw.prompt_eval_count;
    if (ct === undefined && raw && typeof raw === 'object') ct = raw.eval_count;
    var tt = u ? firstDefined(u.totalTokens, u.total_tokens) : undefined;
    if (typeof tt !== 'number') {
      tt = (typeof pt === 'number' ? pt : 0) + (typeof ct === 'number' ? ct : 0);
    }
    return {
      promptTokens: (typeof pt === 'number') ? pt : 0,
      completionTokens: (typeof ct === 'number') ? ct : 0,
      totalTokens: tt
    };
  }

  function normalizeProviderResult(spec, req, raw, durationMs, streamed, chunks) {
    var text = providerText(raw);
    var data = {
      text: text,
      usage: providerUsage(raw),
      model: (raw && typeof raw === 'object' && raw.model) ? String(raw.model) : (req.model || spec.model || '')
    };
    var calls = providerToolCalls(raw);
    if (calls) data.toolCalls = calls;
    if (streamed) { data.streamed = true; data.chunks = chunks || 0; }
    return { ok: true, source: 'plugin', data: data, text: text, duration_ms: durationMs };
  }

  function providerFailure(spec, code, message, durationMs) {
    return {
      ok: false,
      source: 'plugin',
      error: { code: code, message: message },
      text: 'provider error [' + code + ']: ' + message,
      duration_ms: durationMs || 0,
      provider: spec ? spec.name : undefined
    };
  }

  P.completeVia = async function completeVia(name, req, ctx) {
    ctx = ctx || {};
    var spec = lookupProvider(name);
    if (!spec) {
      return providerFailure(null, 'E_CAPABILITY', 'Provider "' + name + '" is not registered (expects defineProvider/registerProvider)', 0);
    }
    var cbName = 'provider:' + spec.name;
    var cbConfig = spec.circuitBreaker || { failureThreshold: 5, resetTimeoutMs: 30000 };
    var cbCheck = checkCircuit(cbName, cbConfig);
    if (!cbCheck.ok) return providerFailure(spec, 'E_CIRCUIT_OPEN', cbCheck.error.message, 0);

    var timeoutMs = (typeof ctx.timeoutMs === 'number' && ctx.timeoutMs > 0) ? ctx.timeoutMs
                  : (typeof spec.timeoutMs === 'number' && spec.timeoutMs > 0) ? spec.timeoutMs
                  : DEFAULT_PROVIDER_TIMEOUT_MS;
    var normalized = normalizeProviderRequest(spec, req);
    var outcome = await runWithWatchdog('Provider ' + spec.name, timeoutMs, ctx.signal || null, function (signal) {
      var pctx = buildProviderCtx(spec, ctx, signal);
      return spec.complete.call(spec, normalized, pctx);
    });
    if (!outcome.ok) {
      recordCircuitFailure(cbName, cbConfig);
      return providerFailure(spec, outcome.code, outcome.message, outcome.durationMs);
    }
    recordCircuitSuccess(cbName);
    return normalizeProviderResult(spec, normalized, outcome.value, outcome.durationMs, false, 0);
  };

  function streamEventText(evt) {
    if (typeof evt === 'string') return evt;
    if (!evt || typeof evt !== 'object') return '';
    if (typeof evt.text === 'string') return evt.text;
    if (typeof evt.content === 'string') return evt.content;
    if (evt.delta && typeof evt.delta === 'object') {
      if (typeof evt.delta.content === 'string') return evt.delta.content;
      if (typeof evt.delta.text === 'string') return evt.delta.text;
    }
    if (evt.data && typeof evt.data === 'object') return streamEventText(evt.data);
    return '';
  }

  P.streamVia = async function streamVia(name, req, ctx, onEvent) {
    ctx = ctx || {};
    var emit = (typeof onEvent === 'function') ? onEvent : noop;
    var spec = lookupProvider(name);
    if (!spec) {
      return providerFailure(null, 'E_CAPABILITY', 'Provider "' + name + '" is not registered (expects defineProvider/registerProvider)', 0);
    }
    var cbName = 'provider:' + spec.name;
    var cbConfig = spec.circuitBreaker || { failureThreshold: 5, resetTimeoutMs: 30000 };
    var cbCheck = checkCircuit(cbName, cbConfig);
    if (!cbCheck.ok) return providerFailure(spec, 'E_CIRCUIT_OPEN', cbCheck.error.message, 0);

    var timeoutMs = (typeof ctx.timeoutMs === 'number' && ctx.timeoutMs > 0) ? ctx.timeoutMs
                  : (typeof spec.timeoutMs === 'number' && spec.timeoutMs > 0) ? spec.timeoutMs
                  : DEFAULT_PROVIDER_TIMEOUT_MS;
    var normalized = normalizeProviderRequest(spec, req);
    var acc = '';
    var chunks = 0;
    var toolCalls = null;

    /* A consumer callback must never break the stream, so every emit is
     * guarded. Events are forwarded verbatim. */
    var safeEmit = function (evt) {
      chunks++;
      var t = streamEventText(evt);
      if (t) acc += t;
      if (evt && typeof evt === 'object') {
        var calls = evt.toolCalls || (evt.delta && evt.delta.tool_calls);
        if (calls) toolCalls = calls;
      }
      try { emit(evt); } catch (eEmit) {}
    };

    var outcome = await runWithWatchdog('Provider ' + spec.name, timeoutMs, ctx.signal || null, async function (signal) {
      var pctx = buildProviderCtx(spec, ctx, signal);
      if (typeof spec.stream === 'function') {
        return spec.stream.call(spec, normalized, pctx, safeEmit);
      }
      /* Adapters without streaming still work: one synthetic event. */
      var res = await spec.complete.call(spec, normalized, pctx);
      safeEmit({ type: 'text', text: providerText(res) });
      return res;
    });
    if (!outcome.ok) {
      recordCircuitFailure(cbName, cbConfig);
      return providerFailure(spec, outcome.code, outcome.message, outcome.durationMs);
    }
    recordCircuitSuccess(cbName);

    var raw = outcome.value;
    if (raw && typeof raw === 'object' && typeof providerText(raw) === 'string' && providerText(raw) && acc !== providerText(raw)) {
      /* Adapters that return the final text as well: prefer the aggregate
       * only when the stream emitted nothing. */
      if (!acc) acc = providerText(raw);
    }
    var res2 = normalizeProviderResult(spec, normalized, raw, outcome.durationMs, true, chunks);
    res2.data.text = acc || res2.text || '';
    res2.text = res2.data.text;
    if (toolCalls) res2.data.toolCalls = toolCalls;
    return res2;
  };

  /* ══ module: store ═══════════════════════════════════════════════════
   * Category 3 — storage & vector adapters, plus the dual-write mirror.
   * The local QTSQ write stays synchronous; the external upsert runs on a
   * per-store serialised background queue that never blocks and never
   * throws into the caller. Failures are counted and logged every 10. */

  P.defineStorage = function defineStorage(spec) {
    if (!spec || typeof spec !== 'object') throw makeError('E_SCHEMA', 'defineStorage requires spec');
    var name = String(spec.name || '').trim();
    if (!name) throw makeError('E_SCHEMA', 'defineStorage requires name');
    spec.name = name;
    spec.category = 'store';
    spec.kind = spec.kind || 'kv';
    P.STORES[name] = spec;
    return spec;
  };

  P.defineVectorStore = function defineVectorStore(spec) {
    if (!spec || typeof spec !== 'object') throw makeError('E_SCHEMA', 'defineVectorStore requires spec');
    var name = String(spec.name || '').trim();
    if (!name) throw makeError('E_SCHEMA', 'defineVectorStore requires name');
    if (typeof spec.upsert !== 'function' && typeof spec.query !== 'function') {
      throw makeError('E_SCHEMA', 'defineVectorStore requires upsert() and/or query() for ' + name);
    }
    spec.name = name;
    spec.category = 'store';
    spec.kind = spec.kind || 'vector';
    P.STORES[name] = spec;
    return spec;
  };

  var MIRRORS = {};

  function queueRecord(key, kind, apply, opts) {
    if (MIRRORS[key]) return MIRRORS[key];
    MIRRORS[key] = {
      key: key,
      kind: kind,
      apply: apply,
      pending: [],
      inFlight: null,
      timer: null,
      enqueued: 0,
      written: 0,
      failed: 0,
      lastError: null,
      batch: (opts && typeof opts.batch === 'number' && opts.batch > 0) ? Math.floor(opts.batch) : 32,
      flushMs: (opts && typeof opts.flushMs === 'number' && opts.flushMs >= 0) ? Math.floor(opts.flushMs) : 250
    };
    return MIRRORS[key];
  }

  function logQueueFailure(rec) {
    if (!rec.failed || rec.failed % 10 !== 0) return;
    try {
      console.error('[mirror:' + rec.key + '] ' + rec.failed + ' background write(s) failed; last error: ' + rec.lastError);
    } catch (eLog) {}
  }

  function armQueue(rec) {
    if (rec.timer || !rec.pending.length) return;
    rec.timer = setTimeout(function () {
      rec.timer = null;
      drainQueue(rec);
    }, rec.flushMs);
    if (rec.timer && typeof rec.timer.unref === 'function') rec.timer.unref();
  }

  /* Drains one queue. The returned Promise NEVER rejects: a failing
   * external store is counted, not propagated. */
  function drainQueue(rec) {
    if (rec.inFlight) return rec.inFlight;
    if (!rec.pending.length) return Promise.resolve(0);
    rec.inFlight = (async function () {
      var done = 0;
      try {
        while (rec.pending.length) {
          var batch = rec.pending.splice(0, rec.batch);
          for (var i = 0; i < batch.length; i++) {
            try {
              await rec.apply(batch[i]);
              rec.written++;
              done++;
            } catch (e) {
              rec.failed++;
              rec.lastError = String((e && e.message) || e);
              logQueueFailure(rec);
            }
          }
        }
      } finally {
        rec.inFlight = null;
        if (rec.timer) { clearTimeout(rec.timer); rec.timer = null; }
        if (rec.pending.length) armQueue(rec);
      }
      return done;
    })();
    handled(rec.inFlight);
    return rec.inFlight;
  }

  function enqueueQueue(rec, item) {
    rec.pending.push(item);
    rec.enqueued++;
    if (rec.pending.length >= rec.batch) drainQueue(rec);
    else armQueue(rec);
  }

  function kickQueue(rec) {
    if (rec.timer) { clearTimeout(rec.timer); rec.timer = null; }
    return drainQueue(rec);
  }

  function queueStats(rec) {
    return {
      queued: rec.pending.length,
      written: rec.written,
      failed: rec.failed,
      lastError: rec.lastError,
      enqueued: rec.enqueued,
      batch: rec.batch,
      flushMs: rec.flushMs
    };
  }

  P.mirrorStats = function mirrorStats() {
    var out = {};
    var keys = Object.keys(MIRRORS);
    for (var i = 0; i < keys.length; i++) {
      var rec = MIRRORS[keys[i]];
      if (rec.kind !== 'vector') continue;
      out[rec.key] = queueStats(rec);
    }
    return out;
  };

  P.flushMirror = function flushMirror() {
    var keys = Object.keys(MIRRORS);
    var jobs = [];
    for (var i = 0; i < keys.length; i++) {
      var rec = MIRRORS[keys[i]];
      jobs.push(kickQueue(rec));
    }
    return Promise.all(jobs).then(function () { return P.mirrorStats(); });
  };

  P.attachVectorStore = function attachVectorStore(brain, storeName, opts) {
    if (!brain || typeof brain.remember !== 'function' || typeof brain.recall !== 'function') {
      throw makeError('E_CAPABILITY', 'attachVectorStore needs an open QTSQ brain (sofuu.memory.open)');
    }
    opts = opts || {};
    var name = String(storeName || '').trim();
    if (!name) throw makeError('E_SCHEMA', 'attachVectorStore requires a store name');
    var store = P.STORES[name];
    if (!store) throw makeError('E_CAPABILITY', 'Vector store "' + name + '" is not registered (call defineVectorStore first)');
    if (typeof store.upsert !== 'function') throw makeError('E_SCHEMA', 'Vector store "' + name + '" has no upsert()');

    var rec = queueRecord(name, 'vector', function (item) {
      return store.upsert({ id: item.id, vector: item.vector, text: item.text, metadata: item.metadata });
    }, opts);
    var seq = 0;

    function toVec(v) {
      if (typeof Float32Array !== 'undefined' && v instanceof Float32Array) return v;
      try { return new Float32Array(v || []); } catch (e) { return []; }
    }
    function remember(vector, text, meta) {
      meta = (meta && typeof meta === 'object') ? meta : {};
      var vec = toVec(vector);
      var txt = (text === undefined || text === null) ? '' : String(text);
      /* Fast path: the local write is synchronous and happens first. */
      var local = brain.remember(vec, txt, meta.role || 'user', meta.kvPageId | 0,
                                 meta.source === undefined ? undefined : String(meta.source));
      try {
        seq++;
        var id = meta.id;
        if (id === undefined || id === null || id === '') { id = name + ':' + seq; try { meta.id = id; } catch (eId) {} }
        enqueueQueue(rec, { id: id, vector: vec, text: txt, metadata: meta });
      } catch (eEnq) {
        rec.failed++;
        rec.lastError = String((eEnq && eEnq.message) || eEnq);
        logQueueFailure(rec);
      }
      return local;
    }
    /* recall() is synchronous on a local hit (the common path). With
     * opts.remoteFirst and ZERO local hits it returns a Promise for the
     * external store instead — await recall(...) handles both shapes, and
     * recallAsync(...) always returns a Promise. */
    function recall(vector, limit) {
      var k = (typeof limit === 'number' && limit > 0) ? Math.floor(limit) : 5;
      var hits = brain.recall(toVec(vector), k);
      var empty = !hits || (Array.isArray(hits) && hits.length === 0);
      if (!empty || !opts.remoteFirst || typeof store.query !== 'function') return hits;
      return Promise.resolve()
        .then(function () { return store.query({ vector: toVec(vector), limit: k }); })
        .then(function (rows) { return Array.isArray(rows) ? rows : []; })
        .catch(function (eR) {
          rec.failed++;
          rec.lastError = String((eR && eR.message) || eR);
          logQueueFailure(rec);
          return [];
        });
    }
    function recallAsync(vector, limit) {
      return Promise.resolve(recall(vector, limit));
    }
    function count() {
      if (typeof brain.count !== 'function') return 0;
      return brain.count();
    }
    function flush() {
      var local = true;
      try { if (typeof brain.flush === 'function') local = brain.flush(); } catch (e) { local = false; }
      return kickQueue(rec).then(function () { return local; });
    }
    function stats() { return queueStats(rec); }

    return {
      name: name,
      brain: brain,
      store: name,
      remember: remember,
      recall: recall,
      recallAsync: recallAsync,
      count: count,
      flush: flush,
      stats: stats
    };
  };

  P.attachStorage = function attachStorage(storeName, opts) {
    opts = opts || {};
    var name = String(storeName || '').trim();
    if (!name) throw makeError('E_SCHEMA', 'attachStorage requires a store name');
    var store = P.STORES[name];
    if (!store) throw makeError('E_CAPABILITY', 'Storage store "' + name + '" is not registered (call defineStorage first)');

    var rec = queueRecord('storage:' + name, 'storage', function (op) {
      if (op.op === 'set') {
        if (typeof store.set !== 'function') throw makeError('E_CAPABILITY', 'Storage store "' + name + '" has no set()');
        return store.set(op.key, op.value);
      }
      if (op.op === 'delete') {
        if (typeof store.delete === 'function') return store.delete(op.key);
        if (typeof store.del === 'function') return store.del(op.key);
        return true;
      }
      return true;
    }, opts);

    var cache = {};

    async function get(key) {
      var k = String(key);
      if (Object.prototype.hasOwnProperty.call(cache, k)) return cache[k];
      if (typeof store.get !== 'function') return undefined;
      try {
        var v = await store.get(k);
        if (v !== undefined) cache[k] = v;
        return v;
      } catch (e) {
        rec.failed++;
        rec.lastError = String((e && e.message) || e);
        logQueueFailure(rec);
        throw e;
      }
    }
    function set(key, value) {
      var k = String(key);
      cache[k] = value;
      enqueueQueue(rec, { op: 'set', key: k, value: value });
      return true;
    }
    function del(key) {
      var k = String(key);
      delete cache[k];
      enqueueQueue(rec, { op: 'delete', key: k });
      return true;
    }
    async function keys() {
      var out = Object.keys(cache);
      if (typeof store.keys === 'function') {
        try {
          var ext = await store.keys();
          if (Array.isArray(ext)) {
            for (var i = 0; i < ext.length; i++) {
              var s = String(ext[i]);
              if (out.indexOf(s) < 0) out.push(s);
            }
          }
        } catch (e) {
          rec.failed++;
          rec.lastError = String((e && e.message) || e);
          logQueueFailure(rec);
        }
      }
      return out;
    }

    return {
      name: name,
      get: get,
      set: set,
      delete: del,
      del: del,
      keys: keys,
      flush: function () { return kickQueue(rec).then(function () { return queueStats(rec); }); },
      stats: function () { return queueStats(rec); }
    };
  };

/* ══ module: service ═════════════════════════════════════════════════
   * Category 4 — lifecycle & middleware. Dispatch contract:
   *   emitTurnStart(evt)  -> RESOLVES to evt (never rejects); a service may
   *                          rewrite evt.prompt in place or return a new
   *                          object, which is merged onto evt.
   *   emitToolCall(evt)   -> resolves to evt; a service may mutate evt.args
   *                          in place; a service THROW rejects with E_DENIED
   *                          ("Blocked by service \"name\": ...") because
   *                          that throw is the security interceptor.
   *   emitToolResult(evt) -> never rejects (observability only).
   * Every other hook failure is swallowed: a broken service must never
   * break a turn. */

  var SERVICE_ORDER = [];

  function serviceLog(service, hook, e) {
    try {
      console.error('[service:' + service.name + '] ' + hook + ' failed: ' + String((e && e.message) || e));
    } catch (eLog) {}
  }

  P.defineService = function defineService(spec) {
    if (!spec || typeof spec !== 'object') throw makeError('E_SCHEMA', 'defineService requires spec');
    var name = String(spec.name || '').trim();
    if (!name) throw makeError('E_SCHEMA', 'defineService requires name');

    var service = {
      name: name,
      description: String(spec.description || ''),
      category: 'service',
      spec: spec,
      started: false,

      start: async function (ctx) {
        if (typeof spec.onStart === 'function') {
          await spec.onStart.call(spec, ctx || { env: envOf(), log: console.log });
        }
        this.started = true;
      },
      stop: async function (ctx) {
        if (typeof spec.onStop === 'function') {
          await spec.onStop.call(spec, ctx || {});
        }
        this.started = false;
      },
      /* Direct hook calls keep the 0.2.0 failure policy: swallow + log. */
      turnStart: async function (evt) {
        if (typeof spec.onTurnStart === 'function') {
          try { await spec.onTurnStart.call(spec, evt || {}); } catch (e) { serviceLog(service, 'onTurnStart', e); }
        }
      },
      turnEnd: async function (evt) {
        if (typeof spec.onTurnEnd === 'function') {
          try { await spec.onTurnEnd.call(spec, evt || {}); } catch (e) { serviceLog(service, 'onTurnEnd', e); }
        }
      },
      /* Deliberately NOT swallowed: this is the block signal. */
      toolCall: async function (evt) {
        if (typeof spec.onToolCall !== 'function') return evt;
        return await spec.onToolCall.call(spec, evt || {});
      },
      toolResult: async function (evt) {
        if (typeof spec.onToolResult === 'function') {
          try { await spec.onToolResult.call(spec, evt || {}); } catch (e) { serviceLog(service, 'onToolResult', e); }
        }
      }
    };

    if (!P.SERVICES[name] && SERVICE_ORDER.indexOf(name) < 0) SERVICE_ORDER.push(name);
    P.SERVICES[name] = service;
    return service;
  };

  function mergeServiceResult(evt, res, skipArgs) {
    if (!res || typeof res !== 'object' || res === evt) return;
    if (!skipArgs && res.args && typeof res.args === 'object') evt.args = res.args;
    for (var k in res) {
      if (!Object.prototype.hasOwnProperty.call(res, k)) continue;
      if (!skipArgs && k === 'args') continue;
      if (res[k] === undefined) continue;
      evt[k] = res[k];
    }
  }

  P.emitTurnStart = async function emitTurnStart(evt) {
    evt = (evt && typeof evt === 'object') ? evt : {};
    var names = SERVICE_ORDER.slice();
    for (var i = 0; i < names.length; i++) {
      var s = P.SERVICES[names[i]];
      if (!s || !s.spec || typeof s.spec.onTurnStart !== 'function') continue;
      try {
        var res = await s.spec.onTurnStart.call(s.spec, evt);
        mergeServiceResult(evt, res, true);
      } catch (e) {
        serviceLog(s, 'onTurnStart', e);
      }
    }
    return evt;
  };

  P.emitTurnEnd = async function emitTurnEnd(evt) {
    evt = (evt && typeof evt === 'object') ? evt : {};
    var names = SERVICE_ORDER.slice();
    for (var i = 0; i < names.length; i++) {
      var s = P.SERVICES[names[i]];
      if (!s || !s.spec || typeof s.spec.onTurnEnd !== 'function') continue;
      try { await s.spec.onTurnEnd.call(s.spec, evt); } catch (e) { serviceLog(s, 'onTurnEnd', e); }
    }
    return evt;
  };

  P.emitToolCall = async function emitToolCall(evt) {
    evt = (evt && typeof evt === 'object') ? evt : {};
    if (!evt.args || typeof evt.args !== 'object') evt.args = {};
    var names = SERVICE_ORDER.slice();
    for (var i = 0; i < names.length; i++) {
      var s = P.SERVICES[names[i]];
      if (!s || !s.spec || typeof s.spec.onToolCall !== 'function') continue;
      var res;
      try {
        res = await s.spec.onToolCall.call(s.spec, evt);
      } catch (e) {
        /* The ONLY intentional block path: typed, and it names the service
         * so a blocked tool_result is diagnosable in the run log. */
        throw makeError(
          'E_DENIED',
          'Blocked by service "' + s.name + '": ' + String((e && e.message) || e),
          { service: s.name, tool: evt.tool, turnId: evt.turnId }
        );
      }
      mergeServiceResult(evt, res, false);
    }
    return evt;
  };

  P.emitToolResult = async function emitToolResult(evt) {
    evt = (evt && typeof evt === 'object') ? evt : {};
    var names = SERVICE_ORDER.slice();
    for (var i = 0; i < names.length; i++) {
      var s = P.SERVICES[names[i]];
      if (!s || !s.spec || typeof s.spec.onToolResult !== 'function') continue;
      try { await s.spec.onToolResult.call(s.spec, evt); } catch (e) { serviceLog(s, 'onToolResult', e); }
    }
    return evt;
  };

  P.startServices = async function startServices(ctx) {
    var started = [], errors = [];
    for (var i = 0; i < SERVICE_ORDER.length; i++) {
      var s = P.SERVICES[SERVICE_ORDER[i]];
      if (!s || s.started) continue;
      try {
        await s.start(ctx);
        started.push(s.name);
      } catch (e) {
        errors.push({ name: s.name, message: String((e && e.message) || e) });
        serviceLog(s, 'onStart', e);
      }
    }
    return { started: started, errors: errors };
  };

  P.stopServices = async function stopServices(ctx) {
    var stopped = [], errors = [];
    for (var i = SERVICE_ORDER.length - 1; i >= 0; i--) {
      var s = P.SERVICES[SERVICE_ORDER[i]];
      if (!s || !s.started) continue;
      try {
        await s.stop(ctx);
        stopped.push(s.name);
      } catch (e) {
        errors.push({ name: s.name, message: String((e && e.message) || e) });
        serviceLog(s, 'onStop', e);
      }
    }
    return { stopped: stopped, errors: errors };
  };

  /* ══ module: openapi ═════════════════════════════════════════════════
   * Category 5 — protocol auto-bridging. Contains the dependency-free YAML
   * subset parser (P._yaml), the local-$ref resolver, the OpenAPI 3.0/3.1 +
   * Swagger 2.0 detectors and the MCP ingestion wrapper. */

  /* ── YAML subset parser ───────────────────────────────────────────── *
   * Supports: indentation-nested mappings and sequences ('- '), inline
   * flow collections [a, b] / {a: b}, quoted and unquoted scalars, '#'
   * comments, booleans/numbers/null, block scalars '|' and '>'.
   * Rejects tab indentation and YAML anchors/aliases/tags with a clear
   * E_SCHEMA (silently mis-parsing an OpenAPI document is worse). */

  function yamlErr(msg, lineNo) {
    return makeError('E_SCHEMA', 'YAML: ' + msg + (lineNo ? ' (line ' + lineNo + ')' : ''));
  }

  function stripInlineComment(s) {
    var q = '', i;
    for (i = 0; i < s.length; i++) {
      var c = s.charAt(i);
      if (q) {
        if (q === "'" && c === "'") {
          if (s.charAt(i + 1) === "'") { i++; continue; }
          q = '';
        } else if (q === '"' && c === '\\') { i++; }
        else if (q === '"' && c === '"') { q = ''; }
        continue;
      }
      if (c === '"' || c === "'") { q = c; continue; }
      if (c === '#' && (i === 0 || /\s/.test(s.charAt(i - 1)))) return s.slice(0, i);
    }
    return s;
  }

  function yamlParseScalar(str) {
    var s = String(str).trim();
    if (s === '') return null;
    var c0 = s.charAt(0);
    if (c0 === '"' && s.charAt(s.length - 1) === '"' && s.length > 1) {
      var body = s.slice(1, -1);
      var out = '', i = 0;
      while (i < body.length) {
        var ch = body.charAt(i);
        if (ch === '\\') {
          var nx = body.charAt(i + 1);
          if (nx === 'n') out += '\n';
          else if (nx === 't') out += '\t';
          else if (nx === 'r') out += '\r';
          else if (nx === '"') out += '"';
          else if (nx === '\\') out += '\\';
          else if (nx === '/') out += '/';
          else if (nx === 'u' && /^[0-9a-fA-F]{4}$/.test(body.substr(i + 2, 4))) {
            out += String.fromCharCode(parseInt(body.substr(i + 2, 4), 16)); i += 4;
          } else out += nx;
          i += 2; continue;
        }
        out += ch; i++;
      }
      return out;
    }
    if (c0 === "'" && s.charAt(s.length - 1) === "'" && s.length > 1) {
      return s.slice(1, -1).replace(/''/g, "'");
    }
    if (s === 'null' || s === 'Null' || s === 'NULL' || s === '~') return null;
    if (s === 'true' || s === 'True' || s === 'TRUE') return true;
    if (s === 'false' || s === 'False' || s === 'FALSE') return false;
    if (/^[-+]?\d+$/.test(s)) return parseInt(s, 10);
    if (/^[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?$/.test(s) && /[.eE]/.test(s)) return parseFloat(s);
    return s;
  }

  function yamlSplitKey(text) {
    var q = '', depth = 0, i;
    for (i = 0; i < text.length; i++) {
      var c = text.charAt(i);
      if (q) {
        if (q === "'" && c === "'") { if (text.charAt(i + 1) === "'") i++; else q = ''; }
        else if (q === '"' && c === '\\') i++;
        else if (q === '"' && c === '"') q = '';
        continue;
      }
      if (c === '"' || c === "'") { q = c; continue; }
      if (c === '[' || c === '{') depth++;
      else if (c === ']' || c === '}') depth--;
      else if (c === ':' && depth === 0) {
        var nxt = text.charAt(i + 1);
        if (nxt === '' || nxt === ' ' || nxt === '\t') {
          var key = text.slice(0, i).trim();
          if (!key) return null;
          if ((key.charAt(0) === '"' && key.charAt(key.length - 1) === '"') ||
              (key.charAt(0) === "'" && key.charAt(key.length - 1) === "'")) {
            key = String(yamlParseScalar(key));
          }
          return { key: key, rest: text.slice(i + 1).replace(/^[ \t]+/, '') };
        }
      }
    }
    return null;
  }

  function yamlParseFlow(text, lineNo) {
    var i = 0;
    function ws() { while (i < text.length && /[ \t]/.test(text.charAt(i))) i++; }
    function fail(m) { throw yamlErr(m + ' in flow collection', lineNo); }
    function token() {
      ws();
      var c = text.charAt(i);
      if (c === '[' || c === '{') return value();
      if (c === '"' || c === "'") {
        var q = c, start = i; i++;
        while (i < text.length) {
          var ch = text.charAt(i);
          if (q === '"' && ch === '\\') { i += 2; continue; }
          if (ch === q) { if (q === "'" && text.charAt(i + 1) === "'") { i += 2; continue; } i++; break; }
          i++;
        }
        return yamlParseScalar(text.slice(start, i));
      }
      var st = i;
      while (i < text.length && ',]}'.indexOf(text.charAt(i)) < 0) i++;
      return yamlParseScalar(text.slice(st, i));
    }
    function value() {
      ws();
      var c = text.charAt(i);
      if (c === '[') {
        i++; var arr = []; ws();
        if (text.charAt(i) === ']') { i++; return arr; }
        while (true) {
          arr.push(token()); ws();
          if (text.charAt(i) === ',') { i++; ws(); if (text.charAt(i) === ']') { i++; return arr; } continue; }
          if (text.charAt(i) === ']') { i++; return arr; }
          fail('expected , or ]');
        }
      }
      if (c === '{') {
        i++; var obj = {}; ws();
        if (text.charAt(i) === '}') { i++; return obj; }
        while (true) {
          ws();
          if (text.charAt(i) === '}') { i++; return obj; }
          var kstart = i, q = text.charAt(i);
          var key;
          if (q === '"' || q === "'") {
            i++;
            while (i < text.length) { var ch = text.charAt(i); if (q === '"' && ch === '\\') { i += 2; continue; } if (ch === q) { i++; break; } i++; }
            key = String(yamlParseScalar(text.slice(kstart, i)));
          } else {
            while (i < text.length && ':}'.indexOf(text.charAt(i)) < 0) i++;
            key = String(yamlParseScalar(text.slice(kstart, i)));
          }
          ws();
          if (text.charAt(i) !== ':') fail('expected : after key "' + key + '"');
          i++;
          obj[key] = token(); ws();
          if (text.charAt(i) === ',') { i++; continue; }
          if (text.charAt(i) === '}') { i++; return obj; }
          fail('expected , or }');
        }
      }
      return token();
    }
    var out = value();
    ws();
    if (i < text.length) throw yamlErr('trailing text after flow collection: "' + text.slice(i) + '"', lineNo);
    return out;
  }

  function yamlParse(src) {
    if (typeof src !== 'string') throw yamlErr('source must be a string');
    var raw = src.replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').split('\n');
    var li;
    for (li = 0; li < raw.length; li++) {
      var lead = /^[ \t]*/.exec(raw[li])[0];
      if (lead.indexOf('\t') >= 0) throw yamlErr('indentation must use spaces (tab at line ' + (li + 1) + ')', li + 1);
    }
    var pos = 0;
    function indentOf(l) { var n = 0; while (l.charAt(n) === ' ') n++; return n; }
    function blank(l) { return /^[ \t]*$/.test(l); }
    function commentLine(l) { return /^ *#/.test(l); }
    function skipNoise() {
      while (pos < raw.length) {
        var l = raw[pos];
        if (blank(l) || commentLine(l)) { pos++; continue; }
        var t = l.trim();
        if (t === '---' || t === '...') { pos++; continue; }
        break;
      }
    }
    function peek() {
      skipNoise();
      if (pos >= raw.length) return null;
      var l = raw[pos], ind = indentOf(l);
      return { indent: ind, text: l.slice(ind), line: pos + 1 };
    }

    function parseBlockScalar(header, parentIndent, lineNo) {
      var style = header.charAt(0);
      var meta = /^[|>]([+-]?)(\d*)([+-]?)/.exec(header) || [];
      var chomp = (meta[1] || meta[3] || '');
      var explicit = meta[2] ? parseInt(meta[2], 10) : 0;
      var blockIndent = explicit ? parentIndent + explicit : -1;
      var lines = [];
      while (pos < raw.length) {
        var l = raw[pos];
        if (blank(l)) { lines.push(''); pos++; continue; }
        var ind = indentOf(l);
        if (ind <= parentIndent) break;
        if (blockIndent < 0) blockIndent = ind;
        if (ind < blockIndent) break;
        lines.push(l.slice(blockIndent));
        pos++;
      }
      var trail = 0;
      while (lines.length && lines[lines.length - 1] === '') { lines.pop(); trail++; }
      var text;
      if (style === '|') {
        text = lines.join('\n');
      } else {
        var buf = [], prevBlank = true;
        for (var k = 0; k < lines.length; k++) {
          var cur = lines[k];
          if (cur === '') { buf.push('\n'); prevBlank = true; continue; }
          if (cur.charAt(0) === ' ' || cur.charAt(0) === '\t') {
            if (!prevBlank) buf.push('\n');
            buf.push(cur); prevBlank = false; continue;
          }
          if (!prevBlank) buf.push(' ');
          buf.push(cur); prevBlank = false;
        }
        text = buf.join('');
      }
      if (chomp === '-') return text;
      if (chomp === '+') return text + '\n' + new Array(trail + 1).join('\n');
      return text === '' ? '' : text + '\n';
    }

    function parseValue(rest, parentIndent, lineNo) {
      rest = String(rest === undefined || rest === null ? '' : rest).replace(/^[ \t]+/, '');
      if (rest === '') {
        var nt = peek();
        if (!nt || nt.indent <= parentIndent) return null;
        return parseNode(nt.indent);
      }
      var c0 = rest.charAt(0);
      if (c0 === '|' || c0 === '>') return parseBlockScalar(rest, parentIndent, lineNo);
      if (c0 === '[' || c0 === '{') return yamlParseFlow(stripInlineComment(rest).trim(), lineNo);
      if ((c0 === '&' || c0 === '*') && /^[&*][A-Za-z_$]/.test(rest)) {
        throw yamlErr('anchors/aliases are not supported by the built-in YAML parser', lineNo);
      }
      if (c0 === '!' && /^!!?/.test(rest)) {
        throw yamlErr('YAML tags are not supported by the built-in YAML parser', lineNo);
      }
      return yamlParseScalar(stripInlineComment(rest));
    }

    function parseMapping(indent) {
      var obj = {};
      while (true) {
        var t = peek();
        if (!t) break;
        if (t.indent < indent) break;
        if (t.indent > indent) throw yamlErr('unexpected indentation', t.line);
        var kv = yamlSplitKey(t.text);
        if (!kv) break;
        pos++;
        obj[kv.key] = parseValue(kv.rest, indent, t.line);
      }
      return obj;
    }

    function parseSeqItem(rest, dashIndent, lineNo) {
      if (rest === '' || /^[ \t]*$/.test(rest)) {
        var nt = peek();
        return (nt && nt.indent > dashIndent) ? parseNode(nt.indent) : null;
      }
      var extraSpaces = rest.length - rest.replace(/^[ \t]+/, '').length;
      if (rest === '-' || rest.slice(0, 2) === '- ') {
        var nestedIndent = dashIndent + 2 + extraSpaces;
        var innerFirst = parseSeqItem(rest === '-' ? '' : rest.slice(2), nestedIndent, lineNo);
        return [innerFirst].concat(parseSequence(nestedIndent));
      }
      var itemIndent = dashIndent + 2 + extraSpaces;
      var kv = yamlSplitKey(rest);
      if (!kv) return parseValue(rest, dashIndent, lineNo);
      var item = {};
      item[kv.key] = parseValue(kv.rest, itemIndent, lineNo);
      var cont = peek();
      if (cont && cont.indent > dashIndent) {
        if (cont.indent < itemIndent) throw yamlErr('bad sequence item indentation', cont.line);
        var more = parseMapping(cont.indent);
        for (var ek in more) if (Object.prototype.hasOwnProperty.call(more, ek)) item[ek] = more[ek];
      }
      return item;
    }

    function parseSequence(indent) {
      var arr = [];
      while (true) {
        var t = peek();
        if (!t) break;
        if (t.indent < indent) break;
        if (t.indent !== indent) throw yamlErr('unexpected indentation in sequence', t.line);
        if (!(t.text === '-' || t.text.slice(0, 2) === '- ')) break;
        var rest = t.text === '-' ? '' : t.text.slice(2);
        pos++;
        arr.push(parseSeqItem(rest, indent, t.line));
      }
      return arr;
    }

    function parseNode(indent) {
      var t = peek();
      if (!t) return null;
      if (t.indent < indent) return null;
      if (t.text === '-' || t.text.slice(0, 2) === '- ') return parseSequence(t.indent);
      return parseMapping(t.indent);
    }

    var root = parseNode(0);
    skipNoise();
    if (pos < raw.length) throw yamlErr('unparsed content: "' + raw[pos].trim().slice(0, 40) + '"', pos + 1);
    return root;
  }

  P._yaml = { parse: yamlParse };

  /* ══ module: typescript ══════════════════════════════════════════════════
   * Textual TypeScript -> JavaScript stripper. It is a scanner, not a
   * regex pile: it walks the source once, is aware of strings, template
   * literals and comments, and decides where a ':' introduces a TYPE by
   * looking at the enclosing construct. The 0.2.0 version stripped
   * 'name: value' pairs inside object literals (z.object({ units: x }))
   * and left optional parameters (a?: T) untouched — both silently broke
   * plugins that had been written correctly.
   *
   * It is deliberately conservative: when a construct cannot be classified
   * it is left ALONE (a .ts plugin is not a compiler; a leftover annotation
   * fails loudly at parse time, a wrongly deleted value fails silently).
   */
  var TS_KEYWORDS = {
    string: 1, number: 1, boolean: 1, any: 1, void: 1, unknown: 1, never: 1,
    object: 1, null: 1, undefined: 1, bigint: 1, symbol: 1, this: 1, true: 1, false: 1
  };

  /* Skips a string / template / comment starting at i; returns the index
   * just past it (or i when the char is not one of those). */
  function tsSkipLiteral(s, i) {
    var c = s.charAt(i);
    if (c === '/' && s.charAt(i + 1) === '/') {
      while (i < s.length && s.charAt(i) !== '\n') i++;
      return i;
    }
    if (c === '/' && s.charAt(i + 1) === '*') {
      i += 2;
      while (i < s.length && !(s.charAt(i) === '*' && s.charAt(i + 1) === '/')) i++;
      return Math.min(s.length, i + 2);
    }
    if (c === '"' || c === "'" || c === BT) {
      var q = c;
      i++;
      while (i < s.length) {
        var ch = s.charAt(i);
        if (ch === '\\') { i += 2; continue; }
        i++;
        if (ch === q) break;
      }
      return i;
    }
    return -1;
  }

  /* Regex-literal skipper. A regex may contain quotes (`/[\"']/`), so a
   * scanner that only knows strings and comments desynchronises on it and
   * then treats real code as string content — that is exactly how
   * `s.replace(/..href=\"([^\"]*)\"/gi, function (all: string) {})`
   * kept its annotations.
   * Only called when the '/' cannot be division (prev significant char is
   * an operator or an opening bracket), so `a / b` is never eaten. */
  function tsSkipRegex(s, i) {
    var k = i + 1;
    var inClass = false;
    while (k < s.length) {
      var c = s.charAt(k);
      if (c === '\\') { k += 2; continue; }
      if (c === '\n') return -1;
      if (inClass) { if (c === ']') inClass = false; k++; continue; }
      if (c === '[') { inClass = true; k++; continue; }
      if (c === '/') { k++; while (k < s.length && /[a-z]/i.test(s.charAt(k))) k++; return k; }
      k++;
    }
    return -1;
  }
  function tsRegexAllowed(prev) {
    if (!prev) return true;
    return '(,=:[!&|?{};+-*%~^<>'.indexOf(prev) >= 0;
  }

  /* Matching bracket index for s[i] === open, or -1. Literal-aware. */
  function tsMatch(s, i) {
    var open = s.charAt(i);
    var close = open === '(' ? ')' : (open === '{' ? '}' : ']');
    var depth = 0;
    while (i < s.length) {
      var skip = tsSkipLiteral(s, i);
      if (skip >= 0) { i = skip; continue; }
      var c = s.charAt(i);
      if (c === open) depth++;
      else if (c === close) { depth--; if (depth === 0) return i; }
      i++;
    }
    return -1;
  }

  function tsFail(msg) { return makeError('E_SCHEMA', 'TypeScript strip: ' + msg); }

  /* Parses one type expression at i. Returns the end index, or -1 when the
   * text there is not type-shaped (a value call, a dotted expression...). */
  function tsTypeEnd(s, i, stopAt) {
    function ws(k) { while (k < s.length && /\s/.test(s.charAt(k))) k++; return k; }
    var k = ws(i);
    var matched = false;
    for (;;) {
      k = ws(k);
      var c = s.charAt(k);
      /* A '{' AFTER a type atom is the function BODY, not part of the
       * type: `function f(a: string): string[] {` ended by swallowing the
       * whole body here, tsTypeEnd ran off the end, and every top-level
       * function kept its annotations. Only a leading '{' is an object
       * type literal. */
      if (c === '{') {
        if (matched) break;
        var e = tsMatch(s, k); if (e < 0) return -1; k = e + 1; matched = true;
      }
      else if (c === '[') { var e2 = tsMatch(s, k); if (e2 < 0) return -1; k = e2 + 1; matched = true; }
      else if (c === '(') { var e3 = tsMatch(s, k); if (e3 < 0) return -1; k = e3 + 1; matched = true; }
      else if (c === '"' || c === "'" || c === BT) { var e4 = tsSkipLiteral(s, k); if (e4 < 0) return -1; k = e4; matched = true; }
      else if (/[A-Za-z_$]/.test(c)) {
        var st = k;
        while (k < s.length && /[A-Za-z0-9_$.]/.test(s.charAt(k))) k++;
        var word = s.slice(st, k);
        /* a call is NOT a type: foo() / z.string() */
        var after = ws(k);
        if (s.charAt(after) === '(') return -1;
        if (word.indexOf('.') >= 0 && !/[A-Z]/.test(word)) return -1;
        if (s.charAt(after) === '<') {
          var td = 0;
          var j = after;
          while (j < s.length) {
            var cc = s.charAt(j);
            if (cc === '<') td++;
            else if (cc === '>') { td--; if (td === 0) { j++; break; } }
            else if (cc === ';' || cc === '{' || cc === '}') return -1;
            j++;
          }
          k = j;
        }
        matched = true;
      }
      else if (c === '|' || c === '&') { k++; continue; }
      else if (c === '<') { var td2 = 0, j2 = k; while (j2 < s.length) { var c2 = s.charAt(j2); if (c2 === '<') td2++; else if (c2 === '>') { td2--; if (td2 === 0) { j2++; break; } } j2++; } k = j2; matched = true; }
      else if (c === '?' || c === ':' ) { k++; continue; }  /* conditional / readonly-ish noise */
      else break;
      /* array suffix */
      var a2 = ws(k);
      if (s.charAt(a2) === '[' && s.charAt(ws(a2 + 1)) === ']') { k = ws(a2 + 1) + 1; continue; }
      if (s.charAt(a2) === '|' || s.charAt(a2) === '&') { k = a2 + 1; continue; }
      break;
    }
    if (!matched) return -1;
    var end = ws(k);
    if (end > s.length) return -1;
    var stop = s.charAt(end);
    if (stopAt && stopAt.indexOf(stop) >= 0) return end;
    /* end of file or a newline from a stripped block is also a terminator */
    if (end >= s.length) return end;
    return -1;
  }

  /* Remove generic type ARGUMENTS at a call site — the plan's own signature is
   * defineTool<TSchema, TOutput>({...}), and 'new Map<string, number>()' is
   * ordinary plugin code. A regex cannot do it (the arguments nest:
   * z.infer<any>), so scan for a balanced <...> that is immediately followed
   * by '(' and drop only that span. Comparisons are left alone. */
  function stripGenericCallArgs(src) {
    var out = '';
    var i = 0;
    while (i < src.length) {
      var ch = src.charAt(i);
      if (ch === '<' && i > 0 && /[\w$]/.test(src.charAt(i - 1))) {
        var depth = 0;
        var j = i;
        var closed = false;
        for (; j < src.length; j++) {
          var c = src.charAt(j);
          if (c === '<') { depth++; }
          else if (c === '>') {
            /* '=>' is an arrow, not the end of a type argument list. */
            if (src.charAt(j - 1) === '=') continue;
            depth--;
            if (depth === 0) { closed = true; break; }
          }
          else if (c === ';' || c === '\n') break;
        }
        if (closed && /^[ \t]*\(/.test(src.slice(j + 1))) { i = j + 1; continue; }
      }
      out += ch;
      i++;
    }
    return out;
  }

  function stripTs(src) {
    if (!src || typeof src !== 'string') return '';
    var s = String(src);

    /* 1. import / export shape rewrites (line-oriented, quote-aware). */
    s = s.replace(/^[ \t]*import\s+type\s+[\s\S]*?from\s*['"][^'"]*['"]\s*;?[ \t]*$/gm, '');
    s = s.replace(/^[ \t]*import\s+[\s\S]*?from\s*['"][^'"]*['"]\s*;?[ \t]*$/gm, '');
    s = s.replace(/^[ \t]*import\s*['"][^'"]*['"]\s*;?[ \t]*$/gm, '');
    s = s.replace(/^[ \t]*export\s+type\s+[\s\S]*?;[ \t]*$/gm, '');
    s = s.replace(/^[ \t]*declare\s+[^\n]*$/gm, '');
    /* 1b. export shapes: the default export is the plugin discovery
     *     sentinel, a named export is just its declaration. */
    s = s.replace(/\bexport\s+default\s+/g, 'globalThis.__sofuu_last_export = ');
    s = s.replace(/\bexport\s+(const|let|var|function|class|async\s+function)\s+/g, '$1 ');
    /* 1c. generic type parameters on function declarations. */
    s = s.replace(/\b(function\s+[A-Za-z_$][\w$]*)\s*<[^<>()\n]*>(?=\s*\()/g, '$1');
    /* generic ARROW functions: const f = <T>(x: T) => x */
    s = s.replace(/([=(,:]\s*)<[^<>\n()]+>(?=\s*\()/g, '$1');
    /* 1d. generic type ARGUMENTS at a call site: the plan's own signature is
     *     defineTool<TSchema, TOutput>({...}), and 'new Map<string, number>()'
     *     is ordinary plugin code. Left in place, the runtime sees
     *     'defineTool < TSchema' as a comparison and dies with
     *     "'string' is not defined". Only an identifier immediately followed
     *     by <...> and then ( is rewritten, so comparisons are untouched. */
    s = stripGenericCallArgs(s);

    /* 2. interface { ... } blocks — brace matched, so a '}' inside a
     *    string or a method signature cannot end it early. */
    for (var guard = 0; guard < 200; guard++) {
      var m = /(^|\n)[ \t]*(export\s+)?(interface|namespace|module)\s+[A-Za-z_$][\w$]*[^{\n]*\{/.exec(s);
      if (!m) break;
      var braceAt = m.index + m[0].length - 1;
      var closeAt = tsMatch(s, braceAt);
      if (closeAt < 0) throw tsFail('unbalanced ' + m[3] + ' block for ' + (m[0].slice(0, 40)));
      s = s.slice(0, m.index + m[1].length) + s.slice(closeAt + 1);
    }

    /* 3. type aliases: 'type X = ...;' (the ';' may be far away). */
    for (var g2 = 0; g2 < 200; g2++) {
      var tm = /(^|\n)[ \t]*(export\s+)?type\s+[A-Za-z_$][\w$]*(\s*<[^>\n]*>)?\s*=/.exec(s);
      if (!tm) break;
      var from = tm.index + tm[1].length;
      var j = tm.index + tm[0].length;
      var depth = 0;
      var endAt = -1;
      while (j < s.length) {
        var sk = tsSkipLiteral(s, j);
        if (sk >= 0) { j = sk; continue; }
        var ch = s.charAt(j);
        if (ch === '{' || ch === '(' || ch === '[') depth++;
        else if (ch === '}' || ch === ')' || ch === ']') depth--;
        else if (ch === ';' && depth <= 0) { endAt = j + 1; break; }
        else if (ch === '\n' && depth <= 0) { endAt = j; break; }
        j++;
      }
      if (endAt < 0) endAt = s.length;
      s = s.slice(0, from) + s.slice(endAt);
    }

    /* 4. enum blocks -> frozen object literals. */
    for (var g3 = 0; g3 < 100; g3++) {
      var em = /(^|\n)[ \t]*(export\s+)?(const\s+)?enum\s+([A-Za-z_$][\w$]*)\s*\{/.exec(s);
      if (!em) break;
      var nb = em.index + em[0].length - 1;
      var nc = tsMatch(s, nb);
      if (nc < 0) throw tsFail('unbalanced enum block');
      var body = s.slice(nb + 1, nc);
      var members = body.split(',');
      var obj = {};
      var auto = 0;
      for (var mi = 0; mi < members.length; mi++) {
        var mem = members[mi].trim();
        if (!mem) continue;
        var eq = mem.indexOf('=');
        var mname = (eq >= 0 ? mem.slice(0, eq) : mem).trim();
        if (!/^[A-Za-z_$][\w$]*$/.test(mname)) throw tsFail('enum ' + em[4] + ' member ' + mname + ' is not supported');
        if (eq >= 0) {
          var raw = mem.slice(eq + 1).trim();
          var parsed = null;
          try { parsed = JSON.parse(raw.replace(/'/g, '"')); } catch (eMem) { parsed = null; }
          if (parsed === null) throw tsFail('enum ' + em[4] + ' member ' + mname + ' value is not a literal');
          obj[mname] = parsed;
          if (typeof parsed === 'number') auto = parsed + 1;
        } else { obj[mname] = auto++; }
      }
      var decl = 'var ' + em[4] + ' = Object.freeze(' + JSON.stringify(obj) + ');';
      s = s.slice(0, em.index + em[1].length) + decl + s.slice(nc + 1);
    }

    /* 5. 'as X' / 'satisfies X' assertions. */
    s = s.replace(/\s+satisfies\s+[A-Za-z_$][\w$.]*\s*(<[^;\n]*>)?/g, '');
    s = s.replace(/\s+as\s+const\b/g, '');
    s = s.replace(/\s+as\s+[A-Za-z_$][\w$.]*(<[^;\n]*>)?(\[\])?/g, '');

    /* 6. annotations, by CONSTRUCT (see the header). */
    var out = '';
    var i = 0;
    var prevSig = '';
    while (i < s.length) {
      var c = s.charAt(i);
      if (c === '/' && s.charAt(i + 1) !== '/' && s.charAt(i + 1) !== '*' && tsRegexAllowed(prevSig)) {
        var rEnd = tsSkipRegex(s, i);
        if (rEnd > i) { out += s.slice(i, rEnd); prevSig = s.charAt(rEnd - 1); i = rEnd; continue; }
      }
      var skip = tsSkipLiteral(s, i);
      if (skip >= 0) { out += s.slice(i, skip); prevSig = s.charAt(skip - 1); i = skip; continue; }

      if (c === '(') {
        var close = tsMatch(s, i);
        if (close > i) {
          var after = close + 1;
          while (after < s.length && /\s/.test(s.charAt(after))) after++;
          var isParams = (s.charAt(after) === '{') || (s.charAt(after) === ':' && s.charAt(after + 1) === ':') ||
                         (s.charAt(after) === '=' && s.charAt(after + 1) === '>');
          if (!isParams && s.charAt(after) === ':') {
            /* function f(...): ReturnType { }  /  (...) : T => */
            var rt = tsTypeEnd(s, after + 1, '{=>');
            if (rt > 0) isParams = true;
          }
          if (isParams) {
            prevSig = ')';
            out += '(' + stripParamList(s.slice(i + 1, close)) + ')';
            i = close + 1;
            /* return type */
            var k2 = i;
            while (k2 < s.length && /\s/.test(s.charAt(k2))) k2++;
            if (s.charAt(k2) === ':') {
              var rtEnd = tsTypeEnd(s, k2 + 1, '{=>');
              if (rtEnd > 0) { out += ' '; i = rtEnd; }
            }
            continue;
          }
        }
        out += c; if (!/\s/.test(c)) prevSig = c; i++; continue;
      }
      out += c; if (!/\s/.test(c)) prevSig = c; i++;
    }
    s = out;

    /* 7. variable declarations: const x: T = / let x: T; */
    s = s.replace(/\b(const|let|var)(\s+)([A-Za-z_$][\w$]*)\s*\??\s*:\s*([^=;\n]+?)(?=\s*[=;]|\s*$)/gm, function (all, kw, sp, nm) {
      return kw + sp + nm;
    });

    /* 8. class/interface leftovers: modifiers and member annotations. */
    s = s.replace(/^[ \t]*(public|private|protected|readonly)\s+/gm, '  ');
    s = s.replace(/^([ \t]*)([A-Za-z_$][\w$]*)\s*\??\s*:\s*[A-Za-z_$][\w$<>,.\[\] |&{}();:'"]+;[ \t]*$/gm, '$1');
    /* member annotations that carry an initialiser (class fields). */
    s = s.replace(/^([ \t]+)([A-Za-z_$][\w$]*)\s*\??\s*:\s*[A-Za-z_$][\w$<>,.\[\] |&]+(?=\s*=)/gm, '$1$2 ');
    return s;
  }

  /* Parameter list stripper: ','-split at depth 0, each 'name?: Type[ = default]'. */
  function stripParamList(list) {
    var parts = [];
    var depth = 0;
    var cur = '';
    var i = 0;
    while (i < list.length) {
      var skip = tsSkipLiteral(list, i);
      if (skip >= 0) { cur += list.slice(i, skip); i = skip; continue; }
      var c = list.charAt(i);
      if (c === '(' || c === '{' || c === '[') depth++;
      else if (c === ')' || c === '}' || c === ']') depth--;
      if (c === ',' && depth === 0) { parts.push(cur); cur = ''; i++; continue; }
      cur += c; i++;
    }
    if (cur.trim()) parts.push(cur);
    var outParts = [];
    for (var p = 0; p < parts.length; p++) {
      var seg = parts[p];
      var colon = -1;
      var d2 = 0;
      for (var q = 0; q < seg.length; q++) {
        var cq = seg.charAt(q);
        if (cq === '(' || cq === '{' || cq === '[') d2++;
        else if (cq === ')' || cq === '}' || cq === ']') d2--;
        else if (cq === ':' && d2 === 0) { colon = q; break; }
      }
      if (colon < 0) { outParts.push(seg); continue; }
      var eq = seg.indexOf('=', colon);
      var name = seg.slice(0, colon).replace(/\?\s*$/, '');
      /* parameter properties: constructor(private a: number) */
      name = name.replace(/^\s*(public|private|protected|readonly)\s+/, ' ');
      outParts.push(name + (eq >= 0 ? seg.slice(eq) : ''));
    }
    return outParts.join(',');
  }

  P._stripTs = stripTs;

  /* ══ module: provider — the public registration surface ══════════════════
   * defineProvider() is what a plugin file calls. registerProvider() is the
   * runtime entry used by the first-party adapters driver; the difference is
   * only ergonomics (defaults + a friendlier error), so both land in the
   * same registry with the same shape. */
  P.defineProvider = function defineProvider(spec) {
    if (!spec || typeof spec !== 'object') throw makeError('E_SCHEMA', 'defineProvider requires a spec object');
    var name = String(spec.name || '').trim();
    if (!name) throw makeError('E_SCHEMA', 'defineProvider requires a non-empty name');
    if (typeof spec.complete !== 'function' && typeof spec.stream !== 'function') {
      throw makeError('E_SCHEMA', 'defineProvider requires complete() or stream() for ' + name);
    }
    if (typeof spec.timeoutMs !== 'number' || !(spec.timeoutMs > 0)) spec.timeoutMs = DEFAULT_PROVIDER_TIMEOUT_MS;
    if (typeof spec.circuitBreaker !== 'object' || !spec.circuitBreaker) spec.circuitBreaker = null;
    return P.registerProvider(spec);
  };

  /* ══ module: openapi — public entry point ════════════════════════════════
   * useOpenAPI(spec, opts) turns a REST description into 5-shield tools.
   * The spec may be a JSON string, an object, or OpenAPI YAML; 3.0/3.1 and
   * Swagger 2.0 are both understood, and local $ref pointers are resolved.
   *
   * SECURITY: the generated tools are pinned to the host of the base URL
   * the caller supplied (permissions.net), so an auto-generated tool can
   * reach the API it was written for and nothing else. Without that pin a
   * loopback API (the common local-dev case) would be denied by Shield 2
   * and the feature would look broken. */
  function oaRef(root, node) {
    if (!node || typeof node !== 'object' || typeof node.$ref !== 'string') return node;
    var ref = node.$ref;
    if (ref.charAt(0) !== '#') return node;
    var cur = root;
    var parts = ref.replace(/^#\/?/, '').split('/');
    for (var i = 0; i < parts.length; i++) {
      var key = parts[i].replace(/~1/g, '/').replace(/~0/g, '~');
      if (cur === null || typeof cur !== 'object') return node;
      cur = cur[key];
    }
    return (cur === undefined) ? node : cur;
  }
  function oaSchemaFor(param) {
    if (!param) return { type: 'string' };
    if (param.schema && typeof param.schema === 'object') return param.schema;
    var s = {};
    ['type', 'format', 'enum', 'items', 'default', 'description'].forEach(function (k) {
      if (param[k] !== undefined) s[k] = param[k];
    });
    if (!s.type) s.type = 'string';
    return s;
  }
  function oaSlug(s) {
    return String(s).replace(/\{([^}]+)\}/g, 'by_$1').replace(/[^A-Za-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  }
  P.lastOpenApi = null;
  /* PLAN §2 Cat 5: "Pass a URL or local path to an OpenAPI 3.0/3.1 JSON/YAML
   * file". The spec argument is therefore three things: a document object, an
   * inline document string, or a source reference. A string is treated as a
   * reference only when it does not already look like a document, so inline
   * YAML never triggers a file read. */
  async function loadSpecSource(src) {
    if (/^https?:\/\//i.test(src)) {
      var f = (typeof fetch === 'function' && fetch) ? fetch : ((sofuu && sofuu.fetch) || null);
      if (!f) throw makeError('E_CAPABILITY', 'useOpenAPI: no fetch available to load ' + src);
      var res = await f(src, { headers: { accept: 'application/json, application/yaml, text/yaml, */*' } });
      if (!res || !res.ok) throw makeError('E_TOOL', 'useOpenAPI: fetch ' + src + ' -> HTTP ' + ((res && res.status) || 0));
      return String(await res.text());
    }
    if (sofuu.fs && typeof sofuu.fs.readFile === 'function') {
      var t = sofuu.fs.readFile(src);
      if (t && typeof t.then === 'function') t = await t;
      if (t === undefined || t === null) throw makeError('E_SCHEMA', 'useOpenAPI: cannot read ' + src);
      return String(t);
    }
    throw makeError('E_CAPABILITY', 'useOpenAPI: no filesystem available to read ' + src);
  }
  P.useOpenAPI = async function useOpenAPI(spec, opts) {
    opts = opts || {};
    var doc = spec;
    if (typeof spec === 'string') {
      var txt = String(spec).trim();
      var looksLikeDoc = (txt.charAt(0) === '{' || txt.charAt(0) === '[') ||
                         /(^|\n)\s*(openapi|swagger)\s*:/.test(txt);
      if (!looksLikeDoc) txt = String(await loadSpecSource(txt)).trim();
      if (txt.charAt(0) === '{' || txt.charAt(0) === '[') {
        try { doc = JSON.parse(txt); }
        catch (eJson) { throw makeError('E_SCHEMA', 'useOpenAPI: invalid JSON spec: ' + eJson.message); }
      } else {
        doc = P._yaml.parse(txt);
      }
    }
    if (!doc || typeof doc !== 'object') throw makeError('E_SCHEMA', 'useOpenAPI requires an object or a JSON/YAML string');
    var version = doc.openapi ? String(doc.openapi) : (doc.swagger ? String(doc.swagger) : '');
    if (!version) throw makeError('E_SCHEMA', 'useOpenAPI: the document has neither an "openapi" nor a "swagger" field');
    if (version.indexOf('3.') !== 0 && version.indexOf('2.') !== 0) {
      throw makeError('E_SCHEMA', 'useOpenAPI: unsupported spec version ' + version + ' (need OpenAPI 3.x or Swagger 2.0)');
    }
    var paths = doc.paths || {};
    var baseUrl = String(opts.baseUrl || '');
    if (!baseUrl && Array.isArray(doc.servers) && doc.servers.length) baseUrl = String(doc.servers[0].url || '');
    if (!baseUrl && version.indexOf('2.') === 0 && doc.host) {
      baseUrl = String((doc.schemes && doc.schemes[0]) || 'https') + '://' + String(doc.host) + String(doc.basePath || '');
    }
    if (!baseUrl) throw makeError('E_SCHEMA', 'useOpenAPI: no server URL (pass opts.baseUrl)');
    baseUrl = baseUrl.replace(/\/+$/, '');
    var ns = opts.namespace ? (String(opts.namespace) + '_') : '';
    var parsed = parseURL(baseUrl);
    var pinHost = parsed ? parsed.hostname : '';
    var perms = { net: pinHost ? [pinHost] : [] };
    var methods = ['get', 'post', 'put', 'patch', 'delete', 'head'];
    var made = [];
    var pathKeys = Object.keys(paths);
    for (var pi = 0; pi < pathKeys.length; pi++) {
      var pathKey = pathKeys[pi];
      var item = oaRef(doc, paths[pathKey]) || {};
      for (var mi = 0; mi < methods.length; mi++) {
        var method = methods[mi];
        var op = item[method];
        if (!op || typeof op !== 'object') continue;
        if (typeof opts.filter === 'function') {
          var keep = true;
          try { keep = opts.filter(op) !== false; } catch (eF) { keep = true; }
          if (!keep) continue;
        }
        var opId = op.operationId ? String(op.operationId) : (method + '_' + oaSlug(pathKey));
        var toolName = ns + oaSlug(opId);
        var params = [];
        var raws = (item.parameters || []).concat(op.parameters || []);
        for (var qi = 0; qi < raws.length; qi++) {
          var rp = oaRef(doc, raws[qi]);
          if (!rp || !rp.name) continue;
          /* Swagger 2.0 body parameter -> request body */
          if (rp.in === 'body') { params.push({ name: rp.name, where: 'body', schema: rp.schema || { type: 'object' }, required: !!rp.required, desc: rp.description || '' }); continue; }
          params.push({ name: String(rp.name), where: String(rp.in || 'query'), schema: oaSchemaFor(rp), required: !!rp.required, desc: rp.description || '' });
        }
        var bodySchema = null;
        if (op.requestBody) {
          var rb = oaRef(doc, op.requestBody) || {};
          var content = rb.content || {};
          var cKeys = Object.keys(content);
          if (cKeys.length) {
            var media = content[cKeys[0]] || {};
            bodySchema = media.schema ? oaRef(doc, media.schema) : null;
          }
        }
        (function (toolName, method, pathKey, op, params, bodySchema, pinHost) {
          var props = {};
          var required = [];
          for (var x = 0; x < params.length; x++) {
            var pr = params[x];
            if (pr.where === 'body') continue;
            props[pr.name] = pr.schema;
            if (pr.desc && !props[pr.name].description) props[pr.name].description = pr.desc;
            if (pr.required) required.push(pr.name);
          }
          if (bodySchema && bodySchema.properties) {
            var bp = Object.keys(bodySchema.properties);
            for (var b = 0; b < bp.length; b++) {
              props[bp[b]] = bodySchema.properties[bp[b]];
              if (bodySchema.required && bodySchema.required.indexOf(bp[b]) >= 0) required.push(bp[b]);
            }
          }
          var inputSchema = { type: 'object', properties: props };
          if (required.length) inputSchema.required = required;
          P.defineTool({
            name: toolName,
            description: String(op.summary || op.description || (method.toUpperCase() + ' ' + pathKey)).slice(0, 300),
            schema: inputSchema,
            permissions: { net: [pinHost] },
            timeoutMs: opts.timeoutMs || 20000,
            async run(args, ctx) {
              var url = baseUrl + pathKey;
              var qs = [];
              var headers = Object.assign({ Accept: 'application/json' }, opts.headers || {});
              var bodyObj = null;
              for (var i = 0; i < params.length; i++) {
                var p2 = params[i];
                var v = (args || {})[p2.name];
                if (v === undefined || v === null) continue;
                if (p2.where === 'path') url = url.replace('{' + p2.name + '}', encodeURIComponent(String(v)));
                else if (p2.where === 'query') qs.push(encodeURIComponent(p2.name) + '=' + encodeURIComponent(String(v)));
                else if (p2.where === 'header') headers[p2.name] = String(v);
              }
              if (bodySchema) {
                bodyObj = {};
                var bKeys = Object.keys(bodySchema.properties || {});
                for (var b2 = 0; b2 < bKeys.length; b2++) {
                  if (args && args[bKeys[b2]] !== undefined) bodyObj[bKeys[b2]] = args[bKeys[b2]];
                }
                if (!bodySchema.properties) bodyObj = args || {};
              }
              if (qs.length) url += (url.indexOf('?') >= 0 ? '&' : '?') + qs.join('&');
              var init = { method: method.toUpperCase(), headers: headers };
              if (bodyObj) { init.body = JSON.stringify(bodyObj); init.headers['Content-Type'] = 'application/json'; }
              var res = await ctx.fetch(url, init);
              var text = await res.text();
              var data = text;
              try { data = JSON.parse(text); } catch (eParse) {}
              if (!res.ok) {
                throw makeError('E_TOOL', 'OpenAPI ' + method.toUpperCase() + ' ' + pathKey + ' -> HTTP ' + res.status + ': ' + text.slice(0, 300));
              }
              /* The payload IS the tool result: an LLM reads fewer
               * envelopes than a {status,data} wrapper, and the wrapper
               * is still available to JS callers as a data property. */
              if (data && typeof data === 'object') return data;
              return { status: res.status, data: data };
            }
          });
          made.push(toolName);
        })(toolName, method, pathKey, op, params, bodySchema, pinHost);
      }
    }
    if (!made.length) throw makeError('E_SCHEMA', 'useOpenAPI: the document declares no operations');
    P.lastOpenApi = {
      version: version,
      title: String((doc.info && doc.info.title) || ''),
      toolCount: made.length,
      names: made.slice()
    };
    return made.map(function (n) { return P.TOOLS[n]; });
  };

  /* ══ module: discovery — ./plugins, ~/.sofuu/plugins, sofuu.json ═════════ */
  function fsReadFile(p) {
    if (!sofuu.fs || typeof sofuu.fs.readFile !== 'function') return null;
    return sofuu.fs.readFile(p);
  }
  function fsListDir(p) {
    if (!sofuu.fs) return null;
    var fn = sofuu.fs.readdir || sofuu.fs.listDir;
    return fn ? fn.call(sofuu.fs, p) : null;
  }
  function pluginExecSource(src, isTs) {
    var execSrc = String(src)
      .replace(/import(?:\s+[\s\S]*?\s+from)?\s+['"][^'"]+['"]\s*;?/g, '')
      .replace(/export\s+default\s+/g, 'globalThis.__sofuu_last_export = ')
      .replace(/export\s+(const|let|var|function|class)\s+/g, '$1 ');
    if (isTs) execSrc = stripTs(execSrc);
    return execSrc;
  }
  async function adoptDefaultExport(lastExp) {
    /* A plugin may export a FACTORY — the `plugin init --category=openapi`
     * starter does exactly that, and a factory that returns useOpenAPI(...)
     * used to be reported as "loaded but registered nothing". Call it. */
    if (typeof lastExp === 'function') lastExp = await lastExp();
    /* useOpenAPI() registers its tools itself and returns them as an array;
     * the registry diff already saw them, so there is nothing to adopt. */
    if (Array.isArray(lastExp)) return null;
    if (!lastExp || typeof lastExp !== 'object') return null;
    /* A spec that already carries a category came back from its factory, i.e.
     * it registered itself (export default defineTool(...) evaluates the
     * factory first). Re-registering would double-count it, so report it. */
    if (lastExp.category) return lastExp;
    if (typeof lastExp.run === 'function' || typeof lastExp.execute === 'function') return P.defineTool(lastExp);
    if (typeof lastExp.complete === 'function' || typeof lastExp.stream === 'function') return P.defineProvider(lastExp);
    if (typeof lastExp.onStart === 'function' || typeof lastExp.onTurnStart === 'function' || typeof lastExp.onStop === 'function') return P.defineService(lastExp);
    if (typeof lastExp.upsert === 'function' || typeof lastExp.query === 'function') return P.defineVectorStore(lastExp);
    if (typeof lastExp.get === 'function' && typeof lastExp.set === 'function') return P.defineStorage(lastExp);
    return null;
  }
  /* Snapshots hold object IDENTITIES, not just names: a plugin that
   * overwrites a built-in registration (a cookbook `ollama` provider replacing
   * the shipped adapter) is a real registration even though the key already
   * existed, and a name-only diff called it "registered nothing". */
  function registrySnapshot() {
    return {
      tools: Object.assign({}, P.TOOLS), providers: Object.assign({}, P.PROVIDERS),
      stores: Object.assign({}, P.STORES), services: Object.assign({}, P.SERVICES)
    };
  }
  /* Discovery tags each registration with the file that produced it:
   * `plugin list` can then report "how many tools/providers came from this
   * plugin", which is what the plan's COUNT column means, and an overwritten
   * built-in still shows its origin. */
  function registryFor(kind) {
    if (kind === 'tool') return P.TOOLS;
    if (kind === 'provider') return P.PROVIDERS;
    if (kind === 'store') return P.STORES;
    if (kind === 'service') return P.SERVICES;
    return null;
  }
  function tagOrigin(items, origin) {
    for (var i = 0; i < items.length; i++) {
      var parts = String(items[i]).split(':');
      var reg = registryFor(parts[0]);
      var nm = parts.slice(1).join(':');
      if (reg && reg[nm]) { try { reg[nm]._origin = origin; } catch (eT) {} }
    }
  }
  function registryDiff(before) {
    var after = registrySnapshot();
    var out = [];
    ['tools', 'providers', 'stores', 'services'].forEach(function (k) {
      var keys = Object.keys(after[k]);
      for (var i = 0; i < keys.length; i++) {
        var fresh = !Object.prototype.hasOwnProperty.call(before[k], keys[i]);
        if (fresh || before[k][keys[i]] !== after[k][keys[i]]) {
          out.push(k.replace(/s$/, '') + ':' + keys[i]);
        }
      }
    });
    return out;
  }
  P.discover = async function discover(dir) {
    dir = dir || './plugins';
    var out = { loaded: [], broken: [] };
    if (!sofuu.fs) return out;
    var exists = false;
    try { exists = await sofuu.fs.exists(dir); } catch (e0) { return out; }
    if (!exists) return out;
    var entries = [];
    try { entries = (await fsListDir(dir)) || []; } catch (e1) { return out; }
    if (entries && entries.length && typeof entries[0] === 'object') {
      entries = entries.map(function (e) { return String(e.name || e.path || e); });
    }
    var files = [];
    for (var i = 0; i < entries.length; i++) {
      var e = String(entries[i]);
      if (e.charAt(0) === '.') continue;
      if (e.slice(-3) !== '.js' && e.slice(-3) !== '.ts') continue;
      if (e.slice(-5) === '.d.ts') continue;
      files.push(e);
    }
    files.sort();
    for (var f = 0; f < files.length; f++) {
      var fname = files[f];
      var fpath = dir.replace(/\/+$/, '') + '/' + fname;
      var before = registrySnapshot();
      try {
        var src = String(await fsReadFile(fpath));
        var execSrc = pluginExecSource(src, /\.ts$/.test(fname));
        g.__sofuu_last_export = undefined;
        (0, eval)(execSrc);
        var adopted = await adoptDefaultExport(g.__sofuu_last_export);
        var added = registryDiff(before);
        if (!added.length && adopted) added.push(String(adopted.category || 'plugin') + ':' + String(adopted.name || fname));
        tagOrigin(added, fpath);
        if (!added.length) out.broken.push({ file: fname, error: 'loaded but registered nothing (no defineTool/defineProvider/defineService/defineStorage export)' });
        else out.loaded.push({ file: fname, items: added });
      } catch (eErr) {
        out.broken.push({ file: fname, error: String((eErr && eErr.message) || eErr) });
      } finally {
        g.__sofuu_last_export = undefined;
      }
    }
    return out;
  };
  P.loadManifest = async function loadManifest(manifestPath) {
    manifestPath = manifestPath || './sofuu.json';
    if (!sofuu.fs || !sofuu.fs.readFile) return { loaded: 0, sources: [] };
    var text;
    try { text = await fsReadFile(manifestPath); } catch (e) { return { loaded: 0, sources: [] }; }
    var manifest;
    try { manifest = JSON.parse(String(text)); }
    catch (eJson) { return { loaded: 0, error: 'Malformed sofuu.json: ' + eJson.message, sources: [] }; }
    var items = (manifest && manifest.plugins) || [];
    var total = 0;
    var sources = [];
    for (var i = 0; i < items.length; i++) {
      var it = items[i];
      if (typeof it === 'string') it = { source: it };
      if (!it || !it.source || it.enabled === false) continue;
      var src2 = String(it.source);
      if (src2.charAt(0) !== '.' && src2.charAt(0) !== '/') continue;
      try {
        var res = await P.discover(src2);
        total += res.loaded.length;
        sources.push({ source: src2, loaded: res.loaded.length, broken: res.broken.length });
      } catch (eLoad) { sources.push({ source: src2, error: String((eLoad && eLoad.message) || eLoad) }); }
    }
    return { loaded: total, sources: sources };
  };
  P.discoverAll = async function discoverAll() {
    var results = { local: null, global: null, manifest: null };
    try { results.local = await P.discover('./plugins'); } catch (e1) {}
    var home = envOf().HOME || envOf().USERPROFILE || '';
    if (home) { try { results.global = await P.discover(home + '/.sofuu/plugins'); } catch (e2) {} }
    try { results.manifest = await P.loadManifest('./sofuu.json'); } catch (e3) {}
    return results;
  };

  /* ══ module: mcp — ingestion of an external MCP server's tools ═══════════
   * MCP tools are the one plugin category with a transport: the client
   * lives in the native layer (sofuu.mcp.connect -> client.call). Wrapping
   * them in defineTool gives an MCP tool the SAME 5 shields, schema
   * validation and circuit breaker as a local one, so a prompt-injected
   * server cannot behave differently from local code. */
  P.ingestMcp = async function ingestMcp(serverName, opts) {
    opts = opts || {};
    if (!sofuu.mcp || typeof sofuu.mcp.connect !== 'function') {
      return { ok: false, error: { code: 'E_CAPABILITY', message: 'MCP client is not available in this runtime' } };
    }
    var name = String(serverName || '');
    if (!name) return { ok: false, error: { code: 'E_SCHEMA', message: 'ingestMcp requires a server name' } };
    var spec = { server: name, name: name };
    /* An HTTP/SSE MCP server needs a URL, not a command; the transport
     * switch lives in the host client (transport 'http' + url). */
    if (typeof opts.url === 'string' && opts.url) { spec.url = opts.url; spec.transport = opts.transport || 'http'; }
    if (typeof opts.command === 'string' && opts.command) spec.command = opts.command;
    if (Array.isArray(opts.args)) spec.args = opts.args;
    if (opts.env && typeof opts.env === 'object') spec.env = opts.env;
    if (typeof opts.cwd === 'string' && opts.cwd) spec.cwd = opts.cwd;
    if (!spec.command && typeof sofuu.mcp.loadConfig === 'function') {
      try {
        var cfg = sofuu.mcp.loadConfig();
        if (cfg && typeof cfg.then === 'function') cfg = await cfg;
        var sets = (cfg && (cfg.servers || cfg.mcpServers)) || [];
        for (var i = 0; i < sets.length; i++) {
          var s = sets[i];
          if (s && (s.name === name || s.server === name) && s.command) { spec.command = s.command; if (s.args) spec.args = s.args; if (s.env) spec.env = s.env; break; }
        }
      } catch (eCfg) {}
    }
    if (!spec.command && !spec.url) {
      return { ok: false, error: { code: 'E_CAPABILITY', message: 'MCP server "' + name + '" is not configured (pass opts.command for stdio or opts.url for http/sse)' } };
    }
    var client;
    try {
      client = await sofuu.mcp.connect(spec, { server: name });
    } catch (eConn) {
      return { ok: false, error: { code: 'E_CAPABILITY', message: 'MCP connect failed: ' + String((eConn && eConn.message) || eConn) } };
    }
    var listed;
    try { listed = await client.listTools(); }
    catch (eList) { return { ok: false, error: { code: 'E_TOOL', message: 'MCP listTools failed: ' + String((eList && eList.message) || eList) } }; }
    var tools = (listed && listed.tools) || [];
    var prefix = opts.prefix !== false ? (name.replace(/[^A-Za-z0-9]+/g, '_') + '_') : '';
    var made = [];
    for (var t = 0; t < tools.length; t++) {
      var mt = tools[t];
      if (!mt || !mt.name) continue;
      var toolName = prefix + String(mt.name).replace(/[^A-Za-z0-9_]+/g, '_');
      var inputSchema = (mt.inputSchema && typeof mt.inputSchema === 'object') ? mt.inputSchema : { type: 'object', properties: {} };
      try {
        P.defineTool({
          name: toolName,
          description: String(mt.description || ('MCP tool ' + mt.name + ' from ' + name)).slice(0, 300),
          schema: inputSchema,
          timeoutMs: opts.timeoutMs || 30000,
          requiresApproval: opts.requiresApproval,
          async run(args) {
            var res = await client.call(String(mt.name), args || {});
            if (res && res.isError) throw makeError('E_TOOL', 'MCP tool ' + mt.name + ' reported an error');
            return res;
          }
        });
        made.push(toolName);
      } catch (eDef) {
        /* One unregisterable tool must not abort the whole server. */
      }
    }
    P.MCP_CLIENTS[name] = client;
    return { ok: true, server: name, tools: made, count: made.length, source: 'mcp' };
  };

  /* ══ module: cli — the `sofuu plugin ...` helpers ════════════════════════
   * main.rs owns argument parsing; these functions own behaviour, so the
   * CLI and a JS caller get byte-identical results. */
  function cliExit(code) {
    if (typeof process !== 'undefined' && process.exit) process.exit(code);
    return code;
  }
  P.cliList = async function (wantJson) {
    try {
      var disc = await P.discoverAll();
      var list = P.list();
      if (wantJson) {
        console.log(JSON.stringify({ plugins: list, discovery: disc }, null, 2));
        return;
      }
      if (!list.length) {
        console.log('(no plugins discovered -- place .js/.ts files in ./plugins/ or ~/.sofuu/plugins/)');
        return;
      }
      /* COUNT is "how many registrations came from this plugin file" — the
       * plan's Tool/Provider Count column. A file that generates an OpenAPI
       * namespace reports the whole family; a hand-written plugin reports 1. */
      var perOrigin = {};
      for (var c0 = 0; c0 < list.length; c0++) {
        var ok0 = list[c0].origin || '';
        perOrigin[ok0] = (perOrigin[ok0] || 0) + 1;
      }
      console.log('NAME                CATEGORY    COUNT  CIRCUIT     TIMEOUT   DESCRIPTION');
      for (var i = 0; i < list.length; i++) {
        var p = list[i];
        var okey = p.origin || '';
        var count = okey ? String(perOrigin[okey]) : '1';
        console.log(
          String(p.name).padEnd(20) +
          String(p.category).padEnd(12) +
          String(count).padEnd(7) +
          String(p.circuit).padEnd(12) +
          String(p.timeoutMs + 'ms').padEnd(10) +
          String(p.description || '')
        );
      }
      if (disc && disc.local && disc.local.broken && disc.local.broken.length) {
        console.log('');
        for (var b = 0; b < disc.local.broken.length; b++) {
          console.log('  broken: ' + disc.local.broken[b].file + ' -- ' + disc.local.broken[b].error);
        }
      }
    } catch (e) {
      console.error(String((e && e.message) || e));
      return cliExit(1);
    }
  };
  function warnOnMissingSchemas(items) {
    for (var i = 0; i < items.length; i++) {
      var parts = String(items[i]).split(':');
      if (parts[0] !== 'tool') continue;
      var tool = P.TOOLS[parts.slice(1).join(':')];
      if (!tool || !tool.warnings || !tool.warnings.length) continue;
      for (var w = 0; w < tool.warnings.length; w++) {
        console.log('  warning: ' + parts[1] + ': ' + tool.warnings[w]);
      }
    }
  }
  P.cliValidate = async function (file) {
    try {
      var src = String(await fsReadFile(file));
      var execSrc = pluginExecSource(src, /\.ts$/.test(file));
      var before = registrySnapshot();
      g.__sofuu_last_export = undefined;
      (0, eval)(execSrc);
      var adopted2 = await adoptDefaultExport(g.__sofuu_last_export);
      var found = registryDiff(before);
      if (!found.length && adopted2) found.push(String(adopted2.category || 'plugin') + ':' + String(adopted2.name || file));
      tagOrigin(found, file);
      warnOnMissingSchemas(found);
      if (!found.length) {
        console.error('Validation failed: ' + file + ' did not register or export any tool, provider, store, or service');
        return cliExit(1);
      }
      console.log('Plugin ' + file + ' is valid (' + found.join(', ') + ')');
    } catch (e) {
      console.error('Validation failed: ' + String((e && e.message) || e));
      return cliExit(1);
    }
  };
  P.cliRun = async function (toolName, rawArgs, opts) {
    opts = opts || {};
    var started = Date.now();
    try {
      await P.discoverAll();
      var args = {};
      try { args = JSON.parse(rawArgs || '{}'); }
      catch (eJson) {
        console.error('Error: invalid JSON arguments: ' + eJson.message);
        return cliExit(1);
      }
      var res = await P.invoke(toolName, args);
      var cn = (res && res.error && res.error.code) || 'E_TOOL';
      var ms = Date.now() - started;
      if (!res || !res.ok) {
        if (opts.test) console.log('ok=false code=' + cn + ' latency=' + ms + 'ms');
        console.error('Tool Error [' + cn + ']: ' + ((res && res.error && res.error.message) || 'tool call failed'));
        if (res && res.error && res.error.path) console.error('  at path: ' + res.error.path);
        return cliExit(1);
      }
      if (opts.test) console.log('ok=true code=OK latency=' + ms + 'ms');
      /* A tool that returns {ok:true, ...} IS the envelope, so its payload
       * hides in the envelope's own keys: printing res.text printed 'ok' and
       * threw the actual answer away. Print the payload either way. */
      var payload = res.data;
      if (payload === undefined || payload === null) {
        var extra = {};
        var rKeys = Object.keys(res);
        for (var rk = 0; rk < rKeys.length; rk++) {
          var k2 = rKeys[rk];
          if (k2 === 'ok' || k2 === 'source' || k2 === 'duration_ms' || k2 === 'text') continue;
          extra[k2] = res[k2];
        }
        payload = Object.keys(extra).length ? extra : (res.text || 'ok');
      }
      if (payload && typeof payload === 'object') console.log(JSON.stringify(payload, null, 2));
      else console.log(payload);
    } catch (e) {
      if (opts.test) console.log('ok=false code=E_TOOL latency=' + (Date.now() - started) + 'ms');
      console.error(String((e && e.message) || e));
      return cliExit(1);
    }
  };
  P.cliTest = function (toolName, rawArgs) {
    /* "complete isolation" (plan §3): a breaker that opened during an earlier
     * run must not turn a fresh test into a fast-fail, so the target's circuit
     * is reset first. Only that tool's state is touched. */
    try { P.resetCircuit(String(toolName)); } catch (eRc) {}
    return P.cliRun(toolName, rawArgs, { test: true });
  };

  var PLUGIN_TEMPLATES = {
    tool: function (name) {
      return [
        '// plugins/' + name + '.ts -- an action tool with the 5 shields.',
        '// Run: sofuu plugin validate plugins/' + name + '.ts',
        "import { defineTool, z } from 'sofuu';",
        '',
        'export default defineTool({',
        '  name: ' + JSON.stringify(name) + ',',
        "  description: 'Describe what this tool does for the model',",
        '  schema: z.object({',
        "    query: z.string().describe('What to look up'),",
        '  }),',
        '  timeoutMs: 10000,',
        '  async run(args, ctx) {',
        '    ctx.log(' + JSON.stringify(name) + ' + " received " + args.query);',
        '    return { ok: true, query: args.query };',
        '  }',
        '});',
        ''
      ].join('\n');
    },
    provider: function (name) {
      return [
        '// plugins/' + name + '.ts -- an inference provider adapter.',
        "import { defineProvider } from 'sofuu';",
        '',
        'export default defineProvider({',
        '  name: ' + JSON.stringify(name) + ',',
        "  kind: 'custom',",
        "  baseUrl: process.env.MY_MODEL_HOST || 'http://127.0.0.1:8080',",
        '  async complete(req, ctx) {',
        '    const body = { model: req.model, messages: req.messages };',
        '    const res = await ctx.fetch(this.baseUrl + \'/v1/chat/completions\', {',
        "      method: 'POST',",
        "      headers: { 'Content-Type': 'application/json' },",
        '      body: JSON.stringify(body),',
        '    });',
        '    const json = await res.json();',
        '    const choice = (json.choices && json.choices[0]) || {};',
        '    return {',
        '      text: (choice.message && choice.message.content) || \'\',',
        '      model: json.model || req.model,',
        '      usage: {',
        '        promptTokens: (json.usage && json.usage.prompt_tokens) || 0,',
        '        completionTokens: (json.usage && json.usage.completion_tokens) || 0,',
        '      },',
        '    };',
        '  }',
        '});',
        ''
      ].join('\n');
    },
    vectorstore: function (name) {
      return [
        '// plugins/' + name + '.ts -- an external vector store (dual-write mirror).',
        "import { defineVectorStore } from 'sofuu';",
        '',
        'export default defineVectorStore({',
        '  name: ' + JSON.stringify(name) + ',',
        '  async init(cfg) {},',
        '  async upsert(item) {',
        '    // item: { id, vector, text, metadata }',
        '  },',
        '  async query(q) {',
        '    return [];',
        '  },',
        '  async delete(id) {},',
        '  async count() { return 0; }',
        '});',
        ''
      ].join('\n');
    },
    store: function (name) {
      return [
        '// plugins/' + name + '.ts -- a key/value storage adapter.',
        "import { defineStorage } from 'sofuu';",
        '',
        'export default defineStorage({',
        '  name: ' + JSON.stringify(name) + ',',
        '  async init(cfg) {},',
        '  async get(key) { return null; },',
        '  async set(key, value) {},',
        '  async delete(key) {},',
        '  async keys() { return []; }',
        '});',
        ''
      ].join('\n');
    },
    service: function (name) {
      return [
        '// plugins/' + name + '.ts -- a lifecycle service (hooks + interceptor).',
        "import { defineService } from 'sofuu';",
        '',
        'export default defineService({',
        '  name: ' + JSON.stringify(name) + ',',
        '  async onStart(ctx) { ctx.log(' + JSON.stringify(name) + ' + " started"); },',
        '  async onTurnStart(turn) {',
        '    return turn;',
        '  },',
        '  async onToolCall(call) {',
        '    // Throw an Error with code E_DENIED to block the call.',
        '  },',
        '  async onToolResult(res) {},',
        '  async onTurnEnd(turn) {},',
        '  async onStop() {}',
        '});',
        ''
      ].join('\n');
    },
    openapi: function (name) {
      /* Self-contained on purpose: the spec is an inline document, so the
       * scaffold validates and produces a callable tool the moment it is
       * written. Swap SPEC for a URL, a local path, or your own document. */
      return [
        '// plugins/' + name + '.ts -- bridge a REST API from its OpenAPI document.',
        '// Replace SPEC with your own OpenAPI 3.0/3.1 (or Swagger 2.0) document:',
        '// a URL, a local path, or an inline object.',
        "import { useOpenAPI } from 'sofuu';",
        '',
        'const SPEC = {',
        "  openapi: '3.0.3',",
        '  info: { title: ' + JSON.stringify(name) + ", version: '1.0.0' },",
        "  servers: [{ url: 'http://127.0.0.1:8080' }],",
        '  paths: {',
        "    '/ping': {",
        '      get: {',
        "        operationId: 'ping',",
        "        summary: 'Health probe',",
        "        responses: { '200': { description: 'ok' } }",
        '      }',
        '    }',
        '  }',
        '};',
        '',
        'export default async function register() {',
        "  return useOpenAPI(SPEC, { namespace: " + JSON.stringify(name) + ' });',
        '}',
        ''
      ].join('\n');
    }
  };
  P.cliInit = async function (name, category) {
    try {
      var cat = String(category || 'tool').toLowerCase();
      if (cat === 'vector_store' || cat === 'vector') cat = 'vectorstore';
      if (!PLUGIN_TEMPLATES[cat]) {
        console.error('unknown category "' + cat + '" (tool | provider | store | vectorstore | service | openapi)');
        return cliExit(1);
      }
      var dir = './plugins';
      if (sofuu.fs && sofuu.fs.mkdir) { try { await sofuu.fs.mkdir(dir, { recursive: true }); } catch (eMk) {} }
      var safe = String(name || '').replace(/[^A-Za-z0-9_-]+/g, '_');
      if (!safe) { console.error('plugin init requires a name'); return cliExit(1); }
      var file = dir + '/' + safe + '.ts';
      var exists = false;
      try { exists = await sofuu.fs.exists(file); } catch (eEx) {}
      if (exists) { console.error('refusing to overwrite existing ' + file); return cliExit(1); }
      await sofuu.fs.writeFile(file, PLUGIN_TEMPLATES[cat](safe));
      console.log('scaffolded ' + file + ' (' + cat + ')');
      console.log('next: sofuu plugin validate ' + file);
    } catch (e) {
      console.error(String((e && e.message) || e));
      return cliExit(1);
    }
  };

  /* ── Global primitive exports (the `import { ... } from 'sofuu'` path
   *    resolves to sofuu.plugins.*, these globals are the script shorthand) */
  g.defineTool = P.defineTool;
  g.defineProvider = P.defineProvider;
  g.defineStorage = P.defineStorage;
  g.defineVectorStore = P.defineVectorStore;
  g.defineService = P.defineService;
  g.useOpenAPI = P.useOpenAPI;

})();

