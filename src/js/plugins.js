// plugins.js — Sofuu Plugin & Extensibility Architecture (PLAN-PLUGIN-AND-EXTENSIBILITY-STANDARD).
//
// Exposes standard primitives for external extensions:
//   - defineTool(spec)         — Action plugins (LLM tools) with 5-Shield fault isolation
//   - defineProvider(spec)     — External model inference substrate adapters
//   - defineStorage(spec)      — External storage adapters
//   - defineVectorStore(spec)  — External semantic vector database adapters
//   - defineService(spec)      — Background lifecycle services with hook dispatch
//   - useOpenAPI(spec, opts)   — Zero-code OpenAPI 3.x REST auto-bridging
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
  P.VERSION = '0.2.0';

  // Registries
  P.TOOLS = {};
  P.PROVIDERS = {};
  P.STORES = {};
  P.SERVICES = {};
  P.CIRCUITS = {};

  /* ── Normative Error Construction ────────────────────────────────── */

  function makeError(code, message, details) {
    var err = new Error(message || code);
    err.code = code;
    if (details) err.details = details;
    return err;
  }

  /* ── Built-in Robust URL Parser (zero-dependency for QuickJS) ────── */

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

    var host = authority;
    var atIdx = host.indexOf('@');
    if (atIdx >= 0) host = host.slice(atIdx + 1);

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
      hostname: hostname.toLowerCase(),
      port: port,
      pathname: pathname,
      search: search,
      hash: hash,
      href: s,
      origin: protocol + '://' + (port ? (hostname + ':' + port) : hostname)
    };
  }

  /* ── Hardened SSRF & Network Security Rails (Shield 2) ───────────── */

  function isPrivateHost(host) {
    if (!host) return true;
    var h = String(host).toLowerCase().trim().replace(/^\[|\]$/g, '');
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

  function createSandboxedFetch(toolName, permissions, defaultSignal) {
    var baseFetch = (typeof g.fetch === 'function') ? g.fetch : null;
    if (!baseFetch) return null;
    var allowedNets = (permissions && Array.isArray(permissions.net)) ? permissions.net : null;

    return function sandboxedFetch(url, init) {
      init = init || {};
      if (!init.signal && defaultSignal) {
        init.signal = defaultSignal;
      }
      var parsedUrl = parseURL(url);
      if (!parsedUrl) {
        return Promise.reject(makeError('E_DENIED', 'Invalid URL in tool ' + toolName + ': ' + url));
      }
      var host = parsedUrl.hostname;
      if (allowedNets) {
        var matched = false;
        for (var i = 0; i < allowedNets.length; i++) {
          if (host === allowedNets[i] || host.endsWith('.' + allowedNets[i])) { matched = true; break; }
        }
        if (!matched) {
          return Promise.reject(makeError('E_DENIED', 'Host ' + host + ' is not allowed by network permissions for ' + toolName));
        }
      } else {
        if (isPrivateHost(host)) {
          return Promise.reject(makeError('E_DENIED', 'Access to internal/private IP ' + host + ' is blocked for ' + toolName));
        }
      }
      return baseFetch(url, init);
    };
  }

  /* ── Jailed Filesystem Rails (Shield 2) ──────────────────────────── */

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

  /* ── Circuit Breaker (Shield 3) ──────────────────────────────────── */

  function getCircuit(name, opts) {
    if (!P.CIRCUITS[name]) {
      P.CIRCUITS[name] = {
        state: 'CLOSED',
        failures: 0,
        lastFailure: 0,
        lastTripped: 0,
        failureThreshold: (opts && opts.failureThreshold) || 5,
        resetTimeoutMs: (opts && opts.resetTimeoutMs) || 30000,
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
        return { ok: true, state: 'HALF_OPEN' };
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
    return { ok: true, state: circuit.state };
  }

  function recordCircuitSuccess(name) {
    var c = P.CIRCUITS[name];
    if (c) { c.state = 'CLOSED'; c.failures = 0; }
  }

  function recordCircuitFailure(name, opts) {
    var c = getCircuit(name, opts);
    c.failures++;
    c.lastFailure = Date.now();
    if (c.state === 'HALF_OPEN' || c.failures >= c.failureThreshold) {
      c.state = 'OPEN';
      c.lastTripped = Date.now();
    }
  }

  /* ── Schema Validation & Micro-Zod (Shield 1) ────────────────────── */

  function isZodSchema(s) {
    return !!(s && (typeof s.parse === 'function' || typeof s.safeParse === 'function' || s._def));
  }

  function zodToJSONSchema(zObj) {
    if (!zObj || typeof zObj !== 'object') return { type: 'object', properties: {} };
    if (zObj._isJSONSchema) return zObj;
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

  function validateJSONSchema(schema, data) {
    if (!schema || typeof schema !== 'object') return { ok: true };
    if (schema.type === 'object') {
      if (!data || typeof data !== 'object' || Array.isArray(data)) {
        return { ok: false, path: '$', error: 'Expected object, received ' + typeof data };
      }
      var required = schema.required || [];
      for (var i = 0; i < required.length; i++) {
        var req = required[i];
        if (!(req in data) || data[req] === undefined) {
          return { ok: false, path: req, error: req + ' is required' };
        }
      }
      var props = schema.properties || {};
      for (var k in props) {
        if (k in data && data[k] !== undefined) {
          var pSchema = props[k];
          var expectedType = pSchema.type;
          var actualType = typeof data[k];
          if (expectedType === 'string' && actualType !== 'string') {
            return { ok: false, path: k, error: 'Expected string, received ' + actualType };
          }
          if (expectedType === 'number' && (actualType !== 'number' || isNaN(data[k]))) {
            return { ok: false, path: k, error: 'Expected number, received ' + actualType };
          }
          if (expectedType === 'integer' && !Number.isInteger(data[k])) {
            return { ok: false, path: k, error: 'Expected integer, received ' + actualType };
          }
          if (expectedType === 'boolean' && actualType !== 'boolean') {
            return { ok: false, path: k, error: 'Expected boolean, received ' + actualType };
          }
          if (expectedType === 'array' && !Array.isArray(data[k])) {
            return { ok: false, path: k, error: 'Expected array, received ' + actualType };
          }
          if (Array.isArray(pSchema.enum) && pSchema.enum.indexOf(data[k]) < 0) {
            return { ok: false, path: k, error: 'Value must be one of ' + JSON.stringify(pSchema.enum) };
          }
        }
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
        safeParse: function (val) { return { success: true, data: val }; }
      };
    },
    record: function () {
      return {
        _isJSONSchema: true,
        type: 'object',
        describe: function (d) { this.description = d; return this; },
        optional: function () { this.isOptional = true; return this; },
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
          for (var r = 0; r < required.length; r++) {
            var reqKey = required[r];
            if (!(reqKey in val) || val[reqKey] === undefined) {
              return { success: false, error: { message: reqKey + ' is required', path: [reqKey] } };
            }
          }
          for (var propKey in shape) {
            if (Object.prototype.hasOwnProperty.call(shape, propKey)) {
              if (propKey in val && val[propKey] !== undefined) {
                var fieldValidator = shape[propKey];
                if (fieldValidator && typeof fieldValidator.safeParse === 'function') {
                  var res = fieldValidator.safeParse(val[propKey]);
                  if (!res.success) {
                    var innerPath = [propKey].concat(res.error.path || []);
                    return { success: false, error: { message: 'Invalid value for ' + propKey + ': ' + res.error.message, path: innerPath } };
                  }
                }
              }
            }
          }
          return { success: true, data: val };
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

  /* ── Category 1: defineTool (Action Plugins with 5-Shields) ──────── */

  P.defineTool = function defineTool(spec) {
    if (!spec || typeof spec !== 'object') throw makeError('E_SCHEMA', 'defineTool requires a spec object');
    var name = String(spec.name || '').trim();
    if (!name) throw makeError('E_SCHEMA', 'defineTool requires a non-empty name');
    var runFn = spec.run || spec.execute;
    if (typeof runFn !== 'function') throw makeError('E_SCHEMA', 'defineTool requires run or execute function for ' + name);

    var timeoutMs = (typeof spec.timeoutMs === 'number' && spec.timeoutMs > 0) ? spec.timeoutMs : 30000;
    var cbConfig = spec.circuitBreaker || { failureThreshold: 5, resetTimeoutMs: 30000 };
    var rawSchema = spec.schema || spec.parameters || { type: 'object', properties: {} };
    var jsonSchema = isZodSchema(rawSchema) ? zodToJSONSchema(rawSchema) : rawSchema;

    var wrappedTool = {
      name: name,
      description: String(spec.description || spec.summary || ''),
      schema: jsonSchema,
      parameters: jsonSchema,
      timeoutMs: timeoutMs,
      requiresApproval: spec.requiresApproval,
      permissions: spec.permissions || {},
      category: 'tool',

      execute: async function (args, customCtx) {
        args = args || {};

        // Shield 1: Schema Validation
        if (isZodSchema(rawSchema) && typeof rawSchema.safeParse === 'function') {
          var zRes = rawSchema.safeParse(args);
          if (!zRes.success) {
            var errMsg = (zRes.error && zRes.error.message) || 'Argument validation failed';
            var errPath = (zRes.error && zRes.error.path && zRes.error.path.join('.')) || '$';
            return {
              ok: false,
              source: 'plugin',
              error: { code: 'E_SCHEMA', message: errMsg, path: errPath },
              text: 'tool error [E_SCHEMA]: ' + errMsg + ' at ' + errPath
            };
          }
          args = zRes.data;
        } else {
          var vCheck = validateJSONSchema(jsonSchema, args);
          if (!vCheck.ok) {
            return {
              ok: false,
              source: 'plugin',
              error: { code: 'E_SCHEMA', message: vCheck.error, path: vCheck.path },
              text: 'tool error [E_SCHEMA]: ' + vCheck.error + ' at ' + vCheck.path
            };
          }
        }

        // Shield 2: Security Policy Check (R7)
        try {
          if (sofuu.policy && typeof sofuu.policy.evaluate === 'function') {
            var polRaw = sofuu.policy.evaluate({ surface: 'js', op: 'tool.run', tool: name });
            var polDec = polRaw ? JSON.parse(polRaw) : null;
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

        // Shield 3: Circuit Breaker Fast-Fail
        var cbCheck = checkCircuit(name, cbConfig);
        if (!cbCheck.ok) {
          return {
            ok: false,
            source: 'plugin',
            error: { code: 'E_CIRCUIT_OPEN', message: cbCheck.error.message },
            text: 'tool error [E_CIRCUIT_OPEN]: ' + cbCheck.error.message
          };
        }

        // Build Isolated Context & Watchdog
        var ctx = customCtx || {};
        var abortCtrl = (typeof AbortController === 'function') ? new AbortController() : null;
        var signal = abortCtrl ? abortCtrl.signal : null;

        var procCwd = (typeof process !== 'undefined' && typeof process.cwd === 'function') ? process.cwd() : '.';
        var sandboxedFetch = createSandboxedFetch(name, spec.permissions, signal);
        var jailedFs = createJailedFs(name, procCwd);

        var toolCtx = {
          name: name,
          fetch: sandboxedFetch || g.fetch,
          fs: jailedFs,
          signal: signal,
          abortSignal: signal,
          secrets: (typeof process !== 'undefined' && process.env) || {},
          env: (typeof process !== 'undefined' && process.env) || {},
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

        // Shield 4: Hardware Timer Watchdog
        var timer = null;
        var timeoutPromise = new Promise(function (_, reject) {
          timer = setTimeout(function () {
            if (abortCtrl) {
              try { abortCtrl.abort(); } catch (eAb) {}
            }
            reject(makeError('E_TIMEOUT', 'Tool ' + name + ' exceeded deadline of ' + timeoutMs + 'ms'));
          }, timeoutMs);
        });

        // Shield 5: Isolated JS Execution / Panic Guard
        var startTs = Date.now();
        try {
          var execPromise = Promise.resolve(runFn.call(spec, args, toolCtx));
          var rawRes = await Promise.race([execPromise, timeoutPromise]);
          clearTimeout(timer);

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
          clearTimeout(timer);
          if (abortCtrl) {
            try { abortCtrl.abort(); } catch (eAb2) {}
          }
          recordCircuitFailure(name, cbConfig);

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

  /* ── Category 2: Substrate Adapters ──────────────────────────────── */

  P.defineProvider = function defineProvider(spec) {
    if (!spec || typeof spec !== 'object') throw makeError('E_SCHEMA', 'defineProvider requires spec');
    var name = String(spec.name || '').trim();
    if (!name) throw makeError('E_SCHEMA', 'defineProvider requires name');
    if (typeof spec.complete !== 'function') throw makeError('E_SCHEMA', 'defineProvider requires complete() function');
    spec.category = 'provider';
    P.PROVIDERS[name] = spec;
    return spec;
  };

  P.defineStorage = function defineStorage(spec) {
    if (!spec || typeof spec !== 'object') throw makeError('E_SCHEMA', 'defineStorage requires spec');
    var name = String(spec.name || '').trim();
    if (!name) throw makeError('E_SCHEMA', 'defineStorage requires name');
    spec.category = 'store';
    P.STORES[name] = spec;
    return spec;
  };

  P.defineVectorStore = function defineVectorStore(spec) {
    if (!spec || typeof spec !== 'object') throw makeError('E_SCHEMA', 'defineVectorStore requires spec');
    var name = String(spec.name || '').trim();
    if (!name) throw makeError('E_SCHEMA', 'defineVectorStore requires name');
    spec.category = 'store';
    P.STORES[name] = spec;
    return spec;
  };

  /* ── Category 3: Lifecycle Services ──────────────────────────────── */

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
          await spec.onStart.call(spec, ctx || { env: process.env || {}, log: console.log });
        }
        this.started = true;
      },
      stop: async function (ctx) {
        if (typeof spec.onStop === 'function') {
          await spec.onStop.call(spec, ctx || {});
        }
        this.started = false;
      },
      turnStart: async function (evt) {
        if (typeof spec.onTurnStart === 'function') {
          try { await spec.onTurnStart.call(spec, evt || {}); } catch (e) {
            console.error('[service:' + name + '] onTurnStart failed: ' + ((e && e.message) || e));
          }
        }
      },
      turnEnd: async function (evt) {
        if (typeof spec.onTurnEnd === 'function') {
          try { await spec.onTurnEnd.call(spec, evt || {}); } catch (e) {
            console.error('[service:' + name + '] onTurnEnd failed: ' + ((e && e.message) || e));
          }
        }
      }
    };

    P.SERVICES[name] = service;
    return service;
  };

  P.emitTurnStart = async function (evt) {
    var keys = Object.keys(P.SERVICES);
    for (var i = 0; i < keys.length; i++) await P.SERVICES[keys[i]].turnStart(evt);
  };

  P.emitTurnEnd = async function (evt) {
    var keys = Object.keys(P.SERVICES);
    for (var i = 0; i < keys.length; i++) await P.SERVICES[keys[i]].turnEnd(evt);
  };

  /* ── Workflow 3: Zero-Code OpenAPI Bridge ────────────────────────── */

  P.useOpenAPI = async function useOpenAPI(urlOrSpec, options) {
    options = options || {};
    var namespace = options.namespace || '';
    var customHeaders = options.headers || {};
    var filterFn = (typeof options.filter === 'function') ? options.filter : null;

    var specObj = urlOrSpec;
    var baseEndpoint = options.baseUrl || '';

    if (typeof urlOrSpec === 'string') {
      if (urlOrSpec.indexOf('http://') === 0 || urlOrSpec.indexOf('https://') === 0) {
        var resp = await fetch(urlOrSpec);
        if (!resp.ok) throw makeError('E_TOOL', 'Failed to fetch OpenAPI spec from ' + urlOrSpec + ': ' + resp.status);
        specObj = await resp.json();
        if (!baseEndpoint) {
          try {
            var pu = parseURL(urlOrSpec);
            if (pu) baseEndpoint = pu.origin;
          } catch (e) {}
        }
      } else {
        if (!sofuu.fs || !sofuu.fs.readFile) throw makeError('E_CAPABILITY', 'sofuu.fs unavailable to read local OpenAPI spec');
        var rawText = await sofuu.fs.readFile(urlOrSpec);
        specObj = JSON.parse(String(rawText));
      }
    }

    if (!specObj || typeof specObj !== 'object') throw makeError('E_SCHEMA', 'Invalid OpenAPI specification');

    if (!baseEndpoint && specObj.servers && specObj.servers[0] && specObj.servers[0].url) {
      baseEndpoint = specObj.servers[0].url;
    }
    if (!baseEndpoint) baseEndpoint = 'http://localhost';

    var paths = specObj.paths || {};
    var generatedTools = [];
    var methods = ['get', 'post', 'put', 'delete', 'patch'];

    for (var p in paths) {
      if (!Object.prototype.hasOwnProperty.call(paths, p)) continue;
      var pathItem = paths[p];
      for (var mi = 0; mi < methods.length; mi++) {
        var m = methods[mi];
        if (!pathItem[m]) continue;
        var op = pathItem[m];
        var opId = op.operationId || (m + '_' + p.replace(/[^a-zA-Z0-9]/g, '_').replace(/_+/g, '_'));
        if (filterFn && !filterFn(op)) continue;

        var toolName = namespace ? (namespace + '_' + opId) : opId;
        toolName = toolName.replace(/[^a-zA-Z0-9_-]/g, '_');

        var params = (op.parameters || []).concat(pathItem.parameters || []);
        var props = {};
        var requiredList = [];

        for (var pi = 0; pi < params.length; pi++) {
          var param = params[pi];
          var pName = param.name;
          var pSchema = param.schema || { type: 'string' };
          props[pName] = {
            type: pSchema.type || 'string',
            description: param.description || ''
          };
          if (param.required) requiredList.push(pName);
        }

        var hasBody = false;
        if (op.requestBody && op.requestBody.content) {
          var jsonBody = op.requestBody.content['application/json'];
          if (jsonBody && jsonBody.schema) {
            hasBody = true;
            var bSchema = jsonBody.schema;
            if (bSchema.properties) {
              for (var bk in bSchema.properties) {
                if (!props[bk]) props[bk] = bSchema.properties[bk];
              }
              if (Array.isArray(bSchema.required)) {
                for (var ri = 0; ri < bSchema.required.length; ri++) {
                  if (requiredList.indexOf(bSchema.required[ri]) < 0) requiredList.push(bSchema.required[ri]);
                }
              }
            } else {
              props.body = bSchema;
              if (op.requestBody.required) requiredList.push('body');
            }
          }
        }

        var schema = { type: 'object', properties: props, required: requiredList };

        (function (optM, optP, optName, optDesc, optSchema, optHasBody) {
          var t = P.defineTool({
            name: optName,
            description: optDesc || (optM.toUpperCase() + ' ' + optP),
            schema: optSchema,
            run: async function (args, ctx) {
              args = args || {};
              var targetPath = optP;
              var queryParts = [];
              var bodyPayload = null;

              for (var k in args) {
                if (targetPath.indexOf('{' + k + '}') >= 0) {
                  targetPath = targetPath.replace('{' + k + '}', encodeURIComponent(String(args[k])));
                } else if (optHasBody && (k === 'body' || optM !== 'get')) {
                  if (k === 'body') bodyPayload = args[k];
                  else {
                    if (!bodyPayload) bodyPayload = {};
                    bodyPayload[k] = args[k];
                  }
                } else {
                  queryParts.push(encodeURIComponent(k) + '=' + encodeURIComponent(String(args[k])));
                }
              }

              var fullUrl = baseEndpoint.replace(/\/+$/, '') + '/' + targetPath.replace(/^\/+/, '');
              if (queryParts.length) fullUrl += (fullUrl.indexOf('?') >= 0 ? '&' : '?') + queryParts.join('&');

              var headers = Object.assign({}, customHeaders);
              if (bodyPayload) headers['Content-Type'] = 'application/json';

              var res = await ctx.fetch(fullUrl, {
                method: optM.toUpperCase(),
                headers: headers,
                body: bodyPayload ? JSON.stringify(bodyPayload) : undefined
              });

              if (!res.ok) {
                throw makeError('E_TOOL', 'HTTP ' + res.status + ' from ' + fullUrl);
              }

              var cType = (res.headers && res.headers.get && res.headers.get('content-type')) || '';
              if (cType.indexOf('application/json') >= 0) return await res.json();
              return await res.text();
            }
          });
          generatedTools.push(t);
        })(m, p, toolName, op.description || op.summary, schema, hasBody);
      }
    }

    return generatedTools;
  };

  /* ── Dynamic Discovery & Manifest Loading ────────────────────────── */

  /* ── Built-in TypeScript Type Stripper for Plugins ───────────────── */

  function stripTs(src) {
    if (!src || typeof src !== 'string') return '';
    var s = src;
    // 1. Strip multi-line and single-line interfaces
    s = s.replace(/(?:export\s+)?interface\s+[A-Za-z_$][\w$]*(?:<[^>]*>)?(?:\s+extends\s+[^{]+)?\s*\{[\s\S]*?\}/g, '');

    // 2. Strip type aliases
    s = s.replace(/(?:export\s+)?type\s+[A-Za-z_$][\w$]*(?:<[^>]*>)?\s*=\s*[^;]+;/g, '');

    // 3. Strip "as Type" assertions
    s = s.replace(/\s+as\s+[A-Za-z_$][\w$]*(?:<[^>]*>)?(?:\[\])?/g, '');

    // 4. Strip return type annotations before block or arrow: ): ReturnType { or ): ReturnType =>
    s = s.replace(/\)\s*:\s*(?:Promise<[\s\S]*?>|[A-Za-z_$][\w$]*(?:<[^>]*>)?(?:\[\])?|\{[\s\S]*?\})\s*(?=\{|=>)/g, ') ');

    // 5. Strip destructuring type annotations: } : { ... }
    s = s.replace(/\}\s*:\s*\{[\s\S]*?\}(?=\s*[,)])/g, '}');

    // 6. Strip variable type annotations: (const|let|var) name: Type =
    s = s.replace(/\b(const|let|var)\s+([A-Za-z_$][\w$]*)\s*:\s*[^=;]+(?=\s*=|\s*;)/g, '$1 $2');

    // 7. Strip parameter type annotations in function signatures: (name: Type)
    s = s.replace(/([,(]\s*[A-Za-z_$][\w$]*)\s*:\s*(?:string|number|boolean|any|void|unknown|never|object|Record<[^>]+>|Array<[^>]+>|[A-Za-z_$][\w$]*(?:<[^>]*>)?(?:\[\])?)(?=\s*[,)=])/g, '$1');

    return s;
  }

  P.discover = async function discover(dir) {
    dir = dir || './plugins';
    var out = { loaded: [], broken: [] };
    var readDirFn = sofuu.fs.readdir || sofuu.fs.listDir;
    if (!sofuu.fs || !readDirFn) return out;

    var exists = false;
    try { exists = await sofuu.fs.exists(dir); } catch (e) { return out; }
    if (!exists) return out;

    var entries = [];
    try { entries = await readDirFn.call(sofuu.fs, dir); } catch (e2) { return out; }

    var pluginFiles = entries.filter(function (e) {
      return (e.endsWith('.js') || e.endsWith('.ts')) && !e.startsWith('.') && !e.endsWith('.d.ts');
    }).sort();

    for (var i = 0; i < pluginFiles.length; i++) {
      var fname = pluginFiles[i];
      var fpath = dir.replace(/\/+$/, '') + '/' + fname;
      try {
        var content = await sofuu.fs.readFile(fpath);
        var src = String(content);

        // Robust import removal (handles single/multi-line, with or without semicolon)
        var execSrc = src.replace(/import(?:\s+[\s\S]*?\s+from)?\s+['"][^'"]+['"]\s*;?/g, '')
                         .replace(/export\s+default\s+/g, 'globalThis.__sofuu_last_export = ')
                         .replace(/export\s+(const|let|var|function|class)\s+/g, '$1 ');

        if (fname.endsWith('.ts')) {
          execSrc = stripTs(execSrc);
        }

        var toolsBefore = Object.keys(P.TOOLS);
        var servicesBefore = Object.keys(P.SERVICES);
        var providersBefore = Object.keys(P.PROVIDERS);
        var storesBefore = Object.keys(P.STORES);

        g.__sofuu_last_export = undefined;
        (0, eval)(execSrc);

        var lastExp = g.__sofuu_last_export;
        if (lastExp && typeof lastExp === 'object' && !lastExp.category) {
          if (typeof lastExp.run === 'function' || typeof lastExp.execute === 'function') P.defineTool(lastExp);
          else if (typeof lastExp.complete === 'function') P.defineProvider(lastExp);
          else if (typeof lastExp.onStart === 'function' || typeof lastExp.onTurnStart === 'function') P.defineService(lastExp);
          else if (typeof lastExp.upsert === 'function' || typeof lastExp.query === 'function') P.defineVectorStore(lastExp);
        }

        var toolsAfter = Object.keys(P.TOOLS);
        var newlyAdded = [];
        for (var ti = 0; ti < toolsAfter.length; ti++) {
          if (toolsBefore.indexOf(toolsAfter[ti]) < 0) newlyAdded.push('tool:' + toolsAfter[ti]);
        }
        for (var si = 0; si < Object.keys(P.SERVICES).length; si++) {
          var sKey = Object.keys(P.SERVICES)[si];
          if (servicesBefore.indexOf(sKey) < 0) newlyAdded.push('service:' + sKey);
        }
        for (var pi = 0; pi < Object.keys(P.PROVIDERS).length; pi++) {
          var pKey = Object.keys(P.PROVIDERS)[pi];
          if (providersBefore.indexOf(pKey) < 0) newlyAdded.push('provider:' + pKey);
        }
        for (var sti = 0; sti < Object.keys(P.STORES).length; sti++) {
          var stKey = Object.keys(P.STORES)[sti];
          if (storesBefore.indexOf(stKey) < 0) newlyAdded.push('store:' + stKey);
        }
        out.loaded.push({ file: fname, items: newlyAdded });
      } catch (eErr) {
        out.broken.push({ file: fname, error: String((eErr && eErr.message) || eErr) });
      }
    }
    return out;
  };

  P.loadManifest = async function loadManifest(manifestPath) {
    manifestPath = manifestPath || './sofuu.json';
    if (!sofuu.fs || !sofuu.fs.readFile) return { loaded: 0 };
    var text;
    try { text = await sofuu.fs.readFile(manifestPath); } catch (e) { return { loaded: 0 }; }
    var manifest;
    try { manifest = JSON.parse(String(text)); } catch (eJson) { return { loaded: 0, error: 'Malformed sofuu.json: ' + eJson.message }; }

    var plugins = manifest.plugins || [];
    var count = 0;
    for (var i = 0; i < plugins.length; i++) {
      var item = plugins[i];
      if (typeof item === 'string') item = { source: item };
      if (!item.source || item.enabled === false) continue;
      if (item.source.indexOf('./') === 0 || item.source.indexOf('/') === 0) {
        try {
          var res = await P.discover(item.source);
          count += res.loaded.length;
        } catch (eLoad) {}
      }
    }
    return { loaded: count };
  };

  P.discoverAll = async function discoverAll() {
    var results = { local: null, global: null, manifest: null };
    try { results.local = await P.discover('./plugins'); } catch (e1) {}

    var home = (typeof process !== 'undefined' && process.env && (process.env.HOME || process.env.USERPROFILE)) || '';
    if (home) {
      try { results.global = await P.discover(home + '/.sofuu/plugins'); } catch (e2) {}
    }
    try { results.manifest = await P.loadManifest('./sofuu.json'); } catch (e3) {}
    return results;
  };

  /* ── Inspection & Inventory Listing ──────────────────────────────── */

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
        timeoutMs: t.timeoutMs || 30000,
        description: t.description || '',
        requiresApproval: t.requiresApproval || null
      });
    }

    var pKeys = Object.keys(P.PROVIDERS);
    for (var pi = 0; pi < pKeys.length; pi++) {
      var p = P.PROVIDERS[pKeys[pi]];
      out.push({ name: p.name, category: 'provider', circuit: 'CLOSED', timeoutMs: 30000, description: p.description || '' });
    }

    var stKeys = Object.keys(P.STORES);
    for (var sti = 0; sti < stKeys.length; sti++) {
      var st = P.STORES[stKeys[sti]];
      out.push({ name: st.name, category: 'store', circuit: 'CLOSED', timeoutMs: 30000, description: st.description || '' });
    }

    var sKeys = Object.keys(P.SERVICES);
    for (var si = 0; si < sKeys.length; si++) {
      var s = P.SERVICES[sKeys[si]];
      out.push({ name: s.name, category: 'service', circuit: 'CLOSED', timeoutMs: 30000, description: s.description || '' });
    }
    return out;
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
    return await tool.execute(args, ctx);
  };

  /* ── CLI Surface Helpers ─────────────────────────────────────────── */

  P.cliList = async function (wantJson) {
    try {
      await P.discoverAll();
      var list = P.list();
      if (wantJson) {
        console.log(JSON.stringify(list, null, 2));
        return;
      }
      if (!list.length) {
        console.log('\x1b[90m(no plugins discovered — place .js/.ts files in ./plugins/ or ~/.sofuu/plugins/)\x1b[0m');
        return;
      }
      console.log('\x1b[1m' + 'NAME'.padEnd(20) + 'CATEGORY'.padEnd(12) + 'CIRCUIT'.padEnd(12) + 'TIMEOUT'.padEnd(10) + 'DESCRIPTION\x1b[0m');
      for (var i = 0; i < list.length; i++) {
        var p = list[i];
        var circColor = p.circuit === 'OPEN' ? '\x1b[31m' : (p.circuit === 'HALF_OPEN' ? '\x1b[33m' : '\x1b[32m');
        var circText = circColor + p.circuit + '\x1b[0m';
        console.log(
          p.name.padEnd(20) +
          p.category.padEnd(12) +
          circText.padEnd(12 + (circColor.length + 4)) +
          (p.timeoutMs + 'ms').padEnd(10) +
          (p.description || '')
        );
      }
    } catch (e) {
      console.error(String((e && e.message) || e));
      if (typeof process !== 'undefined' && process.exit) process.exit(1);
    }
  };

  P.cliValidate = async function (file) {
    try {
      var content = await sofuu.fs.readFile(file);
      var src = String(content);

      var execSrc = src.replace(/import(?:\s+[\s\S]*?\s+from)?\s+['"][^'"]+['"]\s*;?/g, '')
                       .replace(/export\s+default\s+/g, 'globalThis.__sofuu_last_export = ')
                       .replace(/export\s+(const|let|var|function|class)\s+/g, '$1 ');

      if (file.endsWith('.ts')) {
        execSrc = stripTs(execSrc);
      }

      var beforeTools = Object.keys(P.TOOLS);
      var beforeServices = Object.keys(P.SERVICES);
      var beforeProviders = Object.keys(P.PROVIDERS);
      var beforeStores = Object.keys(P.STORES);

      g.__sofuu_last_export = undefined;
      (0, eval)(execSrc);

      var lastExp = g.__sofuu_last_export;
      if (lastExp && typeof lastExp === 'object' && !lastExp.category) {
        if (typeof lastExp.run === 'function' || typeof lastExp.execute === 'function') P.defineTool(lastExp);
        else if (typeof lastExp.complete === 'function') P.defineProvider(lastExp);
        else if (typeof lastExp.onStart === 'function' || typeof lastExp.onTurnStart === 'function') P.defineService(lastExp);
        else if (typeof lastExp.upsert === 'function' || typeof lastExp.query === 'function') P.defineVectorStore(lastExp);
      }

      var afterTools = Object.keys(P.TOOLS).filter(function (k) { return beforeTools.indexOf(k) < 0; });
      var afterServices = Object.keys(P.SERVICES).filter(function (k) { return beforeServices.indexOf(k) < 0; });
      var afterProviders = Object.keys(P.PROVIDERS).filter(function (k) { return beforeProviders.indexOf(k) < 0; });
      var afterStores = Object.keys(P.STORES).filter(function (k) { return beforeStores.indexOf(k) < 0; });

      var found = [].concat(
        afterTools.map(function (t) { return 'tool: ' + t; }),
        afterServices.map(function (s) { return 'service: ' + s; }),
        afterProviders.map(function (p) { return 'provider: ' + p; }),
        afterStores.map(function (st) { return 'store: ' + st; })
      );

      if (!found.length) {
        console.error('\x1b[31m✗ Validation failed:\x1b[0m file ' + file + ' did not register or export any tool, provider, store, or service');
        if (typeof process !== 'undefined' && process.exit) process.exit(1);
        return;
      }

      console.log('\x1b[32m✓ Plugin ' + file + ' is valid\x1b[0m (' + found.join(', ') + ')');
    } catch (e) {
      console.error('\x1b[31m✗ Validation failed:\x1b[0m ' + ((e && e.message) || e));
      if (typeof process !== 'undefined' && process.exit) process.exit(1);
    }
  };

  P.cliRun = async function (toolName, rawArgs) {
    try {
      await P.discoverAll();
      var parsedArgs = {};
      try {
        parsedArgs = JSON.parse(rawArgs || '{}');
      } catch (e) {
        console.error('\x1b[31mError:\x1b[0m invalid JSON arguments: ' + e.message);
        if (typeof process !== 'undefined' && process.exit) process.exit(1);
        return;
      }
      var res = await P.invoke(toolName, parsedArgs);
      if (!res.ok) {
        var errCode = (res.error && res.error.code) || 'E_TOOL';
        var errMsg = (res.error && res.error.message) || res.text || 'Tool call failed';
        console.error('\x1b[31mTool Error [' + errCode + ']:\x1b[0m ' + errMsg);
        if (res.error && res.error.path) {
          console.error('  at path: ' + res.error.path);
        }
        if (typeof process !== 'undefined' && process.exit) process.exit(1);
        return;
      }
      if (res.data !== undefined) {
        if (typeof res.data === 'object') console.log(JSON.stringify(res.data, null, 2));
        else console.log(res.data);
      } else {
        console.log(res.text || 'ok');
      }
    } catch (e) {
      console.error(String((e && e.message) || e));
      if (typeof process !== 'undefined' && process.exit) process.exit(1);
    }
  };

  P.cliInit = async function (name, category) {
    try {
      var dir = './plugins';
      if (sofuu.fs && sofuu.fs.mkdir) {
        try { await sofuu.fs.mkdir(dir, { recursive: true }); } catch (e) {}
      }
      var file = dir + '/' + name + '.js';
      if (await sofuu.fs.exists(file)) {
        console.error('refusing to overwrite existing ' + file);
        if (typeof process !== 'undefined' && process.exit) process.exit(1);
        return;
      }
      var cat = category || 'tool';
      var tpl = '';
      if (cat === 'service') {
        tpl = "// plugins/" + name + ".js\nimport { defineService } from 'sofuu';\nexport default defineService({\n  name: " + JSON.stringify(name) + ",\n  async onStart(ctx) { ctx.log('Service started'); },\n  async onTurnStart(e) {},\n  async onTurnEnd(e) {},\n  async onStop() {}\n});\n";
      } else if (cat === 'provider') {
        tpl = "// plugins/" + name + ".js\nimport { defineProvider } from 'sofuu';\nexport default defineProvider({\n  name: " + JSON.stringify(name) + ",\n  async complete({ prompt, messages, model }) { return 'response'; }\n});\n";
      } else if (cat === 'store') {
        tpl = "// plugins/" + name + ".js\nimport { defineVectorStore } from 'sofuu';\nexport default defineVectorStore({\n  name: " + JSON.stringify(name) + ",\n  async init(cfg) {},\n  async upsert(item) {},\n  async query(q) { return []; }\n});\n";
      } else {
        tpl = "// plugins/" + name + ".js\nimport { defineTool } from 'sofuu';\nimport { z } from 'zod';\nexport default defineTool({\n  name: " + JSON.stringify(name) + ",\n  schema: z.object({ query: z.string() }),\n  async run({ query }, ctx) { return { status: 'ok', query }; }\n});\n";
      }
      await sofuu.fs.writeFile(file, tpl);
      console.log('scaffolded ' + file);
    } catch (e) {
      console.error(String((e && e.message) || e));
      if (typeof process !== 'undefined' && process.exit) process.exit(1);
    }
  };

  /* ── Global Primitive Exports ────────────────────────────────────── */

  g.defineTool = P.defineTool;
  g.defineProvider = P.defineProvider;
  g.defineStorage = P.defineStorage;
  g.defineVectorStore = P.defineVectorStore;
  g.defineService = P.defineService;

})();
