/* Kaze's async coordinator. The native continuation owns all interpreter
 * state; this file only forwards bounded JSON requests through the existing
 * policy-aware Sofuu APIs and returns their real results. */
(function (global) {
  'use strict';
  var sofuu = global.sofuu;
  var kaze = sofuu && sofuu.kaze;
  if (!kaze || typeof kaze._asyncStart !== 'function') return;

  function errorFrom(step) {
    var source = step && step.error && typeof step.error === 'object' ? step.error : (step || {});
    var err = new Error(String(source.error || source.message || 'Kaze async run failed'));
    if (source.code) err.code = String(source.code);
    if (source.diagnostic) err.diagnostic = source.diagnostic;
    return err;
  }

  function boundedError(error, request) {
    var code = error && error.code;
    if (typeof code !== 'string' || !/^[A-Z][A-Z0-9_]{1,63}$/.test(code)) code = 'E_KAZE_HOST';
    if (code === 'E_KAZE_OUTCOME_UNKNOWN') {
      return {
        code: code,
        message: 'The tool request may have completed, but its final result was not received. Do not retry it automatically.'
      };
    }
    var family = request && request.family;
    if (family === 'ai' && code === 'E_CAP_MISSING') {
      return {
        code: code,
        message: 'AI capability is unavailable. Configure a model in runtime configuration, SOFUU_AI_MODEL, or options.ai.model, and ensure an AI backend is available.'
      };
    }
    if (family === 'ai' && code === 'E_KAZE_RESULT') {
      return { code: code, message: 'AI classification returned a label outside the requested set' };
    }
    if (family === 'ai' && code === 'E_KAZE_ARGS') {
      return {
        code: code,
        message: 'AI request arguments are invalid; ai.classify needs text input and 1 to 128 unique labels of at most 256 characters.'
      };
    }
    var label = family === 'ai' ? 'AI provider request failed' : 'tool or MCP request failed';
    var message = label + ' (' + code + ')';
    return { code: code, message: message };
  }

  function unknownOutcome(error) {
    var code = error && error.code;
    if (code === 'E_POLICY_DENIED' || code === 'E_POLICY_APPROVAL_REQUIRED' ||
        code === 'E_POLICY_UNAVAILABLE' || code === 'E_PERMISSION_DENIED' ||
        code === 'E_PATH_OUTSIDE' || code === 'E_KAZE_ARGS' || code === 'E_CAP_MISSING' ||
        code === 'E_TOOL' || code === 'E_MCP_CONFIG' || code === 'E_MCP_SECRET_UNAVAILABLE' ||
        code === 'E_MCP_QUEUE_FULL' || code === 'E_MCP_CIRCUIT_OPEN' ||
        code === 'E_MCP_SERVER_NOT_FOUND' || code === 'E_MCP_TOOL_NOT_FOUND' ||
        code === 'E_MCP_AMBIGUOUS' || code === 'E_MCP_METHOD_NOT_SUPPORTED' ||
        code === 'E_SCHEMA') return error;
    var uncertain = new Error('The tool request may have completed, but its final result was not received. Do not retry it automatically.');
    uncertain.code = 'E_KAZE_OUTCOME_UNKNOWN';
    return uncertain;
  }

  function abortError() {
    var e = new Error('Kaze asynchronous run was cancelled');
    e.code = 'E_KAZE_CANCELLED';
    return e;
  }

  function timeoutError() {
    var e = new Error('Kaze asynchronous run deadline elapsed; a dispatched effect may have an unknown outcome');
    e.code = 'E_TIMEOUT';
    return e;
  }

  function ensureActive(signal) {
    if (signal && signal.aborted) throw abortError();
  }

  function bridgeSignal(parent) {
    var listeners = [];
    var signal = {
      aborted: Boolean(parent && parent.aborted),
      addEventListener: function (name, fn) {
        if (name !== 'abort' || typeof fn !== 'function') return;
        listeners.push(fn);
        if (signal.aborted) fn();
      },
      removeEventListener: function (name, fn) {
        if (name !== 'abort') return;
        listeners = listeners.filter(function (listener) { return listener !== fn; });
      }
    };
    function abort() {
      if (signal.aborted) return;
      signal.aborted = true;
      var pending = listeners.slice();
      listeners.length = 0;
      pending.forEach(function (listener) { try { listener(); } catch (e) { /* abort listeners cannot block cancellation */ } });
    }
    var onParentAbort = abort;
    if (parent && typeof parent.addEventListener === 'function') {
      if (parent.aborted) abort();
      else parent.addEventListener('abort', onParentAbort, { once: true });
    }
    return {
      signal: signal,
      abort: abort,
      dispose: function () {
        if (parent && typeof parent.removeEventListener === 'function') parent.removeEventListener('abort', onParentAbort);
        listeners.length = 0;
      }
    };
  }

  function waitFor(promise, ms, signal, onTimeout) {
    return new Promise(function (resolve, reject) {
      var settled = false;
      var timer;
      var onAbort = function () { finish(reject, abortError()); };
      function finish(callback, value) {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (signal && typeof signal.removeEventListener === 'function') signal.removeEventListener('abort', onAbort);
        callback(value);
      }
      timer = setTimeout(function () {
        if (typeof onTimeout === 'function') onTimeout();
        finish(reject, timeoutError());
      }, Math.max(1, ms));
      if (signal && typeof signal.addEventListener === 'function') {
        if (signal.aborted) onAbort();
        else signal.addEventListener('abort', onAbort, { once: true });
      }
      Promise.resolve(promise).then(function (value) { finish(resolve, value); }, function (error) { finish(reject, error); });
    });
  }

  /* 12-factor configuration channel for a compiled agent: an artifact has no
   * CLI flag or config file of its own for the model, so the coordinator reads
   * the documented SOFUU_* environment variables. An explicit options.ai
   * object always wins over the environment. The provider's API key is left to
   * the runtime, which already resolves OPENAI_API_KEY/ANTHROPIC_API_KEY. */
  function environmentAiOptions() {
    var env = (typeof process !== 'undefined' && process && process.env) ? process.env : null;
    if (!env) return {};
    var out = {};
    function take(key, field) {
      var value = env[key];
      if (typeof value === 'string' && value.trim() !== '') out[field] = value.trim();
    }
    take('SOFUU_AI_PROVIDER', 'provider');
    take('SOFUU_AI_MODEL', 'model');
    take('SOFUU_AI_BASE_URL', 'base_url');
    var rawTemperature = env.SOFUU_AI_TEMPERATURE;
    if (typeof rawTemperature === 'string' && rawTemperature.trim() !== '') {
      var temperature = Number(rawTemperature);
      if (isFinite(temperature) && temperature >= 0) out.temperature = temperature;
    }
    var rawMaxTokens = env.SOFUU_AI_MAX_TOKENS;
    if (typeof rawMaxTokens === 'string' && rawMaxTokens.trim() !== '') {
      var maxTokens = parseInt(rawMaxTokens, 10);
      if (isFinite(maxTokens) && maxTokens > 0) out.maxTokens = maxTokens;
    }
    return out;
  }

  /* Exposed for the runtime's own tests; not part of the agent-facing API. */
  kaze._envAiOptions = environmentAiOptions;

  function classifyError(code, message) {
    var error = new Error(message);
    error.code = code;
    return error;
  }

  function validateClassifyLabels(labels) {
    if (!Array.isArray(labels) || labels.length === 0 || labels.length > 128) {
      throw classifyError('E_KAZE_ARGS', 'ai.classify needs between 1 and 128 text labels');
    }
    for (var i = 0; i < labels.length; i++) {
      if (typeof labels[i] !== 'string' || labels[i].length === 0 ||
          labels[i].length > 256 || labels[i] !== labels[i].trim() ||
          /[\u0000-\u001f\u007f]/.test(labels[i]) || labels.indexOf(labels[i]) !== i) {
        throw classifyError('E_KAZE_ARGS', 'ai.classify labels must be unique, plain text up to 256 characters');
      }
    }
    return labels;
  }

  function validateClassifyResult(text, labels) {
    if (typeof text !== 'string') {
      throw classifyError('E_KAZE_RESULT', 'AI classification did not return text');
    }
    var candidate = text.trim();
    for (var i = 0; i < labels.length; i++) {
      if (candidate === labels[i]) return labels[i];
    }
    throw classifyError('E_KAZE_RESULT', 'AI classification returned a label outside the requested set');
  }

  /* Internal pure validator for deterministic runtime tests; not a Kaze
   * capability and never invokes or impersonates a provider. */
  kaze._validateClassifyResult = validateClassifyResult;

  async function invokeAi(request, options, remainingMs, signal) {
    ensureActive(signal);
    if (!sofuu.ai || typeof sofuu.ai.complete !== 'function') {
      var missing = new Error('sofuu.ai.complete is unavailable');
      missing.code = 'E_CAP_MISSING';
      throw missing;
    }
    var args = request.args || [];
    var prompt;
    var labels = null;
    if (request.op === 'classify') {
      labels = validateClassifyLabels(args[1]);
      if (typeof args[0] !== 'string') {
        throw classifyError('E_KAZE_ARGS', 'ai.classify expects text input');
      }
      prompt = 'Classify the input using exactly one label from this JSON array: ' +
        JSON.stringify(labels) + '\nReturn only the selected label.\nInput:\n' + args[0];
    } else {
      prompt = args[0];
    }
    if (typeof prompt !== 'string') {
      var invalid = new Error('ai.' + request.op + ' expects text input');
      invalid.code = 'E_KAZE_ARGS';
      throw invalid;
    }
    var aiOptions = environmentAiOptions();
    if (options && options.ai && typeof options.ai === 'object') {
      Object.keys(options.ai).forEach(function (key) {
        if (options.ai[key] !== undefined && options.ai[key] !== null) aiOptions[key] = options.ai[key];
      });
    }
    if (!aiOptions.base_url && aiOptions.baseUrl) aiOptions.base_url = aiOptions.baseUrl;
    if (!aiOptions.api_key && aiOptions.apiKey) aiOptions.api_key = aiOptions.apiKey;
    aiOptions.timeout = Math.max(1, remainingMs);
    var response;
    try {
      response = await sofuu.ai.complete(prompt, aiOptions);
    } catch (error) {
      /* The native API rejects an absent model with a TypeError rather than a
       * coded error. Keep that known configuration failure distinct from a
       * provider/network failure, without forwarding arbitrary provider text. */
      var detail = error && typeof error.message === 'string' ? error.message : '';
      if (/no model configured/i.test(detail)) {
        var noModel = new Error('No AI model is configured');
        noModel.code = 'E_CAP_MISSING';
        throw noModel;
      }
      throw error;
    }
    if (!response || typeof response.text !== 'string') {
      var malformed = new Error('AI backend completed without a text result');
      malformed.code = 'E_KAZE_HOST';
      throw malformed;
    }
    if (request.op === 'classify') return validateClassifyResult(response.text, labels);
    return response.text;
  }

  function toolRequest(args) {
    var name;
    var server;
    var input;
    if (args[0] && typeof args[0] === 'object' && !Array.isArray(args[0])) {
      var spec = args[0];
      name = String(spec.tool || spec.name || '');
      server = spec.server == null ? '' : String(spec.server);
      input = spec.arguments || spec.args || {};
    } else {
      name = String(args[0] == null ? '' : args[0]);
      input = args.length > 1 ? args[1] : {};
      var separator = name.indexOf('::');
      if (separator > 0) {
        server = name.slice(0, separator);
        name = name.slice(separator + 2);
      } else {
        server = '';
      }
    }
    if (!name) {
      var invalid = new Error('tool.call requires a tool name');
      invalid.code = 'E_KAZE_ARGS';
      throw invalid;
    }
    if (input == null) input = {};
    return { name: name, server: server, input: input };
  }

  async function invokeMcp(request, options, remainingMs, signal) {
    ensureActive(signal);
    if (!sofuu.mcp || typeof sofuu.mcp.connect !== 'function' || typeof sofuu.mcp.loadConfig !== 'function') {
      var missing = new Error('the MCP client is unavailable');
      missing.code = 'E_CAP_MISSING';
      throw missing;
    }
    var wanted = toolRequest(request.args || []);
    var mcpOpts = options && options.mcp && typeof options.mcp === 'object' ? options.mcp : {};
    var loaded = await sofuu.mcp.loadConfig(mcpOpts);
    if (!loaded || !loaded.ok) {
      var configError = new Error('MCP configuration could not be loaded');
      configError.code = 'E_MCP_CONFIG';
      throw configError;
    }
    var servers = loaded.servers || [];
    ensureActive(signal);
    var candidates = servers.filter(function (config) {
      return !wanted.server || String(config.name) === wanted.server;
    });
    if (wanted.server && candidates.length !== 1) {
      var serverAbsent = new Error('the selected MCP server is missing or ambiguous');
      serverAbsent.code = 'E_MCP_SERVER_NOT_FOUND';
      throw serverAbsent;
    }
    if (!wanted.server && candidates.length > 1) {
      var explicit = new Error('multiple MCP servers are configured; specify "server::tool"');
      explicit.code = 'E_MCP_AMBIGUOUS';
      throw explicit;
    }
    if (candidates.length === 0) {
      var absent = new Error('no configured MCP server exposes tool ' + wanted.name);
      absent.code = 'E_MCP_TOOL_NOT_FOUND';
      throw absent;
    }
    var config = candidates[0];
    ensureActive(signal);
    var client = await sofuu.mcp.connect(config, {
      server: config.name,
      /* A dispatched write or tool call is never replayed automatically. */
      reconnect: { enabled: false }
    });
    try {
      ensureActive(signal);
      /* Discovery is also bounded. The default listTools helper has no
       * timeout argument, while call(method, params, opts) does. */
      var inventory = await client.call('tools/list', {}, { timeoutMs: Math.max(1, remainingMs) });
      ensureActive(signal);
      var tools = inventory && Array.isArray(inventory.tools) ? inventory.tools : [];
      if (!tools.some(function (tool) { return tool && tool.name === wanted.name; })) {
        var absentTool = new Error('the selected MCP server does not expose tool ' + wanted.name);
        absentTool.code = 'E_MCP_TOOL_NOT_FOUND';
        throw absentTool;
      }
      ensureActive(signal);
      var result;
      try {
        request.__effectDispatched = true;
        result = await client.call(wanted.name, wanted.input, { timeoutMs: Math.max(1, remainingMs) });
      } catch (error) {
        throw unknownOutcome(error);
      }
      if (result && result.isError) {
        throw unknownOutcome(null);
      }
      /* The same normalizer the rest of Sofuu uses: an MCP result becomes one
       * bounded, structured { ok, text } value the Kaze string operators can
       * render. The raw transport shape is never the data contract. */
      if (sofuu.tools && typeof sofuu.tools.mcpResult === 'function') {
        var normalized = sofuu.tools.mcpResult(result);
        if (normalized && typeof normalized.text === 'string') {
          return { ok: true, text: normalized.text };
        }
      }
      return result;
    } finally {
      if (client && typeof client.disconnect === 'function') {
        try { await client.disconnect(); } catch (e) { /* a received tool result remains authoritative */ }
      }
    }
  }

  async function invokeTool(request, options, remainingMs, signal) {
    ensureActive(signal);
    var args = request.args || [];
    var wanted = toolRequest(args);
    if (!wanted.server && sofuu.tools && sofuu.tools.TOOLS &&
        Object.prototype.hasOwnProperty.call(sofuu.tools.TOOLS, wanted.name) &&
        typeof sofuu.tools.execute === 'function') {
      try {
        request.__effectDispatched = true;
        return await sofuu.tools.execute(wanted.name, wanted.input, {
          timeoutMs: remainingMs,
          runId: 'kaze'
        });
      } catch (error) {
        throw unknownOutcome(error);
      }
    }
    return await invokeMcp(request, options, remainingMs, signal);
  }

  async function dispatch(request, options, remainingMs, signal) {
    ensureActive(signal);
    if (request.family === 'ai') return await invokeAi(request, options, remainingMs, signal);
    if (request.family === 'tool') return await invokeTool(request, options, remainingMs, signal);
    var unsupported = new Error('no async adapter is registered for ' + request.family + '.' + request.op);
    unsupported.code = 'E_CAP_MISSING';
    throw unsupported;
  }

  kaze.runAsync = async function (pathOrSource, args, options) {
    args = args || {};
    options = options || {};
    var signal = options.signal;
    if (signal && signal.aborted) throw abortError();
    var step = kaze._asyncStart(pathOrSource, args, options);
    var runId = step && step.runId;
    if (!step || step.state === 'error') throw errorFrom(step);
    try {
      while (step && (step.state === 'yield' || step.state === 'capability')) {
        if (signal && signal.aborted) throw abortError();
        var remaining = Math.max(1, Number(step.remainingMs) || 1);
        if (step.state === 'yield') {
          await waitFor(new Promise(function (resolve) { setTimeout(resolve, 0); }), remaining, signal);
          step = kaze._asyncResume(runId, '0', null, null);
          continue;
        }
        var value;
        var failure = null;
        var dispatchControl = bridgeSignal(signal);
        try {
          var dispatched = dispatch(step, options, remaining, dispatchControl.signal);
          value = await waitFor(dispatched, remaining, dispatchControl.signal, dispatchControl.abort);
        } catch (error) {
          dispatchControl.abort();
          failure = boundedError(step.__effectDispatched ? unknownOutcome(error) : error, step);
        } finally {
          dispatchControl.dispose();
        }
        step = kaze._asyncResume(runId, step.requestId, value, failure);
        if (step && step.state === 'late') throw new Error('Kaze run ended before the capability response arrived');
        if (step && step.state === 'error') throw errorFrom(step);
      }
      if (step && step.state === 'complete') return step;
      throw errorFrom(step);
    } catch (error) {
      if (runId) kaze._asyncCancel(runId);
      throw error;
    }
  };
})(globalThis);
