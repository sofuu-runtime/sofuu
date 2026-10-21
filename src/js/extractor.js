/* src/js/extractor.js — sofuu.extractor: automated local-first & hybrid fact extractor
 * (PLAN-CMA-V2-AND-STRUCTURED-STORAGE Phase 2).
 *
 * Extracts durable user facts, preferences, project decisions, and life updates
 * from conversational turns into structured JSON, and integrates them into
 * the Cognitive Memory Architecture (CMA) with contradiction resolution and
 * duplicate suppression.
 *
 * Capabilities:
 *   1. Zero-cost heuristic pre-filtering (< 1 µs) to bypass model invocation
 *      on queries and chit-chat that contain no personal user facts.
 *   2. Dual local/cloud extraction: runs on local GGUF models (e.g. SmolLM2,
 *      Qwen-0.6B/1.5B) via sofuu.ai.inferLocal or sofuu.ai.complete, and on
 *      remote cloud models (OpenAI, Anthropic, Groq).
 *   3. Contradiction Resolution: automated detection of conflicting or superseded
 *      facts, tombstoning/decaying old memories, and committing fresh facts.
 *   4. Deduplication: semantic similarity search before insertion to prevent
 *      redundant memory accumulation.
 *   5. Async post-turn queue: background non-blocking execution so chat latency
 *      is 0 ms.
 */

(function () {
  var S = globalThis.sofuu;
  if (!S) return;

  var E = S.extractor || {};

  // Heuristic indicator: does this message likely contain personal user facts/updates?
  var USER_MARKERS_RE = /\b(I|I'm|I've|I'll|I'd|my|mine|we|prefer|preferred|prefers|preference|switched|migrated|love|hate|always|never|work as|live in|located in|building|stack|using|favorite|moved to|hobby|name is)\b/i;

  // Generic definitions or non-user pattern rejection:
  var GENERIC_DEF_RE = /^(a|an|the)\s+[\w\s-]+\s+(is|are|refers to|means|stands for)\b/i;

  function shouldExtract(text) {
    if (!text || typeof text !== 'string') return false;
    var trimmed = text.trim();
    if (trimmed.length < 5 || trimmed.length > 8000) return false;
    return USER_MARKERS_RE.test(trimmed);
  }

  function cleanJsonText(raw) {
    if (!raw) return '[]';
    var s = String(raw).trim();
    // Strip markdown code fences if present: ```json ... ``` or ``` ... ```
    if (s.indexOf('```') !== -1) {
      s = s.replace(/```(?:json)?([\s\S]*?)```/g, '$1').trim();
    }
    // Find outermost '[' and ']'
    var start = s.indexOf('[');
    var end = s.lastIndexOf(']');
    if (start !== -1 && end !== -1 && end > start) {
      return s.slice(start, end + 1);
    }
    // Single object fallback: '{...}' -> '[{...}]'
    var objStart = s.indexOf('{');
    var objEnd = s.lastIndexOf('}');
    if (objStart !== -1 && objEnd !== -1 && objEnd > objStart) {
      return '[' + s.slice(objStart, objEnd + 1) + ']';
    }
    return '[]';
  }

  function parseFactsJson(raw) {
    var jsonText = cleanJsonText(raw);
    try {
      var arr = JSON.parse(jsonText);
      if (!Array.isArray(arr)) return [];
      var results = [];
      for (var i = 0; i < arr.length; i++) {
        var item = arr[i];
        if (!item || typeof item !== 'object') continue;
        var factStr = typeof item.fact === 'string' ? item.fact.trim() : '';
        // If fact was nested as an object:
        if (!factStr && item.fact && typeof item.fact === 'object') {
          factStr = JSON.stringify(item.fact);
        }
        if (!factStr || factStr.length < 3) continue;

        // Skip placeholder tags like <...>
        if (factStr.indexOf('<') !== -1 && factStr.indexOf('>') !== -1) {
          continue;
        }
        // Skip generic dictionary definitions
        if (GENERIC_DEF_RE.test(factStr) && !/\b(user|I|developer)\b/i.test(factStr)) {
          continue;
        }

        var action = (item.action || 'ADD').toUpperCase();
        var supersedes = '';
        if (typeof item.supersedes === 'string') {
          supersedes = item.supersedes.trim();
        } else if (Array.isArray(item.supersedes) && item.supersedes.length > 0) {
          supersedes = String(item.supersedes[0]).trim();
        }

        if (supersedes && action === 'ADD') {
          action = 'UPDATE';
        }
        if (action !== 'ADD' && action !== 'UPDATE' && action !== 'CONTRADICT') {
          action = 'ADD';
        }

        var entity = typeof item.entity === 'string' ? item.entity.trim() : (typeof item.entity === 'object' && item.entity ? JSON.stringify(item.entity) : '');

        results.push({
          action: action,
          fact: factStr,
          supersedes: supersedes,
          entity: entity
        });
      }
      return results;
    } catch (e) {
      return [];
    }
  }

  function buildExtractionPrompt(userText, assistantText) {
    var prompt = "<|im_start|>system\n" +
      "You are a memory extractor. Extract durable facts ABOUT THE USER from the input.\n" +
      "Output a JSON list of objects with \"action\" (\"ADD\" or \"UPDATE\"), \"fact\", \"supersedes\" (optional).\n" +
      "If no personal user facts, output [].\n\n" +
      "Examples:\n" +
      "User: Can you explain quicksort?\n" +
      "Assistant: []\n\n" +
      "User: I live in Seattle and I build microservices in Go.\n" +
      "Assistant: [{\"action\": \"ADD\", \"fact\": \"User lives in Seattle\"}, {\"action\": \"ADD\", \"fact\": \"User builds microservices in Go\"}]\n\n" +
      "User: Actually, I moved from Seattle to Boston.\n" +
      "Assistant: [{\"action\": \"UPDATE\", \"fact\": \"User lives in Boston\", \"supersedes\": \"User lives in Seattle\"}]<|im_end|>\n" +
      "<|im_start|>user\n" +
      userText + "<|im_end|>\n" +
      "<|im_start|>assistant\n[";
    return prompt;
  }

  async function callModel(prompt, options) {
    options = options || {};
    var model = options.model || '';
    var provider = options.provider || 'local';

    // 1. Direct local synchronous inference if inferLocal is available and it's a local GGUF path
    if (S.ai && typeof S.ai.inferLocal === 'function' && (provider === 'local' || model.indexOf('.gguf') !== -1)) {
      try {
        var raw = S.ai.inferLocal(model, prompt, { maxTokens: options.maxTokens || 200, temperature: 0.1 });
        if (raw) {
          // If the prompt ended with '[', restore the leading '['
          if (prompt.endsWith('[') && raw.trim().indexOf('[') !== 0) {
            raw = '[' + raw;
          }
          return raw;
        }
      } catch (eLocal) {
        // Fall through to ai.complete
      }
    }

    // 2. Standard ai.complete (supports both local and cloud providers)
    if (S.ai && typeof S.ai.complete === 'function') {
      var res = await S.ai.complete({
        provider: provider,
        model: model,
        prompt: prompt,
        max_tokens: options.maxTokens || 200,
        temperature: 0.1
      });
      var text = (res && res.text) || '';
      if (prompt.endsWith('[') && text.trim().indexOf('[') !== 0) {
        text = '[' + text;
      }
      return text;
    }

    throw new Error('No inference engine available for extraction');
  }

  async function extract(userText, options) {
    options = options || {};
    if (options.heuristicFilter !== false && !shouldExtract(userText)) {
      return [];
    }

    var prompt = buildExtractionPrompt(userText, options.assistantText || '');
    var raw = await callModel(prompt, options);
    return parseFactsJson(raw);
  }

  function resolveBrain(b) {
    if (b) return b;
    if (S.brain && typeof S.brain.open === 'function') {
      return S.brain;
    }
    if (S.memory && typeof S.memory.open === 'function') {
      return S.memory;
    }
    return null;
  }

  async function applyFacts(facts, brainInstance) {
    if (!facts || !facts.length) return { added: 0, updated: 0, skipped: 0 };
    var b = resolveBrain(brainInstance);
    if (!b) return { added: 0, updated: 0, skipped: 0 };

    var added = 0;
    var updated = 0;
    var skipped = 0;

    for (var i = 0; i < facts.length; i++) {
      var f = facts[i];
      var factText = f.fact;
      var role = 'user';
      var source = 'extractor' + (f.entity ? ':' + f.entity : '');

      if (f.action === 'UPDATE' || f.action === 'CONTRADICT') {
        // 1. Search for older facts to supersede
        var searchQuery = f.supersedes || f.fact;
        var existing = [];
        try {
          if (typeof b.search === 'function') {
            existing = b.search(searchQuery, 3);
          } else if (typeof b.recall === 'function') {
            existing = b.recall(searchQuery, 3);
          }
        } catch (eSearch) {}

        // 2. Tombstone or decay superseded records
        if (Array.isArray(existing)) {
          for (var j = 0; j < existing.length; j++) {
            var hit = existing[j];
            // If distance is close or score is high, supersede it
            if (hit && (hit.distance < 0.45 || hit.score > 0.55)) {
              try {
                if (typeof b.forget === 'function' && hit.id !== undefined) {
                  b.forget(hit.id);
                }
              } catch (eForget) {}
            }
          }
        }

        // 3. Learn new fact
        if (typeof b.learn === 'function') {
          b.learn(factText, role, source);
        } else if (typeof b.remember === 'function') {
          b.remember(factText, role, source);
        }
        updated++;
      } else {
        // ADD action: Deduplicate check
        var dupHits = [];
        try {
          if (typeof b.search === 'function') {
            dupHits = b.search(factText, 1);
          } else if (typeof b.recall === 'function') {
            dupHits = b.recall(factText, 1);
          }
        } catch (eDup) {}

        if (Array.isArray(dupHits) && dupHits.length > 0 && dupHits[0].distance < 0.08) {
          // Already present almost verbatim
          skipped++;
          continue;
        }

        if (typeof b.learn === 'function') {
          b.learn(factText, role, source);
        } else if (typeof b.remember === 'function') {
          b.remember(factText, role, source);
        }
        added++;
      }
    }

    try {
      if (typeof b.flush === 'function') b.flush();
    } catch (eFlush) {}

    return { added: added, updated: updated, skipped: skipped };
  }

  async function processTurn(userText, assistantText, options) {
    options = options || {};
    var facts = await extract(userText, Object.assign({}, options, { assistantText: assistantText }));
    var result = await applyFacts(facts, options.brain);
    return {
      facts: facts,
      added: result.added,
      updated: result.updated,
      skipped: result.skipped
    };
  }

  function processTurnAsync(userText, assistantText, options) {
    return new Promise(function (resolve) {
      setTimeout(function () {
        processTurn(userText, assistantText, options)
          .then(resolve)
          .catch(function (err) {
            resolve({ facts: [], error: String((err && err.message) || err) });
          });
      }, 0);
    });
  }

  E.shouldExtract = shouldExtract;
  E.cleanJsonText = cleanJsonText;
  E.parseFactsJson = parseFactsJson;
  E.extract = extract;
  E.applyFacts = applyFacts;
  E.processTurn = processTurn;
  E.processTurnAsync = processTurnAsync;

  S.extractor = E;

  // Extend CMA prototype with extractAndLearn convenience method
  if (S.brain && S.brain.open) {
    S.brain.extractor = E;
    var realOpen = S.brain.open;
    S.brain.open = function () {
      var inst = realOpen.apply(S.brain, arguments);
      if (inst && !inst.extractAndLearn) {
        inst.extractAndLearn = function (userText, opts) {
          opts = opts || {};
          return processTurn(userText, opts.assistantText || '', Object.assign({}, opts, { brain: inst }));
        };
      }
      return inst;
    };
  }
})();
