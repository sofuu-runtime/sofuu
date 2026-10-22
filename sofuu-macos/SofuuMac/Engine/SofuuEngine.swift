import Foundation

/// The in-process Sofuu runtime, driven from a dedicated worker thread.
///
/// A direct port of the Android app's `SofuuEngine.kt` (which itself mirrors
/// the desktop backend's `engine.rs` + `host.rs`). Same contract:
///
///   * ONE thread owns the runtime. The UI never touches it.
///   * Turn driving is fire-and-forget + a JS event queue:
///       1. `sofuu.chat.submit(text, {onEvent: globalThis.__sofuu_push})`
///          starts the turn as a promise; the eval returns immediately.
///       2. A pump drains `globalThis.__evq` every ≤50 ms, delivering events
///          to observers on the main thread. Each drain also runs the libuv
///          loop, so the turn progresses between polls.
///       3. Cancel + approvals ride `chat.js`'s `__host_poke` on the same
///          thread — the thread is never blocked mid-turn.
///   * Engine reads (sessions, state, model lists, …) come back through a
///     second queue, `globalThis.__rq`, resolved into the awaiting Task.
///     Replies are strictly sequential — one query in flight at a time,
///     matched by arrival order.
///
/// The only structural difference from Android: `CompletableFuture` becomes
/// Swift `async`/`await` (checked continuations), and the mailbox is an
/// `NSCondition` instead of a `LinkedBlockingQueue`.
final class SofuuEngine: @unchecked Sendable {

    let homeDir: String
    /// The active workspace. Mutated by `setProject` so session-file cleanup
    /// always targets the folder the engine is actually working in.
    private(set) var projectDir: String

    // ── Event fan-out ────────────────────────────────────────────────

    private var observers: [(EngineEvent) -> Void] = []
    private let observersLock = NSLock()

    func observe(_ fn: @escaping (EngineEvent) -> Void) {
        observersLock.lock()
        observers.append(fn)
        observersLock.unlock()
    }

    // ── Threading ────────────────────────────────────────────────────

    private let mailboxCond = NSCondition()
    private var mailbox: [() -> Void] = []

    private let replyLock = NSLock()
    /// One parked caller waiting for the engine's reply.
    ///
    /// `armed` is what makes a timeout safe: whichever of the reply and the
    /// timer gets there first flips it, so a late reply is dropped instead
    /// of being handed to whichever query happens to be next.
    private final class ReplySlot: @unchecked Sendable {
        // Only ever touched under replyLock, so handing the slot to the
        // timer queue is safe even though the compiler cannot see that.
        let cont: CheckedContinuation<Any?, Never>
        var armed = true
        init(_ cont: CheckedContinuation<Any?, Never>) { self.cont = cont }
    }

    private var pendingReplies: [ReplySlot] = []

    private let runLock = NSLock()
    private var _running = false
    private var isRunning: Bool {
        runLock.lock(); defer { runLock.unlock() }
        return _running
    }

    private let readyLock = NSLock()
    private var _ready = false
    var isReady: Bool {
        readyLock.lock(); defer { readyLock.unlock() }
        return _ready
    }

    private var thread: Thread?
    /// The live runtime. Written on the engine thread at boot and cleared
    /// there at shutdown, but READ from any caller thread — so it needs a
    /// lock. Without one, a caller between `start()` and the new thread's
    /// boot can pick up the PREVIOUS runtime and evaluate on it from the
    /// wrong thread, which the ABI forbids outright.
    private let bridgeLock = NSLock()
    private var _bridge: SofuuBridge?

    private var bridge: SofuuBridge? {
        get { bridgeLock.lock(); defer { bridgeLock.unlock() }; return _bridge }
        set { bridgeLock.lock(); _bridge = newValue; bridgeLock.unlock() }
    }

    /// The C ABI version of the linked libsofuu.
    var abiVersion: Int { SofuuBridge.abiVersion() }

    init(homeDir: String, projectDir: String) {
        self.homeDir = homeDir
        self.projectDir = projectDir
    }

    // ── Boot ─────────────────────────────────────────────────────────

    func start() {
        runLock.lock()
        if _running { runLock.unlock(); return }
        _running = true
        runLock.unlock()
        let t = Thread { [weak self] in self?.engineMain() }
        t.name = "sofuu-engine"
        t.stackSize = 8 << 20
        thread = t
        t.start()
    }

    func shutdown() {
        runLock.lock()
        if !_running { runLock.unlock(); return }
        _running = false
        runLock.unlock()
        mailboxCond.lock()
        mailboxCond.broadcast()
        mailboxCond.unlock()
        // Do not join from the engine thread itself.
        if Thread.current != thread {
            let deadline = Date().addingTimeInterval(4)
            while thread?.isFinished == false && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.02)
            }
        }
        thread = nil
    }

    private func log(_ msg: String) {
        #if DEBUG
        NSLog("[sofuu-engine] %@", msg)
        #endif
    }

    private func engineMain() {
        do {
            try? FileManager.default.createDirectory(atPath: projectDir, withIntermediateDirectories: true)
            log("boot: home=\(homeDir) project=\(projectDir) abi=\(SofuuBridge.abiVersion())")
            // Real $HOME on the Mac — ~/.sofuu is the CLI's shared directory
            // (no restamping like the iOS container pin), and config_root is
            // also passed explicitly (the documented override).
            SofuuBridge.setHome(homeDir)
            SofuuBridge.chdir(projectDir)
            // Engine console output → the device log (filter `sofuu-js`).
            // Installed BEFORE rt_new so boot warnings are not lost.
            SofuuBridge.setLogEnabled(true)

            var cfg: JSONObject = [:]
            cfg["embedded"] = true
            cfg["config_root"] = (homeDir as NSString).appendingPathComponent(".sofuu")
            cfg["provider"] = ""
            cfg["model"] = ""
            // Boot-inject every Keychain key the Tauri app would have
            // injected (desktop engine.rs step 5a): the runtime's
            // thread-local settings snapshot is fixed at rt_new, so
            // post-boot writes to the process-global key list are invisible
            // to this thread. The config file stays the per-turn channel
            // (chat.js re-reads it); this covers the boot instant and noise
            // sources outside the chat turn engine.
            var keys: JSONObject = [:]
            if let raw = try? String(contentsOf: MacPaths.configFile, encoding: .utf8),
               let obj = parseJSONObject(raw) {
                for e in (obj.optArray("providers") ?? []).compactMap({ $0 as? JSONObject }) {
                    let n = e.optString("name")
                    let k = e.optString("api_key")
                    if !n.isEmpty, !k.isEmpty { keys[n] = k }
                }
            }
            cfg["api_keys"] = keys

            log("boot: rt_new")
            let rt = try SofuuBridge.create(jsonEncode(cfg))
            bridge = rt
            hostToolsLock.lock(); _hostTools.removeAll(); hostToolsLock.unlock()
            log("boot: rt created")

            // Bootstrap (event + reply queues) and chat.init must land before
            // the pump starts, so no event is lost and the very first drain
            // sees an initialized engine.
            enqueue { [weak self] in
                guard let self else { return }
                self.log("boot: bootstrap…")
                let boot = rt.eval(Self.bootstrapJS)
                self.log("boot: bootstrap=\(boot ?? "nil")")
                // Host tools BEFORE chat.init: the runtime publishes each
                // one into both __sofuu_host_tools (resolved when a turn
                // runs) and sofuu.tools.TOOLS (resolved when an agent is
                // defined), and registering after init would leave the
                // first turn's agent without them.
                let registered = MacTools.registerAll(into: rt)
                // Recorded here rather than through registerHostTool: boot
                // hands the bridge straight to MacTools, so the engine
                // wrapper was never on the path and the list stayed empty.
                hostToolsLock.lock(); _hostTools = registered; hostToolsLock.unlock()
                self.log("boot: host tools = \(registered)")
                var initCfg: JSONObject = [:]
                initCfg["host"] = "macos"
                initCfg["project"] = self.projectDir
                initCfg["permissions"] = "prompt"
                // Approvals wait for a human, not 30s — the desktop's value.
                initCfg["toolTimeoutMs"] = 3_600_000
                initCfg["pastTurns"] = 40          // desktop resume depth
                let r = rt.eval("sofuu.chat.init(\(jsonEncode(initCfg)))")
                self.log("boot: chat.init=\(r ?? "nil")")
                if boot == nil || r == nil {
                    self.emit(.bootError("engine boot failed (chat.js unavailable)"))
                } else {
                    self.readyLock.lock(); self._ready = true; self.readyLock.unlock()
                    self.emit(.engineReady)
                }
            }

            // Pump loop: commands first, then a drain of both JS queues.
            // 20 ms cadence: ~50 UI updates/s makes streamed text read as a
            // smooth typewriter instead of chunky bursts (the ABI tick is
            // cheap); idle cost is negligible.
            while isRunning {
                if let cmd = dequeue(timeout: 0.02) {
                    if !isRunning { break }
                    cmd()
                }
                if !isRunning { break }
                drainEvents(rt)
            }
            log("pump exit")

            // Free the runtime HERE, while this is still the thread that
            // created it — docs/EMBEDDING.md: a runtime belongs to exactly
            // one thread, so it has to be released from that thread. Not
            // doing this leaked an entire QuickJS heap + libuv loop on
            // every shutdown, and restartEngineForEnv() can be tapped
            // repeatedly from Settings.
            releaseRuntime()
        } catch {
            emit(.bootError("\(error)"))
            readyLock.lock(); _ready = false; readyLock.unlock()
            releaseRuntime()
        }
    }

    /// Tear the runtime down and strand nothing.
    ///
    /// The free itself must happen on the engine thread (see above), so
    /// `engineMain` is what calls this. What it does here is unblock
    /// every caller still parked on a reply: without that, a `Task` that
    /// was awaiting when the engine went away waits forever, and the
    /// `importLocalModel` flow would sit at "checking it…" indefinitely.
    private func releaseRuntime() {
        bridgeLock.lock()
        let rt = bridge
        bridge = nil
        bridgeLock.unlock()
        rt?.close()

        replyLock.lock()
        let stranded = pendingReplies
        pendingReplies.removeAll()
        replyLock.unlock()
        for slot in stranded where slot.armed {
            slot.armed = false
            slot.cont.resume(returning: nil)
        }
    }

    // ── Mailbox ──────────────────────────────────────────────────────

    private func enqueue(_ fn: @escaping () -> Void) {
        mailboxCond.lock()
        mailbox.append(fn)
        mailboxCond.signal()
        mailboxCond.unlock()
    }

    private func dequeue(timeout: TimeInterval) -> (() -> Void)? {
        mailboxCond.lock()
        defer { mailboxCond.unlock() }
        if mailbox.isEmpty {
            mailboxCond.wait(until: Date().addingTimeInterval(timeout))
        }
        if mailbox.isEmpty { return nil }
        return mailbox.removeFirst()
    }

    /// Enqueue a block that runs on the engine thread with the runtime.
    /// The bridge's calls surface failures as `nil`, never as throws — the
    /// C ABI's contract is an error envelope, not an exception.
    private func onEngine(_ block: @escaping (SofuuBridge) -> Void) {
        enqueue { [weak self] in
            guard let self, let rt = self.bridge else { return }
            block(rt)
        }
    }

    // ── Event pump ───────────────────────────────────────────────────

    /// Drain both JS queues; deliver events to observers and complete pending
    /// reply continuations. Runs on the engine thread.
    private func drainEvents(_ rt: SofuuBridge) {
        // evalNb, NOT eval: the blocking eval drains the whole libuv loop,
        // so a drain during an active turn would run it to completion and
        // hand every delta to the UI at once. One non-blocking tick per
        // pump iteration advances the turn by a step (the Android pump's
        // design — sofuu_rt_eval_nb exists for exactly this).
        //
        // evalNb hands back the {ok,result} ENVELOPE like eval does —
        // unwrap result before parsing the {ev,rp} payload, or nothing is
        // ever drained (events and replies silently never arrive).
        guard let raw = rt.evalNb(Self.drainJS),
              let envelope = parseJSONObject(raw),
              envelope.optBool("ok"),
              let payload = envelope.optString("result").nilIfEmpty,
              let obj = parseJSONObject(payload) else { return }

        if let rp = obj.optArray("rp") {
            for item in rp {
                guard let s = item as? String else { continue }
                completeNext(parseReply(s))
            }
        }

        guard let ev = obj.optArray("ev") else { return }
        for item in ev {
            guard let e = item as? JSONObject else { continue }
            if let parsed = parseEvent(e) { emit(parsed) }
        }
    }

    private func completeNext(_ value: Any?) {
        replyLock.lock()
        let slot = pendingReplies.isEmpty ? nil : pendingReplies.removeFirst()
        replyLock.unlock()
        guard let slot, slot.armed else { return }
        slot.armed = false
        slot.cont.resume(returning: value)
    }

    /// Deliver to observers on the main thread (UI state lives there).
    private func emit(_ ev: EngineEvent) {
        observersLock.lock()
        let obs = observers
        observersLock.unlock()
        DispatchQueue.main.async { for o in obs { o(ev) } }
    }

    private func parseEvent(_ e: JSONObject) -> EngineEvent? {
        let kind = e.optString("kind")
        let p = e.optObject("payload") ?? [:]
        switch kind {
        case "ctx":
            return .ctx(
                used: max(0, p.optLong("used")),
                window: max(0, p.optLong("window")),
                source: p.optString("source"),
                model: p.optString("model").nilIfEmpty,
                sid: p.optString("sid").nilIfEmpty,
                clampedConfig: p.optBool("clampedConfig"),
                cacheHit: p.has("cacheHit") && !(p["cacheHit"] is NSNull) ? p.optDouble("cacheHit") : nil,
                sections: p.optObject("sections"))
        case "start":
            return .turnStarted(sid: p.optString("sessionId").nilIfEmpty)
        case "done":
            return .turnFinished(
                error: p.optObject("error")?.optString("message").nilIfEmpty,
                usage: parseUsage(p))
        case "approval_request":
            return .approvalRequested(
                id: p.optString("id"),
                tool: p.optString("tool").nilIfEmpty ?? "tool",
                args: p.optObject("args") ?? [:],
                risky: p.optBool("risky", true))
        case "approval_resolved":
            return .approvalResolved(id: p.optString("id"))
        case "warn":
            return .warn(p.optString("message"))
        default:
            return kind.isEmpty ? nil : .step(kind: kind, raw: e)
        }
    }

    /// A reply is JSON-encoded once by `__reply_put`. When the queried
    /// expression already produced a JSON STRING (every chat.js read is
    /// `JSON.stringify(...)`), the wire value is that string encoded again —
    /// so unwrap a String result one more time.
    private func parseReply(_ text: String) -> Any? {
        let first = parseJSONFragment(text)
        if let s = first as? String {
            return parseJSONFragment(s) ?? s
        }
        return first
    }

    private func parseUsage(_ p: JSONObject) -> Usage? {
        let u = p.optObject("usage") ?? p
        let servedBy = u.optString("servedBy").nilIfEmpty
        let prompt = u.has("promptTokens") ? u.optLong("promptTokens", -1) : -1
        let completion = u.has("completionTokens") ? u.optLong("completionTokens", -1) : -1
        let cost = u.has("costUsd") ? u.optDouble("costUsd", -1) : -1
        let steps = u.has("steps") ? u.optInt("steps", -1) : -1
        if prompt < 0 && completion < 0 && cost < 0 && servedBy == nil && steps < 0 { return nil }
        return Usage(
            promptTokens: prompt >= 0 ? prompt : nil,
            completionTokens: completion >= 0 ? completion : nil,
            costUsd: cost >= 0 ? cost : nil,
            steps: steps >= 0 ? steps : nil,
            servedBy: servedBy)
    }

    // ── Public surface (safe from any thread) ────────────────────────

    func submitTurn(_ text: String, noTools: Bool, images: [String] = []) {
        onEngine { rt in
            let textJs = jsonQuote(text)
            let noToolsJs = noTools ? "true" : "false"
            let imagesJs = jsonEncode(images)
            // evalNb: the submit script arms a promise; a blocking eval
            // would drain the loop here and starve this mailbox for the
            // whole turn.
            _ = rt.evalNb(
                "(function(){ try {" +
                    "sofuu.chat.submit(\(textJs), {onEvent: globalThis.__sofuu_push, noTools: \(noToolsJs), images: \(imagesJs)})" +
                    ".then(function(){}, function(e){ globalThis.__sofuu_push({kind:'done', payload:{error:{message:String(e&&e.message||e)}}}); });" +
                    "return true;" +
                    "} catch(e) { globalThis.__sofuu_push({kind:'done', payload:{error:{message:String(e&&e.message||e)}}}); return false; } })()")
        }
    }

    func cancelTurn() {
        onEngine { rt in _ = rt.evalNb("__host_poke('{\"type\":\"cancel\"}')") }
    }

    func resolveApproval(_ id: String, allow: Bool, always: Bool) {
        onEngine { rt in
            var payload: JSONObject = [:]
            payload["type"] = "approval"
            payload["id"] = id
            payload["allow"] = allow
            payload["always"] = always
            _ = rt.evalNb("__host_poke(\(jsonQuote(jsonEncode(payload))))")
        }
    }

    // ── Queries (async) ──────────────────────────────────────────────

    func compactSession() async -> JSONObject? {
        await queryObject("Promise.resolve(sofuu.chat.compact())")
    }

    /// The engine's chat.sessions() returns {sessions:[…]} on current
    /// chat.js (a bare array on older builds) — hand back the parsed value
    /// RAW and let the store normalize both shapes.
    func sessions() async -> Any? {
        await query("JSON.stringify((sofuu.chat && sofuu.chat.sessions) ? sofuu.chat.sessions() : [])")
    }

    func sessionTurns(_ id: String) async -> [Any]? {
        await queryArray("JSON.stringify((sofuu.chat && sofuu.chat.sessionTurns) ? sofuu.chat.sessionTurns(\(jsonQuote(id))) : [])")
    }

    func newSession() async -> JSONObject? {
        await queryObject("JSON.stringify((sofuu.chat && sofuu.chat.newSession) ? sofuu.chat.newSession() : {error:'chat engine unavailable'})")
    }

    func resumeSession(_ id: String) async -> JSONObject? {
        await queryObject("JSON.stringify((sofuu.chat && sofuu.chat.resume) ? sofuu.chat.resume(\(jsonQuote(id))) : {error:'chat engine unavailable'})")
    }

    func setProject(_ path: String) async -> JSONObject? {
        projectDir = path
        return await queryObject("JSON.stringify((sofuu.chat && sofuu.chat.setProject) ? sofuu.chat.setProject(\(jsonQuote(path))) : {error:'chat engine unavailable'})")
    }

    func setPermissions(_ profile: String) async -> JSONObject? {
        await queryObject("JSON.stringify((sofuu.chat && sofuu.chat.setPermissions) ? sofuu.chat.setPermissions(\(jsonQuote(profile))) : {error:'chat engine unavailable'})")
    }

    func chatState() async -> JSONObject? {
        await queryObject("JSON.stringify((sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {error:'chat engine unavailable'})")
    }

    func deleteSession(_ id: String) async -> JSONObject? {
        await queryObject("JSON.stringify((sofuu.chat && sofuu.chat.deleteSession) ? sofuu.chat.deleteSession(\(jsonQuote(id))) : {ok:false, error:'chat engine unavailable'})")
    }

    func clearSessions() async -> JSONObject? {
        await queryObject("JSON.stringify((sofuu.chat && sofuu.chat.clearSessions) ? sofuu.chat.clearSessions() : {ok:false, error:'chat engine unavailable'})")
    }

    func usageReport() async -> JSONObject? {
        await queryObject("JSON.stringify((sofuu.chat && sofuu.chat.usage) ? sofuu.chat.usage() : {error:'chat engine unavailable'})")
    }

    func brainRemember(_ fact: String) async -> JSONObject? {
        await queryObject("JSON.stringify((function(){ try { if (sofuu.memory && sofuu.memory.remember) { sofuu.memory.remember(\(jsonQuote(fact))); return {ok:true}; } return {ok:false, error:'brain unavailable (enable /brain)'}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())")
    }

    func brainWhy() async -> JSONObject? {
        await queryObject("JSON.stringify((function(){ try { var st = (sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {}; return {ok:true, lastRecall:(st && st.lastRecall) || null, recall:(st && st.lastRecallHits) || null}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())")
    }

    func mlInfo() async -> JSONObject? {
        await queryObject("JSON.stringify((function(){ try { var info = (sofuu.ml && sofuu.ml.info) ? sofuu.ml.info() : null; return {ok:true, info:info}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())")
    }

    func contextDump() async -> JSONObject? {
        await queryObject("JSON.stringify((function(){ try { var st = (sofuu.chat && sofuu.chat.state) ? sofuu.chat.state() : {}; return {ok:true, state:st}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())")
    }

    func listTools() async -> JSONObject? {
        await queryObject("JSON.stringify((function(){ try { var defs = []; if (sofuu.tools && sofuu.tools.TOOLS) { for (var k in sofuu.tools.TOOLS) defs.push(k); } var mcp = []; try { if (sofuu.chat && sofuu.chat.state) { var st = sofuu.chat.state(); if (st && st.mcpTools) mcp = st.mcpTools; } } catch(e){} return {ok:true, builtin:defs, mcp:mcp}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())")
    }

    func listAgents() async -> JSONObject? {
        await queryObject("JSON.stringify((function(){ try { var list = (sofuu.agent && sofuu.agent.list) ? sofuu.agent.list() : []; return {ok:true, agents:list}; } catch(e){ return {ok:false, error:String(e&&e.message||e)}; } })())")
    }

    // ── Local GGUF models ──────────────────────────────────────────────

    /// Read the real caps out of a GGUF header: `ai.modelCaps` opens the
    /// file and returns its context window, max output, and whether it
    /// carries a thinking channel. Mirrors what the CLI's `/model x.gguf`
    /// does so a phone model is configured the same way a desktop one is.
    func localModelCaps(_ path: String) async -> JSONObject? {
        await evalJson("(function(){ try { var c = sofuu.ai.modelCaps(\(jsonQuote(path))); return { ok:true, caps:c||{} }; } catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()")
    }

    /// Confirm the engine can actually open a model before the app points
    /// its config at it. `modelCaps` only parses the header, so this also
    /// runs one short generation — that is what surfaces a model whose
    /// tensors are the wrong shape or unsupported.
    ///
    /// Synchronous on the engine thread by design; a local model has no
    /// network round-trip to wait for.
    func probeLocalModel(_ path: String, prompt: String = "hi", maxTokens: Int = 8) async -> JSONObject? {
        await evalJson("(function(){ try { var t = sofuu.ai.inferLocal(\(jsonQuote(path)), \(jsonQuote(prompt)), {maxTokens:\(max(1, min(maxTokens, 64))), temperature:0}); return { ok:true, text: String(t == null ? '' : t) }; } catch(e){ return { ok:false, error:String(e&&e.message||e) }; } })()", timeout: 180)
    }

    // ── Voice (sofuu.ai.transcribe) ─────────────────────────────────────

    /// Does this engine build ship the ai.transcribe surface at all?
    /// The app-side half of the voice gate — the engine itself STILL
    /// rejects Local/Anthropic providers and missing models at call time;
    /// the store adds the provider-shaped half of the check.
    func voiceSurfaceProbe() async -> Bool {
        await evalJson("(function(){ try { return {ok:(typeof sofuu==='object'&&!!sofuu&&typeof sofuu.ai==='object'&&!!sofuu.ai&&typeof sofuu.ai.transcribe==='function')}; } catch(e) { return {ok:false}; } })()", timeout: 15)
            .flatMap { $0.optBool("ok") } ?? false
    }

    /// WAV bytes (base64) → transcript, through the engine's
    /// OpenAI-compatible audio funnel (sofuu.ai.transcribe returns a
    /// Promise). Same pattern as probeLocalModel: evalJson drives the
    /// promise over evalNb + the reply queue, so the engine loop keeps
    /// ticking (streaming, approvals) during the upload — a blocking eval
    /// here would freeze the whole runtime for seconds.
    func voiceTranscribe(_ audioBase64: String) async -> JSONObject? {
        await evalJson("(function(){ try { return sofuu.ai.transcribe({audio_b64:\(jsonQuote(audioBase64))}).then(function(r){ return {ok:true, text:(r&&r.text!=null)?String(r.text):''}; }); } catch(e) { return {ok:false, error:String(e&&e.message||e)}; } })()", timeout: 180)
    }

    // ── Host tools ─────────────────────────────────────────────────────

    /// Native tools registered on the engine thread at boot.
    ///
    /// Surfaced so Settings can show what the phone adds over a terminal,
    /// and so a host tool registration can be verified instead of assumed.
    private let hostToolsLock = NSLock()
    private var _hostTools: [String] = []
    var hostTools: [String] {
        hostToolsLock.lock(); defer { hostToolsLock.unlock() }
        return _hostTools
    }

    /// Expose a native capability to the agent as a tool. Engine thread
    /// only, and only while the runtime is alive.
    func registerHostTool(name: String, description: String, parametersJson: String, tool: @escaping SofuuBridge.HostTool) -> Bool {
        guard let rt = bridge else { return false }
        let ok = rt.registerHostTool(name: name, description: description, parametersJson: parametersJson, tool: tool)
        if ok {
            hostToolsLock.lock(); _hostTools.append(name); hostToolsLock.unlock()
        }
        return ok
    }

    /// Settle a call whose host tool returned nil. Safe from any thread.
    @discardableResult
    func resolveHostTool(_ callId: UInt64, resultJson: String? = nil, error: String? = nil) -> Bool {
        bridge?.resolveHostTool(callId, resultJson: resultJson, error: error) ?? false
    }

    // ── Provider model lists (ported from commands.rs) ───────────────

    /// Fetch a provider's model list through the engine's own fetch (keys
    /// never leave the runtime). Returns {ok, models, note, error}.
    func fetchProviderModels(_ name: String, endpoint: String, profile: String, key: String) async -> JSONObject? {
        guard let url = Self.modelsUrl(endpoint, profile) else {
            return ["ok": false, "error": "no models URL"]
        }
        let urlJs = jsonQuote(url)
        let keyJs = jsonQuote(key)
        return await queryObject(
            """
            (function(){ var url = \(urlJs); var key = \(keyJs);
              var headers = key ? { 'Authorization': 'Bearer ' + key } : {};
              /* Exactly one answer per query, whichever arrives first. A
                 second one would be consumed by the NEXT caller: replies
                 are matched by arrival order, not by query, so a slow
                 provider's late timeout frame would hand its error to an
                 unrelated query and shift every later answer by one. */
              var done = false, timer = null;
              function settle(o){ if (done) return; done = true; if (timer) clearTimeout(timer); __reply_put(JSON.stringify(o)); }
              timer = setTimeout(function () { settle({ok:false, error:'timeout', models:[]}); }, 12000);
              try {
                fetch(url, { headers: headers }).then(function (res) {
                  if (!res.ok) { settle({ok:false, error:'HTTP ' + res.status, models:[]}); return; }
                  return res.json();
                }).then(function (j) {
                  var raw = (j && j.data && Array.isArray(j.data)) ? j.data
                          : (j && j.models && Array.isArray(j.models)) ? j.models : [];
                  var arr = raw.map(function (m) { return (m && (m.id || m.name)) || m; })
                                 .filter(function (x) { return x && String(x).trim(); });
                  settle({ok:true, models: arr, raw: JSON.stringify(j)});
                }).catch(function (e) {
                  settle({ok:false, error:String(e&&e.message||e), models:[]});
                });
              } catch (e) {
                settle({ok:false, error:String(e&&e.message||e), models:[]});
              } })()
            """
        )
    }

    // ── Query plumbing ───────────────────────────────────────────────

    /// Enqueue a JS expression whose *promise or value* is answered through
    /// `__reply_put`. Resolves with the parsed JSON value.
    private func query(_ expr: String) async -> Any? {
        await withCheckedContinuation { (cont: CheckedContinuation<Any?, Never>) in
            replyLock.lock()
            pendingReplies.append(ReplySlot(cont))
            replyLock.unlock()
            enqueue { [weak self] in
                guard let self else { return }
                guard let rt = self.bridge else {
                    self.completeNext(nil)
                    return
                }
                _ = rt.evalNb(self.queryScript(expr))
            }
        }
    }
    private func queryObject(_ expr: String) async -> JSONObject? { await query(expr) as? JSONObject }
    private func queryArray(_ expr: String) async -> [Any]? { await query(expr) as? [Any] }

    /// Evaluate a JS expression that already evaluates to a JSON object,
    /// without the `{ok,result}` envelope the query channel wraps.
    ///
    /// `RuntimeSurface` uses this because those snippets build their own
    /// `{ok:..., error:...}` object so a failure inside the brain or the
    /// ML gates is reported rather than collapsing to nil — the whole
    /// difference between "the brain is off" and "the call threw".
    func evalJson(_ script: String) async -> JSONObject? {
        guard let raw = await queryValue(script) else { return nil }
        return asJSONObject(raw)
    }

    /// `evalJson` with a bound, because a headless agent run takes tens of
    /// seconds on a phone and must not hang a caller forever.
    func evalJson(_ script: String, timeout: TimeInterval) async -> JSONObject? {
        guard let raw = await queryValue(script, timeout: timeout) else { return nil }
        return asJSONObject(raw)
    }

    /// A reply arrives either already parsed or still as JSON text,
    /// depending on whether the snippet serialized it.
    private func asJSONObject(_ raw: Any?) -> JSONObject? {
        if let obj = raw as? JSONObject { return obj }
        if let s = raw as? String { return parseJSONObject(s) }
        return nil
    }

    func queryFuture(_ expr: String, timeout: TimeInterval) async -> JSONObject? {
        await query(expr, timeout: timeout) as? JSONObject
    }

    /// Queue an expression and return its JSON text, without the envelope.
    ///
    /// Goes through the reply queue like `query` (so it lands on the engine
    /// thread) but hands back the raw string, because these snippets
    /// serialize their own object.
    private func queryValue(_ expr: String, timeout: TimeInterval = 30) async -> Any? {
        let script = "(function(){ try { Promise.resolve(eval(\(jsonQuote(expr)))).then(function (v) { " +
            "__reply_put(typeof v === 'string' ? v : JSON.stringify(v)); }, function (e) { " +
            "__reply_put(JSON.stringify({ok:false, error:String(e&&e.message||e)})); }); } " +
            "catch(e) { __reply_put(JSON.stringify({ok:false, error:String(e&&e.message||e)})); } })()"

        return await withCheckedContinuation { (cont: CheckedContinuation<Any?, Never>) in
            replyLock.lock()
            pendingReplies.append(ReplySlot(cont))
            replyLock.unlock()
            enqueue { [weak self] in
                guard let self else { return }
                guard let rt = self.bridge else {
                    self.completeNext(nil)
                    return
                }
                _ = rt.evalNb(script)
            }
        }
    }

    /// `query` with a deadline.
    ///
    /// Deliberately NOT `withTaskGroup`: `query` parks on
    /// `withCheckedContinuation`, which is not cancellable, and a task
    /// group waits for every child before returning — so the group could
    /// not return until the continuation was resumed anyway, and the
    /// timeout never fired. Instead the continuation itself is resumed by
    /// whichever arrives first: the engine's reply, or the timer.
    ///
    /// `pendingReplies` keeps the entry either way. If the timer wins, the
    /// late reply is consumed by the next `completeNext`, so the slot is
    /// removed here and the reply is dropped rather than handed to an
    /// unrelated caller — a stale reply would otherwise shift every
    /// subsequent answer by one.
    private func query(_ expr: String, timeout: TimeInterval) async -> Any? {
        let deadline = timeout <= 0 ? .distantFuture : DispatchTime.now() + timeout
        return await withCheckedContinuation { (cont: CheckedContinuation<Any?, Never>) in
            let slot = ReplySlot(cont)

            replyLock.lock()
            pendingReplies.append(slot)
            replyLock.unlock()

            DispatchQueue.global().asyncAfter(deadline: deadline) { [weak self] in
                guard let self else { return }
                self.replyLock.lock()
                let won = slot.armed
                slot.armed = false
                if won, let idx = self.pendingReplies.firstIndex(where: { $0 === slot }) {
                    self.pendingReplies.remove(at: idx)
                }
                self.replyLock.unlock()
                if won { cont.resume(returning: nil) }
            }

            self.enqueue { [weak self] in
                guard let self else { return }
                guard let rt = self.bridge else { self.completeNext(nil); return }
                _ = rt.evalNb(self.queryScript(expr))
            }
        }
    }

    private func queryScript(_ expr: String) -> String {
        "(function(){ try { Promise.resolve(eval(\(jsonQuote(expr)))).then(function (v) { " +
            "__reply_put(JSON.stringify(v)); }, function (e) { " +
            "__reply_put(JSON.stringify({ok:false, error:String(e&&e.message||e)})); }); } " +
            "catch(e) { __reply_put(JSON.stringify({ok:false, error:String(e&&e.message||e)})); } })()"
    }

    // ── JS bootstrap ─────────────────────────────────────────────────

    private static let bootstrapJS = """
        (function(){
          globalThis.__evq = [];
          globalThis.__sofuu_push = function (e) { try { globalThis.__evq.push(e); } catch (x) {} };
          globalThis.__rq = [];
          globalThis.__reply_put = function (j) { try { globalThis.__rq.push(j); } catch (x) {} };
          return true;
        })()
        """

    private static let drainJS = """
        (function(){
          var ev = globalThis.__evq.splice(0, globalThis.__evq.length);
          var rp = globalThis.__rq.splice(0, globalThis.__rq.length);
          return JSON.stringify({ev: ev, rp: rp});
        })()
        """

    // ── models_url / profile detection (ported from commands.rs) ─────

    static func isAnthropic(_ endpoint: String, _ profile: String) -> Bool {
        let prof = profile.trimmingCharacters(in: .whitespaces).lowercased()
        let low = endpoint.lowercased()
        return prof == "anthropic" || (prof.isEmpty && (low.contains("anthropic") || low.contains("claude")))
    }

    static func isLocal(_ endpoint: String, _ profile: String) -> Bool {
        let prof = profile.trimmingCharacters(in: .whitespaces).lowercased()
        let low = endpoint.lowercased()
        return prof == "local" || (prof.isEmpty && (low.contains("11434") || low.contains("localhost") || low.contains("127.0.0.1")))
    }

    static func modelsUrl(_ endpoint: String, _ profile: String) -> String? {
        if isAnthropic(endpoint, profile) { return nil }
        if isLocal(endpoint, profile) {
            let trimmed = endpoint.trimmingCharacters(in: .whitespaces)
            let root = trimmed.isEmpty ? "http://127.0.0.1:11434" : trimmed.trimmingTrailingSlashes
            return "\(root)/api/tags"
        }
        let ep = endpoint.trimmingCharacters(in: .whitespaces).trimmingTrailingSlashes
        if ep.isEmpty { return nil }
        let root = stripUntilCore(ep, ["/chat/completions", "/completions"])
        return "\(root)/models"
    }

    private static func stripUntilCore(_ endpoint: String, _ suffixes: [String]) -> String {
        for s in suffixes {
            if let r = endpoint.range(of: s), r.lowerBound != endpoint.startIndex {
                return String(endpoint[endpoint.startIndex..<r.lowerBound])
            }
        }
        return endpoint
    }
}

// MARK: - Events

/// Everything the UI can observe from the engine.
enum EngineEvent {
    case engineReady
    case bootError(String)
    case turnStarted(sid: String?)
    case turnFinished(error: String?, usage: Usage?)
    case ctx(used: Int, window: Int, source: String, model: String?, sid: String?,
             clampedConfig: Bool = false, cacheHit: Double? = nil, sections: JSONObject? = nil)
    case approvalRequested(id: String, tool: String, args: JSONObject, risky: Bool)
    case approvalResolved(id: String)
    case warn(String)
    case step(kind: String, raw: JSONObject)
}

// MARK: - Small helpers

extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
    var trimmingTrailingSlashes: String {
        var s = self
        while s.hasSuffix("/") { s.removeLast() }
        return s
    }
}
