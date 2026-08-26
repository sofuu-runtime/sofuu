// Smoke test: sofuu.chat turn engine boots + session mesh writes.
const assert = (cond, msg) => { if (!cond) throw new Error('FAIL: ' + msg); };

assert(sofuu.chat, 'sofuu.chat namespace exists');
assert(typeof sofuu.chat.init === 'function', 'chat.init is a function');
assert(typeof sofuu.chat.submit === 'function', 'chat.submit is a function');

const init = sofuu.chat.init({
  host: 'smoke', project: '/tmp/chat-js-smoke', permissions: 'prompt',
  toolTimeoutMs: 60000, pastTurns: 5,
});
console.log('init:', JSON.stringify(init));
assert(init.ok === true, 'init.ok');
assert(typeof init.sessionId === 'string' && init.sessionId.indexOf('s-') === 0, 'sessionId format');
// The mesh root is canonicalized (/tmp → /private/tmp on macOS) so it
// matches session.rs's password derivation byte for byte.
assert(/\/tmp\/chat-js-smoke$/.test(init.project), 'project set (canonicalized): ' + init.project);

const st = sofuu.chat.state();
console.log('state:', JSON.stringify(st));
assert(st.sessionId === init.sessionId, 'state.sessionId matches');
assert(st.permissions === 'prompt', 'permissions');
assert(st.streaming === false, 'not streaming');

// Session mesh artifacts exist.
assert(sofuu.fs.exists('/tmp/chat-js-smoke/.sofuu/sessions/registry.json'), 'registry.json written');
assert(sofuu.fs.exists('/tmp/chat-js-smoke/.sofuu/sessions/' + init.sessionId + '.qtsq'), '.qtsq written');

// Sync reads through the engine.
const turns = sofuu.chat.sessionTurns(init.sessionId);
assert(Array.isArray(turns) && turns.length === 0, 'sessionTurns empty for fresh session');

// New session replaces the id.
const ns = sofuu.chat.newSession();
assert(ns.ok === true && ns.sessionId !== init.sessionId, 'newSession mints a fresh id');

// resolveApproval with an unknown id is a clean no-op error.
const ra = sofuu.chat.resolveApproval('ap-nope', true, false);
assert(ra.ok === false, 'unknown approval rejected cleanly');

// clear + compact on empty history.
assert(sofuu.chat.clear().ok === true, 'clear works');
const comp = await sofuu.chat.compact();
assert(comp.ok === false, 'compact on empty history reports nothing to fold');

// setProject re-roots the mesh (root canonicalized, same as init).
const sp = sofuu.chat.setProject('/tmp/chat-js-smoke2');
assert(sp.ok === true && /\/tmp\/chat-js-smoke2$/.test(sp.project), 'setProject works: ' + sp.project);
assert(sofuu.fs.exists('/tmp/chat-js-smoke2/.sofuu/sessions/registry.json'), 'new project registry written');

console.log('chat.js smoke: ALL OK');
