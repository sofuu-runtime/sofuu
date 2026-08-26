// Interop check: chat.js session files must be readable by the Rust mesh.
const init = sofuu.chat.init({
  host: 'interop', project: '/tmp/chat-js-smoke',
  permissions: 'prompt', toolTimeoutMs: 60000, pastTurns: 5,
});
console.log('ROOT=' + init.project);
console.log('SID=' + init.sessionId);
