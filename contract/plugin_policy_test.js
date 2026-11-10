// Real R7 evaluator and in-process plugin tool; no network or model backend.
// Run with: sofuu run contract/plugin_policy_test.js
(async function () {
  var calls = 0;
  var tool = sofuu.plugins.defineTool({
    name: 'policy_contract_arithmetic',
    schema: { type: 'object', properties: {} },
    run: function () { calls++; return { value: 2 + 3 }; }
  });

  function assert(condition, message) {
    if (!condition) {
      console.error('FAIL: ' + message);
      process.exit(1);
    }
    console.log('PASS: ' + message);
  }

  sofuu.policy.clear();
  var allowed = await tool.execute({});
  assert(allowed.ok === true && calls === 1, 'real policy permits the tool');

  sofuu.policy.set({ tools: { deny: ['policy_contract_arithmetic'] } });
  var denied = await tool.execute({});
  assert(denied.ok === false && denied.error.code === 'E_DENIED' &&
    denied.error.rule === 'R7.6/tools.deny' && calls === 1,
    'real policy denial prevents handler execution');
  var deniedInvoke = await sofuu.plugins.invoke('policy_contract_arithmetic', {}, { approved: true });
  assert(deniedInvoke.ok === false && deniedInvoke.error.code === 'E_DENIED' && calls === 1,
    'a policy denial takes precedence over host approval');

  sofuu.policy.set({ approval: { mode: 'always' } });
  var asked = await sofuu.plugins.invoke('policy_contract_arithmetic', {});
  assert(asked.ok === false && asked.error.code === 'E_APPROVAL' && calls === 1,
    'unattended plugin invocation does not grant approval');
  var approved = await sofuu.plugins.invoke('policy_contract_arithmetic', {}, { approved: true });
  assert(approved.ok === true && calls === 2, 'explicit host approval permits an ask decision');

  sofuu.policy.clear();
  var recovered = await tool.execute({});
  assert(recovered.ok === true && calls === 3, 'clearing the policy permits the tool again');
  console.log('Plugin policy integration checks passed.');
})().catch(function (error) {
  console.error('FAIL: plugin policy check failed: ' + String(error));
  process.exit(1);
});
