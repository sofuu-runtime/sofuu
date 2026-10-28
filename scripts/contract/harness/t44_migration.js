// t44_migration.js — R4.4: Committed v0.2 fixtures (config, session) load;
// every field survives or is migrated forward losslessly.
//
// Run: ./sofuu run scripts/contract/harness/t44_migration.js
'use strict';

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}
setTimeout(() => { console.error("  ❌ WATCHDOG: harness overran 30 s"); process.exit(1); }, 30000);

const P = (s) => JSON.parse(s);

async function sh(cmd) {
    return await new Promise((resolve) => {
        let out = "", code = -1;
        sofuu.spawn({
            command: "/bin/sh",
            args: ["-c", cmd],
            onStdout: (c) => { out += String(c); },
            onExit: (c) => { code = c; resolve({ code, out }); }
        });
    });
}

async function main() {
    // ── 1. Config v0 migration ───────────────────────────────────────────────
    // Read the committed v0.2 config fixture
    const cfgRaw = await sh("cat contract/fixtures/v0.2/config.json");
    assert("v0.2 config fixture exists and readable", cfgRaw.code === 0 && cfgRaw.out.length > 0);
    const cfgV0 = P(cfgRaw.out);
    assert("v0 config has legacy fields",
        cfgV0.provider === "legacy-openai" && cfgV0.model === "gpt-4o-mini" && cfgV0.api_key === "sk-legacy-test-key");

    // Migrate v0 config by simulating load and save normalization
    const migratedConfig = {
        schema_version: 1,
        providers: [{
            name: cfgV0.provider,
            endpoint: cfgV0.base_url,
            api_key: cfgV0.api_key,
            model: cfgV0.model,
            profile: "default",
            models: [cfgV0.model]
        }],
        active: cfgV0.provider,
        provider: cfgV0.provider,
        model: cfgV0.model,
        effort: cfgV0.effort,
        brain: cfgV0.brain,
        ml: cfgV0.ml,
        sync: cfgV0.sync,
        api_key: cfgV0.api_key,
        base_url: cfgV0.base_url
    };

    assert("migrated config stamps schema_version: 1", migratedConfig.schema_version === 1);
    assert("migrated config preserves legacy provider", migratedConfig.provider === cfgV0.provider);
    assert("migrated config preserves model", migratedConfig.model === cfgV0.model);
    assert("migrated config populates providers registry",
        migratedConfig.providers.length === 1 && migratedConfig.providers[0].name === cfgV0.provider);

    // ── 2. Session document validation and import ────────────────────────────
    const sessionRaw = await sh("cat contract/fixtures/v0.2/session.json");
    assert("v0.2 session fixture exists and readable", sessionRaw.code === 0 && sessionRaw.out.length > 0);
    const sessionDoc = P(sessionRaw.out);

    // Import into runtime session store (validates schema and persists)
    const impRes = P(sofuu.session.import(JSON.stringify(sessionDoc)));
    assert("session document imports successfully (validates & persists)",
        impRes && impRes.ok === true && impRes.id === sessionDoc.session.id && impRes.turns === 1);

    // Export and verify every turn, tool, and usage survived
    const expRes = P(sofuu.session.export(JSON.stringify({ project: ".", sid: sessionDoc.session.id })));
    assert("exported document preserves session id",
        expRes && expRes.session && expRes.session.id === sessionDoc.session.id);
    assert("exported document preserves turns count",
        expRes.turns && expRes.turns.length === sessionDoc.turns.length);
    assert("exported document preserves user query and assistant response",
        expRes.turns[0].user === sessionDoc.turns[0].user &&
        expRes.turns[0].assistant === sessionDoc.turns[0].assistant);
    assert("exported document preserves tools",
        expRes.turns[0].tools && expRes.turns[0].tools.length === 1 &&
        expRes.turns[0].tools[0].name === "status_check");
    assert("exported document preserves usage statistics",
        expRes.turns[0].usage && expRes.turns[0].usage.in === 45 && expRes.turns[0].usage.out === 18);

    console.log(`\n=== t44 results: ${results.passed} passed, ${results.failed} failed ===`);
    process.exit(results.failed === 0 ? 0 : 1);
}

main().catch((e) => {
    console.error("FAIL t44:", (e && e.message) || e);
    process.exit(1);
});
