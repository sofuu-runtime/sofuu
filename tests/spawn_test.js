// tests/spawn_test.js — Child process spawn test (assert-based, A0.2)

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

console.log("=== Spawn Test ===\n");

assert("sofuu.spawn is function", typeof sofuu.spawn === "function");

let stdoutData = "";
let stderrData = "";
let exitCode = null;

const p = sofuu.spawn({
    command: "sh",
    args: ["-c", "echo 'Hello from child'; sleep 0.1; echo 'Goodbye'; >&2 echo 'Error output'; exit 42"],
    onStdout: (data) => { stdoutData += data; },
    onStderr: (data) => { stderrData += data; },
    onExit: (code) => { exitCode = code; }
});

assert("spawn returned object", typeof p === "object");

// Wait for the child to finish
setTimeout(() => {
    assert("stdout received", stdoutData.includes("Hello from child"));
    assert("stderr received", stderrData.includes("Error output"));
    assert("exit code === 42", exitCode === 42);

    console.log("\n=== RESULTS ===");
    console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
    if (results.failed > 0) process.exit(1);
    console.log("\n✅ All spawn tests PASSED");
}, 500);
