// tests/fs_test.js — Async file I/O test (assert-based, A0.2).
// Tests: writeFile, readFile, appendFile, exists, readdir, mkdir, rm.
// P2-32 (AUDIT-2026-09-01): paths are timestamp-unique (like agent_test.js)
// and readdir asserts INSIDE the test's own directory — the old fixed
// /tmp/sofuu_test* paths went stale-green across runs and the readdir(/tmp)
// scan was order/parallelism dependent.

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

async function main() {
    console.log("=== Sofuu Async File I/O Test ===\n");

    const TDIR = "/tmp/sofuu_fs_test_" + Date.now();
    await sofuu.fs.mkdir(TDIR);
    assert("mkdir (test root) succeeded", true);
    assert("test root exists", (await sofuu.fs.exists(TDIR)) === true);

    const FILE = TDIR + "/data.txt";

    // Write a file
    await sofuu.fs.writeFile(FILE, "Hello from Sofuu!\nLine 2\nLine 3");
    assert("writeFile succeeded", true);

    // Read it back
    const content = await sofuu.fs.readFile(FILE);
    assert("readFile returns content", content && content.includes("Hello from Sofuu!"));

    // Append to it
    await sofuu.fs.appendFile(FILE, "\nAppended line!");
    assert("appendFile succeeded", true);

    // Read again
    const updated = await sofuu.fs.readFile(FILE);
    assert("readFile after append has new line", updated && updated.includes("Appended line!"));

    // Check exists
    const exists = await sofuu.fs.exists(FILE);
    assert("exists(test file) === true", exists === true);

    const noexist = await sofuu.fs.exists(TDIR + "/definitely_not_here_XYZ.txt");
    assert("exists(missing file) === false", noexist === false);

    // readdir — scoped to the test's own directory
    const files = await sofuu.fs.readdir(TDIR);
    const hasSofuuFile = files.some(f => f === "data.txt");
    assert("readdir of the test dir contains data.txt", hasSofuuFile);

    // mkdir + rm inside the test root
    const SUB = TDIR + "/subdir";
    await sofuu.fs.mkdir(SUB);
    assert("mkdir subdir succeeded", true);
    const dirExists = await sofuu.fs.exists(SUB);
    assert("mkdir subdir exists", dirExists === true);
    await sofuu.fs.rm(SUB);
    assert("rm subdir succeeded", true);
    const dirGone = await sofuu.fs.exists(SUB);
    assert("rm subdir gone", dirGone === false);

    // Clean the whole test root (leaving stale dirs in /tmp was part of
    // the old fixed-path problem — this run removes everything it made).
    // rm is file/EMPTY-dir only (no recursive flag), so remove the file
    // first, then the root.
    await sofuu.fs.rm(FILE);
    await sofuu.fs.rm(TDIR);
    assert("test root cleaned up", (await sofuu.fs.exists(TDIR)) === false);

    console.log("\n=== RESULTS ===");
    console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
    if (results.failed > 0) process.exit(1);
    console.log("\n✅ All fs tests PASSED");
}

main().catch(err => {
    console.error("Test failed:", err);
    process.exit(1);
});
