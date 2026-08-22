// tests/fs_test.js — Async file I/O test (assert-based, A0.2)
// Tests: writeFile, readFile, appendFile, exists, readdir, mkdir, rm

const results = { passed: 0, failed: 0 };
function assert(label, cond) {
    if (cond) { console.log("  ✅ " + label); results.passed++; }
    else { console.error("  ❌ FAIL: " + label); results.failed++; }
}

async function main() {
    console.log("=== Sofuu Async File I/O Test ===\n");

    // Write a file
    await sofuu.fs.writeFile("/tmp/sofuu_test.txt", "Hello from Sofuu!\nLine 2\nLine 3");
    assert("writeFile succeeded", true);

    // Read it back
    const content = await sofuu.fs.readFile("/tmp/sofuu_test.txt");
    assert("readFile returns content", content && content.includes("Hello from Sofuu!"));

    // Append to it
    await sofuu.fs.appendFile("/tmp/sofuu_test.txt", "\nAppended line!");
    assert("appendFile succeeded", true);

    // Read again
    const updated = await sofuu.fs.readFile("/tmp/sofuu_test.txt");
    assert("readFile after append has new line", updated && updated.includes("Appended line!"));

    // Check exists
    const exists = await sofuu.fs.exists("/tmp/sofuu_test.txt");
    assert("exists('/tmp/sofuu_test.txt') === true", exists === true);

    const noexist = await sofuu.fs.exists("/tmp/definitely_not_here_XYZ.txt");
    assert("exists('/tmp/not_here') === false", noexist === false);

    // readdir
    const files = await sofuu.fs.readdir("/tmp");
    const hasSofuuFile = files.some(f => f === "sofuu_test.txt");
    assert("readdir /tmp contains sofuu_test.txt", hasSofuuFile);

    // mkdir + rm
    await sofuu.fs.mkdir("/tmp/sofuu_test_dir");
    assert("mkdir succeeded", true);
    const dirExists = await sofuu.fs.exists("/tmp/sofuu_test_dir");
    assert("mkdir dir exists", dirExists === true);
    await sofuu.fs.rm("/tmp/sofuu_test_dir");
    assert("rm succeeded", true);
    const dirGone = await sofuu.fs.exists("/tmp/sofuu_test_dir");
    assert("rm dir gone", dirGone === false);

    console.log("\n=== RESULTS ===");
    console.log(`Passed: ${results.passed} | Failed: ${results.failed}`);
    if (results.failed > 0) process.exit(1);
    console.log("\n✅ All fs tests PASSED");
}

main().catch(err => {
    console.error("Test failed:", err);
    process.exit(1);
});
