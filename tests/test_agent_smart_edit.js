// tests/test_agent_smart_edit.js -- Phase 5 verification battery.
//
// Covers the two additions to edit_file (src/js/tools.js):
//   1. whitespace-tolerant line matching (unique-match only, re-indented to
//      the file, real bytes in the stashed diff),
//   2. the syntax dry-run (sofuu.parse / JSON.parse) that refuses to write a
//      candidate which would break a file that parses today.
//
// Run: ./sofuu run tests/test_agent_smart_edit.js   (cwd = repo root; the
// tool is jailed to cwd, so every fixture is a bare filename in this dir).

const NL = String.fromCharCode(10);
const CR = String.fromCharCode(13);
const CRLF = CR + NL;

let passed = 0;
let failed = 0;
function ok(cond, label) {
  if (cond) { passed++; console.log("  PASS " + label); }
  else { failed++; console.log("  FAIL " + label); }
}
function lines() { return Array.prototype.slice.call(arguments).join(NL); }

const FILES = [];
async function fixture(name, content) {
  await sofuu.fs.writeFile(name, content);
  if (FILES.indexOf(name) < 0) FILES.push(name);
  return name;
}
async function read(p) { return await sofuu.fs.readFile(p, "utf8"); }
function lastDiff() { return globalThis.__sofuu_last_diff || null; }
async function expectThrow(fn) {
  try { await fn(); return null; } catch (e) { return e; }
}

async function main() {
  console.log("=== Phase 5: resilient edit_file (tolerant match + syntax dry-run) ===");
  const ef = sofuu.tools.TOOLS.edit_file.execute;

  console.log("Test 1: literal path is unchanged");
  {
    const p = await fixture("smart_edit_literal.txt", lines("AA-BB-CC", ""));
    globalThis.__sofuu_last_diff = null;
    const r = await ef({ path: p, old_string: "BB", new_string: "XX" });
    ok((await read(p)) === lines("AA-XX-CC", ""), "literal edit still splices exactly");
    ok(String(r).indexOf("edited " + p + ": 1 replacement (+0 chars)") === 0, "result string shape unchanged");
    ok(String(r).indexOf("whitespace-tolerant") < 0, "literal path does not claim a tolerant match");
  }

  console.log("Test 2: whitespace-tolerant match re-indents to the file");
  {
    const p = await fixture("smart_edit_tolerant.js", lines(
      "class A {",
      "    run() {",
      "        if (ready) {",
      "            go();",
      "        }",
      "    }",
      "}", ""));
    const oldS = lines("if (ready) {", "    go();", "}");
    const newS = lines("if (ready) {", "    go();", "} else {", "    stop();", "}");
    globalThis.__sofuu_last_diff = null;
    const r = await ef({ path: p, old_string: oldS, new_string: newS });
    const want = lines(
      "class A {",
      "    run() {",
      "        if (ready) {",
      "            go();",
      "        } else {",
      "            stop();",
      "        }",
      "    }",
      "}", "");
    ok((await read(p)) === want, "block lands at the file's indentation, relative nesting kept");
    ok(String(r).indexOf("[whitespace-tolerant match]") > 0, "tolerant result is labelled");
    ok(String(r).indexOf("1 replacement") > 0, "tolerant edit reports one replacement");
    const d = lastDiff();
    ok(!!d && d.name === "edit_file", "tolerant edit still stashes a diff");
    ok(!!d && d.diff.indexOf("- " + "        if (ready) {") >= 0, "diff shows the file's real removed bytes");
    ok(!!d && d.diff.indexOf("+ " + "        } else {") >= 0, "diff shows the re-indented added bytes");
  }

  console.log("Test 3: an ambiguous tolerant match is still an error");
  {
    const p = await fixture("smart_edit_ambig.txt", lines(
      "function a() {",
      "    if (ready) {",
      "        go();",
      "    }",
      "}",
      "function b() {",
      "        if (ready) {",
      "            go();",
      "        }",
      "}", ""));
    const before = await read(p);
    const err = await expectThrow(function () {
      return ef({ path: p, old_string: lines("if (ready) {", "    go();", "}"), new_string: "NOPE" });
    });
    ok(!!err, "ambiguous tolerant match throws");
    ok(!!err && String(err.message).indexOf("ignoring indentation") > 0, "message names the relaxed match");
    ok(!!err && String(err.message).indexOf("lines 2, 7") > 0, "message lists both candidate lines");
    ok((await read(p)) === before, "ambiguous edit leaves the file untouched");
  }

  console.log("Test 4: replace_all rewrites every tolerant match in its own indent");
  {
    const p = await fixture("smart_edit_all.txt", lines(
      "function a() {",
      "    if (ready) {",
      "        go();",
      "    }",
      "}",
      "function b() {",
      "        if (ready) {",
      "            go();",
      "        }",
      "}", ""));
    const r = await ef({
      path: p,
      old_string: lines("if (ready) {", "    go();", "}"),
      new_string: lines("if (ready) {", "    go();", "} else {", "    stop();", "}"),
      replace_all: true,
    });
    const want = lines(
      "function a() {",
      "    if (ready) {",
      "        go();",
      "    } else {",
      "        stop();",
      "    }",
      "}",
      "function b() {",
      "        if (ready) {",
      "            go();",
      "        } else {",
      "            stop();",
      "        }",
      "}", "");
    ok((await read(p)) === want, "both blocks rewritten, each at its own indentation");
    ok(String(r).indexOf("2 replacements") > 0, "reports 2 replacements");
  }

  console.log("Test 5: syntax dry-run rejects a candidate that breaks JS");
  {
    const p = await fixture("smart_edit_gate.js", lines(
      "const ok = 1;", "function fine() {", "  return ok;", "}", ""));
    const before = await read(p);
    globalThis.__sofuu_last_diff = null;
    const err = await expectThrow(function () {
      return ef({ path: p, old_string: "function fine() {", new_string: "function fine( {" });
    });
    ok(!!err, "breaking JS candidate is rejected");
    ok(!!err && String(err.message).indexOf("syntax dry-run") > 0, "rejection names the dry run");
    ok(!!err && String(err.message).indexOf("SyntaxError") > 0, "rejection carries the compiler class");
    ok(!!err && String(err.message).indexOf("nothing was written") > 0, "rejection states nothing was written");
    ok((await read(p)) === before, "rejected edit leaves the file byte-identical");
    ok(lastDiff() === null, "rejected edit stashes no diff");
  }

  console.log("Test 6: JSON gate rejects a broken document");
  {
    const p = await fixture("smart_edit_gate.json", lines('{ "a": 1, "b": [1, 2] }', ""));
    const before = await read(p);
    const err = await expectThrow(function () {
      return ef({ path: p, old_string: '"b": [1, 2]', new_string: '"b": [1, 2,],' });
    });
    ok(!!err, "invalid JSON candidate is rejected");
    ok(!!err && String(err.message).indexOf("JSON.parse") > 0, "rejection names JSON.parse");
    ok((await read(p)) === before, "JSON rejection leaves the file untouched");
  }

  console.log("Test 7: a file that was already broken stays editable");
  {
    const p = await fixture("smart_edit_broken.js", lines("const broken = ;", "// target line", ""));
    const r = await ef({ path: p, old_string: "// target line", new_string: "// updated line" });
    ok((await read(p)).indexOf("// updated line") >= 0, "edit applied despite the pre-existing error");
    ok(String(r).indexOf("[warning]") > 0, "warning rider present");
    ok(String(r).indexOf("still does not parse") > 0, "warning rider explains the leftover error");
  }

  console.log("Test 8: ESM is not a false positive");
  {
    const p = await fixture("smart_edit_esm.js", lines(
      "export function f() {", "  return 1;", "}", "// note", ""));
    const err = await expectThrow(function () {
      return ef({ path: p, old_string: "// note", new_string: "// changed" });
    });
    ok(!err, "export/import file is not rejected by classic-script mode");
    ok((await read(p)).indexOf("// changed") >= 0, "ESM edit applied");
  }

  console.log("Test 9: TypeScript is stripped before parsing");
  {
    const p = await fixture("smart_edit_types.ts", lines(
      "interface Point { x: number; y: number }",
      "const pt: Point = { x: 1, y: 2 };",
      "// old", ""));
    const err = await expectThrow(function () {
      return ef({ path: p, old_string: "// old", new_string: "// new" });
    });
    ok(!err, "typed file is not rejected");
    ok((await read(p)).indexOf("// new") >= 0, "TS edit applied");
  }

  console.log("Test 10: non-code extensions are never gated");
  {
    const p = await fixture("smart_edit_notes.md", lines(
      "this is ((( not javascript", "second line", ""));
    const err = await expectThrow(function () {
      return ef({ path: p, old_string: "second line", new_string: "third line" });
    });
    ok(!err, "markdown is not syntax-gated");
    ok((await read(p)).indexOf("third line") >= 0, "markdown edit applied");
  }

  console.log("Test 11: a total miss still reports not-found");
  {
    const p = await fixture("smart_edit_miss.txt", lines("alpha", ""));
    const err = await expectThrow(function () {
      return ef({ path: p, old_string: "zzz-no-such-block", new_string: "q" });
    });
    ok(!!err, "zero matches throws");
    ok(!!err && String(err.message).indexOf("not found") > 0, "message says not found");
    ok(!!err && String(err.message).indexOf("ignoring indentation") > 0, "message notes the relaxed attempt");
  }

  console.log("Test 12: CRLF files keep their line endings through a tolerant edit");
  {
    const p = await fixture("smart_edit_crlf.txt",
      ["head", "    if (ready) {", "        go();", "    }", "tail"].join(CRLF));
    await ef({
      path: p,
      old_string: lines("if (ready) {", "    go();", "}"),
      new_string: lines("if (ready) {", "    go();", "} else {", "    stop();", "}"),
    });
    const want = ["head", "    if (ready) {", "        go();", "    } else {", "        stop();", "    }", "tail"].join(CRLF);
    ok((await read(p)) === want, "CRLF preserved and the block re-indented");
  }

  console.log("");
  if (failed === 0) console.log("ALL PHASE 5 SMART-EDIT TESTS PASSED (" + passed + " checks)");
  else console.log("PHASE 5 SMART-EDIT TESTS FAILED: " + failed + " of " + (passed + failed) + " checks");
}

main()
  .then(async function () {
    for (const f of FILES) { try { await sofuu.fs.rm(f); } catch (e) {} }
    process.exit(failed === 0 ? 0 : 1);
  })
  .catch(async function (err) {
    console.error("Test crashed:", (err && (err.stack || err.message)) || err);
    for (const f of FILES) { try { await sofuu.fs.rm(f); } catch (e) {} }
    process.exit(1);
  });
