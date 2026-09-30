/* tools_edit_literal_test.js — js-6 (AUDIT-2026-09-07) regression driver.
 *
 * edit_file single-match mode must splice new_string LITERALLY. The old
 * implementation ran text.replace(oldS, newS), which interprets the four
 * two-char dollar sequences in new_string as replacement patterns (matched
 * text / literal dollar sign / left context / right context) and silently
 * corrupted the file. The fix splices with slice() instead.
 *
 * Invoked via tools_edit_literal_e2e.sh, which cds into a scratch project
 * dir first — this driver uses cwd-relative bare filenames only.
 */
const results = { passed: 0, failed: 0 };
function assert(label, cond) {
  if (cond) { results.passed++; console.log("PASS " + label); }
  else { results.failed++; console.log("FAIL " + label); }
}

/* The dollar sign is built at runtime so the replacement-pattern sequences
 * never appear in this source file. */
const D = String.fromCharCode(36);

async function main() {
  const ef = sofuu.tools.TOOLS.edit_file.execute;
  const f1 = "edit_literal_1.txt";
  const f2 = "edit_literal_2.txt";
  const f3 = "edit_literal_3.txt";
  const f4 = "edit_literal_4.txt";

  try {
    /* (1) Single match, ALL four dollar sequences in new_string — the
     * exact set the old String.replace branch would have reinterpreted. */
    await sofuu.fs.writeFile(f1, "AA-BB-CC\n");
    const pat = D + "&" + " " + D + D + " " + D + "`" + " " + D + "'";
    const want1 = "AA-[" + pat + "]-CC\n";
    const a1 = { path: f1, old_string: "BB", new_string: "[" + pat + "]" };
    const r1 = await ef(a1);
    const t1 = await sofuu.fs.readFile(f1, "utf8");
    assert("single-match: all four dollar sequences land literally", t1 === want1);
    assert("single-match: result reports 1 replacement", String(r1).indexOf("1 replacement") >= 0);

    /* (2) The right-context sequence alone (the audit's worked example:
     * old code expanded it to everything after the match). */
    await sofuu.fs.writeFile(f2, "one two three\n");
    const a2 = { path: f2, old_string: "two", new_string: D + "'" };
    await ef(a2);
    const t2 = await sofuu.fs.readFile(f2, "utf8");
    assert("single-match: right-context sequence stays literal", t2 === "one " + D + "'" + " three\n");

    /* (3) replace_all (split/join — already literal before the fix; the
     * splice rewrite must not have changed it). */
    await sofuu.fs.writeFile(f3, "x y x y\n");
    const a3 = { path: f3, old_string: "y", new_string: "y " + D + D, replace_all: true };
    const r3 = await ef(a3);
    const t3 = await sofuu.fs.readFile(f3, "utf8");
    assert("replace_all: literal-dollar stays literal", t3 === "x y " + D + D + " x y " + D + D + "\n");
    assert("replace_all: result reports 2 replacements", String(r3).indexOf("2 replacements") >= 0);

    /* (4) A plain edit with no special characters still works. */
    await sofuu.fs.writeFile(f4, "hello world\n");
    const a4 = { path: f4, old_string: "world", new_string: "there" };
    await ef(a4);
    const t4 = await sofuu.fs.readFile(f4, "utf8");
    assert("plain edit unaffected", t4 === "hello there\n");

    /* (5) Zero matches still throw (match counting survived the rewrite). */
    let threw = false;
    try {
      const a5 = { path: f4, old_string: "no-such-text-here", new_string: "x" };
      await ef(a5);
    } catch (e0) { threw = true; }
    assert("zero matches throw", threw);
  } finally {
    for (const f of [f1, f2, f3, f4]) {
      try { await sofuu.fs.rm(f); } catch (eC) {}
    }
  }

  console.log("");
  if (results.failed === 0) console.log("EDIT-LITERAL TEST: ALL PASSED (" + results.passed + " checks)");
  else console.log("EDIT-LITERAL TEST: " + results.failed + " FAILED");
  process.exit(results.failed === 0 ? 0 : 1);
}

main().catch(e => { console.error("Test failed:", e); process.exit(1); });
