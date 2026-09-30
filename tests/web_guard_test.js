// tests/web_guard_test.js — SSRF-guard regression battery (P1-15, AUDIT-2026-09-07).
//
// guardInternalUrl() classifies the HOST of an outbound URL before any
// network I/O. Its host extraction used to strip only the trailing port,
// so userinfo rode through unclassified:
//   http://x@169.254.169.254/  → "host" seen by the guard: "x@169.254.169.254"
//   → not an IP literal, not localhost → guard passed → fetch dialed the
//   metadata endpoint. The fix splits userinfo (LAST '@', since userinfo
//   may contain ':' but never an unencoded '@'), then bracket-wrapped
//   IPv6, then the port.
//
// No mocks and no network needed: every blocked vector must throw the
// guard error BEFORE fetch runs (verdict === 'guard'), and the
// pass-through controls use the RFC-2606 .invalid TLD, which fails DNS
// without ever producing a guard error — so a guard throw and a fetch
// failure are distinguishable.
//
// Run:  ./sofuu run tests/web_guard_test.js
// NOTE: do NOT arm SOFUU_WEB_ALLOW_INTERNAL here — web_test.js arms it
// for its loopback mocks; this battery needs the guard LIVE.

let failures = 0;
function check(name, cond) {
  if (cond) { console.log(`PASS ${name}`); }
  else { failures++; console.log(`FAIL ${name}`); }
}

/* 'guard' = refused by the SSRF guard (the only acceptable verdict for a
 * blocked vector). Any other error string means the guard let the URL
 * through and fetch itself failed — i.e. a bypass. */
async function verdict(url) {
  try {
    await sofuu.web.open(url, { timeoutMs: 1500 });
    return `ok`;
  } catch (e) {
    const m = String((e && e.message) || e);
    return m.indexOf(`SSRF guard`) >= 0 ? `guard` : m.slice(0, 80);
  }
}

async function main() {
  check(`sofuu.web.open exists`, !!(sofuu.web && typeof sofuu.web.open === `function`));
  delete process.env.SOFUU_WEB_ALLOW_INTERNAL;

  /* P1-15 bypass family — internal targets smuggled through userinfo. */
  const blocked = [
    [`http://x@169.254.169.254/latest/meta-data/`, `userinfo metadata (audit vector)`],
    [`http://x@10.0.0.5:9/`, `userinfo RFC1918`],
    [`http://x@localhost:9/`, `userinfo localhost`],
    [`http://x@127.0.0.1:9/`, `userinfo loopback`],
    [`http://x@[::1]:9/`, `userinfo bracketed IPv6 loopback`],
    [`http://x@y@10.0.0.5:9/`, `double userinfo (last @ wins)`],
    [`http://user:pw@localhost:9/`, `user:password userinfo`],
    /* pre-existing plain-host blocks must stay blocked */
    [`http://169.254.169.254/latest/meta-data/`, `plain metadata (pre-existing)`],
    [`http://192.168.1.4:9/`, `plain RFC1918 (pre-existing)`],
  ];
  for (const [url, label] of blocked) {
    check(`blocked: ${label}`, (await verdict(url)) === `guard`);
  }

  /* js-7 — alternate IPv4 encodings + IPv4-mapped IPv6. curl's inet_aton
   * accepts hex/octal/mixed-dot forms and ::ffff: IPv4-mapped tails; a
   * guard that only string-matches the canonical dotted quad is bypassed
   * by all of them dialing the same address. */
  const blockedJ7 = [
    [`http://0x7f000001:9/`, `hex IPv4 (0x7f000001 = 127.0.0.1)`],
    [`http://2130706433:9/`, `dword decimal IPv4 (2130706433 = 127.0.0.1)`],
    [`http://127.1:9/`, `shortened dotted IPv4 (127.1)`],
    [`http://0177.0.0.1:9/`, `octal IPv4 (0177 = 127)`],
    [`http://x@0x7f000001:9/`, `hex IPv4 behind userinfo`],
    [`http://[::ffff:169.254.169.254]/latest/meta-data/`, `IPv4-mapped metadata (decimal tail)`],
    [`http://[::ffff:a9fe:a9fe]/latest/meta-data/`, `IPv4-mapped metadata (hex tail)`],
    [`http://x@[::ffff:127.0.0.1]:9/`, `IPv4-mapped loopback behind userinfo`],
  ];
  for (const [url, label] of blockedJ7) {
    check(`blocked: ${label}`, (await verdict(url)) === `guard`);
  }

  /* Pass-through: the guard must stay silent for public hosts so real
   * fetches still work — including when userinfo is present. */
  for (const [url, label] of [
    [`http://p115-guard-canary.invalid/`, `plain public host`],
    [`http://user:pw@p115-guard-canary.invalid/`, `userinfo on a public host`],
  ]) {
    check(`passes guard: ${label}`, (await verdict(url)) !== `guard`);
  }

  console.log(failures === 0
    ? `\nWEB GUARD TEST: ALL PASSED`
    : `\nWEB GUARD TEST: ${failures} FAILURE(S)`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch(e => {
  console.error(`FAIL exception: ` + (e && e.stack || e));
  process.exit(1);
});
