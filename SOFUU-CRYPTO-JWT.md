# `sofuu.crypto` — RS256 JWT verification

- **Status:** implemented 2026-10-26 in the local Sofuu checkout
- **Target:** Sofuu `0.2.0-beta`
- **Motivation:** Clerk/OIDC session tokens are RS256 JWTs verified against a
  public JWKS. The runtime had no crypto surface, so hosted applications had
  no way to authenticate a bearer token locally.
- **Gap closed:** before this change `sofuu` had no `crypto.subtle`,
  `crypto.getRandomValues`, `atob`/`btoa`, `TextEncoder`/`TextDecoder`,
  `sofuu.crypto`, `sofuu.jwt`, or `require("crypto")`. RS256 verification was
  impossible from JavaScript. JWKS fetching/caching stays application-level;
  this module provides the signature and claim verification primitive.

## What changed

| File | Change |
| --- | --- |
| `crates/sofuu-core/src/rt/crypto.rs` | **New.** Base64url, SHA-256 wrapper, RSA PKCS#1 v1.5 verification, JWT/JWKS verification, JS registration, unit tests. |
| `crates/sofuu-core/src/rt/mod.rs` | Registers `pub mod crypto;`. |
| `crates/sofuu-core/src/rt/engine.rs` | Calls `mod_crypto_register(ctx)` after `sofuu.async` registration. |
| `crates/sofuu-core/Cargo.toml` | Adds `num-bigint` (pure Rust, `std`, default features off) for public-key modular exponentiation. |
| `Cargo.lock` | Updated by the build to include `num-bigint` and its existing `num-integer` / `num-traits` dependencies. |

No C code, no OpenSSL, no platform Security.framework. The module reuses the
pure-Rust SHA-256 already in `rt/wal.rs` and keeps the dependency footprint to
one pure-Rust crate. RSA verification only uses public values (`n`, `e`,
signature), so no secret key material is held by the runtime.

## JavaScript API

```js
sofuu.crypto.sha256(text)                  // lowercase hex SHA-256
sofuu.crypto.base64urlEncode(text)         // unpadded base64url
sofuu.crypto.base64urlDecode(text)         // UTF-8 string, or null if invalid
sofuu.crypto.verifyJWT(token, jwks, opts)  // object, never throws
```

### `verifyJWT`

```js
const result = sofuu.crypto.verifyJWT(
  request.headers.authorization.replace(/^Bearer /, ""),
  { keys: [/* JWKS from https://<frontend-api>/.well-known/jwks.json */] },
  {
    issuer: "https://veliora-app.rikuhana.com",
    authorizedParties: ["https://veliora-app.rikuhana.com"],
    clockSkewSeconds: 5,       // optional, default 5
    nowMs: Date.now(),         // optional, for tests
  },
);

if (!result.ok) {
  // result.error = { code, message }
}
// result.payload.sub, result.payload.sid, result.payload.azp, ...
```

Success:

```json
{
  "ok": true,
  "header": { "alg": "RS256", "kid": "ins_...", "typ": "JWT" },
  "payload": { "sub": "user_...", "sid": "sess_...", "iss": "...", "exp": 0 },
  "kid": "ins_..."
}
```

Failure:

```json
{
  "ok": false,
  "error": { "code": "invalid_signature", "message": "digest mismatch" }
}
```

`jwks` and `opts` may be objects or JSON strings. `token` and `jwks` are
required; `opts` is optional.

### Checks performed

- Token must have exactly three non-empty dot-separated segments.
- Header `alg` must be exactly `RS256` (no `none`, no HMAC confusion).
- Key is selected by `kid`; a single-key JWKS may omit `kid`.
- JWK must be `kty: "RSA"`, `use` absent or `"sig"`, `alg` absent or `"RS256"`.
- RSA modulus must be 2048–8192 bits; signature length must equal modulus length.
- DigestInfo must be the SHA-256 PKCS#1 v1.5 encoding (RFC 8017 §9.2).
- `exp` is required; expired tokens fail.
- `nbf`, when present, is enforced with clock skew.
- `iss`, when `opts.issuer` is set, must match exactly.
- Clerk's `azp`, when `opts.authorizedParties` is non-empty, must be allow-listed.
- `aud`, when `opts.audience` is set, must match (string or array).
- `sub` must be a non-empty string.

The function never throws and never returns `ok: true` with an unverified
payload. Invalid base64url, unknown `kid`, wrong issuer, expired tokens, and
tampered signatures all return `{ok:false, error}`.

## Build and test

```bash
cd /Users/priyanshuboruah/ai-native-js-runtime

# Fast unit tests
cargo test -p sofuu-core --lib rt::crypto::tests

# Full release binary
make
```

The unit tests cover:

- SHA-256 FIPS vector for `"abc"`.
- Base64url round-trips, optional padding, non-canonical trailing bits,
  embedded padding, and `+`/`/` rejection.
- A real 2048-bit RS256 JWT fixture verifies.
- A tampered signature is rejected.
- Expired token and wrong issuer are rejected.
- Unknown `kid` is rejected.

A JS-level smoke test:

```js
console.log(sofuu.crypto.sha256("abc"));
// ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad

const result = sofuu.crypto.verifyJWT(token, { keys: [jwk] }, {
  issuer: "https://veliora-app.rikuhana.com",
  authorizedParties: ["https://veliora-app.rikuhana.com"],
});
console.log(result.ok, result.payload && result.payload.sub);
```

## Notes for future work

- Only `RS256` is implemented. Clerk uses RS256. Adding `RS384`/`RS512`
  would mean adding their DigestInfo prefixes; ES256/EdDSA would need new
  verification code.
- JWKS fetching and caching are intentionally left to the application. The
  runtime primitive is synchronous and network-free; `fetch` plus a short
  cache in the app is enough.
- If Sofuu later grows WebCrypto (`crypto.subtle`), `sofuu.crypto.verifyJWT`
  can become a thin wrapper or stay as the hosted-auth fast path.
- The 5 MB release gate still passes on the local macOS arm64 build:
  `make size-check` reports **5,220,432 bytes ≤ 5,242,880** (22,448 bytes of
  headroom). The `num-bigint` addition is a small pure-Rust cost, but the
  margin is thin; re-run `make size-check` after any further crypto work.
