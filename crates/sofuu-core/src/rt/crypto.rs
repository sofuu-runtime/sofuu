// rt/crypto.rs — the cryptographic primitives hosted-auth needs.
//
// JS surface (JSON in, JSON out for structured results):
//   sofuu.crypto.sha256(text)                    lowercase hex digest
//   sofuu.crypto.base64urlEncode(text)           unpadded base64url
//   sofuu.crypto.base64urlDecode(text)           UTF-8 string or null
//   sofuu.crypto.verifyJWT(token, jwks, opts?)   RS256 verification + claims
//
// Why this exists: OIDC-style providers hand a client an RS256-signed session
// JWT and expect the backend to verify it against the project's public JWKS.
// QuickJS has no WebCrypto, no atob, no TextEncoder, and the runtime
// deliberately has no TLS/OpenSSL dependency. The module keeps that posture:
// SHA-256 is the pure-Rust implementation already used by the WAL, base64url is
// ~40 lines, and RSA PKCS#1 v1.5 verification is BigUint modular
// exponentiation with the public exponent. No secret key ever enters the
// runtime, so constant-time comparison is a correctness detail, not a
// side-channel boundary.

use num_bigint::BigUint;
use serde_json::{json, Value};
use sofuu_ffi::qjs::{self, JSContext, JSValue, JSValueConst};
use std::ffi::CString;
use std::os::raw::c_int;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::rt::wal::Sha256;

const MAX_TOKEN_LEN: usize = 64 * 1024;
const MIN_RSA_BITS: u64 = 2048;
const MAX_RSA_BITS: u64 = 8192;
const CLOCK_SKEW_SECONDS: i64 = 5;

/// PKCS#1 v1.5 DigestInfo prefix for SHA-256 (RFC 8017 §9.2, note 1).
const SHA256_DIGEST_INFO: [u8; 19] = [
    0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01,
    0x05, 0x00, 0x04, 0x20,
];

#[derive(Debug, Clone)]
pub struct AuthError {
    pub code: &'static str,
    pub message: String,
}

impl AuthError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

type Result<T> = std::result::Result<T, AuthError>;

/// SHA-256 of a UTF-8 string, lowercase hex.
pub fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Base64url (RFC 4648 §5) without `=` padding.
pub fn base64url_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0];
        out.push(ALPHABET[(b0 >> 2) as usize] as char);
        if chunk.len() == 1 {
            out.push(ALPHABET[((b0 & 0x03) << 4) as usize] as char);
            break;
        }
        let b1 = chunk[1];
        out.push(ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() == 2 {
            out.push(ALPHABET[((b1 & 0x0f) << 2) as usize] as char);
            break;
        }
        let b2 = chunk[2];
        out.push(ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        out.push(ALPHABET[(b2 & 0x3f) as usize] as char);
    }
    out
}

/// Strict base64url decoder. Accepts optional `=` padding only at the end.
pub fn base64url_decode(input: &str) -> Result<Vec<u8>> {
    if input.len() > MAX_TOKEN_LEN {
        return Err(AuthError::new("invalid_encoding", "base64url input is too large"));
    }
    let bytes = input.as_bytes();
    let mut end = bytes.len();
    let mut pad = 0usize;
    while end > 0 && bytes[end - 1] == b'=' {
        end -= 1;
        pad += 1;
    }
    if pad > 2 {
        return Err(AuthError::new("invalid_encoding", "too much base64 padding"));
    }
    let body = &bytes[..end];
    if body.contains(&b'=') {
        return Err(AuthError::new(
            "invalid_encoding",
            "base64 padding may only appear at the end",
        ));
    }
    let rem = body.len() % 4;
    if rem == 1 {
        return Err(AuthError::new("invalid_encoding", "base64url length is invalid"));
    }
    if pad > 0 {
        let expected = 4 - rem;
        if (expected == 4 && pad != 0) || (expected != 4 && expected != pad) || input.len() % 4 != 0 {
            return Err(AuthError::new("invalid_encoding", "base64 padding is invalid"));
        }
    }
    // Non-padded input is valid at lengths 2 and 3 mod 4; the unused low bits
    // must be zero (canonical encoding), otherwise two strings decode alike.
    let mut out = Vec::with_capacity(body.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for &c in body {
        let value = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            _ => {
                return Err(AuthError::new(
                    "invalid_encoding",
                    "base64url contains a non-alphabet character",
                ));
            }
        };
        acc = (acc << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= if bits == 0 { 0 } else { (1u32 << bits) - 1 };
        }
    }
    if bits > 0 && acc != 0 {
        return Err(AuthError::new(
            "invalid_encoding",
            "base64url has non-zero trailing bits",
        ));
    }
    Ok(out)
}

/// RSASSA-PKCS1-v1_5 with SHA-256 (the JOSE `RS256` algorithm).
pub fn verify_rs256(
    signing_input: &[u8],
    signature: &[u8],
    modulus: &[u8],
    exponent: &[u8],
) -> Result<()> {
    if modulus.is_empty() || exponent.is_empty() {
        return Err(AuthError::new("invalid_key", "RSA key is missing n or e"));
    }
    if exponent.iter().all(|&b| b == 0) {
        return Err(AuthError::new("invalid_key", "RSA public exponent is zero"));
    }
    let n = BigUint::from_bytes_be(modulus);
    let e = BigUint::from_bytes_be(exponent);
    let bits = n.bits();
    if bits < MIN_RSA_BITS || bits > MAX_RSA_BITS {
        return Err(AuthError::new(
            "invalid_key",
            format!("RSA modulus must be {MIN_RSA_BITS}..={MAX_RSA_BITS} bits, got {bits}"),
        ));
    }
    let k = bits.div_ceil(8) as usize;
    // JOSE signatures are exactly the modulus length. Shorter values are not
    // accepted with implicit left padding (that is a well-known cross-parser
    // source of signature malleability).
    if signature.len() != k {
        return Err(AuthError::new(
            "invalid_signature",
            format!("signature length {} does not match RSA modulus {k}", signature.len()),
        ));
    }
    let s = BigUint::from_bytes_be(signature);
    if s >= n {
        return Err(AuthError::new(
            "invalid_signature",
            "signature is not smaller than the RSA modulus",
        ));
    }
    let message = s.modpow(&e, &n).to_bytes_be();
    if message.len() > k {
        return Err(AuthError::new("invalid_signature", "RSA result is too large"));
    }
    let mut em = vec![0u8; k - message.len()];
    em.extend_from_slice(&message);

    let mut hasher = Sha256::new();
    hasher.update(signing_input);
    let digest = hasher.finalize();
    let mut expected = Vec::with_capacity(SHA256_DIGEST_INFO.len() + digest.len());
    expected.extend_from_slice(&SHA256_DIGEST_INFO);
    expected.extend_from_slice(&digest);

    if em[0] != 0x00 || em[1] != 0x01 {
        return Err(AuthError::new("invalid_signature", "bad PKCS#1 block type"));
    }
    if k < expected.len() + 11 {
        return Err(AuthError::new("invalid_key", "RSA modulus is too small for SHA-256"));
    }
    let t_start = k - expected.len();
    let ps = &em[2..t_start - 1];
    if ps.len() < 8 || ps.iter().any(|&b| b != 0xff) {
        return Err(AuthError::new("invalid_signature", "bad PKCS#1 padding"));
    }
    if em[t_start - 1] != 0x00 {
        return Err(AuthError::new("invalid_signature", "bad PKCS#1 separator"));
    }
    let mut diff = 0u8;
    for (got, want) in em[t_start..].iter().zip(expected.iter()) {
        diff |= got ^ want;
    }
    if diff != 0 {
        return Err(AuthError::new("invalid_signature", "digest mismatch"));
    }
    Ok(())
}

fn jwk_parts(jwk: &Value) -> Result<(Vec<u8>, Vec<u8>)> {
    if jwk.get("kty").and_then(Value::as_str) != Some("RSA") {
        return Err(AuthError::new("invalid_key", "JWK is not an RSA key"));
    }
    if let Some(alg) = jwk.get("alg").and_then(Value::as_str) {
        if alg != "RS256" {
            return Err(AuthError::new(
                "unsupported_alg",
                format!("JWK algorithm {alg:?} is not RS256"),
            ));
        }
    }
    if let Some(use_) = jwk.get("use").and_then(Value::as_str) {
        if use_ != "sig" {
            return Err(AuthError::new("invalid_key", "JWK is not a signing key"));
        }
    }
    let n = jwk
        .get("n")
        .and_then(Value::as_str)
        .ok_or_else(|| AuthError::new("invalid_key", "JWK is missing n"))?;
    let e = jwk
        .get("e")
        .and_then(Value::as_str)
        .ok_or_else(|| AuthError::new("invalid_key", "JWK is missing e"))?;
    Ok((base64url_decode(n)?, base64url_decode(e)?))
}

fn select_jwk<'a>(jwks: &'a Value, kid: Option<&str>) -> Result<&'a Value> {
    let keys = jwks.get("keys").and_then(Value::as_array);
    if let Some(keys) = keys {
        if let Some(kid) = kid {
            return keys
                .iter()
                .find(|key| key.get("kid").and_then(Value::as_str) == Some(kid))
                .ok_or_else(|| AuthError::new("unknown_kid", format!("no JWKS key for kid {kid:?}")));
        }
        if keys.len() == 1 {
            return Ok(&keys[0]);
        }
        return Err(AuthError::new(
            "unknown_kid",
            "token has no kid and the JWKS has multiple keys",
        ));
    }
    // A bare JWK (not wrapped in {keys:[...]}) is accepted for local setups.
    if jwks.get("kty").is_some() {
        return Ok(jwks);
    }
    Err(AuthError::new("invalid_jwks", "expected {keys:[...]} or a single JWK"))
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn option_i64(options: &Value, snake: &str, camel: &str) -> Option<i64> {
    options
        .get(snake)
        .or_else(|| options.get(camel))
        .and_then(Value::as_i64)
}

fn validate_claims(payload: &Value, options: &Value) -> Result<()> {
    let now = option_i64(options, "now_ms", "nowMs").unwrap_or_else(now_ms);
    let skew = option_i64(options, "clock_skew_seconds", "clockSkewSeconds")
        .unwrap_or(CLOCK_SKEW_SECONDS)
        .max(0);

    let exp = payload
        .get("exp")
        .and_then(Value::as_i64)
        .ok_or_else(|| AuthError::new("missing_exp", "token is missing exp"))?;
    if now / 1000 >= exp.saturating_add(skew) {
        return Err(AuthError::new("token_expired", "token has expired"));
    }
    if let Some(nbf) = payload.get("nbf").and_then(Value::as_i64) {
        if now / 1000 < nbf.saturating_sub(skew) {
            return Err(AuthError::new("token_not_yet_valid", "token is not valid yet"));
        }
    }

    if let Some(issuer) = options.get("issuer").and_then(Value::as_str) {
        if payload.get("iss").and_then(Value::as_str) != Some(issuer) {
            return Err(AuthError::new("issuer_mismatch", "token issuer does not match"));
        }
    }

    let parties = options
        .get("authorized_parties")
        .or_else(|| options.get("authorizedParties"))
        .and_then(Value::as_array);
    if let Some(parties) = parties {
        if !parties.is_empty() {
            // OIDC providers commonly define `azp` as the authorized party
            // (the browser Origin). Native SDK tokens may omit it because they
            // have no browser origin; a missing claim is therefore allowed.
            // When present, it must be in the allow-list.
            if let Some(azp) = payload.get("azp").and_then(Value::as_str) {
                let ok = parties.iter().any(|party| party.as_str() == Some(azp));
                if !ok {
                    return Err(AuthError::new(
                        "azp_mismatch",
                        "token authorized party is not in the allow-list",
                    ));
                }
            }
        }
    }

    if let Some(audience) = options.get("audience") {
        let expected: Vec<&str> = if let Some(s) = audience.as_str() {
            vec![s]
        } else if let Some(items) = audience.as_array() {
            items.iter().filter_map(Value::as_str).collect()
        } else {
            Vec::new()
        };
        if !expected.is_empty() {
            let actual = payload.get("aud");
            let ok = match actual {
                Some(Value::String(s)) => expected.contains(&s.as_str()),
                Some(Value::Array(items)) => items
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|item| expected.contains(&item)),
                _ => false,
            };
            if !ok {
                return Err(AuthError::new("audience_mismatch", "token audience does not match"));
            }
        }
    }

    if payload.get("sub").and_then(Value::as_str).unwrap_or("").is_empty() {
        return Err(AuthError::new("missing_sub", "token is missing sub"));
    }
    Ok(())
}

fn verify_jwt_inner(token: &str, jwks: &Value, options: &Value) -> Result<Value> {
    if token.len() > MAX_TOKEN_LEN {
        return Err(AuthError::new("invalid_token", "token is too large"));
    }
    let mut parts = token.split('.');
    let (Some(encoded_header), Some(encoded_payload), Some(encoded_signature), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(AuthError::new(
            "invalid_token",
            "token must have exactly three dot-separated segments",
        ));
    };
    if encoded_header.is_empty() || encoded_payload.is_empty() || encoded_signature.is_empty() {
        return Err(AuthError::new("invalid_token", "token has an empty segment"));
    }

    let header_bytes = base64url_decode(encoded_header)?;
    let payload_bytes = base64url_decode(encoded_payload)?;
    let signature = base64url_decode(encoded_signature)?;
    let header: Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| AuthError::new("invalid_token", format!("header is not JSON: {e}")))?;
    let payload: Value = serde_json::from_slice(&payload_bytes)
        .map_err(|e| AuthError::new("invalid_token", format!("payload is not JSON: {e}")))?;

    let alg = header.get("alg").and_then(Value::as_str).unwrap_or("");
    if alg != "RS256" {
        return Err(AuthError::new(
            "unsupported_alg",
            format!("token algorithm {alg:?} is not RS256"),
        ));
    }
    let kid = header.get("kid").and_then(Value::as_str);
    let jwk = select_jwk(jwks, kid)?;
    let (modulus, exponent) = jwk_parts(jwk)?;
    let signing_input = format!("{encoded_header}.{encoded_payload}");
    verify_rs256(signing_input.as_bytes(), &signature, &modulus, &exponent)?;
    validate_claims(&payload, options)?;

    Ok(json!({
        "ok": true,
        "header": header,
        "payload": payload,
        "kid": kid,
    }))
}

/// Verify an RS256 JWT against a JWKS (or a single JWK). Never panics and
/// never throws: failures are `{ok:false, error:{code,message}}`.
pub fn verify_jwt(token: &str, jwks: &Value, options: &Value) -> Value {
    match verify_jwt_inner(token, jwks, options) {
        Ok(mut value) => {
            if value.get("kid").map(Value::is_null).unwrap_or(true) {
                if let Some(obj) = value.as_object_mut() {
                    obj.remove("kid");
                }
            }
            value
        }
        Err(error) => json!({
            "ok": false,
            "error": {"code": error.code, "message": error.message},
        }),
    }
}

// ── JS surface ───────────────────────────────────────────────────────────────

fn json_str(v: &str) -> CString {
    CString::new(v).unwrap_or_default()
}

unsafe fn arg_string(ctx: *mut JSContext, arg: JSValueConst) -> Option<String> {
    if qjs::sofuu_js_is_string(arg) == 0 {
        return None;
    }
    let pointer = qjs::sofuu_js_to_cstring(ctx, arg);
    if pointer.is_null() {
        return None;
    }
    let out = std::ffi::CStr::from_ptr(pointer).to_string_lossy().into_owned();
    qjs::sofuu_js_free_cstring(ctx, pointer);
    Some(out)
}

unsafe fn arg_json(ctx: *mut JSContext, arg: JSValueConst) -> Option<Value> {
    if qjs::is_object(arg) {
        let stringified =
            qjs::JS_JSONStringify(ctx, arg, qjs::sofuu_js_undefined(), qjs::sofuu_js_undefined());
        if qjs::is_exception(stringified) || qjs::is_undefined(stringified) {
            qjs::sofuu_js_free_value(ctx, stringified);
            return None;
        }
        let pointer = qjs::sofuu_js_to_cstring(ctx, stringified);
        let text = if pointer.is_null() {
            None
        } else {
            let out = std::ffi::CStr::from_ptr(pointer).to_string_lossy().into_owned();
            qjs::sofuu_js_free_cstring(ctx, pointer);
            Some(out)
        };
        qjs::sofuu_js_free_value(ctx, stringified);
        text.and_then(|text| serde_json::from_str(&text).ok())
    } else {
        let text = arg_string(ctx, arg)?;
        serde_json::from_str(&text).ok()
    }
}

unsafe fn json_result(ctx: *mut JSContext, value: &Value) -> JSValue {
    let text = serde_json::to_string(value).unwrap_or_else(|_| "{}".into());
    let c_text = json_str(&text);
    qjs::JS_ParseJSON(ctx, c_text.as_ptr(), c_text.as_bytes().len(), c"<crypto>".as_ptr())
}

unsafe fn text_result(ctx: *mut JSContext, value: &str) -> JSValue {
    let c_text = json_str(value);
    qjs::sofuu_js_new_string(ctx, c_text.as_ptr())
}

unsafe extern "C" fn js_crypto_sha256(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_null();
    }
    match arg_string(ctx, *argv) {
        Some(text) => text_result(ctx, &sha256_hex(&text)),
        None => qjs::sofuu_js_null(),
    }
}

unsafe extern "C" fn js_crypto_base64url_encode(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_null();
    }
    match arg_string(ctx, *argv) {
        Some(text) => text_result(ctx, &base64url_encode(text.as_bytes())),
        None => qjs::sofuu_js_null(),
    }
}

unsafe extern "C" fn js_crypto_base64url_decode(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::sofuu_js_null();
    }
    let Some(text) = arg_string(ctx, *argv) else {
        return qjs::sofuu_js_null();
    };
    match base64url_decode(&text).ok().and_then(|bytes| String::from_utf8(bytes).ok()) {
        Some(text) => text_result(ctx, &text),
        None => qjs::sofuu_js_null(),
    }
}

unsafe extern "C" fn js_crypto_verify_jwt(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return json_result(
            ctx,
            &json!({"ok": false, "error": {
                "code": "invalid_arguments",
                "message": "verifyJWT requires (token, jwks, options?)",
            }}),
        );
    }
    let Some(token) = arg_string(ctx, *argv) else {
        return json_result(
            ctx,
            &json!({"ok": false, "error": {
                "code": "invalid_arguments",
                "message": "token must be a string",
            }}),
        );
    };
    let Some(jwks) = arg_json(ctx, *argv.add(1)) else {
        return json_result(
            ctx,
            &json!({"ok": false, "error": {
                "code": "invalid_jwks",
                "message": "jwks must be a JSON object or string",
            }}),
        );
    };
    let options = if argc >= 3 {
        arg_json(ctx, *argv.add(2)).unwrap_or_else(|| json!({}))
    } else {
        json!({})
    };
    json_result(ctx, &verify_jwt(&token, &jwks, &options))
}

/// Register `sofuu.crypto`. Called from `engine_register_builtins`.
pub unsafe fn mod_crypto_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
    if qjs::is_undefined(sofuu_obj) {
        sofuu_obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(
            ctx,
            global,
            c"sofuu".as_ptr(),
            qjs::sofuu_js_dup_value(ctx, sofuu_obj),
        );
    }
    let obj = qjs::sofuu_js_new_object(ctx);
    let functions: [(
        &std::ffi::CStr,
        unsafe extern "C" fn(*mut JSContext, JSValueConst, c_int, *const JSValueConst) -> JSValue,
    ); 4] = [
        (c"sha256", js_crypto_sha256),
        (c"base64urlEncode", js_crypto_base64url_encode),
        (c"base64urlDecode", js_crypto_base64url_decode),
        (c"verifyJWT", js_crypto_verify_jwt),
    ];
    for (name, func) in functions {
        qjs::sofuu_js_set_property_str(
            ctx,
            obj,
            name.as_ptr(),
            qjs::sofuu_js_new_cfunction(ctx, func, name.as_ptr(), 2),
        );
    }
    qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"crypto".as_ptr(), obj);
    qjs::sofuu_js_free_value(ctx, sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_JWK: &str = r#"{
        "kty": "RSA",
        "n": "ntahXsbkEUCoSvUhRJtDJSf_XE4uHz4KDeGeSRblbbrhIJdHfxl8vVWMGRI9VHXDZbjBkBAKsOjRM_cRNO2n7K9TksqSAnmc4rDfzMIc4wAIYiWYW5txI23Zk26FigaFS-Px-KkBKSmH_2tEB9D-Ens8avhnciaT1b3Z1Q425vgG8MNCOHaYhmIuMtNZRYGSQe9AYfirdhGn8FqyhcGgeDea8iZXwQ1mDAypJgN5X4k5yDs2engA_dzPbjWvnZOPJiftIgT4ZpgvahY7zCZscofz4BKgPRPmIpXOdIKTHqQSpyDvDmg9j8HBcgo_x7iPWQB0iHmB60sb7Lg_QdYnaw",
        "e": "AQAB",
        "alg": "RS256",
        "use": "sig",
        "kid": "test-key-1"
    }"#;

    const TEST_JWT: &str = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InRlc3Qta2V5LTEifQ.eyJpc3MiOiJodHRwczovL2FwcC5leGFtcGxlLmNvbSIsInN1YiI6InVzZXJfdGVzdF8xMjMiLCJzaWQiOiJzZXNzX3Rlc3RfNDU2IiwiYXpwIjoiaHR0cHM6Ly9hcHAuZXhhbXBsZS5jb20iLCJleHAiOjQxMDI0NDQ4MDAsIm5iZiI6MTcwMDAwMDAwMCwiaWF0IjoxNzAwMDAwMDAwfQ.GhyZRPrfTY4lDCrM4iaJ_WXdu1tonF8UF32GwQNNswKM2fpSf4HK7vqkCm2TzzlmUlRVsCjNpKftBKJ8R5Byf0IS1Yp65_kXKZY47sYqiv5W2tcYUB1IH1b4m2sx6PSUmNhpqzDUoN8wx9CCsJ_r-QZpvphenUr9OYC4Cx2NfuyuitZLSuDzFZV54knUf9Obm_tmaP8QqfJfmJLFsmeUUQGctf7VXTDJv7CPvLVAYWxyMBzRyt48ZuxH0HLJOqHEa2fFBHrXbbPrBGdRZHrcMLpDbcUa2jZ1y3g98tbOFF8ZnjBZecs3yMiybxfga0FYRg5Cs4PMsdEGS3UHYGDpug";

    fn jwks() -> Value {
        serde_json::from_str::<Value>(TEST_JWK).unwrap()
    }

    fn options() -> Value {
        json!({
            "issuer": "https://app.example.com",
            "authorizedParties": ["https://app.example.com"],
            "nowMs": 1_800_000_000_000i64,
        })
    }

    #[test]
    fn sha256_matches_the_fips_vector() {
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn base64url_round_trips_with_and_without_padding() {
        assert_eq!(base64url_encode(b"hello"), "aGVsbG8");
        assert_eq!(base64url_decode("aGVsbG8").unwrap(), b"hello");
        assert_eq!(base64url_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64url_encode(b"hi"), "aGk");
        assert_eq!(base64url_decode("aGk=").unwrap(), b"hi");
        assert!(base64url_decode("a").is_err());
        assert!(base64url_decode("aGk=").is_ok());
        assert!(base64url_decode("aGk==").is_err());
        assert!(base64url_decode("aGk===").is_err());
        assert!(base64url_decode("a+b/").is_err());
        assert!(base64url_decode("aGk=a").is_err());
    }

    #[test]
    fn valid_rs256_token_verifies() {
        let out = verify_jwt(TEST_JWT, &json!({"keys": [jwks()]}), &options());
        assert_eq!(out["ok"], true, "{out}");
        assert_eq!(out["payload"]["sub"], "user_test_123");
        assert_eq!(out["header"]["kid"], "test-key-1");
    }

    #[test]
    fn tampered_signature_is_rejected() {
        let tampered = format!(
            "{}.{}",
            TEST_JWT.rsplit_once('.').unwrap().0,
            base64url_encode(&[b'x'; 256])
        );
        let out = verify_jwt(&tampered, &json!({"keys": [jwks()]}), &options());
        assert_eq!(out["ok"], false, "{out}");
        assert_eq!(out["error"]["code"], "invalid_signature");
    }

    #[test]
    fn expired_and_wrong_issuer_tokens_are_rejected() {
        let opts = json!({
            "issuer": "https://app.example.com",
            "authorizedParties": ["https://app.example.com"],
            "nowMs": 4_102_444_800_000i64 + 10_000,
        });
        let expired = verify_jwt(TEST_JWT, &json!({"keys": [jwks()]}), &opts);
        assert_eq!(expired["ok"], false, "{expired}");
        assert_eq!(expired["error"]["code"], "token_expired");

        let wrong = json!({
            "issuer": "https://evil.example",
            "nowMs": 1_800_000_000_000i64,
        });
        let mismatch = verify_jwt(TEST_JWT, &json!({"keys": [jwks()]}), &wrong);
        assert_eq!(mismatch["ok"], false, "{mismatch}");
        assert_eq!(mismatch["error"]["code"], "issuer_mismatch");
    }

    #[test]
    fn missing_azp_is_allowed_but_a_wrong_one_is_rejected() {
        let payload = json!({
            "iss": "https://app.example.com",
            "sub": "user_native",
            "exp": 4_102_444_800i64,
        });
        assert!(
            validate_claims(&payload, &options()).is_ok(),
            "native tokens may omit azp"
        );

        let wrong = json!({
            "iss": "https://app.example.com",
            "sub": "user_native",
            "exp": 4_102_444_800i64,
            "azp": "https://evil.example",
        });
        let error = validate_claims(&wrong, &options()).unwrap_err();
        assert_eq!(error.code, "azp_mismatch");
    }

    #[test]
    fn unknown_kid_is_rejected() {
        let header = base64url_encode(br#"{"alg":"RS256","kid":"missing"}"#);
        let payload = TEST_JWT.split('.').nth(1).unwrap();
        let signature = TEST_JWT.split('.').nth(2).unwrap();
        let token = format!("{header}.{payload}.{signature}");
        let out = verify_jwt(&token, &json!({"keys": [jwks()]}), &options());
        assert_eq!(out["ok"], false, "{out}");
        assert_eq!(out["error"]["code"], "unknown_kid");
    }
}
