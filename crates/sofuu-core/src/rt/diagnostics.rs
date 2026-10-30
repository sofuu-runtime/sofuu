// rt/diagnostics.rs — PLAN-MODERN-NPM pillar C: failures that explain themselves.
//
// Three shapes of failure used to surface as a bare `ReferenceError: Cannot
// find module 'x'` (or, for a `.node` entry point, a syntax error from
// compiling a Mach-O file as JavaScript). None of them told a developer that
// the runtime is fine and the PACKAGE is the mismatch:
//
//   * a compiled V8 addon (`.node`) — impossible in a zero-JIT runtime;
//   * a Node built-in this single-address-space runtime does not provide;
//   * a package that resolved to nothing (missing entry point or subpath).
//
// Every message is two lines: what happened, then what to reach for instead.
// The first line is stable enough to grep; the thrown Error carries a
// machine-readable `.code` for scripts that want to branch on the class.
//
// The recommendation table is deliberately small and nominative: it names
// packages only to tell a developer what to install instead, never to imply
// an affiliation with them.

use std::ffi::CString;

use sofuu_ffi::qjs::{self, JSContext, JSValue};

pub const E_NATIVE_ADDON: &str = "ERR_SOFUU_NATIVE_ADDON";
pub const E_NODE_API: &str = "ERR_SOFUU_UNSUPPORTED_NODE_API";
pub const E_RESOLVE: &str = "ERR_SOFUU_RESOLUTION";
pub const E_REQUIRE_ESM: &str = "ERR_SOFUU_REQUIRE_ESM";

/// Node built-ins this runtime does not provide, and the in-runtime answer.
const UNSUPPORTED_NODE_APIS: &[(&str, &str)] = &[
    (
        "child_process",
        "use sofuu.spawn(cmd, args) or sofuu.exec for subprocesses; there is no fork() of \
         JavaScript siblings in a single-address-space runtime",
    ),
    (
        "v8",
        "there are no V8 internals here; use standard JavaScript (no --allow-natives-syntax)",
    ),
    (
        "worker_threads",
        "Sofuu runs one address space: use scheduler.yield() for cooperative concurrency, or \
         sofuu.swarm / an agent mesh for parallel work",
    ),
    (
        "cluster",
        "Sofuu runs one address space: use sofuu.swarm or an agent mesh for parallel work",
    ),
    (
        "inspector",
        "attach a debugger through the CLI instead; Node's inspector protocol is not implemented",
    ),
];

/// Packages whose published entry point is (or requires) a compiled addon,
/// with a pure-JS or in-runtime alternative.
const NATIVE_PACKAGES: &[(&str, &str)] = &[
    ("sharp", "a pure-JS/WASM image pipeline (Sofuu cannot load V8 addons)"),
    ("canvas", "a pure-JS/WASM canvas implementation"),
    ("sqlite3", "sofuu.storage (the QTSQ structured store) for local tables"),
    ("better-sqlite3", "sofuu.storage (the QTSQ structured store) for local tables"),
    ("bcrypt", "the pure-JS bcryptjs package, or @noble/hashes for scrypt/pbkdf2"),
    ("puppeteer", "sofuu.web.open / sofuu.web.search — no browser process is spawned"),
    ("node-sass", "the pure-JS sass (dart-sass) package"),
    ("grpc", "HTTP/JSON through sofuu.http or fetch()"),
    ("node-gyp", "a pure-JS alternative; Sofuu never runs a C++ build toolchain"),
];

/// The recommendation for a package known to ship a native addon.
pub fn native_package_hint(pkg: &str) -> Option<&'static str> {
    NATIVE_PACKAGES
        .iter()
        .find(|(name, _)| *name == pkg)
        .map(|(_, hint)| *hint)
}

/// `Some(recommendation)` when the requested built-in is deliberately absent.
pub fn unsupported_node_api(name: &str) -> Option<&'static str> {
    UNSUPPORTED_NODE_APIS
        .iter()
        .find(|(api, _)| *api == name)
        .map(|(_, hint)| *hint)
}

/// The package name behind a resolved path, for a `.node` diagnostic:
/// `…/node_modules/@scope/pkg/build/x.node` → `@scope/pkg`.
pub fn package_of_path(path: &str) -> Option<String> {
    let marker = "node_modules/";
    let at = path.rfind(marker)? + marker.len();
    let rest = &path[at..];
    let mut parts = rest.split('/');
    let first = parts.next()?;
    if first.starts_with('@') {
        let second = parts.next()?;
        Some(format!("{first}/{second}"))
    } else {
        Some(first.to_string())
    }
}

pub fn native_addon_message(pkg: &str, detail: &str) -> String {
    let hint = native_package_hint(pkg)
        .map(|h| h.to_string())
        .unwrap_or_else(|| "use a pure-JS or WebAssembly alternative".to_string());
    format!(
        "Incompatible package: '{pkg}' contains a compiled C++ V8 addon ({detail}). \
         Sofuu runs a zero-JIT engine and cannot load V8 native binaries.\n  \
         → Recommendation: {hint}."
    )
}

/// `require()` of an ES module. Node answers `ERR_REQUIRE_ESM`; without this
/// the module would be wrapped in the CommonJS function and fail as
/// "unsupported keyword: export", which points at the wrong thing.
pub fn require_esm_message(spec: &str, path: &str) -> String {
    format!(
        "Cannot require() the ES module '{spec}' ({path}).\n  \
         → Recommendation: load it with `import {spec}` (or `await import(\"{spec}\")`) instead."
    )
}

pub fn unsupported_api_message(api: &str, hint: &str) -> String {
    format!(
        "Unsupported Node API: '{api}' is not available in Sofuu's single-address-space runtime.\n  \
         → Recommendation: {hint}."
    )
}

/// `"Cannot find module 'x' — <reason>"` (the reason comes from
/// `npm::explain_module_not_found`).
pub fn resolution_message(spec: &str, reason: &str) -> String {
    format!("Cannot find module '{spec}' — {reason}")
}

/// Throw `err` (created by one of the `JS_Throw*` helpers) with `.code`
/// stamped on it, so scripts can branch on the failure class.
///
/// # Safety
/// `ctx` live.
pub unsafe fn throw_with_code(
    ctx: *mut JSContext,
    kind: &std::ffi::CStr,
    msg: &str,
    code: &str,
) -> JSValue {
    let m = CString::new(msg).unwrap_or_default();
    /* JS_Throw* sets the pending exception; take it back, stamp the field,
     * and re-throw the same object so the stack survives. */
    let _ = match kind.to_bytes() {
        b"ReferenceError" => qjs::JS_ThrowReferenceError(ctx, c"%s".as_ptr(), m.as_ptr()),
        b"TypeError" => qjs::JS_ThrowTypeError(ctx, c"%s".as_ptr(), m.as_ptr()),
        _ => qjs::JS_ThrowInternalError(ctx, c"%s".as_ptr(), m.as_ptr()),
    };
    let exc = qjs::JS_GetException(ctx);
    if !qjs::is_exception(exc) {
        if let Ok(c) = CString::new(code) {
            let v = qjs::JS_NewString(ctx, c.as_ptr());
            qjs::JS_SetPropertyStr(ctx, exc, c"code".as_ptr(), v);
        }
    }
    qjs::JS_Throw(ctx, exc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_of_path_handles_scoped_and_plain_packages() {
        assert_eq!(
            package_of_path("/p/node_modules/sharp/build/Release/sharp.node").as_deref(),
            Some("sharp")
        );
        assert_eq!(
            package_of_path("/p/node_modules/@scope/pkg/x.node").as_deref(),
            Some("@scope/pkg")
        );
        assert_eq!(package_of_path("/p/lib/x.js"), None);
    }

    #[test]
    fn known_native_packages_have_advice() {
        assert!(native_package_hint("sharp").unwrap().contains("image"));
        assert!(native_package_hint("better-sqlite3").unwrap().contains("sofuu.storage"));
        assert!(native_package_hint("left-pad").is_none());
    }

    #[test]
    fn unsupported_apis_are_named_and_advised() {
        for api in ["child_process", "v8", "worker_threads", "cluster"] {
            assert!(unsupported_node_api(api).is_some(), "{api}");
        }
        assert!(unsupported_node_api("fs").is_none());
    }

    #[test]
    fn messages_carry_both_the_fact_and_the_advice() {
        let m = native_addon_message("sharp", "sharp.node");
        assert!(m.contains("zero-JIT"), "{m}");
        assert!(m.contains("Recommendation"), "{m}");
        let m = unsupported_api_message("worker_threads", "use scheduler.yield()");
        assert!(m.contains("single-address-space"), "{m}");
        let m = resolution_message("pkg/nope", "package 'pkg@1.0.0' does not provide './nope'");
        assert!(m.starts_with("Cannot find module 'pkg/nope'"), "{m}");
    }
}
