// keychain.rs — provider API keys in the macOS Keychain (PLAN-DESKTOP §4).
//
// The desktop reads keys from the Keychain first and injects them into the
// runtime via embed_config::set_api_key at engine boot; the CLI's env/config
// behavior is unchanged. Keys never reach the WebView — the frontend only
// ever sees a has_key boolean.

const SERVICE: &str = "com.sofuu.desktop";

/// Store (or replace) the API key for a provider name ("openai"/…).
pub fn set_key(provider: &str, key: &str) -> Result<(), String> {
    security_framework::passwords::set_generic_password(SERVICE, provider, key.as_bytes())
        .map_err(|e| format!("keychain set failed: {e}"))
}

/// Read the API key for a provider, if one is stored.
pub fn get_key(provider: &str) -> Option<String> {
    match security_framework::passwords::get_generic_password(SERVICE, provider) {
        Ok(bytes) => String::from_utf8(bytes).ok(),
        Err(_) => None,
    }
}

/// Delete the stored API key for a provider (ok if it was absent).
pub fn delete_key(provider: &str) -> Result<(), String> {
    match security_framework::passwords::delete_generic_password(SERVICE, provider) {
        Ok(_) => Ok(()),
        // errSecItemNotFound — nothing stored; treat as success.
        Err(e) if e.code() == -25300 => Ok(()),
        Err(e) => Err(format!("keychain delete failed: {e}")),
    }
}
