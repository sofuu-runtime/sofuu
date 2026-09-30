// keychain.rs — provider API keys in the OS credential store (PLAN-DESKTOP §4).
//
// The desktop reads keys from the credential store first and injects them
// into the runtime via embed_config::set_api_key at engine boot; the CLI's
// env/config behavior is unchanged. Keys never reach the WebView — the
// frontend only ever sees a has_key boolean.
//
// Backends: macOS Keychain (security_framework), Windows Credential Manager
// + Linux Secret Service (keyring; see the per-target features in
// Cargo.toml), everything else → in-process stub.

const SERVICE: &str = "com.sofuu.desktop";

#[cfg(target_os = "macos")]
mod imp {
    use super::SERVICE;

    pub fn set_key(provider: &str, key: &str) -> Result<(), String> {
        security_framework::passwords::set_generic_password(SERVICE, provider, key.as_bytes())
            .map_err(|e| format!("keychain set failed: {e}"))
    }

    pub fn get_key(provider: &str) -> Option<String> {
        match security_framework::passwords::get_generic_password(SERVICE, provider) {
            Ok(bytes) => String::from_utf8(bytes).ok(),
            Err(_) => None,
        }
    }

    pub fn delete_key(provider: &str) -> Result<(), String> {
        match security_framework::passwords::delete_generic_password(SERVICE, provider) {
            Ok(_) => Ok(()),
            // errSecItemNotFound — nothing stored; treat as success.
            Err(e) if e.code() == -25300 => Ok(()),
            Err(e) => Err(format!("keychain delete failed: {e}")),
        }
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
mod imp {
    use super::SERVICE;

    fn entry(provider: &str) -> Result<keyring::Entry, String> {
        keyring::Entry::new(SERVICE, provider).map_err(|e| format!("credential entry: {e}"))
    }

    pub fn set_key(provider: &str, key: &str) -> Result<(), String> {
        entry(provider)?
            .set_password(key)
            .map_err(|e| format!("credential set failed: {e}"))
    }

    pub fn get_key(provider: &str) -> Option<String> {
        entry(provider).ok()?.get_password().ok()
    }

    pub fn delete_key(provider: &str) -> Result<(), String> {
        match entry(provider)?.delete_credential() {
            Ok(_) => Ok(()),
            // Nothing stored — treat as success (mirrors errSecItemNotFound).
            Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(format!("credential delete failed: {e}")),
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
mod imp {
    use std::collections::HashMap;
    use std::sync::Mutex;

    static STUB: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);

    pub fn set_key(provider: &str, key: &str) -> Result<(), String> {
        STUB.lock().unwrap().get_or_insert_with(HashMap::new).insert(provider.into(), key.into());
        Ok(())
    }

    pub fn get_key(provider: &str) -> Option<String> {
        STUB.lock().unwrap().as_ref()?.get(provider).cloned()
    }

    pub fn delete_key(provider: &str) -> Result<(), String> {
        STUB.lock().unwrap().as_mut()?.remove(provider);
        Ok(())
    }
}

pub use imp::{delete_key, get_key, set_key};
