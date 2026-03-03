use crate::errors::CoreError;
use base64::{
    engine::general_purpose::{self, STANDARD_NO_PAD, URL_SAFE, URL_SAFE_NO_PAD},
    Engine as _,
};
use pasetors::claims::ClaimsValidationRules;
use pasetors::footer::Footer;
use pasetors::keys::AsymmetricPublicKey;
use pasetors::token::UntrustedToken;
use pasetors::{public, version4::V4, Public};
use sodiumoxide::crypto::box_::PublicKey;
use std::env;

pub const SEALED_BOX_ALGORITHM: &str = "libsodium.sealedbox.v1";
pub const CONTAINER_AUTHORIZATION_HEADER_NAME: &str = "X-Synth-Container-Authorization";
pub const SYNTH_CONTAINER_TRUSTED_PUBKEYS_NAME: &str = "SYNTH_CONTAINER_TRUSTED_PUBKEYS";
pub const SYNTH_CONTAINER_AUTH_ISSUER_NAME: &str = "SYNTH_CONTAINER_AUTH_ISSUER";
pub const SYNTH_CONTAINER_AUTH_AUDIENCE: &str = "synth-container";
pub const SYNTH_CONTAINER_AUTH_DEFAULT_ISSUER: &str = "synth-backend";

fn decode_base64_bytes(encoded: &str) -> Option<Vec<u8>> {
    let input = encoded.trim();
    if input.is_empty() {
        return None;
    }
    general_purpose::STANDARD
        .decode(input)
        .ok()
        .or_else(|| STANDARD_NO_PAD.decode(input).ok())
        .or_else(|| URL_SAFE.decode(input).ok())
        .or_else(|| URL_SAFE_NO_PAD.decode(input).ok())
}

#[derive(Clone)]
struct TrustedContainerPublicKey {
    kid: Option<String>,
    key: AsymmetricPublicKey<V4>,
}

fn parse_trusted_public_key_entry(entry: &str) -> Option<(Option<String>, Vec<u8>)> {
    let trimmed = entry.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some((kid, encoded)) = trimmed.split_once(':') {
        let kid = kid.trim();
        if !kid.is_empty() {
            if let Some(decoded) = decode_base64_bytes(encoded) {
                return Some((Some(kid.to_string()), decoded));
            }
        }
    }
    decode_base64_bytes(trimmed).map(|decoded| (None, decoded))
}

fn parse_trusted_public_keys() -> Result<Vec<TrustedContainerPublicKey>, CoreError> {
    let raw = env::var(SYNTH_CONTAINER_TRUSTED_PUBKEYS_NAME).unwrap_or_default();
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut keys = Vec::new();
    for entry in raw.split(',') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }

        let Some((kid, decoded)) = parse_trusted_public_key_entry(trimmed) else {
            return Err(CoreError::InvalidInput(format!(
                "{} contains invalid base64 key entry",
                SYNTH_CONTAINER_TRUSTED_PUBKEYS_NAME
            )));
        };
        let key = AsymmetricPublicKey::<V4>::from(decoded.as_slice()).map_err(|_| {
            CoreError::InvalidInput(format!(
                "{} entry is not a valid Ed25519 public key",
                SYNTH_CONTAINER_TRUSTED_PUBKEYS_NAME
            ))
        })?;
        keys.push(TrustedContainerPublicKey { kid, key });
    }

    Ok(keys)
}

fn extract_bearer_token(header_value: &str) -> Option<String> {
    let trimmed = header_value.trim();
    let token = if trimmed.len() >= 7 && trimmed[..7].eq_ignore_ascii_case("bearer ") {
        trimmed[7..].trim()
    } else {
        trimmed
    };
    if token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

fn scope_claim_values(claims: &pasetors::claims::Claims) -> Vec<String> {
    let mut scopes = Vec::new();
    for claim_name in ["scopes", "scope"] {
        if let Some(value) = claims.get_claim(claim_name) {
            if let Some(single) = value.as_str() {
                let token = single.trim();
                if !token.is_empty() {
                    scopes.push(token.to_string());
                }
            } else if let Some(items) = value.as_array() {
                for item in items {
                    if let Some(token) = item.as_str() {
                        let scoped = token.trim();
                        if !scoped.is_empty() {
                            scopes.push(scoped.to_string());
                        }
                    }
                }
            }
        }
    }
    scopes
}

fn token_footer_kid(untrusted: &UntrustedToken<Public, V4>) -> Option<String> {
    let footer_bytes = untrusted.untrusted_footer();
    if footer_bytes.is_empty() {
        return None;
    }
    let mut footer = Footer::new();
    if footer.parse_bytes(footer_bytes).is_err() {
        return None;
    }
    footer
        .get_claim("kid")
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
}

pub fn verify_container_paseto_header(
    header_value: &str,
    required_scope: Option<&str>,
) -> Result<(), CoreError> {
    let token = extract_bearer_token(header_value).ok_or_else(|| {
        CoreError::Authentication(format!(
            "{} must be a Bearer token",
            CONTAINER_AUTHORIZATION_HEADER_NAME
        ))
    })?;

    let trusted_keys = parse_trusted_public_keys()?;
    if trusted_keys.is_empty() {
        return Err(CoreError::Config(format!(
            "{} is not configured",
            SYNTH_CONTAINER_TRUSTED_PUBKEYS_NAME
        )));
    }

    let mut validation = ClaimsValidationRules::new();
    validation.validate_audience_with(SYNTH_CONTAINER_AUTH_AUDIENCE);
    let issuer = env::var(SYNTH_CONTAINER_AUTH_ISSUER_NAME)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| SYNTH_CONTAINER_AUTH_DEFAULT_ISSUER.to_string());
    validation.validate_issuer_with(&issuer);

    let untrusted = UntrustedToken::<Public, V4>::try_from(token.as_str()).map_err(|_| {
        CoreError::Authentication("container token is malformed or unsupported".to_string())
    })?;
    let footer_kid = token_footer_kid(&untrusted);
    let verify_candidates: Vec<&TrustedContainerPublicKey> =
        if let Some(kid) = footer_kid.as_deref() {
            let filtered: Vec<&TrustedContainerPublicKey> = trusted_keys
                .iter()
                .filter(|entry| entry.kid.as_deref() == Some(kid))
                .collect();
            if filtered.is_empty() {
                return Err(CoreError::Authentication(format!(
                    "container token kid '{kid}' is not trusted"
                )));
            }
            filtered
        } else {
            trusted_keys.iter().collect()
        };

    let mut last_verify_error: Option<String> = None;
    for key in verify_candidates {
        let trusted = match public::verify(&key.key, &untrusted, &validation, None, None) {
            Ok(token) => token,
            Err(err) => {
                last_verify_error = Some(err.to_string());
                continue;
            }
        };

        let claims = trusted.payload_claims().ok_or_else(|| {
            CoreError::Authentication("container token claims missing".to_string())
        })?;

        if let Some(scope) = required_scope.map(str::trim).filter(|s| !s.is_empty()) {
            let scopes = scope_claim_values(claims);
            if !scopes.iter().any(|candidate| candidate == scope) {
                return Err(CoreError::Authentication(format!(
                    "container token missing required scope '{scope}'"
                )));
            }
        }

        return Ok(());
    }

    Err(CoreError::Authentication(format!(
        "container token verification failed{}",
        last_verify_error
            .map(|msg| format!(": {msg}"))
            .unwrap_or_default()
    )))
}

pub fn encrypt_for_backend(pubkey_b64: &str, secret: &[u8]) -> Result<String, CoreError> {
    if pubkey_b64.trim().is_empty() {
        return Err(CoreError::InvalidInput(
            "public key must be a non-empty base64 string".to_string(),
        ));
    }
    if secret.is_empty() {
        return Err(CoreError::InvalidInput(
            "secret must not be empty".to_string(),
        ));
    }

    let key_bytes = general_purpose::STANDARD
        .decode(pubkey_b64.trim())
        .map_err(|_| CoreError::InvalidInput("public key must be base64-encoded".to_string()))?;
    if key_bytes.len() != 32 {
        return Err(CoreError::InvalidInput(
            "public key must be 32 bytes for X25519".to_string(),
        ));
    }

    if sodiumoxide::init().is_err() {
        return Err(CoreError::Internal("failed to init libsodium".to_string()));
    }

    let pk = PublicKey::from_slice(&key_bytes)
        .ok_or_else(|| CoreError::InvalidInput("invalid public key bytes".to_string()))?;
    let cipher = sodiumoxide::crypto::sealedbox::seal(secret, &pk);
    Ok(general_purpose::STANDARD.encode(cipher))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pasetors::claims::Claims;
    use pasetors::keys::{AsymmetricKeyPair, Generate};
    use pasetors::{public, version4::V4};
    use serde_json::{json, Value};
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;
    use uuid::Uuid;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let previous = env::var(key).ok();
            if let Some(v) = value {
                env::set_var(key, v);
            } else {
                env::remove_var(key);
            }
            Self { key, previous }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                env::set_var(self.key, previous);
            } else {
                env::remove_var(self.key);
            }
        }
    }

    fn signed_test_token(scopes: &[&str], issuer: &str, kid: Option<&str>) -> (String, String) {
        let key_pair = AsymmetricKeyPair::<V4>::generate().expect("keypair");
        let mut claims = Claims::new_expires_in(&Duration::from_secs(300)).expect("claims");
        claims.issuer(issuer).expect("issuer");
        claims
            .audience(SYNTH_CONTAINER_AUTH_AUDIENCE)
            .expect("audience");
        claims
            .token_identifier(&Uuid::new_v4().to_string())
            .expect("jti");
        claims
            .add_additional(
                "scopes",
                Value::Array(
                    scopes
                        .iter()
                        .map(|scope| Value::String((*scope).to_string()))
                        .collect(),
                ),
            )
            .expect("scopes");
        let footer = kid.and_then(|value| {
            let mut footer = Footer::new();
            let footer_json = json!({ "kid": value }).to_string();
            footer.parse_string(&footer_json).ok()?;
            Some(footer)
        });
        let token = public::sign(&key_pair.secret, &claims, footer.as_ref(), None).expect("sign");
        let public_key_b64 = general_purpose::STANDARD.encode(key_pair.public.as_bytes());
        (token, public_key_b64)
    }

    #[test]
    fn verify_container_paseto_header_accepts_valid_scope() {
        let _lock = env_lock().lock().expect("env lock");
        let issuer = "synth-backend";
        let (token, public_key_b64) = signed_test_token(&["rollout", "task_info"], issuer, None);
        let _trusted_keys =
            EnvGuard::set(SYNTH_CONTAINER_TRUSTED_PUBKEYS_NAME, Some(&public_key_b64));
        let _issuer = EnvGuard::set(SYNTH_CONTAINER_AUTH_ISSUER_NAME, Some(issuer));

        let result = verify_container_paseto_header(&format!("Bearer {}", token), Some("rollout"));
        assert!(result.is_ok());
    }

    #[test]
    fn verify_container_paseto_header_rejects_missing_scope() {
        let _lock = env_lock().lock().expect("env lock");
        let issuer = "synth-backend";
        let (token, public_key_b64) = signed_test_token(&["task_info"], issuer, None);
        let _trusted_keys =
            EnvGuard::set(SYNTH_CONTAINER_TRUSTED_PUBKEYS_NAME, Some(&public_key_b64));
        let _issuer = EnvGuard::set(SYNTH_CONTAINER_AUTH_ISSUER_NAME, Some(issuer));

        let result = verify_container_paseto_header(&format!("Bearer {}", token), Some("rollout"));
        assert!(result.is_err());
    }

    #[test]
    fn verify_container_paseto_header_matches_kid_scoped_key() {
        let _lock = env_lock().lock().expect("env lock");
        let issuer = "synth-backend";
        let (_other_token, other_key) = signed_test_token(&["rollout"], issuer, Some("k1"));
        let (token, active_key) = signed_test_token(&["rollout"], issuer, Some("k2"));
        let trusted = format!("k1:{other_key},k2:{active_key}");
        let _trusted_keys = EnvGuard::set(SYNTH_CONTAINER_TRUSTED_PUBKEYS_NAME, Some(&trusted));
        let _issuer = EnvGuard::set(SYNTH_CONTAINER_AUTH_ISSUER_NAME, Some(issuer));

        let result = verify_container_paseto_header(&format!("Bearer {}", token), Some("rollout"));
        assert!(result.is_ok());
    }

    #[test]
    fn verify_container_paseto_header_rejects_unknown_kid() {
        let _lock = env_lock().lock().expect("env lock");
        let issuer = "synth-backend";
        let (token, _key_for_k2) = signed_test_token(&["rollout"], issuer, Some("k2"));
        let (_other_token, key_for_k1) = signed_test_token(&["rollout"], issuer, Some("k1"));
        let trusted = format!("k1:{key_for_k1}");
        let _trusted_keys = EnvGuard::set(SYNTH_CONTAINER_TRUSTED_PUBKEYS_NAME, Some(&trusted));
        let _issuer = EnvGuard::set(SYNTH_CONTAINER_AUTH_ISSUER_NAME, Some(issuer));

        let result = verify_container_paseto_header(&format!("Bearer {}", token), Some("rollout"));
        assert!(result.is_err());
    }
}
