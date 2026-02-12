//! x402 client-side helpers (HTTP 402 machine payments).
//!
//! Synth uses x402 (Stripe-compatible) headers:
//! - `PAYMENT-REQUIRED` (server -> client)
//! - `PAYMENT-SIGNATURE` (client -> server)
//!
//! This module implements client-side creation of `PAYMENT-SIGNATURE` for the
//! x402 "exact" scheme on EVM networks (EIP-3009 / TransferWithAuthorization).

use base64::{engine::general_purpose, Engine as _};
use chrono::Utc;
use k256::ecdsa::SigningKey;
#[cfg(test)]
use k256::ecdsa::{RecoveryId, Signature, VerifyingKey};
use k256::elliptic_curve::rand_core::{OsRng, RngCore as _};
use serde::{Deserialize, Serialize};
use sha3::{Digest as _, Keccak256};
use thiserror::Error;

const X402_VERSION_V2: i64 = 2;
const DEFAULT_VALIDITY_BUFFER_SECONDS: i64 = 30;

// EIP-712 type strings (must match upstream x402 / eth_account behavior).
const EIP712_DOMAIN_TYPE: &str =
    "EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)";
const TRANSFER_WITH_AUTHORIZATION_TYPE: &str = "TransferWithAuthorization(address from,address to,uint256 value,uint256 validAfter,uint256 validBefore,bytes32 nonce)";

#[derive(Debug, Error)]
pub enum X402Error {
    #[error("missing payment requirements")]
    MissingRequirements,

    #[error("unsupported payment scheme: {0}")]
    UnsupportedScheme(String),

    #[error("invalid base64 header: {0}")]
    Base64(#[from] base64::DecodeError),

    #[error("invalid json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("invalid private key")]
    InvalidPrivateKey,

    #[error("invalid address: {0}")]
    InvalidAddress(String),

    #[error("invalid u256 decimal: {0}")]
    InvalidU256(String),

    #[error("signing failed")]
    SignFailure,

    #[error("malformed payment payload: {0}")]
    MalformedPayload(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(rename = "mimeType", default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentRequirements {
    pub scheme: String,
    pub network: String,
    pub asset: String,
    pub amount: String,
    #[serde(rename = "payTo")]
    pub pay_to: String,
    #[serde(rename = "maxTimeoutSeconds")]
    pub max_timeout_seconds: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentRequired {
    #[serde(rename = "x402Version")]
    pub x402_version: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource: Option<ResourceInfo>,
    pub accepts: Vec<PaymentRequirements>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentPayload {
    #[serde(rename = "x402Version")]
    pub x402_version: i64,
    pub payload: serde_json::Value,
    pub accepted: PaymentRequirements,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource: Option<ResourceInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
struct Eip3009Authorization {
    from: [u8; 20],
    to: [u8; 20],
    value: [u8; 32],
    valid_after: u64,
    valid_before: u64,
    nonce: [u8; 32],
}

#[derive(Clone)]
pub struct X402Payer {
    signing_key: SigningKey,
    address: String,
}

impl X402Payer {
    /// Build a payer from environment configuration.
    ///
    /// Looks for:
    /// - `SYNTH_X402_PRIVATE_KEY`
    /// - `SYNTH_X402_EVM_PRIVATE_KEY`
    /// - `X402_PRIVATE_KEY`
    pub fn from_env() -> Option<Self> {
        let raw = std::env::var("SYNTH_X402_PRIVATE_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .or_else(|| {
                std::env::var("SYNTH_X402_EVM_PRIVATE_KEY")
                    .ok()
                    .filter(|v| !v.trim().is_empty())
            })
            .or_else(|| {
                std::env::var("X402_PRIVATE_KEY")
                    .ok()
                    .filter(|v| !v.trim().is_empty())
            })?;

        let signing_key = parse_signing_key(&raw).ok()?;
        let address = checksum_address(address_from_signing_key(&signing_key));
        Some(Self {
            signing_key,
            address,
        })
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    /// Create a `PAYMENT-SIGNATURE` header value from a `PAYMENT-REQUIRED` header value.
    pub fn build_payment_signature_header(
        &self,
        payment_required_header: &str,
    ) -> Result<String, X402Error> {
        let payment_required = decode_payment_required_header(payment_required_header)?;
        let accepted = select_accepted_requirements(&payment_required)?;
        let payment_payload = self.create_payment_payload(&payment_required, accepted)?;
        encode_payment_signature_header(&payment_payload)
    }

    fn create_payment_payload(
        &self,
        payment_required: &PaymentRequired,
        accepted: &PaymentRequirements,
    ) -> Result<PaymentPayload, X402Error> {
        if accepted.scheme != "exact" {
            return Err(X402Error::UnsupportedScheme(accepted.scheme.clone()));
        }

        let chain_id = evm_chain_id_from_network(&accepted.network)?;
        let verifying_contract = parse_address(&accepted.asset)?;
        let token_name = accepted
            .extra
            .as_ref()
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("USDC");
        let token_version = accepted
            .extra
            .as_ref()
            .and_then(|v| v.get("version"))
            .and_then(|v| v.as_str())
            .unwrap_or("1");

        let mut nonce_bytes = [0u8; 32];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce_hex = format!("0x{}", hex::encode(nonce_bytes));

        let now = Utc::now().timestamp();
        let valid_after = (now - DEFAULT_VALIDITY_BUFFER_SECONDS).max(0) as u64;
        let valid_before = (now + accepted.max_timeout_seconds.max(1)).max(0) as u64;

        let from_addr = address_from_signing_key(&self.signing_key);
        let to_addr = parse_address(&accepted.pay_to)?;
        let value_u256 = u256_from_dec_str(&accepted.amount)?;

        let auth = Eip3009Authorization {
            from: from_addr,
            to: to_addr,
            value: value_u256,
            valid_after,
            valid_before,
            nonce: nonce_bytes,
        };

        let signature_hex = sign_eip3009_authorization(
            &self.signing_key,
            chain_id,
            verifying_contract,
            token_name,
            token_version,
            &auth,
        )?;

        let payload = serde_json::json!({
            "authorization": {
                "from": self.address,
                "to": accepted.pay_to,
                "value": accepted.amount,
                "validAfter": valid_after.to_string(),
                "validBefore": valid_before.to_string(),
                "nonce": nonce_hex,
            },
            "signature": signature_hex,
        });

        Ok(PaymentPayload {
            x402_version: X402_VERSION_V2,
            payload,
            accepted: accepted.clone(),
            resource: payment_required.resource.clone(),
            extensions: payment_required.extensions.clone(),
        })
    }
}

/// Decode a `PAYMENT-SIGNATURE` header value into a typed payload.
pub fn decode_payment_signature_header(value: &str) -> Result<PaymentPayload, X402Error> {
    let decoded = general_purpose::STANDARD.decode(value.trim())?;
    Ok(serde_json::from_slice(&decoded)?)
}

pub fn decode_payment_required_header(value: &str) -> Result<PaymentRequired, X402Error> {
    let decoded = general_purpose::STANDARD.decode(value.trim())?;
    Ok(serde_json::from_slice(&decoded)?)
}

pub fn encode_payment_signature_header(payload: &PaymentPayload) -> Result<String, X402Error> {
    let json = serde_json::to_vec(payload)?;
    Ok(general_purpose::STANDARD.encode(json))
}

/// Recover the payer address (EIP-55 checksummed) from an exact-scheme payment payload.
///
/// This is primarily used for tests and debugging to validate that we are signing
/// EIP-3009 (TransferWithAuthorization) correctly.
#[cfg(test)]
pub(crate) fn recover_payer_address_from_payment_payload(
    payload: &PaymentPayload,
) -> Result<String, X402Error> {
    if payload.accepted.scheme != "exact" {
        return Err(X402Error::UnsupportedScheme(
            payload.accepted.scheme.clone(),
        ));
    }

    let chain_id = evm_chain_id_from_network(&payload.accepted.network)?;
    let verifying_contract = parse_address(&payload.accepted.asset)?;
    let token_name = payload
        .accepted
        .extra
        .as_ref()
        .and_then(|v| v.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("USDC");
    let token_version = payload
        .accepted
        .extra
        .as_ref()
        .and_then(|v| v.get("version"))
        .and_then(|v| v.as_str())
        .unwrap_or("1");

    let auth_obj = payload
        .payload
        .get("authorization")
        .and_then(|v| v.as_object())
        .ok_or_else(|| X402Error::MalformedPayload("missing authorization".to_string()))?;

    let from_str = auth_obj
        .get("from")
        .and_then(|v| v.as_str())
        .ok_or_else(|| X402Error::MalformedPayload("missing from".to_string()))?;
    let to_str = auth_obj
        .get("to")
        .and_then(|v| v.as_str())
        .ok_or_else(|| X402Error::MalformedPayload("missing to".to_string()))?;
    let value_str = auth_obj
        .get("value")
        .and_then(|v| v.as_str())
        .ok_or_else(|| X402Error::MalformedPayload("missing value".to_string()))?;
    let valid_after_str = auth_obj
        .get("validAfter")
        .and_then(|v| v.as_str())
        .ok_or_else(|| X402Error::MalformedPayload("missing validAfter".to_string()))?;
    let valid_before_str = auth_obj
        .get("validBefore")
        .and_then(|v| v.as_str())
        .ok_or_else(|| X402Error::MalformedPayload("missing validBefore".to_string()))?;
    let nonce_str = auth_obj
        .get("nonce")
        .and_then(|v| v.as_str())
        .ok_or_else(|| X402Error::MalformedPayload("missing nonce".to_string()))?;

    let sig_str = payload
        .payload
        .get("signature")
        .and_then(|v| v.as_str())
        .ok_or_else(|| X402Error::MalformedPayload("missing signature".to_string()))?;

    let nonce_hex = nonce_str
        .trim()
        .strip_prefix("0x")
        .unwrap_or(nonce_str.trim());
    let nonce_bytes =
        hex::decode(nonce_hex).map_err(|_| X402Error::InvalidU256(nonce_str.to_string()))?;
    if nonce_bytes.len() != 32 {
        return Err(X402Error::InvalidU256(nonce_str.to_string()));
    }
    let mut nonce = [0u8; 32];
    nonce.copy_from_slice(&nonce_bytes);

    let auth = Eip3009Authorization {
        from: parse_address(from_str)?,
        to: parse_address(to_str)?,
        value: u256_from_dec_str(value_str)?,
        valid_after: valid_after_str
            .parse::<u64>()
            .map_err(|_| X402Error::InvalidU256(valid_after_str.to_string()))?,
        valid_before: valid_before_str
            .parse::<u64>()
            .map_err(|_| X402Error::InvalidU256(valid_before_str.to_string()))?,
        nonce,
    };

    let digest = eip712_signing_hash_transfer_with_authorization(
        chain_id,
        verifying_contract,
        token_name,
        token_version,
        &auth,
    );

    let sig_hex = sig_str.trim().strip_prefix("0x").unwrap_or(sig_str.trim());
    let sig_bytes = hex::decode(sig_hex).map_err(|_| X402Error::SignFailure)?;
    if sig_bytes.len() != 65 {
        return Err(X402Error::SignFailure);
    }
    let v = sig_bytes[64];
    if v < 27 {
        return Err(X402Error::SignFailure);
    }
    let recid = RecoveryId::try_from(v - 27).map_err(|_| X402Error::SignFailure)?;
    let sig = Signature::from_slice(&sig_bytes[..64]).map_err(|_| X402Error::SignFailure)?;

    let recovered = VerifyingKey::recover_from_prehash(&digest, &sig, recid)
        .map_err(|_| X402Error::SignFailure)?;
    Ok(checksum_address(address_from_verifying_key(&recovered)))
}

fn select_accepted_requirements<'a>(
    payment_required: &'a PaymentRequired,
) -> Result<&'a PaymentRequirements, X402Error> {
    if payment_required.accepts.is_empty() {
        return Err(X402Error::MissingRequirements);
    }

    // Prefer exact scheme.
    if let Some(req) = payment_required
        .accepts
        .iter()
        .find(|req| req.scheme == "exact")
    {
        return Ok(req);
    }

    Ok(&payment_required.accepts[0])
}

fn parse_signing_key(raw: &str) -> Result<SigningKey, X402Error> {
    let s = raw.trim();
    let s = s.strip_prefix("0x").unwrap_or(s);
    let bytes = hex::decode(s).map_err(|_| X402Error::InvalidPrivateKey)?;
    if bytes.len() != 32 {
        return Err(X402Error::InvalidPrivateKey);
    }
    let mut buf = [0u8; 32];
    buf.copy_from_slice(&bytes);
    SigningKey::from_bytes(&buf.into()).map_err(|_| X402Error::InvalidPrivateKey)
}

fn address_from_signing_key(signing_key: &SigningKey) -> [u8; 20] {
    // Ethereum address = last 20 bytes of keccak256(uncompressed_pubkey[1..]).
    let verify_key = signing_key.verifying_key();
    let encoded = verify_key.to_encoded_point(false);
    let bytes = encoded.as_bytes();
    // bytes[0] == 0x04, then 32-byte X, 32-byte Y.
    let hash = keccak256(&bytes[1..]);
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&hash[12..]);
    addr
}

#[cfg(test)]
fn address_from_verifying_key(vk: &VerifyingKey) -> [u8; 20] {
    let encoded = vk.to_encoded_point(false);
    let bytes = encoded.as_bytes();
    let hash = keccak256(&bytes[1..]);
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&hash[12..]);
    addr
}

fn checksum_address(address: [u8; 20]) -> String {
    // EIP-55 checksum.
    let hex_lower = hex::encode(address);
    let hash = keccak256(hex_lower.as_bytes());
    let mut out = String::with_capacity(2 + 40);
    out.push_str("0x");

    for (i, ch) in hex_lower.chars().enumerate() {
        let nibble = if i % 2 == 0 {
            (hash[i / 2] >> 4) & 0x0f
        } else {
            hash[i / 2] & 0x0f
        };
        if ch.is_ascii_hexdigit() && ch.is_ascii_alphabetic() && nibble >= 8 {
            out.push(ch.to_ascii_uppercase());
        } else {
            out.push(ch);
        }
    }

    out
}

fn parse_address(raw: &str) -> Result<[u8; 20], X402Error> {
    let s = raw.trim();
    let s = s.strip_prefix("0x").unwrap_or(s);
    if s.len() != 40 {
        return Err(X402Error::InvalidAddress(raw.to_string()));
    }
    let bytes = hex::decode(s).map_err(|_| X402Error::InvalidAddress(raw.to_string()))?;
    let mut out = [0u8; 20];
    out.copy_from_slice(&bytes);
    Ok(out)
}

fn evm_chain_id_from_network(network: &str) -> Result<u64, X402Error> {
    // Only support CAIP-2 "eip155:<chain_id>" for now.
    let network = network.trim();
    let Some(rest) = network.strip_prefix("eip155:") else {
        return Err(X402Error::InvalidU256(network.to_string()));
    };
    rest.parse::<u64>()
        .map_err(|_| X402Error::InvalidU256(network.to_string()))
}

fn keccak256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(data);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn encode_address(addr: [u8; 20]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[12..].copy_from_slice(&addr);
    out
}

fn encode_u64(value: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[24..].copy_from_slice(&value.to_be_bytes());
    out
}

fn u256_from_dec_str(raw: &str) -> Result<[u8; 32], X402Error> {
    let s = raw.trim();
    if s.is_empty() {
        return Err(X402Error::InvalidU256(raw.to_string()));
    }
    let mut out = [0u8; 32];
    for ch in s.chars() {
        if !ch.is_ascii_digit() {
            return Err(X402Error::InvalidU256(raw.to_string()));
        }
        let digit = (ch as u8 - b'0') as u16;

        // out = out * 10 + digit  (big-endian base-256)
        let mut carry: u16 = digit;
        for i in (0..32).rev() {
            let val: u16 = (out[i] as u16) * 10 + carry;
            out[i] = (val & 0xff) as u8;
            carry = val >> 8;
        }
        if carry != 0 {
            return Err(X402Error::InvalidU256(raw.to_string()));
        }
    }
    Ok(out)
}

fn eip712_domain_separator(
    name: &str,
    version: &str,
    chain_id: u64,
    verifying_contract: [u8; 20],
) -> [u8; 32] {
    let type_hash = keccak256(EIP712_DOMAIN_TYPE.as_bytes());
    let name_hash = keccak256(name.as_bytes());
    let version_hash = keccak256(version.as_bytes());
    let chain_id_enc = encode_u64(chain_id);
    let verifying_contract_enc = encode_address(verifying_contract);

    let mut encoded = Vec::with_capacity(32 * 5);
    encoded.extend_from_slice(&type_hash);
    encoded.extend_from_slice(&name_hash);
    encoded.extend_from_slice(&version_hash);
    encoded.extend_from_slice(&chain_id_enc);
    encoded.extend_from_slice(&verifying_contract_enc);
    keccak256(&encoded)
}

fn eip712_transfer_with_authorization_struct_hash(auth: &Eip3009Authorization) -> [u8; 32] {
    let type_hash = keccak256(TRANSFER_WITH_AUTHORIZATION_TYPE.as_bytes());

    let mut encoded = Vec::with_capacity(32 * 7);
    encoded.extend_from_slice(&type_hash);
    encoded.extend_from_slice(&encode_address(auth.from));
    encoded.extend_from_slice(&encode_address(auth.to));
    encoded.extend_from_slice(&auth.value);
    encoded.extend_from_slice(&encode_u64(auth.valid_after));
    encoded.extend_from_slice(&encode_u64(auth.valid_before));
    encoded.extend_from_slice(&auth.nonce);
    keccak256(&encoded)
}

fn eip712_signing_hash_transfer_with_authorization(
    chain_id: u64,
    verifying_contract: [u8; 20],
    token_name: &str,
    token_version: &str,
    auth: &Eip3009Authorization,
) -> [u8; 32] {
    let domain_sep =
        eip712_domain_separator(token_name, token_version, chain_id, verifying_contract);
    let struct_hash = eip712_transfer_with_authorization_struct_hash(auth);

    let mut encoded = Vec::with_capacity(2 + 32 + 32);
    encoded.extend_from_slice(&[0x19, 0x01]);
    encoded.extend_from_slice(&domain_sep);
    encoded.extend_from_slice(&struct_hash);
    keccak256(&encoded)
}

fn sign_eip3009_authorization(
    signing_key: &SigningKey,
    chain_id: u64,
    verifying_contract: [u8; 20],
    token_name: &str,
    token_version: &str,
    auth: &Eip3009Authorization,
) -> Result<String, X402Error> {
    let digest = eip712_signing_hash_transfer_with_authorization(
        chain_id,
        verifying_contract,
        token_name,
        token_version,
        auth,
    );

    let (signature, recid) = signing_key
        .sign_prehash_recoverable(&digest)
        .map_err(|_| X402Error::SignFailure)?;

    let sig64 = signature.to_bytes();
    let recid_u8: u8 = recid.into();
    let v = recid_u8.saturating_add(27);

    let mut sig65 = [0u8; 65];
    sig65[..64].copy_from_slice(sig64.as_slice());
    sig65[64] = v;

    Ok(format!("0x{}", hex::encode(sig65)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eip712_digest_matches_python_x402() {
        // Golden computed with x402 (python) + eth_account:
        // - chain_id=84532
        // - verifying_contract=0x036CbD53842c5426634e7929541eC2318f3dCF7e
        // - name=USDC version=2
        // - from=0xFCAd0B19bB29D4674531d6f115237E16AfCE377c
        // - to=0x1111111111111111111111111111111111111111
        // - value=250000
        // - validAfter=1700000000 validBefore=1700000300
        // - nonce=0x1111... (32 bytes)
        let signing_key =
            parse_signing_key("0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
                .unwrap();
        let from = address_from_signing_key(&signing_key);
        assert_eq!(
            checksum_address(from),
            "0xFCAd0B19bB29D4674531d6f115237E16AfCE377c"
        );

        let auth = Eip3009Authorization {
            from,
            to: parse_address("0x1111111111111111111111111111111111111111").unwrap(),
            value: u256_from_dec_str("250000").unwrap(),
            valid_after: 1_700_000_000u64,
            valid_before: 1_700_000_300u64,
            nonce: [0x11u8; 32],
        };

        let digest = eip712_signing_hash_transfer_with_authorization(
            84532,
            parse_address("0x036CbD53842c5426634e7929541eC2318f3dCF7e").unwrap(),
            "USDC",
            "2",
            &auth,
        );

        assert_eq!(
            format!("0x{}", hex::encode(digest)),
            "0x798f516cfe5a9cc10934b46623d65b0facb181da516e4dc7cfea11a16cc44a81"
        );

        let sig = sign_eip3009_authorization(
            &signing_key,
            84532,
            parse_address("0x036CbD53842c5426634e7929541eC2318f3dCF7e").unwrap(),
            "USDC",
            "2",
            &auth,
        )
        .unwrap();

        assert_eq!(sig, "0xdbed925a525095c7d6933ab969b8421521c160de32d28e4628fa01913908382745a0b499c5562376e4d275e0626b30355fa03342bb7168dfe3dfae277eab4eb41c");
    }
}
