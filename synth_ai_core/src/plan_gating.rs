//! Plan gating helpers.
//!
//! Centralizes plan/feature access rules so wrappers can stay thin.

use serde_json::Value;

use crate::errors::{CoreError, PlanGatingInfo};
use crate::http::HttpClient;

const UPGRADE_URL: &str = "https://usesynth.ai/pricing";
const DEFAULT_FEATURE: &str = "environment_pools";

fn is_plan_allowed(plan: &str, allow_demo: bool) -> bool {
    matches!(plan, "pro" | "team" | "enterprise") || (allow_demo && plan == "demo")
}

fn feature_flag_allows(flags: &Value, feature: &str) -> bool {
    match flags {
        Value::Object(map) => {
            if let Some(flag) = map.get(feature) {
                match flag {
                    Value::Bool(true) => return true,
                    Value::Object(obj) => {
                        if obj.get("enabled").and_then(|v| v.as_bool()) == Some(true) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
            false
        }
        Value::Array(items) => {
            for item in items {
                match item {
                    Value::String(s) if s == feature => return true,
                    Value::Object(obj) => {
                        if obj.get("feature").and_then(|v| v.as_str()) == Some(feature)
                            && obj.get("enabled").and_then(|v| v.as_bool()) == Some(true)
                        {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
            false
        }
        _ => false,
    }
}

/// Check if the current API key has access to a feature, based on `/api/v1/me`.
///
/// Semantics mirror the Python SDK behavior:
/// - If `/me` is unreachable or returns an error, return `{}` (don't block SDK).
/// - If plan is not allowed but a feature flag enables the feature, allow.
/// - Otherwise, return `CoreError::PlanGating`.
pub async fn check_feature_access(
    api_key: &str,
    base_url: &str,
    timeout_secs: u64,
    feature: Option<&str>,
    allow_demo: bool,
) -> Result<Value, CoreError> {
    let feature = feature.filter(|f| !f.trim().is_empty()).unwrap_or(DEFAULT_FEATURE);

    let client = match HttpClient::new(base_url, api_key, timeout_secs) {
        Ok(client) => client,
        Err(_) => return Ok(Value::Object(Default::default())),
    };

    let data: Value = match client.get_json("/api/v1/me", None).await {
        Ok(value) => value,
        Err(_) => return Ok(Value::Object(Default::default())),
    };

    let plan_raw = data
        .get("plan")
        .or_else(|| data.get("tier"))
        .and_then(|v| v.as_str())
        .unwrap_or("free");
    let plan = plan_raw.to_lowercase();

    if is_plan_allowed(&plan, allow_demo) {
        return Ok(data);
    }

    if let Some(flags) = data.get("feature_flags") {
        if feature_flag_allows(flags, feature) {
            return Ok(data);
        }
    }

    Err(CoreError::PlanGating(PlanGatingInfo {
        feature: feature.to_string(),
        current_plan: plan,
        required_plans: vec!["pro".to_string(), "team".to_string()],
        upgrade_url: UPGRADE_URL.to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_pro_allowed() {
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/api/v1/me");
            then.status(200).json_body(json!({"plan":"pro","org_id":"org-1"}));
        });
        let out = check_feature_access("sk_test", &server.base_url(), 5, None, true)
            .await
            .unwrap();
        assert_eq!(out["plan"], "pro");
    }

    #[tokio::test]
    async fn test_demo_allowed_when_enabled() {
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/api/v1/me");
            then.status(200).json_body(json!({"plan":"demo"}));
        });
        let out = check_feature_access("sk_test", &server.base_url(), 5, None, true)
            .await
            .unwrap();
        assert_eq!(out["plan"], "demo");
    }

    #[tokio::test]
    async fn test_free_rejected_without_flags() {
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/api/v1/me");
            then.status(200).json_body(json!({"plan":"free"}));
        });
        let err = check_feature_access("sk_test", &server.base_url(), 5, None, true)
            .await
            .unwrap_err();
        match err {
            CoreError::PlanGating(info) => {
                assert_eq!(info.current_plan, "free");
                assert_eq!(info.feature, "environment_pools");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[tokio::test]
    async fn test_feature_flag_dict_bool_allows() {
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/api/v1/me");
            then.status(200).json_body(json!({
                "plan":"free",
                "feature_flags": {"environment_pools": true}
            }));
        });
        let out = check_feature_access("sk_test", &server.base_url(), 5, None, true)
            .await
            .unwrap();
        assert_eq!(out["plan"], "free");
    }

    #[tokio::test]
    async fn test_feature_flag_dict_object_allows() {
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/api/v1/me");
            then.status(200).json_body(json!({
                "plan":"free",
                "feature_flags": {"environment_pools": {"enabled": true, "reason": "beta"}}
            }));
        });
        let out = check_feature_access("sk_test", &server.base_url(), 5, None, true)
            .await
            .unwrap();
        assert_eq!(out["plan"], "free");
    }

    #[tokio::test]
    async fn test_feature_flag_list_string_allows() {
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/api/v1/me");
            then.status(200).json_body(json!({
                "plan":"free",
                "feature_flags": ["environment_pools", "other_feature"]
            }));
        });
        let out = check_feature_access("sk_test", &server.base_url(), 5, None, true)
            .await
            .unwrap();
        assert_eq!(out["plan"], "free");
    }

    #[tokio::test]
    async fn test_feature_flag_list_dict_allows() {
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/api/v1/me");
            then.status(200).json_body(json!({
                "plan":"free",
                "feature_flags": [{"feature":"environment_pools","enabled": true}]
            }));
        });
        let out = check_feature_access("sk_test", &server.base_url(), 5, None, true)
            .await
            .unwrap();
        assert_eq!(out["plan"], "free");
    }

    #[tokio::test]
    async fn test_feature_flag_disabled_rejected() {
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/api/v1/me");
            then.status(200).json_body(json!({
                "plan":"free",
                "feature_flags": {"environment_pools": false}
            }));
        });
        let err = check_feature_access("sk_test", &server.base_url(), 5, None, true)
            .await
            .unwrap_err();
        assert!(matches!(err, CoreError::PlanGating(_)));
    }

    #[tokio::test]
    async fn test_wrong_feature_flag_rejected() {
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/api/v1/me");
            then.status(200).json_body(json!({
                "plan":"free",
                "feature_flags": {"other_feature": true}
            }));
        });
        let err = check_feature_access("sk_test", &server.base_url(), 5, None, true)
            .await
            .unwrap_err();
        assert!(matches!(err, CoreError::PlanGating(_)));
    }

    #[tokio::test]
    async fn test_empty_feature_flags_rejected() {
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/api/v1/me");
            then.status(200).json_body(json!({
                "plan":"free",
                "feature_flags": {}
            }));
        });
        let err = check_feature_access("sk_test", &server.base_url(), 5, None, true)
            .await
            .unwrap_err();
        assert!(matches!(err, CoreError::PlanGating(_)));
    }

    #[tokio::test]
    async fn test_tier_field_is_used_for_plan() {
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/api/v1/me");
            then.status(200).json_body(json!({"tier":"free"}));
        });
        let err = check_feature_access("sk_test", &server.base_url(), 5, None, true)
            .await
            .unwrap_err();
        match err {
            CoreError::PlanGating(info) => assert_eq!(info.current_plan, "free"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[tokio::test]
    async fn test_trial_rejected() {
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/api/v1/me");
            then.status(200).json_body(json!({"plan":"trial"}));
        });
        let err = check_feature_access("sk_test", &server.base_url(), 5, None, true)
            .await
            .unwrap_err();
        match err {
            CoreError::PlanGating(info) => assert_eq!(info.current_plan, "trial"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[tokio::test]
    async fn test_unreachable_me_falls_through() {
        // Invalid base URL should not block; returns {}.
        let out = check_feature_access("sk_test", "http://[::1", 5, None, true)
            .await
            .unwrap();
        assert!(out.as_object().unwrap().is_empty());
    }
}
