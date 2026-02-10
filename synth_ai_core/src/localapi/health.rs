use crate::errors::CoreError;
use serde_json::Value;
use std::time::Duration;

pub async fn task_app_health(task_app_url: &str) -> Result<Value, CoreError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(CoreError::Http)?;

    async fn try_request(
        client: &reqwest::Client,
        url: &str,
        method: reqwest::Method,
    ) -> Result<Option<Value>, CoreError> {
        let resp = client.request(method, url).send().await;
        let resp = match resp {
            Ok(resp) => resp,
            Err(_) => return Ok(None),
        };
        let status = resp.status();
        if status.is_success() || status.is_redirection() {
            let mut map = serde_json::Map::new();
            map.insert("ok".to_string(), Value::Bool(true));
            map.insert("status".to_string(), Value::Number(status.as_u16().into()));
            return Ok(Some(Value::Object(map)));
        }
        Ok(None)
    }

    if let Some(result) = try_request(&client, task_app_url, reqwest::Method::HEAD).await? {
        return Ok(result);
    }
    if let Some(result) = try_request(&client, task_app_url, reqwest::Method::GET).await? {
        return Ok(result);
    }

    let mut map = serde_json::Map::new();
    map.insert("ok".to_string(), Value::Bool(false));
    map.insert("status".to_string(), Value::Null);
    Ok(Value::Object(map))
}
