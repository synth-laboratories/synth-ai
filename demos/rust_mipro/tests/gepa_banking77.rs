use std::time::Duration;

use futures_util::{FutureExt, StreamExt};
use serde_json::{json, Map, Value};
use synth_ai::{
    create_local_api, LocalApiConfig, LocalApiError, PolicyOptimizationJob,
    PolicyOptimizationJobConfig, RolloutMetrics, RolloutRequest, RolloutResponse, SynthClient,
};

const APP_ID: &str = "banking77";
const APP_NAME: &str = "Banking77 Intent Classification";
const APP_DESC: &str = "Minimal Banking77 task app (Rust demo)";

fn banking77_examples() -> Vec<(&'static str, &'static str)> {
    vec![
        ("My card is not working at all.", "card_not_working"),
        ("I want to change my card pin.", "change_pin"),
        ("My card payment was declined.", "declined_card_payment"),
        ("I did a bank transfer and it never arrived.", "transfer_not_received_by_recipient"),
        ("Cash withdrawal was charged twice.", "transaction_charged_twice"),
        ("How do I top up by bank transfer?", "top_up_by_bank_transfer_charge"),
        ("My card was swallowed by the ATM.", "card_swallowed"),
        ("My balance is wrong after a transfer.", "balance_not_updated_after_bank_transfer"),
    ]
}

fn banking77_labels() -> Vec<&'static str> {
    vec![
        "activate_my_card",
        "age_limit",
        "apple_pay_or_google_pay",
        "atm_support",
        "automatic_top_up",
        "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit",
        "beneficiary_not_allowed",
        "cancel_transfer",
        "card_about_to_expire",
        "card_acceptance",
        "card_arrival",
        "card_delivery_estimate",
        "card_linking",
        "card_not_working",
        "card_payment_fee_charged",
        "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate",
        "card_swallowed",
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
        "change_pin",
        "compromised_card",
        "contactless_not_working",
        "country_support",
        "declined_card_payment",
        "declined_cash_withdrawal",
        "declined_transfer",
        "direct_debit_payment_not_recognised",
        "disposable_card_limits",
        "edit_personal_details",
        "exchange_charge",
        "exchange_rate",
        "exchange_via_app",
        "extra_charge_on_statement",
        "failed_transfer",
        "fiat_currency_support",
        "get_disposable_virtual_card",
        "get_physical_card",
        "getting_spare_card",
        "getting_virtual_card",
        "lost_or_stolen_card",
        "lost_or_stolen_phone",
        "order_physical_card",
        "passcode_forgotten",
        "pending_card_payment",
        "pending_cash_withdrawal",
        "pending_top_up",
        "pending_transfer",
        "pin_blocked",
        "receiving_money",
        "Refund_not_showing_up",
        "request_refund",
        "reverted_card_payment?",
        "supported_cards_and_currencies",
        "terminate_account",
        "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge",
        "top_up_by_cash_or_cheque",
        "top_up_failed",
        "top_up_limits",
        "top_up_reverted",
        "topping_up_by_card",
        "transaction_charged_twice",
        "transfer_fee_charged",
        "transfer_into_account",
        "transfer_not_received_by_recipient",
        "transfer_timing",
        "unable_to_verify_identity",
        "verify_my_identity",
        "verify_source_of_funds",
        "verify_top_up",
        "virtual_card_not_working",
        "visa_or_mastercard",
        "why_verify_identity",
        "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ]
}

fn available_intents() -> String {
    banking77_labels()
        .into_iter()
        .enumerate()
        .map(|(idx, label)| format!("{}. {}", idx + 1, label))
        .collect::<Vec<_>>()
        .join("\n")
}

fn choose_seed(env: &serde_json::Value) -> u64 {
    env.get("seed")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            env.get("config")
                .and_then(|c| c.get("seed"))
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(0)
}

fn lookup_example(seed: u64) -> (&'static str, &'static str) {
    let data = banking77_examples();
    let idx = (seed as usize) % data.len();
    data[idx]
}

async fn wait_for_health_check(base_url: &str, api_key: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let url = format!("{base_url}/health");
    let start = std::time::Instant::now();
    loop {
        let resp = client
            .get(&url)
            .header("X-API-Key", api_key)
            .send()
            .await;
        if let Ok(resp) = resp {
            if resp.status().is_success() {
                break;
            }
        }
        if start.elapsed() > Duration::from_secs(10) {
            anyhow::bail!("LocalAPI health check failed");
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    Ok(())
}

fn normalize_inference_url(url: &str) -> String {
    let mut candidate = url.trim().to_string();
    if candidate.is_empty() {
        candidate = "https://api.openai.com/v1/chat/completions".to_string();
    }

    let parsed = match url::Url::parse(&candidate) {
        Ok(url) => url,
        Err(_) => return candidate,
    };

    let mut path = parsed.path().trim_end_matches('/').to_string();
    let mut query = parsed.query().unwrap_or("").to_string();

    if !query.is_empty() && query.contains('/') {
        let mut parts = query.splitn(2, '/');
        let base_query = parts.next().unwrap_or("");
        let mut remainder = parts.next().unwrap_or("").to_string();
        let mut extra_query = String::new();
        for sep in ['&', '?'] {
            if let Some(idx) = remainder.find(sep) {
                extra_query = remainder[idx + 1..].to_string();
                remainder = remainder[..idx].to_string();
                break;
            }
        }
        let query_path = format!("/{}", remainder.trim_start_matches('/'));
        let mut merged_query = Vec::new();
        if !base_query.is_empty() {
            merged_query.push(base_query.to_string());
        }
        if !extra_query.is_empty() {
            merged_query.push(extra_query);
        }
        query = merged_query.join("&");
        if !query_path.is_empty() && query_path != "/" {
            if path.is_empty() {
                path = query_path;
            } else {
                path = format!("{}/{}", path.trim_end_matches('/'), query_path.trim_start_matches('/'));
            }
        }
    }

    if path.ends_with("/v1/chat/completions") || path.ends_with("/chat/completions") {
        let mut rebuilt = parsed.clone();
        rebuilt.set_path(&path);
        rebuilt.set_query(if query.is_empty() { None } else { Some(&query) });
        return rebuilt.to_string();
    }

    let new_path = if path.contains("/v1/") || path.ends_with("/v1") {
        format!("{}/chat/completions", path.trim_end_matches('/'))
    } else if path.ends_with("/chat") {
        format!("{}/completions", path.trim_end_matches('/'))
    } else if path.is_empty() {
        "/v1/chat/completions".to_string()
    } else {
        format!("{}/v1/chat/completions", path.trim_end_matches('/'))
    };

    let mut rebuilt = parsed;
    rebuilt.set_path(&new_path);
    rebuilt.set_query(if query.is_empty() { None } else { Some(&query) });
    rebuilt.to_string()
}

async fn call_llm(
    query: &str,
    inference_url: &str,
    model: &str,
    api_key: Option<&str>,
) -> Result<(String, Option<String>, Value, Vec<Value>), LocalApiError> {
    let intents = available_intents();
    let user_msg = format!(
        "Customer Query: {query}\n\nAvailable Intents:\n{intents}\n\nClassify this query into one of the above banking intents using the tool call."
    );
    let messages = vec![
        json!({ "role": "system", "content": "You are an expert banking assistant that classifies customer queries into banking intents. Given a customer message, respond with exactly one intent label from the provided list using the banking77_classify tool." }),
        json!({ "role": "user", "content": user_msg }),
    ];

    let payload = json!({
        "model": model,
        "messages": messages,
        "tools": [{
            "type": "function",
            "function": {
                "name": "banking77_classify",
                "description": "Return the predicted banking77 intent label.",
                "parameters": {
                    "type": "object",
                    "properties": { "intent": { "type": "string" } },
                    "required": ["intent"]
                }
            }
        }],
        "tool_choice": { "type": "function", "function": { "name": "banking77_classify" } }
    });

    let url = normalize_inference_url(inference_url);
    let client = reqwest::Client::new();
    let mut req = client.post(&url).json(&payload);
    if let Some(key) = api_key {
        req = req.header("X-API-Key", key).bearer_auth(key);
    }

    let resp = req
        .timeout(Duration::from_secs(120))
        .send()
        .await
        .map_err(|err| LocalApiError::internal(err.to_string()))?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(LocalApiError::internal(format!(
            "LLM error {}: {}",
            status, body
        )));
    }

    let candidate_id = resp
        .headers()
        .get("x-mipro-candidate-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let data: Value = resp
        .json()
        .await
        .map_err(|err| LocalApiError::internal(err.to_string()))?;

    let choices = data
        .get("choices")
        .and_then(|v| v.as_array())
        .ok_or_else(|| LocalApiError::internal("missing choices".to_string()))?;
    let tool_calls = choices
        .get(0)
        .and_then(|v| v.get("message"))
        .and_then(|v| v.get("tool_calls"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| LocalApiError::internal("missing tool_calls".to_string()))?;
    let args_raw = tool_calls
        .get(0)
        .and_then(|v| v.get("function"))
        .and_then(|v| v.get("arguments"))
        .ok_or_else(|| LocalApiError::internal("missing arguments".to_string()))?;

    let args_val = if args_raw.is_string() {
        let raw = args_raw.as_str().unwrap_or("{}");
        serde_json::from_str::<Value>(raw)
            .map_err(|err| LocalApiError::internal(err.to_string()))?
    } else {
        args_raw.clone()
    };
    let intent = args_val
        .get("intent")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Ok((intent, candidate_id, data, messages))
}

#[tokio::test]
#[ignore]
async fn gepa_banking77_rust_demo() -> anyhow::Result<()> {
    let backend_url =
        std::env::var("SYNTH_BACKEND_URL").unwrap_or_else(|_| "http://localhost:8000".to_string());
    let api_key = std::env::var("SYNTH_API_KEY")
        .expect("SYNTH_API_KEY is required for submitting GEPA jobs");
    let task_api_key =
        std::env::var("ENVIRONMENT_API_KEY").unwrap_or_else(|_| "local-dev-key".to_string());

    let config = LocalApiConfig::new(APP_ID, APP_NAME, APP_DESC, move |req: RolloutRequest| {
        async move {
            let seed = choose_seed(&req.env);
            let (text, label) = lookup_example(seed);
            let policy_cfg = req
                .policy
                .get("config")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()));

            let inference_url = policy_cfg
                .get("inference_url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| LocalApiError::bad_request("missing inference_url"))?
                .to_string();

            let model = policy_cfg
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("gpt-4.1-nano");
            let api_key = policy_cfg
                .get("api_key")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let (prediction, candidate_id, llm_response, llm_messages) =
                call_llm(&text, &inference_url, model, api_key.as_deref()).await?;

            let score = if prediction == label { 1.0 } else { 0.0 };
            let reward = RolloutMetrics {
                outcome_reward: score,
                event_rewards: None,
                outcome_objectives: None,
                event_objectives: None,
                instance_objectives: None,
                details: Default::default(),
            };

            let mut extra = Map::new();
            if let Some(candidate_id) = candidate_id {
                extra.insert(
                    "metadata".to_string(),
                    json!({ "mipro_candidate_id": candidate_id }),
                );
            }

            Ok(RolloutResponse {
                trace_correlation_id: req.trace_correlation_id.clone(),
                reward_info: reward,
                trace: Some(json!({
                    "inference": {
                        "messages": llm_messages,
                        "response": llm_response,
                    }
                })),
                inference_url: Some(inference_url),
                artifact: Some(json!({
                    "text": text,
                    "label": label,
                    "prediction": prediction,
                })),
                success_status: Some("success".to_string()),
                status_detail: None,
                override_application_results: None,
                extra,
            })
        }
        .boxed()
    });

    let mut config = config;
    config.require_api_key = true;
    config.api_keys = vec![task_api_key.clone()];

    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let local_addr = listener.local_addr()?;
    let local_base_url = format!("http://{local_addr}");
    let app = create_local_api(config);
    let router = app.router();
    let server_handle = tokio::spawn(async move {
        axum::Server::from_tcp(listener)
            .unwrap()
            .serve(router.into_make_service())
            .await
            .unwrap();
    });

    wait_for_health_check(&local_base_url, &task_api_key).await?;

    let client = SynthClient::new(backend_url, api_key.clone());

    let system_prompt = "You are an expert banking assistant that classifies customer queries into \
banking intents. Given a customer message, respond with exactly one intent label from the provided \
list using the banking77_classify tool.";
    let user_prompt = "Customer: {{query}}\n\nAvailable intents:\n{{available_intents}}";

    let config_body = json!({
        "prompt_learning": {
            "algorithm": "gepa",
            "task_app_id": APP_ID,
            "task_app_url": local_base_url,
            "initial_prompt": {
                "id": "banking77_pattern",
                "name": "Banking77 Classification",
                "messages": [
                    { "role": "system", "order": 0, "pattern": system_prompt },
                    { "role": "user", "order": 1, "pattern": user_prompt },
                ],
                "wildcards": {
                    "query": "REQUIRED",
                    "available_intents": "OPTIONAL",
                },
            },
            "policy": {
                "model": "gpt-4.1-nano",
                "provider": "openai",
                "inference_mode": "synth_hosted",
                "api_key": api_key,
                "temperature": 0.0,
                "max_completion_tokens": 256,
            },
            "env_config": {
                "split": "train",
                "available_intents": available_intents(),
            },
            "gepa": {
                "env_name": APP_ID,
                "evaluation": {
                    "seeds": [0, 1, 2, 3, 4, 5],
                    "validation_seeds": [6, 7],
                },
                "rollout": {
                    "budget": 20,
                    "max_concurrent": 4,
                    "minibatch_size": 2,
                },
                "population": {
                    "initial_size": 2,
                    "num_generations": 2,
                    "children_per_generation": 1,
                },
            },
        }
    });

    let job = PolicyOptimizationJob::submit(
        client.clone(),
        &PolicyOptimizationJobConfig::from_json(config_body),
    )
    .await?;

    let mut stream = job.stream_events().await?;
    let mut seen = 0usize;
    while let Ok(Some(event)) = tokio::time::timeout(Duration::from_secs(5), stream.next()).await {
        println!("[sse] {:?}", event);
        seen += 1;
        if seen >= 5 {
            break;
        }
    }

    let status = job.status().await?;
    println!("job status: {status}");

    let results = job.results().await?;
    println!(
        "best_score: {:?}, best_prompt: {:?}",
        results.best_score, results.best_prompt
    );

    server_handle.abort();
    Ok(())
}
