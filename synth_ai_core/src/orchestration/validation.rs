//! Prompt learning configuration validation.
//!
//! Mirrors the Python SDK prompt_learning_validation logic so both
//! Rust and Python clients share the same core validation behavior.

use once_cell::sync::Lazy;
use serde_json::{Map, Value};
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct PromptLearningValidationResult {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub info: Vec<String>,
}

impl PromptLearningValidationResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    fn add_error(&mut self, msg: String) {
        self.errors.push(msg);
    }

    fn add_warning(&mut self, msg: String) {
        self.warnings.push(msg);
    }

    fn add_info(&mut self, msg: String) {
        self.info.push(msg);
    }
}

const KNOWN_TOP_LEVEL_SECTIONS: &[&str] = &["prompt_learning", "display", "termination_config"];

const KNOWN_PROMPT_LEARNING_FIELDS: &[&str] = &[
    "algorithm",
    "task_app_url",
    "task_app_api_key",
    "task_app_id",
    "initial_prompt",
    "policy",
    "mipro",
    "gepa",
    "verifier",
    "proxy_models",
    "env_config",
    "env_name",
    "termination_config",
    "results_folder",
    "bootstrap_train_seeds",
    "online_pool",
    "test_pool",
    "reference_pool",
    "auto_discover_patterns",
    "use_byok",
];

const KNOWN_POLICY_FIELDS: &[&str] = &[
    "model",
    "provider",
    "inference_url",
    "inference_mode",
    "temperature",
    "max_completion_tokens",
    "policy_name",
    "config",
    "context_override",
    "timeout",
];

const KNOWN_TERMINATION_CONFIG_FIELDS: &[&str] = &[
    "max_cost_usd",
    "max_trials",
    "max_seconds",
    "max_time_seconds",
    "max_rollouts",
    "max_trials_without_improvement",
    "pessimism_enabled",
    "max_category_costs_usd",
];

const KNOWN_GEPA_FIELDS: &[&str] = &[
    "env_name",
    "env_config",
    "rng_seed",
    "proposer_type",
    "proposer_effort",
    "proposer_output_tokens",
    "metaprompt",
    "modules",
    "rollout",
    "evaluation",
    "mutation",
    "population",
    "archive",
    "token",
    "verifier",
    "proxy_models",
    "adaptive_pool",
    "adaptive_batch",
    "rollout_budget",
    "max_concurrent_rollouts",
    "minibatch_size",
    "evaluation_seeds",
    "validation_seeds",
    "test_pool",
    "validation_pool",
    "validation_top_k",
    "mutation_rate",
    "mutation_llm_model",
    "mutation_llm_provider",
    "mutation_llm_inference_url",
    "mutation_prompt",
    "initial_population_size",
    "num_generations",
    "children_per_generation",
    "crossover_rate",
    "selection_pressure",
    "patience_generations",
    "archive_size",
    "pareto_set_size",
    "pareto_eps",
    "feedback_fraction",
    "max_token_limit",
    "token_counting_model",
    "enforce_pattern_token_limit",
    "max_spend_usd",
    "unified_optimization",
    "baseline_context_override",
    "proposed_prompt_max_tokens",
    "use_byok",
];

const KNOWN_GEPA_ROLLOUT_FIELDS: &[&str] =
    &["budget", "max_concurrent", "minibatch_size", "timeout"];

const KNOWN_GEPA_EVALUATION_FIELDS: &[&str] = &[
    "seeds",
    "train_seeds",
    "validation_seeds",
    "val_seeds",
    "test_pool",
    "validation_pool",
    "validation_top_k",
];

const KNOWN_GEPA_MUTATION_FIELDS: &[&str] = &[
    "rate",
    "llm_model",
    "llm_provider",
    "llm_inference_url",
    "prompt",
];

const KNOWN_GEPA_POPULATION_FIELDS: &[&str] = &[
    "initial_size",
    "num_generations",
    "children_per_generation",
    "crossover_rate",
    "selection_pressure",
    "patience_generations",
];

const KNOWN_GEPA_ARCHIVE_FIELDS: &[&str] =
    &["size", "pareto_set_size", "pareto_eps", "feedback_fraction"];

const KNOWN_GEPA_TOKEN_FIELDS: &[&str] = &[
    "max_limit",
    "counting_model",
    "enforce_pattern_limit",
    "max_spend_usd",
];

const KNOWN_MIPRO_FIELDS: &[&str] = &[
    "task_app_url",
    "task_app_api_key",
    "task_app_id",
    "num_iterations",
    "num_evaluations_per_iteration",
    "batch_size",
    "max_concurrent",
    "env_name",
    "env_config",
    "meta_model",
    "meta_model_provider",
    "meta_model_inference_url",
    "few_shot_score_threshold",
    "results_file",
    "max_wall_clock_seconds",
    "max_total_tokens",
    "policy_config",
    "meta",
    "modules",
    "seeds",
    "proposer_effort",
    "proposer_output_tokens",
    "max_token_limit",
    "max_spend_usd",
    "token_counting_model",
    "enforce_token_limit",
    "tpe",
    "demo",
    "grounding",
    "meta_update",
    "verifier",
    "proxy_models",
    "adaptive_pool",
    "spec_path",
    "spec_max_tokens",
    "spec_include_examples",
    "spec_priority_threshold",
    "metaprompt",
    "bootstrap_train_seeds",
    "online_pool",
    "test_pool",
    "reference_pool",
    "min_bootstrap_demos",
];

const KNOWN_VERIFIER_FIELDS: &[&str] = &[
    "enabled",
    "reward_source",
    "backend_base",
    "backend_api_key_env",
    "backend_provider",
    "backend_model",
    "verifier_graph_id",
    "backend_event_enabled",
    "backend_outcome_enabled",
    "backend_options",
    "concurrency",
    "timeout",
    "weight_env",
    "weight_event",
    "weight_outcome",
    "spec_path",
    "spec_max_tokens",
    "spec_context",
];

const KNOWN_ADAPTIVE_POOL_FIELDS: &[&str] = &[
    "level",
    "anchor_size",
    "pool_init_size",
    "pool_min_size",
    "warmup_iters",
    "anneal_stop_iter",
    "pool_update_period",
    "min_evals_per_example",
    "k_info_prompts",
    "info_buffer_factor",
    "info_epsilon",
    "anchor_selection_method",
    "exploration_strategy",
    "heatup_reserve_pool",
    "heatup_trigger",
    "heatup_size",
    "heatup_cooldown_trials",
    "heatup_schedule",
];

const KNOWN_ADAPTIVE_BATCH_FIELDS: &[&str] = &[
    "level",
    "reflection_minibatch_size",
    "min_local_improvement",
    "val_evaluation_mode",
    "val_subsample_size",
    "candidate_selection_strategy",
];

const KNOWN_PROXY_MODELS_FIELDS: &[&str] = &[
    "hi_provider",
    "hi_model",
    "lo_provider",
    "lo_model",
    "n_min_hi",
    "r2_thresh",
    "r2_stop",
    "sigma_max",
    "sigma_stop",
    "verify_every",
    "proxy_patience_usd",
];

fn deprecated_message(key: &str) -> Option<&'static str> {
    match key {
        "display" => Some(
            "The [display] section is deprecated and ignored by the backend. Remove it from your config.",
        ),
        "results_folder" => Some(
            "'results_folder' is deprecated and ignored by the backend. Remove it from your config.",
        ),
        "rollout_budget" => Some(
            "Use [prompt_learning.gepa.rollout].budget instead of flat rollout_budget.",
        ),
        "max_concurrent_rollouts" => Some(
            "Use [prompt_learning.gepa.rollout].max_concurrent instead.",
        ),
        "evaluation_seeds" => Some(
            "Use [prompt_learning.gepa.evaluation].seeds instead of flat evaluation_seeds.",
        ),
        "validation_seeds" => Some(
            "Use [prompt_learning.gepa.evaluation].validation_seeds instead.",
        ),
        "backend_rubric_id" => Some("Use 'verifier_graph_id' in [prompt_learning.verifier]."),
        _ => None,
    }
}

fn contains_known(known: &[&str], key: &str) -> bool {
    known.iter().any(|k| *k == key)
}

fn check_unknown_fields(
    map: &Map<String, Value>,
    known_fields: &[&str],
    section_path: &str,
    result: &mut PromptLearningValidationResult,
) {
    for key in map.keys() {
        if !contains_known(known_fields, key) {
            result.add_warning(format!(
                "Unknown field '{}' in [{}]. This field will be ignored. Check spelling or remove it.",
                key, section_path
            ));
        }
    }
}

fn check_deprecated_fields(
    map: &Map<String, Value>,
    section_path: &str,
    result: &mut PromptLearningValidationResult,
) {
    for key in map.keys() {
        if let Some(msg) = deprecated_message(key) {
            result.add_warning(format!("[{}] {}", section_path, msg));
        }
    }
}

fn validate_gepa_config(
    gepa: &Map<String, Value>,
    result: &mut PromptLearningValidationResult,
    path_prefix: &str,
) {
    check_unknown_fields(gepa, KNOWN_GEPA_FIELDS, "prompt_learning.gepa", result);

    for field in [
        "rollout_budget",
        "max_concurrent_rollouts",
        "evaluation_seeds",
        "validation_seeds",
    ] {
        if gepa.contains_key(field) {
            result.add_info(format!(
                "Using flat '{}' in [prompt_learning.gepa] - consider migrating to nested structure for clarity",
                field
            ));
        }
    }

    if let Some(Value::Object(rollout)) = gepa.get("rollout") {
        check_unknown_fields(
            rollout,
            KNOWN_GEPA_ROLLOUT_FIELDS,
            "prompt_learning.gepa.rollout",
            result,
        );
    }
    if let Some(Value::Object(evaluation)) = gepa.get("evaluation") {
        check_unknown_fields(
            evaluation,
            KNOWN_GEPA_EVALUATION_FIELDS,
            "prompt_learning.gepa.evaluation",
            result,
        );
    }
    if let Some(Value::Object(mutation)) = gepa.get("mutation") {
        check_unknown_fields(
            mutation,
            KNOWN_GEPA_MUTATION_FIELDS,
            "prompt_learning.gepa.mutation",
            result,
        );
    }
    if let Some(Value::Object(population)) = gepa.get("population") {
        check_unknown_fields(
            population,
            KNOWN_GEPA_POPULATION_FIELDS,
            "prompt_learning.gepa.population",
            result,
        );
    }
    if let Some(Value::Object(archive)) = gepa.get("archive") {
        check_unknown_fields(
            archive,
            KNOWN_GEPA_ARCHIVE_FIELDS,
            "prompt_learning.gepa.archive",
            result,
        );
    }
    if let Some(Value::Object(token)) = gepa.get("token") {
        check_unknown_fields(
            token,
            KNOWN_GEPA_TOKEN_FIELDS,
            "prompt_learning.gepa.token",
            result,
        );
    }
    if let Some(Value::Object(adaptive_pool)) = gepa.get("adaptive_pool") {
        check_unknown_fields(
            adaptive_pool,
            KNOWN_ADAPTIVE_POOL_FIELDS,
            "prompt_learning.gepa.adaptive_pool",
            result,
        );
    }
    if let Some(Value::Object(adaptive_batch)) = gepa.get("adaptive_batch") {
        check_unknown_fields(
            adaptive_batch,
            KNOWN_ADAPTIVE_BATCH_FIELDS,
            "prompt_learning.gepa.adaptive_batch",
            result,
        );
    }
    if let Some(Value::Object(proxy_models)) = gepa.get("proxy_models") {
        check_unknown_fields(
            proxy_models,
            KNOWN_PROXY_MODELS_FIELDS,
            "prompt_learning.gepa.proxy_models",
            result,
        );
    }
    if let Some(Value::Object(verifier)) = gepa.get("verifier") {
        check_unknown_fields(
            verifier,
            KNOWN_VERIFIER_FIELDS,
            "prompt_learning.gepa.verifier",
            result,
        );
    }

    if gepa.is_empty() {
        result.add_warning(format!(
            "{}No [prompt_learning.gepa] section found for GEPA algorithm",
            path_prefix
        ));
    }
}

fn validate_mipro_config(
    mipro: &Map<String, Value>,
    result: &mut PromptLearningValidationResult,
    path_prefix: &str,
) {
    check_unknown_fields(mipro, KNOWN_MIPRO_FIELDS, "prompt_learning.mipro", result);

    if let Some(Value::Object(verifier)) = mipro.get("verifier") {
        check_unknown_fields(
            verifier,
            KNOWN_VERIFIER_FIELDS,
            "prompt_learning.mipro.verifier",
            result,
        );
    }
    if let Some(Value::Object(adaptive_pool)) = mipro.get("adaptive_pool") {
        check_unknown_fields(
            adaptive_pool,
            KNOWN_ADAPTIVE_POOL_FIELDS,
            "prompt_learning.mipro.adaptive_pool",
            result,
        );
    }
    if let Some(Value::Object(proxy_models)) = mipro.get("proxy_models") {
        check_unknown_fields(
            proxy_models,
            KNOWN_PROXY_MODELS_FIELDS,
            "prompt_learning.mipro.proxy_models",
            result,
        );
    }

    if mipro.is_empty() {
        result.add_warning(format!(
            "{}No [prompt_learning.mipro] section found for MIPRO algorithm",
            path_prefix
        ));
    }
}

pub fn validate_prompt_learning_config(
    config: &Value,
    config_path: Option<&str>,
) -> PromptLearningValidationResult {
    let mut result = PromptLearningValidationResult::new();
    let path_prefix = config_path
        .map(|p| format!("({}) ", p))
        .unwrap_or_else(String::new);

    let config_map = match config.as_object() {
        Some(map) => map,
        None => {
            result.add_error(format!("{}Config must be an object", path_prefix));
            return result;
        }
    };

    for key in config_map.keys() {
        if !contains_known(KNOWN_TOP_LEVEL_SECTIONS, key) {
            result.add_warning(format!(
                "{}Unknown top-level section '[{}]'. Known sections: {}",
                path_prefix,
                key,
                KNOWN_TOP_LEVEL_SECTIONS.join(", ")
            ));
        }
    }

    if config_map.contains_key("display") {
        result.add_warning(format!(
            "{}The [display] section is deprecated and ignored by the backend. Remove it to clean up your config.",
            path_prefix
        ));
    }

    let pl_value = config_map.get("prompt_learning");
    let pl_map = match pl_value.and_then(|v| v.as_object()) {
        Some(map) => map,
        None => {
            result.add_error(format!(
                "{}Missing required [prompt_learning] section",
                path_prefix
            ));
            return result;
        }
    };

    check_unknown_fields(
        pl_map,
        KNOWN_PROMPT_LEARNING_FIELDS,
        "prompt_learning",
        &mut result,
    );
    check_deprecated_fields(pl_map, "prompt_learning", &mut result);

    let algorithm = pl_map.get("algorithm").and_then(|v| v.as_str());
    if algorithm.is_none() {
        result.add_error(format!(
            "{}Missing required 'algorithm' field in [prompt_learning]",
            path_prefix
        ));
    } else if !matches!(algorithm, Some("gepa") | Some("mipro")) {
        if let Some(value) = algorithm {
            result.add_error(format!(
                "{}Invalid algorithm '{}'. Must be 'gepa' or 'mipro'",
                path_prefix, value
            ));
        }
    }

    if pl_map
        .get("task_app_url")
        .and_then(|v| v.as_str())
        .is_none()
    {
        result.add_error(format!(
            "{}Missing required 'task_app_url' in [prompt_learning]",
            path_prefix
        ));
    }

    if let Some(Value::Object(policy)) = pl_map.get("policy") {
        check_unknown_fields(
            policy,
            KNOWN_POLICY_FIELDS,
            "prompt_learning.policy",
            &mut result,
        );
    }

    if let Some(Value::Object(termination)) = pl_map.get("termination_config") {
        check_unknown_fields(
            termination,
            KNOWN_TERMINATION_CONFIG_FIELDS,
            "prompt_learning.termination_config",
            &mut result,
        );
        result.add_info(
            "termination_config is supported and will create backend TerminationManager conditions"
                .to_string(),
        );
    }

    if let Some(Value::Object(verifier)) = pl_map.get("verifier") {
        check_unknown_fields(
            verifier,
            KNOWN_VERIFIER_FIELDS,
            "prompt_learning.verifier",
            &mut result,
        );
    }

    if let Some(Value::Object(proxy_models)) = pl_map.get("proxy_models") {
        check_unknown_fields(
            proxy_models,
            KNOWN_PROXY_MODELS_FIELDS,
            "prompt_learning.proxy_models",
            &mut result,
        );
    }

    match algorithm {
        Some("gepa") => {
            if let Some(Value::Object(gepa)) = pl_map.get("gepa") {
                validate_gepa_config(gepa, &mut result, &path_prefix);
            } else {
                result.add_warning(format!(
                    "{}No [prompt_learning.gepa] section found for GEPA algorithm",
                    path_prefix
                ));
            }
        }
        Some("mipro") => {
            if let Some(Value::Object(mipro)) = pl_map.get("mipro") {
                validate_mipro_config(mipro, &mut result, &path_prefix);
            } else {
                result.add_warning(format!(
                    "{}No [prompt_learning.mipro] section found for MIPRO algorithm",
                    path_prefix
                ));
            }
        }
        _ => {}
    }

    result
}

// =============================================================================
// Strict prompt learning config validation (errors only)
// =============================================================================

#[derive(Debug, Clone)]
struct SupportedModels {
    openai: HashSet<String>,
    groq: HashSet<String>,
    google: HashSet<String>,
}

fn extract_model_list(value: Option<&Value>) -> Vec<String> {
    match value.and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_lowercase())
            .collect(),
        None => Vec::new(),
    }
}

fn load_supported_models() -> Option<SupportedModels> {
    let raw = include_str!("../../assets/supported_models.json");
    let value: Value = serde_json::from_str(raw).ok()?;
    let prompt_opt = value.get("prompt_optimization")?.as_object()?;

    let openai = extract_model_list(prompt_opt.get("openai").and_then(|v| v.get("models")));
    let openai_image =
        extract_model_list(prompt_opt.get("openai_image").and_then(|v| v.get("models")));
    let google = extract_model_list(prompt_opt.get("google").and_then(|v| v.get("models")));
    let google_image =
        extract_model_list(prompt_opt.get("google_image").and_then(|v| v.get("models")));
    let groq = extract_model_list(prompt_opt.get("groq").and_then(|v| v.get("models")));

    let mut openai_set = HashSet::new();
    for item in openai.into_iter().chain(openai_image.into_iter()) {
        openai_set.insert(item);
    }
    let mut google_set = HashSet::new();
    for item in google.into_iter().chain(google_image.into_iter()) {
        google_set.insert(item);
    }
    let groq_set: HashSet<String> = groq.into_iter().collect();

    Some(SupportedModels {
        openai: openai_set,
        groq: groq_set,
        google: google_set,
    })
}

static SUPPORTED_MODELS: Lazy<Option<SupportedModels>> = Lazy::new(load_supported_models);

fn is_supported_openai_model(model: &str) -> bool {
    if let Some(models) = SUPPORTED_MODELS.as_ref() {
        let key = model.to_lowercase();
        return models.openai.contains(&key);
    }
    true
}

fn is_supported_groq_model(model: &str) -> bool {
    if let Some(models) = SUPPORTED_MODELS.as_ref() {
        let key = model.to_lowercase();
        return models.groq.contains(&key);
    }
    true
}

fn is_supported_google_model(model: &str) -> bool {
    if let Some(models) = SUPPORTED_MODELS.as_ref() {
        let key = model.to_lowercase();
        return models.google.contains(&key);
    }
    true
}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "None",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn parse_int(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn parse_float(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn validate_model_for_provider(
    model: &str,
    provider: &str,
    field_name: &str,
    allow_nano: bool,
) -> Vec<String> {
    let mut errors = Vec::new();

    if model.trim().is_empty() {
        errors.push(format!("Missing or empty {}", field_name));
        return errors;
    }

    let provider_lower = provider.trim().to_lowercase();
    let model_lower = model.trim().to_lowercase();
    let model_without_prefix = if let Some((_, rest)) = model_lower.split_once('/') {
        rest
    } else {
        model_lower.as_str()
    };

    if model_without_prefix == "gpt-5-pro" {
        errors.push(format!(
            "Model '{}' is not supported for prompt learning (too expensive).\n  gpt-5-pro is excluded due to high cost ($15/$120 per 1M tokens).\n  Please use a supported model instead.",
            model
        ));
        return errors;
    }

    if !allow_nano && model_without_prefix.ends_with("-nano") {
        errors.push(format!(
            "Model '{}' is not supported for {}.\n  ❌ Nano models (e.g., gpt-4.1-nano, gpt-5-nano) are NOT allowed for proposal/mutation models.\n  \n  Why?\n  Proposal and mutation models need to be SMART and capable of generating high-quality,\n  creative prompt variations. Nano models are too small and lack the reasoning capability\n  needed for effective prompt optimization.\n  \n  ✅ Use a larger model instead:\n     - For OpenAI: gpt-4.1-mini, gpt-4o-mini, gpt-4o, or gpt-4.1\n     - For Groq: openai/gpt-oss-120b, llama-3.3-70b-versatile\n     - For Google: gemini-2.5-flash, gemini-2.5-pro\n  \n  Note: Nano models ARE allowed for policy models (task execution), but NOT for\n  proposal/mutation models (prompt generation).",
            model, field_name
        ));
        return errors;
    }

    match provider_lower.as_str() {
        "openai" => {
            if !is_supported_openai_model(model_without_prefix) {
                errors.push(format!(
                    "Unsupported OpenAI model: '{}'\n  Supported OpenAI models for prompt learning:\n    - gpt-4o\n    - gpt-4o-mini\n    - gpt-4.1, gpt-4.1-mini, gpt-4.1-nano\n    - gpt-5, gpt-5-mini, gpt-5-nano\n    - Image generation: gpt-image-1.5, gpt-image-1, gpt-image-1-mini, chatgpt-image-latest\n  Note: gpt-5-pro is excluded (too expensive)\n  Got: '{}'",
                    model, model
                ));
            }
        }
        "groq" => {
            if !is_supported_groq_model(&model_lower) {
                errors.push(format!(
                    "Unsupported Groq model: '{}'\n  Supported Groq models for prompt learning:\n    - gpt-oss-Xb (e.g., gpt-oss-20b, openai/gpt-oss-120b)\n    - llama-3.3-70b (and variants like llama-3.3-70b-versatile)\n    - llama-3.1-8b-instant\n    - qwen/qwen3-32b (and variants)\n  Got: '{}'",
                    model, model
                ));
            }
        }
        "google" => {
            if !is_supported_google_model(model_without_prefix) {
                errors.push(format!(
                    "Unsupported Google/Gemini model: '{}'\n  Supported Google models for prompt learning:\n    - gemini-2.5-pro, gemini-2.5-pro-gt200k\n    - gemini-2.5-flash\n    - gemini-2.5-flash-lite\n    - Image generation: gemini-2.5-flash-image, gemini-3-pro-image-preview\n  Got: '{}'",
                    model, model
                ));
            }
        }
        _ => {
            errors.push(format!(
                "Unsupported provider: '{}'\n  Supported providers for prompt learning: 'openai', 'groq', 'google'\n  Got: '{}'",
                provider, provider
            ));
        }
    }

    errors
}

fn validate_adaptive_pool_config(
    adaptive_pool_section: &Value,
    prefix: &str,
    errors: &mut Vec<String>,
) {
    let section = match adaptive_pool_section.as_object() {
        Some(map) => map,
        None => {
            errors.push(format!("❌ {} must be a table/dict when provided", prefix));
            return;
        }
    };

    if let Some(level) = section.get("level") {
        let level_str = value_to_string(level).unwrap_or_default().to_uppercase();
        let valid = ["NONE", "LOW", "MODERATE", "HIGH"];
        if !valid.contains(&level_str.as_str()) {
            errors.push(format!(
                "❌ {}.level must be one of {:?}, got '{}'",
                prefix, valid, level_str
            ));
        }
    }

    for (field, min_val) in [
        ("anchor_size", 0),
        ("pool_init_size", 0),
        ("pool_min_size", 0),
        ("warmup_iters", 0),
        ("anneal_stop_iter", 0),
        ("pool_update_period", 1),
        ("min_evals_per_example", 1),
        ("k_info_prompts", 0),
    ] {
        if let Some(val) = section.get(field) {
            match parse_int(val) {
                Some(ival) => {
                    if ival < min_val {
                        errors.push(format!(
                            "❌ {}.{} must be >= {}, got {}",
                            prefix, field, min_val, ival
                        ));
                    }
                }
                None => {
                    errors.push(format!(
                        "❌ {}.{} must be an integer, got {}",
                        prefix,
                        field,
                        value_type_name(val)
                    ));
                }
            }
        }
    }

    let pool_init = section.get("pool_init_size").and_then(parse_int);
    let pool_min = section.get("pool_min_size").and_then(parse_int);
    if let (Some(init), Some(min)) = (pool_init, pool_min) {
        if init < min {
            errors.push(format!(
                "❌ {}.pool_init_size ({}) must be >= pool_min_size ({})",
                prefix, init, min
            ));
        }
    }

    let anchor_size = section.get("anchor_size").and_then(parse_int);
    if let (Some(min), Some(anchor)) = (pool_min, anchor_size) {
        if min < anchor {
            errors.push(format!(
                "❌ {}.pool_min_size ({}) must be >= anchor_size ({})",
                prefix, min, anchor
            ));
        }
    }

    for (field, min_val, max_val) in [
        ("info_buffer_factor", 0.0, Some(1.0)),
        ("info_epsilon", 0.0, None),
    ] {
        if let Some(val) = section.get(field) {
            match parse_float(val) {
                Some(fval) => {
                    if fval < min_val {
                        errors.push(format!(
                            "❌ {}.{} must be >= {}, got {}",
                            prefix, field, min_val, fval
                        ));
                    }
                    if let Some(max) = max_val {
                        if fval > max {
                            errors.push(format!(
                                "❌ {}.{} must be <= {}, got {}",
                                prefix, field, max, fval
                            ));
                        }
                    }
                }
                None => {
                    errors.push(format!(
                        "❌ {}.{} must be numeric, got {}",
                        prefix,
                        field,
                        value_type_name(val)
                    ));
                }
            }
        }
    }

    if let Some(val) = section.get("anchor_selection_method") {
        let method = value_to_string(val).unwrap_or_default();
        if !["random", "clustering"].contains(&method.as_str()) {
            errors.push(format!(
                "❌ {}.anchor_selection_method must be 'random' or 'clustering', got '{}'",
                prefix, method
            ));
        }
    }

    if let Some(val) = section.get("exploration_strategy") {
        let method = value_to_string(val).unwrap_or_default();
        if !["random", "diversity"].contains(&method.as_str()) {
            errors.push(format!(
                "❌ {}.exploration_strategy must be 'random' or 'diversity', got '{}'",
                prefix, method
            ));
        }
    }

    if let Some(val) = section.get("heatup_trigger") {
        let trigger = value_to_string(val).unwrap_or_default();
        if !["after_min_size", "immediate", "every_N_trials_after_min"].contains(&trigger.as_str())
        {
            errors.push(format!(
                "❌ {}.heatup_trigger must be 'after_min_size', 'immediate', or 'every_N_trials_after_min', got '{}'",
                prefix, trigger
            ));
        }
    }

    if let Some(val) = section.get("heatup_schedule") {
        let schedule = value_to_string(val).unwrap_or_default();
        if !["repeat", "once"].contains(&schedule.as_str()) {
            errors.push(format!(
                "❌ {}.heatup_schedule must be 'repeat' or 'once', got '{}'",
                prefix, schedule
            ));
        }
    }

    if let Some(val) = section.get("heatup_size") {
        match parse_int(val) {
            Some(ival) => {
                if ival <= 0 {
                    errors.push(format!(
                        "❌ {}.heatup_size must be > 0, got {}",
                        prefix, ival
                    ));
                }
            }
            None => {
                errors.push(format!(
                    "❌ {}.heatup_size must be an integer, got {}",
                    prefix,
                    value_type_name(val)
                ));
            }
        }
    }

    if let Some(val) = section.get("heatup_cooldown_trials") {
        match parse_int(val) {
            Some(ival) => {
                if ival < 0 {
                    errors.push(format!(
                        "❌ {}.heatup_cooldown_trials must be >= 0, got {}",
                        prefix, ival
                    ));
                }
            }
            None => {
                errors.push(format!(
                    "❌ {}.heatup_cooldown_trials must be an integer, got {}",
                    prefix,
                    value_type_name(val)
                ));
            }
        }
    }

    if let Some(val) = section.get("heatup_reserve_pool") {
        match val.as_array() {
            Some(list) => {
                if list.iter().any(|item| parse_int(item).is_none()) {
                    errors.push(format!(
                        "❌ {}.heatup_reserve_pool must contain only integers",
                        prefix
                    ));
                }
            }
            None => {
                errors.push(format!(
                    "❌ {}.heatup_reserve_pool must be a list, got {}",
                    prefix,
                    value_type_name(val)
                ));
            }
        }
    }
}

fn extract_pipeline_modules(initial_prompt: Option<&Value>) -> Vec<String> {
    let mut out = Vec::new();
    let initial_prompt = match initial_prompt.and_then(|v| v.as_object()) {
        Some(map) => map,
        None => return out,
    };
    let metadata = match initial_prompt.get("metadata").and_then(|v| v.as_object()) {
        Some(map) => map,
        None => return out,
    };
    let pipeline_modules = match metadata.get("pipeline_modules").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return out,
    };

    for entry in pipeline_modules {
        if let Some(name) = entry.as_str() {
            let trimmed = name.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            continue;
        }
        if let Some(map) = entry.as_object() {
            let name = map
                .get("name")
                .or_else(|| map.get("module_id"))
                .or_else(|| map.get("stage_id"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if !name.is_empty() {
                out.push(name);
            }
        }
    }

    out
}

pub fn validate_prompt_learning_config_strict(config: &Value) -> Vec<String> {
    let mut errors: Vec<String> = Vec::new();

    let config_map = match config.as_object() {
        Some(map) => map,
        None => {
            errors.push("Missing [prompt_learning] section in config. Expected: [prompt_learning] with algorithm, task_app_url, etc.".to_string());
            return errors;
        }
    };

    let pl_section = match config_map.get("prompt_learning") {
        Some(Value::Object(map)) => map,
        Some(other) => {
            errors.push(format!(
                "[prompt_learning] must be a table/dict, got {}",
                value_type_name(other)
            ));
            return errors;
        }
        None => {
            errors.push(
                "Missing [prompt_learning] section in config. Expected: [prompt_learning] with algorithm, task_app_url, etc."
                    .to_string(),
            );
            return errors;
        }
    };

    let algorithm = pl_section.get("algorithm").and_then(|v| v.as_str());
    if algorithm.is_none() {
        errors.push(
            "Missing required field: prompt_learning.algorithm\n  Must be one of: 'gepa', 'mipro'\n  Example:\n    [prompt_learning]\n    algorithm = \"gepa\""
                .to_string(),
        );
    } else if !matches!(algorithm, Some("gepa") | Some("mipro")) {
        let algo = algorithm.unwrap_or_default();
        errors.push(format!(
            "Invalid algorithm: '{}'\n  Must be one of: 'gepa', 'mipro'\n  Got: '{}'",
            algo, algo
        ));
    }

    let task_app_url = pl_section.get("task_app_url");
    if task_app_url.is_none() {
        errors.push(
            "Missing required field: prompt_learning.task_app_url\n  Example:\n    task_app_url = \"http://127.0.0.1:8102\""
                .to_string(),
        );
    } else if let Some(val) = task_app_url {
        if let Some(url) = val.as_str() {
            if !url.starts_with("http://") && !url.starts_with("https://") {
                errors.push(format!(
                    "task_app_url must start with http:// or https://, got: '{}'",
                    url
                ));
            }
        } else {
            errors.push(format!(
                "task_app_url must be a string, got {}",
                value_type_name(val)
            ));
        }
    }

    if let Some(initial_prompt) = pl_section.get("initial_prompt") {
        if let Some(map) = initial_prompt.as_object() {
            if let Some(messages) = map.get("messages") {
                match messages.as_array() {
                    Some(arr) => {
                        if arr.is_empty() {
                            errors.push("prompt_learning.initial_prompt.messages is empty (must have at least one message)".to_string());
                        }
                    }
                    None => {
                        errors.push(format!(
                            "prompt_learning.initial_prompt.messages must be an array, got {}",
                            value_type_name(messages)
                        ));
                    }
                }
            }
        } else {
            errors.push(format!(
                "prompt_learning.initial_prompt must be a table/dict, got {}",
                value_type_name(initial_prompt)
            ));
        }
    }

    let policy = pl_section.get("policy");
    if let Some(Value::Object(policy_map)) = policy {
        let mode = policy_map
            .get("inference_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if mode.is_empty() {
            errors.push(
                "Missing required field: prompt_learning.policy.inference_mode (must be 'synth_hosted')"
                    .to_string(),
            );
        } else if mode != "synth_hosted" {
            errors.push(
                "prompt_learning.policy.inference_mode must be 'synth_hosted' (bring_your_own unsupported)"
                    .to_string(),
            );
        }

        let provider = policy_map
            .get("provider")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let model = policy_map
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if provider.is_empty() {
            errors.push("Missing required field: prompt_learning.policy.provider".to_string());
        }
        if model.is_empty() {
            errors.push("Missing required field: prompt_learning.policy.model".to_string());
        } else if !provider.is_empty() {
            errors.extend(validate_model_for_provider(
                &model,
                &provider,
                "prompt_learning.policy.model",
                true,
            ));
        }

        for forbidden in ["inference_url", "api_base", "base_url"] {
            if policy_map.contains_key(forbidden) {
                errors.push(format!(
                    "{} must not be specified in [prompt_learning.policy]. The trainer provides the inference URL in rollout requests. Remove {} from your config file.",
                    forbidden, forbidden
                ));
            }
        }
    } else {
        errors.push("Missing [prompt_learning.policy] section or not a table".to_string());
    }

    if let Some(proxy_models) = pl_section.get("proxy_models") {
        match proxy_models.as_object() {
            Some(map) => {
                for field in ["hi_provider", "hi_model", "lo_provider", "lo_model"] {
                    if map
                        .get(field)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .is_empty()
                    {
                        errors.push(format!(
                            "prompt_learning.proxy_models.{} is required",
                            field
                        ));
                    }
                }
                for (field, min_val) in [
                    ("n_min_hi", 0.0),
                    ("r2_thresh", 0.0),
                    ("r2_stop", 0.0),
                    ("sigma_max", 0.0),
                    ("sigma_stop", 0.0),
                    ("verify_every", 0.0),
                ] {
                    if let Some(val) = map.get(field) {
                        match parse_float(val) {
                            Some(fval) => {
                                if (field == "r2_thresh" || field == "r2_stop")
                                    && !(0.0..=1.0).contains(&fval)
                                {
                                    errors.push(format!(
                                        "prompt_learning.proxy_models.{} must be between 0.0 and 1.0, got {}",
                                        field, fval
                                    ));
                                } else if fval < min_val {
                                    errors.push(format!(
                                        "prompt_learning.proxy_models.{} must be >= {}, got {}",
                                        field, min_val, fval
                                    ));
                                }
                            }
                            None => errors.push(format!(
                                "prompt_learning.proxy_models.{} must be numeric, got {}",
                                field,
                                value_type_name(val)
                            )),
                        }
                    }
                }

                let hi_provider = map
                    .get("hi_provider")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let hi_model = map.get("hi_model").and_then(|v| v.as_str()).unwrap_or("");
                if !hi_provider.is_empty() && !hi_model.is_empty() {
                    errors.extend(validate_model_for_provider(
                        hi_model,
                        hi_provider,
                        "prompt_learning.proxy_models.hi_model",
                        true,
                    ));
                }

                let lo_provider = map
                    .get("lo_provider")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let lo_model = map.get("lo_model").and_then(|v| v.as_str()).unwrap_or("");
                if !lo_provider.is_empty() && !lo_model.is_empty() {
                    errors.extend(validate_model_for_provider(
                        lo_model,
                        lo_provider,
                        "prompt_learning.proxy_models.lo_model",
                        true,
                    ));
                }
            }
            None => errors.push(format!(
                "prompt_learning.proxy_models must be a table/dict, got {}",
                value_type_name(proxy_models)
            )),
        }
    }

    if let Some(verifier) = pl_section.get("verifier") {
        match verifier.as_object() {
            Some(map) => {
                let reward_source = map
                    .get("reward_source")
                    .and_then(|v| v.as_str())
                    .unwrap_or("task_app")
                    .trim()
                    .to_lowercase();
                if !reward_source.is_empty()
                    && !matches!(reward_source.as_str(), "task_app" | "verifier" | "fused")
                {
                    errors.push(
                        "prompt_learning.verifier.reward_source must be 'task_app', 'verifier', or 'fused'"
                            .to_string(),
                    );
                }
                if reward_source == "fused" {
                    let weight_event = map.get("weight_event");
                    let weight_outcome = map.get("weight_outcome");
                    let weight_event_f = weight_event.and_then(parse_float);
                    let weight_outcome_f = weight_outcome.and_then(parse_float);
                    if weight_event.is_some() && weight_event_f.is_none() {
                        errors.push(
                            "prompt_learning.verifier.weight_event must be numeric".to_string(),
                        );
                    }
                    if weight_outcome.is_some() && weight_outcome_f.is_none() {
                        errors.push(
                            "prompt_learning.verifier.weight_outcome must be numeric".to_string(),
                        );
                    }
                    if weight_event_f.unwrap_or(0.0) <= 0.0
                        && weight_outcome_f.unwrap_or(0.0) <= 0.0
                    {
                        errors.push(
                            "prompt_learning.verifier.reward_source='fused' requires weight_event > 0 or weight_outcome > 0"
                                .to_string(),
                        );
                    }
                }
            }
            None => errors.push(format!(
                "prompt_learning.verifier must be a table/dict, got {}",
                value_type_name(verifier)
            )),
        }
    }

    let pipeline_modules = extract_pipeline_modules(pl_section.get("initial_prompt"));
    let has_multi_stage = !pipeline_modules.is_empty();

    match algorithm {
        Some("gepa") => {
            let gepa_config = pl_section.get("gepa");
            let gepa_map = match gepa_config.and_then(|v| v.as_object()) {
                Some(map) => map,
                None => {
                    errors.push(
                        "Missing [prompt_learning.gepa] section for GEPA algorithm".to_string(),
                    );
                    return errors;
                }
            };

            if has_multi_stage {
                let modules_config = gepa_map.get("modules");
                match modules_config.and_then(|v| v.as_array()) {
                    Some(arr) if !arr.is_empty() => {
                        let mut module_ids = HashSet::new();
                        for module in arr {
                            if let Some(map) = module.as_object() {
                                if let Some(id) = map
                                    .get("module_id")
                                    .or_else(|| map.get("stage_id"))
                                    .and_then(|v| v.as_str())
                                {
                                    module_ids.insert(id.trim().to_string());
                                }
                            }
                        }
                        let pipeline_set: HashSet<String> =
                            pipeline_modules.iter().cloned().collect();
                        let missing: Vec<String> =
                            pipeline_set.difference(&module_ids).cloned().collect();
                        if !missing.is_empty() {
                            errors.push(format!(
                                "Pipeline modules {:?} are missing from [prompt_learning.gepa.modules]. Each pipeline module must have a corresponding module config with matching module_id.",
                                missing
                            ));
                        }
                    }
                    _ => {
                        errors.push(format!(
                            "GEPA multi-stage pipeline detected (found {} modules in prompt_learning.initial_prompt.metadata.pipeline_modules), but [prompt_learning.gepa.modules] is missing or empty. Define module configs for each pipeline stage.",
                            pipeline_modules.len()
                        ));
                    }
                }
            }

            let pos_int = |name: &str, errors: &mut Vec<String>| {
                if let Some(val) = gepa_map.get(name) {
                    match parse_int(val) {
                        Some(ival) => {
                            if ival <= 0 {
                                errors.push(format!("prompt_learning.gepa.{} must be > 0", name));
                            }
                        }
                        None => {
                            errors.push(format!("prompt_learning.gepa.{} must be an integer", name))
                        }
                    }
                }
            };
            let non_neg_int = |name: &str, errors: &mut Vec<String>| {
                if let Some(val) = gepa_map.get(name) {
                    match parse_int(val) {
                        Some(ival) => {
                            if ival < 0 {
                                errors.push(format!("prompt_learning.gepa.{} must be >= 0", name));
                            }
                        }
                        None => {
                            errors.push(format!("prompt_learning.gepa.{} must be an integer", name))
                        }
                    }
                }
            };
            let rate_float = |name: &str, errors: &mut Vec<String>| {
                if let Some(val) = gepa_map.get(name) {
                    match parse_float(val) {
                        Some(fval) => {
                            if !(0.0..=1.0).contains(&fval) {
                                errors.push(format!(
                                    "prompt_learning.gepa.{} must be between 0.0 and 1.0",
                                    name
                                ));
                            }
                        }
                        None => {
                            errors.push(format!("prompt_learning.gepa.{} must be numeric", name))
                        }
                    }
                }
            };
            let pos_float = |name: &str, errors: &mut Vec<String>| {
                if let Some(val) = gepa_map.get(name) {
                    match parse_float(val) {
                        Some(fval) => {
                            if fval <= 0.0 {
                                errors.push(format!("prompt_learning.gepa.{} must be > 0", name));
                            }
                        }
                        None => {
                            errors.push(format!("prompt_learning.gepa.{} must be numeric", name))
                        }
                    }
                }
            };
            let pos_int_nested = |section: &str, name: &str, errors: &mut Vec<String>| {
                if let Some(Value::Object(section_map)) = gepa_map.get(section) {
                    if let Some(val) = section_map.get(name) {
                        match parse_int(val) {
                            Some(ival) => {
                                if ival <= 0 {
                                    errors.push(format!(
                                        "prompt_learning.gepa.{}.{} must be > 0",
                                        section, name
                                    ));
                                }
                            }
                            None => errors.push(format!(
                                "prompt_learning.gepa.{}.{} must be an integer",
                                section, name
                            )),
                        }
                    }
                }
            };

            for fld in [
                "initial_population_size",
                "num_generations",
                "children_per_generation",
                "max_concurrent_rollouts",
            ] {
                pos_int(fld, &mut errors);
            }
            pos_int_nested("rollout", "budget", &mut errors);
            pos_int_nested("rollout", "max_concurrent", &mut errors);
            pos_int_nested("rollout", "minibatch_size", &mut errors);
            pos_int_nested("population", "initial_size", &mut errors);
            pos_int_nested("population", "num_generations", &mut errors);
            pos_int_nested("population", "children_per_generation", &mut errors);
            rate_float("mutation_rate", &mut errors);
            rate_float("crossover_rate", &mut errors);
            pos_float("selection_pressure", &mut errors);
            if let Some(val) = gepa_map.get("selection_pressure") {
                if let Some(sp) = parse_float(val) {
                    if sp < 1.0 {
                        errors.push(
                            "prompt_learning.gepa.selection_pressure must be >= 1.0".to_string(),
                        );
                    }
                }
            }
            non_neg_int("patience_generations", &mut errors);
            pos_int_nested("archive", "size", &mut errors);
            pos_int_nested("archive", "pareto_set_size", &mut errors);
            pos_float("pareto_eps", &mut errors);
            rate_float("feedback_fraction", &mut errors);

            if let Some(Value::Object(mutation)) = gepa_map.get("mutation") {
                let mutation_model = mutation.get("llm_model").and_then(|v| v.as_str());
                let mutation_provider = mutation
                    .get("llm_provider")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if let Some(model) = mutation_model {
                    if mutation_provider.is_empty() {
                        errors.push(
                            "Missing required field: prompt_learning.gepa.mutation.llm_provider\n  Required when prompt_learning.gepa.mutation.llm_model is set"
                                .to_string(),
                        );
                    } else {
                        errors.extend(validate_model_for_provider(
                            model,
                            &mutation_provider,
                            "prompt_learning.gepa.mutation.llm_model",
                            false,
                        ));
                    }
                }
            }

            if let Some(val) = gepa_map.get("max_spend_usd") {
                match parse_float(val) {
                    Some(fval) => {
                        if fval <= 0.0 {
                            errors.push(
                                "prompt_learning.gepa.max_spend_usd must be > 0 when provided"
                                    .to_string(),
                            );
                        }
                    }
                    None => errors
                        .push("prompt_learning.gepa.max_spend_usd must be numeric".to_string()),
                }
            }

            let rollout_budget = gepa_map
                .get("rollout")
                .and_then(|v| v.get("budget"))
                .or_else(|| gepa_map.get("rollout_budget"));
            if let Some(val) = rollout_budget {
                match parse_int(val) {
                    Some(ival) => {
                        if ival <= 0 {
                            errors.push("prompt_learning.gepa.rollout.budget (or rollout_budget) must be > 0 when provided".to_string());
                        }
                    }
                    None => errors.push("prompt_learning.gepa.rollout.budget (or rollout_budget) must be an integer".to_string()),
                }
            }

            let minibatch_size = gepa_map
                .get("rollout")
                .and_then(|v| v.get("minibatch_size"))
                .or_else(|| gepa_map.get("minibatch_size"));
            if let Some(val) = minibatch_size {
                match parse_int(val) {
                    Some(ival) => {
                        if ival <= 0 {
                            errors.push("prompt_learning.gepa.rollout.minibatch_size (or minibatch_size) must be > 0".to_string());
                        }
                    }
                    None => errors.push("prompt_learning.gepa.rollout.minibatch_size (or minibatch_size) must be an integer".to_string()),
                }
            }

            let proposer_type = gepa_map
                .get("proposer_type")
                .and_then(|v| v.as_str())
                .unwrap_or("dspy");
            if !matches!(proposer_type, "dspy" | "spec" | "synth" | "gepa-ai") {
                errors.push(format!(
                    "Invalid proposer_type: '{}'\n  Must be one of: 'dspy', 'spec', 'synth', 'gepa-ai'\n  Got: '{}'",
                    proposer_type, proposer_type
                ));
            }

            let proposer_effort = gepa_map
                .get("proposer_effort")
                .and_then(|v| v.as_str())
                .unwrap_or("LOW")
                .to_uppercase();
            let valid_effort = ["LOW_CONTEXT", "LOW", "MEDIUM", "HIGH"];
            if !valid_effort.contains(&proposer_effort.as_str()) {
                errors.push(format!(
                    "Invalid proposer_effort: '{}'\n  Must be one of: {}\n  Got: '{}'",
                    proposer_effort,
                    valid_effort.join(", "),
                    proposer_effort
                ));
            }

            let proposer_output_tokens = gepa_map
                .get("proposer_output_tokens")
                .and_then(|v| v.as_str())
                .unwrap_or("FAST")
                .to_uppercase();
            let valid_output = ["RAPID", "FAST", "SLOW"];
            if !valid_output.contains(&proposer_output_tokens.as_str()) {
                errors.push(format!(
                    "Invalid proposer_output_tokens: '{}'\n  Must be one of: {}\n  Got: '{}'",
                    proposer_output_tokens,
                    valid_output.join(", "),
                    proposer_output_tokens
                ));
            }

            if proposer_type == "spec" {
                if gepa_map
                    .get("spec_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .is_empty()
                {
                    errors.push(
                        "Missing required field: prompt_learning.gepa.spec_path\n  Required when proposer_type='spec'\n  Example:\n    [prompt_learning.gepa]\n    proposer_type = \"spec\"\n    spec_path = \"examples/task_apps/banking77/banking77_spec.json\""
                            .to_string(),
                    );
                } else {
                    if let Some(val) = gepa_map.get("spec_max_tokens") {
                        match parse_int(val) {
                            Some(ival) => {
                                if ival <= 0 {
                                    errors.push(
                                        "prompt_learning.gepa.spec_max_tokens must be > 0"
                                            .to_string(),
                                    );
                                }
                            }
                            None => errors.push(
                                "prompt_learning.gepa.spec_max_tokens must be an integer"
                                    .to_string(),
                            ),
                        }
                    }
                    if let Some(val) = gepa_map.get("spec_priority_threshold") {
                        match parse_int(val) {
                            Some(ival) => {
                                if ival < 0 {
                                    errors.push(
                                        "prompt_learning.gepa.spec_priority_threshold must be >= 0"
                                            .to_string(),
                                    );
                                }
                            }
                            None => errors.push(
                                "prompt_learning.gepa.spec_priority_threshold must be an integer"
                                    .to_string(),
                            ),
                        }
                    }
                }
            }

            let archive_size = gepa_map
                .get("archive")
                .and_then(|v| v.get("size"))
                .or_else(|| gepa_map.get("archive_size"));
            if let Some(val) = archive_size {
                match parse_int(val) {
                    Some(ival) => {
                        if ival <= 0 {
                            errors.push(
                                "prompt_learning.gepa.archive.size (or archive_size) must be > 0"
                                    .to_string(),
                            );
                        }
                    }
                    None => errors.push(
                        "prompt_learning.gepa.archive.size (or archive_size) must be an integer"
                            .to_string(),
                    ),
                }
            }

            let eval_config = gepa_map.get("evaluation").and_then(|v| v.as_object());
            if let Some(eval_map) = eval_config {
                let train_seeds = eval_map
                    .get("seeds")
                    .or_else(|| eval_map.get("train_seeds"))
                    .and_then(|v| v.as_array());
                if let Some(seeds_list) = train_seeds {
                    if !seeds_list.is_empty() {
                        let total_seeds = seeds_list.len();
                        let pareto_set_size = gepa_map
                            .get("archive")
                            .and_then(|v| v.get("pareto_set_size"))
                            .or_else(|| gepa_map.get("pareto_set_size"))
                            .and_then(parse_int)
                            .unwrap_or(64);
                        let feedback_fraction = gepa_map
                            .get("archive")
                            .and_then(|v| v.get("feedback_fraction"))
                            .or_else(|| gepa_map.get("feedback_fraction"))
                            .and_then(parse_float)
                            .unwrap_or(0.5);
                        let _ = feedback_fraction;

                        let feedback_count = total_seeds as i64 - pareto_set_size;
                        let min_pareto_set_size = 10;
                        let min_feedback_seeds = 3;

                        if pareto_set_size > total_seeds as i64 {
                            errors.push(format!(
                                "CONFIG ERROR: pareto_set_size={} > total_seeds={}. Increase [prompt_learning.gepa.evaluation].seeds or decrease [prompt_learning.gepa.archive].pareto_set_size. Seeds: {:?}{}",
                                pareto_set_size,
                                total_seeds,
                                seeds_list.iter().take(10).filter_map(value_to_string).collect::<Vec<_>>(),
                                if seeds_list.len() > 10 { "..." } else { "" }
                            ));
                        }
                        if pareto_set_size < min_pareto_set_size {
                            errors.push(format!(
                                "CONFIG ERROR: pareto_set_size={} < MIN_PARETO_SET_SIZE={}. Increase [prompt_learning.gepa.archive].pareto_set_size to at least {}. Below this threshold, accuracy estimates are too noisy for reliable optimization.",
                                pareto_set_size, min_pareto_set_size, min_pareto_set_size
                            ));
                        }
                        if feedback_count < min_feedback_seeds {
                            errors.push(format!(
                                "CONFIG ERROR: feedback_count={} < MIN_FEEDBACK_SEEDS={}. Increase total seeds or decrease pareto_set_size to ensure at least {} feedback seeds. Below this threshold, reflection prompts lack sufficient diversity.",
                                feedback_count, min_feedback_seeds, min_feedback_seeds
                            ));
                        }
                    }
                }
            }

            let pareto_eps = gepa_map
                .get("archive")
                .and_then(|v| v.get("pareto_eps"))
                .or_else(|| gepa_map.get("pareto_eps"));
            if let Some(val) = pareto_eps {
                match parse_float(val) {
                    Some(fval) => {
                        if fval <= 0.0 {
                            errors.push("prompt_learning.gepa.archive.pareto_eps (or pareto_eps) must be > 0".to_string());
                        } else if fval >= 1.0 {
                            errors.push("prompt_learning.gepa.archive.pareto_eps (or pareto_eps) should be < 1.0 (typically 1e-6)".to_string());
                        }
                    }
                    None => errors.push(
                        "prompt_learning.gepa.archive.pareto_eps (or pareto_eps) must be numeric"
                            .to_string(),
                    ),
                }
            }

            let feedback_fraction = gepa_map
                .get("archive")
                .and_then(|v| v.get("feedback_fraction"))
                .or_else(|| gepa_map.get("feedback_fraction"));
            if let Some(val) = feedback_fraction {
                match parse_float(val) {
                    Some(fval) => {
                        if !(0.0..=1.0).contains(&fval) {
                            errors.push("prompt_learning.gepa.archive.feedback_fraction (or feedback_fraction) must be between 0.0 and 1.0".to_string());
                        }
                    }
                    None => errors.push("prompt_learning.gepa.archive.feedback_fraction (or feedback_fraction) must be numeric".to_string()),
                }
            }

            let token_config = gepa_map
                .get("token")
                .or_else(|| gepa_map.get("prompt_budget"));
            let token_counting_model = token_config
                .and_then(|v| v.get("counting_model"))
                .or_else(|| gepa_map.get("token_counting_model"));
            if let Some(val) = token_counting_model {
                let ok = val.as_str().map(|s| !s.trim().is_empty()).unwrap_or(false);
                if !ok {
                    errors.push("prompt_learning.gepa.token.counting_model (or prompt_budget.counting_model, token_counting_model) must be a non-empty string".to_string());
                }
            }

            if has_multi_stage {
                if let Some(Value::Array(modules)) = gepa_map.get("modules") {
                    for (idx, module_entry) in modules.iter().enumerate() {
                        if let Some(map) = module_entry.as_object() {
                            if let Some(val) = map.get("max_instruction_slots") {
                                match parse_int(val) {
                                    Some(ival) => {
                                        if ival < 1 {
                                            errors.push(format!(
                                                "prompt_learning.gepa.modules[{}].max_instruction_slots must be >= 1",
                                                idx
                                            ));
                                        }
                                    }
                                    None => errors.push(format!(
                                        "prompt_learning.gepa.modules[{}].max_instruction_slots must be an integer",
                                        idx
                                    )),
                                }
                            }
                            if let Some(val) = map.get("max_tokens") {
                                match parse_int(val) {
                                    Some(ival) => {
                                        if ival <= 0 {
                                            errors.push(format!(
                                                "prompt_learning.gepa.modules[{}].max_tokens must be > 0",
                                                idx
                                            ));
                                        }
                                    }
                                    None => errors.push(format!(
                                        "prompt_learning.gepa.modules[{}].max_tokens must be an integer",
                                        idx
                                    )),
                                }
                            }
                            if let Some(val) = map.get("allowed_tools") {
                                match val.as_array() {
                                    Some(tools) => {
                                        if tools.is_empty() {
                                            errors.push(format!(
                                                "prompt_learning.gepa.modules[{}].allowed_tools cannot be empty (use null/omit to allow all tools)",
                                                idx
                                            ));
                                        } else {
                                            let mut seen = HashSet::new();
                                            for (tool_idx, tool) in tools.iter().enumerate() {
                                                let name = tool.as_str().unwrap_or("").trim().to_string();
                                                if name.is_empty() {
                                                    errors.push(format!(
                                                        "prompt_learning.gepa.modules[{}].allowed_tools[{}] cannot be empty",
                                                        idx, tool_idx
                                                    ));
                                                } else if seen.contains(&name) {
                                                    errors.push(format!(
                                                        "prompt_learning.gepa.modules[{}].allowed_tools contains duplicate '{}'",
                                                        idx, name
                                                    ));
                                                } else {
                                                    seen.insert(name);
                                                }
                                            }
                                        }
                                    }
                                    None => errors.push(format!(
                                        "prompt_learning.gepa.modules[{}].allowed_tools must be a list",
                                        idx
                                    )),
                                }
                            }
                            let module_policy = map.get("policy");
                            match module_policy {
                                None => errors.push(format!(
                                    "❌ gepa.modules[{}]: [policy] table is REQUIRED. Each module must have its own policy configuration with 'model' and 'provider' fields.",
                                    idx
                                )),
                                Some(Value::Object(policy_map)) => {
                                    if policy_map
                                        .get("provider")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .trim()
                                        .is_empty()
                                    {
                                        errors.push(format!(
                                            "❌ gepa.modules[{}]: [policy].provider is required",
                                            idx
                                        ));
                                    }
                                    let module_model = policy_map.get("model").and_then(|v| v.as_str());
                                    let module_provider = policy_map.get("provider").and_then(|v| v.as_str());
                                    if let (Some(model), Some(provider)) = (module_model, module_provider)
                                    {
                                        errors.extend(validate_model_for_provider(
                                            model,
                                            provider,
                                            &format!(
                                                "prompt_learning.gepa.modules[{}].policy.model",
                                                idx
                                            ),
                                            true,
                                        ));
                                    }
                                    for forbidden in ["inference_url", "api_base", "base_url"] {
                                        if policy_map.contains_key(forbidden) {
                                            errors.push(format!(
                                                "❌ gepa.modules[{}]: [policy].{} must not be specified. The trainer provides the inference URL in rollout requests. Remove {} from module policy.",
                                                idx, forbidden, forbidden
                                            ));
                                        }
                                    }
                                }
                                Some(other) => errors.push(format!(
                                    "❌ gepa.modules[{}]: [policy] must be a table/dict, got {}",
                                    idx,
                                    value_type_name(other)
                                )),
                            }
                        }
                    }
                }
            }
        }
        Some("mipro") => {
            let mipro_config = pl_section.get("mipro");
            let mipro_map = match mipro_config.and_then(|v| v.as_object()) {
                Some(map) => map,
                None => {
                    errors.push(
                        "Missing [prompt_learning.mipro] section for MIPRO algorithm".to_string(),
                    );
                    return errors;
                }
            };

            let pos_int = |name: &str, errors: &mut Vec<String>| {
                if let Some(val) = mipro_map.get(name) {
                    match parse_int(val) {
                        Some(ival) => {
                            if ival <= 0 {
                                errors.push(format!("prompt_learning.mipro.{} must be > 0", name));
                            }
                        }
                        None => errors
                            .push(format!("prompt_learning.mipro.{} must be an integer", name)),
                    }
                }
            };
            for fld in [
                "num_iterations",
                "num_evaluations_per_iteration",
                "batch_size",
                "max_concurrent",
            ] {
                pos_int(fld, &mut errors);
            }
            for fld in [
                "max_demo_set_size",
                "max_demo_sets",
                "max_instruction_sets",
                "full_eval_every_k",
                "instructions_per_batch",
                "max_instructions",
                "duplicate_retry_limit",
            ] {
                pos_int(fld, &mut errors);
            }

            if let Some(meta_model) = mipro_map.get("meta_model").and_then(|v| v.as_str()) {
                let provider = mipro_map
                    .get("meta_model_provider")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if provider.is_empty() {
                    errors.push(
                        "Missing required field: prompt_learning.mipro.meta_model_provider\n  Required when prompt_learning.mipro.meta_model is set"
                            .to_string(),
                    );
                } else {
                    errors.extend(validate_model_for_provider(
                        meta_model,
                        &provider,
                        "prompt_learning.mipro.meta_model",
                        false,
                    ));
                }
            }

            if let Some(val) = mipro_map.get("meta_model_temperature") {
                match parse_float(val) {
                    Some(fval) => {
                        if fval < 0.0 {
                            errors.push(
                                "prompt_learning.mipro.meta_model_temperature must be >= 0.0"
                                    .to_string(),
                            );
                        }
                    }
                    None => errors.push(
                        "prompt_learning.mipro.meta_model_temperature must be numeric".to_string(),
                    ),
                }
            }
            if let Some(val) = mipro_map.get("meta_model_max_tokens") {
                match parse_int(val) {
                    Some(ival) => {
                        if ival <= 0 {
                            errors.push(
                                "prompt_learning.mipro.meta_model_max_tokens must be > 0"
                                    .to_string(),
                            );
                        }
                    }
                    None => errors.push(
                        "prompt_learning.mipro.meta_model_max_tokens must be an integer"
                            .to_string(),
                    ),
                }
            }

            if let Some(val) = mipro_map.get("generate_at_iterations") {
                match val.as_array() {
                    Some(arr) => {
                        for (idx, item) in arr.iter().enumerate() {
                            match parse_int(item) {
                                Some(ival) => {
                                    if ival < 0 {
                                        errors.push(format!(
                                            "prompt_learning.mipro.generate_at_iterations[{}] must be >= 0",
                                            idx
                                        ));
                                    }
                                }
                                None => errors.push(format!(
                                    "prompt_learning.mipro.generate_at_iterations[{}] must be an integer",
                                    idx
                                )),
                            }
                        }
                    }
                    None => errors.push(
                        "prompt_learning.mipro.generate_at_iterations must be a list".to_string(),
                    ),
                }
            }

            if mipro_map
                .get("spec_path")
                .and_then(|v| v.as_str())
                .is_some()
            {
                if let Some(val) = mipro_map.get("spec_max_tokens") {
                    match parse_int(val) {
                        Some(ival) => {
                            if ival <= 0 {
                                errors.push(
                                    "prompt_learning.mipro.spec_max_tokens must be > 0".to_string(),
                                );
                            }
                        }
                        None => errors.push(
                            "prompt_learning.mipro.spec_max_tokens must be an integer".to_string(),
                        ),
                    }
                }
                if let Some(val) = mipro_map.get("spec_priority_threshold") {
                    match parse_int(val) {
                        Some(ival) => {
                            if ival < 0 {
                                errors.push(
                                    "prompt_learning.mipro.spec_priority_threshold must be >= 0"
                                        .to_string(),
                                );
                            }
                        }
                        None => errors.push(
                            "prompt_learning.mipro.spec_priority_threshold must be an integer"
                                .to_string(),
                        ),
                    }
                }
            }

            if let Some(modules) = mipro_map.get("modules").and_then(|v| v.as_array()) {
                let max_instruction_sets = mipro_map
                    .get("max_instruction_sets")
                    .and_then(parse_int)
                    .unwrap_or(128);
                let max_demo_sets = mipro_map
                    .get("max_demo_sets")
                    .and_then(parse_int)
                    .unwrap_or(128);
                let mut seen_module_ids = HashSet::new();
                let mut seen_stage_ids = HashSet::new();

                for (module_idx, module_entry) in modules.iter().enumerate() {
                    let module_map = match module_entry.as_object() {
                        Some(map) => map,
                        None => {
                            errors.push(format!(
                                "prompt_learning.mipro.modules[{}] must be a table/dict",
                                module_idx
                            ));
                            continue;
                        }
                    };

                    let module_id = module_map
                        .get("module_id")
                        .or_else(|| module_map.get("id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or(&format!("module_{}", module_idx))
                        .to_string();
                    if !seen_module_ids.insert(module_id.clone()) {
                        errors.push(format!(
                            "Duplicate module_id '{}' in prompt_learning.mipro.modules",
                            module_id
                        ));
                    }

                    let stages = module_map.get("stages");
                    if let Some(stages_val) = stages {
                        match stages_val.as_array() {
                            Some(stage_list) => {
                                for (stage_idx, stage_entry) in stage_list.iter().enumerate() {
                                    if let Some(stage_map) = stage_entry.as_object() {
                                        let stage_id = stage_map
                                            .get("stage_id")
                                            .or_else(|| stage_map.get("module_stage_id"))
                                            .and_then(|v| v.as_str())
                                            .unwrap_or(&format!("stage_{}", stage_idx))
                                            .to_string();
                                        if !seen_stage_ids.insert(stage_id.clone()) {
                                            errors.push(format!(
                                                "Duplicate stage_id '{}' across modules",
                                                stage_id
                                            ));
                                        }
                                        if let Some(val) = stage_map.get("max_instruction_slots") {
                                            match parse_int(val) {
                                                Some(ival) => {
                                                    if ival < 1 {
                                                        errors.push(format!(
                                                            "prompt_learning.mipro.modules[{}].stages[{}].max_instruction_slots must be >= 1",
                                                            module_idx, stage_idx
                                                        ));
                                                    } else if ival > max_instruction_sets {
                                                        errors.push(format!(
                                                            "prompt_learning.mipro.modules[{}].stages[{}].max_instruction_slots ({}) exceeds max_instruction_sets ({})",
                                                            module_idx, stage_idx, ival, max_instruction_sets
                                                        ));
                                                    }
                                                }
                                                None => errors.push(format!(
                                                    "prompt_learning.mipro.modules[{}].stages[{}].max_instruction_slots must be an integer",
                                                    module_idx, stage_idx
                                                )),
                                            }
                                        }
                                        if let Some(val) = stage_map.get("max_demo_slots") {
                                            match parse_int(val) {
                                                Some(ival) => {
                                                    if ival < 0 {
                                                        errors.push(format!(
                                                            "prompt_learning.mipro.modules[{}].stages[{}].max_demo_slots must be >= 0",
                                                            module_idx, stage_idx
                                                        ));
                                                    } else if ival > max_demo_sets {
                                                        errors.push(format!(
                                                            "prompt_learning.mipro.modules[{}].stages[{}].max_demo_slots ({}) exceeds max_demo_sets ({})",
                                                            module_idx, stage_idx, ival, max_demo_sets
                                                        ));
                                                    }
                                                }
                                                None => errors.push(format!(
                                                    "prompt_learning.mipro.modules[{}].stages[{}].max_demo_slots must be an integer",
                                                    module_idx, stage_idx
                                                )),
                                            }
                                        }
                                    }
                                }
                            }
                            None => errors.push(format!(
                                "prompt_learning.mipro.modules[{}].stages must be a list",
                                module_idx
                            )),
                        }
                    }

                    if let Some(edges_val) = module_map.get("edges") {
                        match edges_val.as_array() {
                            Some(edges) => {
                                let mut stage_ids_in_module = HashSet::new();
                                if let Some(Value::Array(stage_list)) = stages {
                                    for stage_entry in stage_list {
                                        if let Some(stage_map) = stage_entry.as_object() {
                                            if let Some(id) = stage_map
                                                .get("stage_id")
                                                .or_else(|| stage_map.get("module_stage_id"))
                                                .and_then(|v| v.as_str())
                                            {
                                                stage_ids_in_module.insert(id.to_string());
                                            }
                                        }
                                    }
                                }
                                for (edge_idx, edge) in edges.iter().enumerate() {
                                    let (source, target) = if let Some(arr) = edge.as_array() {
                                        if arr.len() == 2 {
                                            (arr[0].clone(), arr[1].clone())
                                        } else {
                                            errors.push(format!(
                                                "prompt_learning.mipro.modules[{}].edges[{}] must be a pair or mapping",
                                                module_idx, edge_idx
                                            ));
                                            continue;
                                        }
                                    } else if let Some(map) = edge.as_object() {
                                        let source = map
                                            .get("from")
                                            .or_else(|| map.get("source"))
                                            .cloned()
                                            .unwrap_or(Value::Null);
                                        let target = map
                                            .get("to")
                                            .or_else(|| map.get("target"))
                                            .cloned()
                                            .unwrap_or(Value::Null);
                                        (source, target)
                                    } else {
                                        errors.push(format!(
                                            "prompt_learning.mipro.modules[{}].edges[{}] must be a pair or mapping",
                                            module_idx, edge_idx
                                        ));
                                        continue;
                                    };

                                    let source_str = value_to_string(&source)
                                        .unwrap_or_default()
                                        .trim()
                                        .to_string();
                                    let target_str = value_to_string(&target)
                                        .unwrap_or_default()
                                        .trim()
                                        .to_string();
                                    if !source_str.is_empty()
                                        && !stage_ids_in_module.contains(&source_str)
                                    {
                                        errors.push(format!(
                                            "prompt_learning.mipro.modules[{}].edges[{}] references unknown source stage '{}'",
                                            module_idx, edge_idx, source_str
                                        ));
                                    }
                                    if !target_str.is_empty()
                                        && !stage_ids_in_module.contains(&target_str)
                                    {
                                        errors.push(format!(
                                            "prompt_learning.mipro.modules[{}].edges[{}] references unknown target stage '{}'",
                                            module_idx, edge_idx, target_str
                                        ));
                                    }
                                }
                            }
                            None => errors.push(format!(
                                "prompt_learning.mipro.modules[{}].edges must be a list",
                                module_idx
                            )),
                        }
                    }
                }
            }

            let bootstrap_seeds = pl_section
                .get("bootstrap_train_seeds")
                .or_else(|| mipro_map.get("bootstrap_train_seeds"));
            let online_pool = pl_section
                .get("online_pool")
                .or_else(|| mipro_map.get("online_pool"));

            match bootstrap_seeds {
                None => errors.push(
                    "Missing required field: prompt_learning.bootstrap_train_seeds\n  MIPRO requires bootstrap seeds for the few-shot bootstrapping phase.\n  Example:\n    [prompt_learning]\n    bootstrap_train_seeds = [0, 1, 2, 3, 4]"
                        .to_string(),
                ),
                Some(Value::Array(arr)) => {
                    if arr.is_empty() {
                        errors.push("prompt_learning.bootstrap_train_seeds cannot be empty".to_string());
                    }
                }
                Some(_) => errors.push("prompt_learning.bootstrap_train_seeds must be an array".to_string()),
            }

            match online_pool {
                None => errors.push(
                    "Missing required field: prompt_learning.online_pool\n  MIPRO requires online_pool seeds for mini-batch evaluation during optimization.\n  Example:\n    [prompt_learning]\n    online_pool = [5, 6, 7, 8, 9]"
                        .to_string(),
                ),
                Some(Value::Array(arr)) => {
                    if arr.is_empty() {
                        errors.push("prompt_learning.online_pool cannot be empty".to_string());
                    }
                }
                Some(_) => errors.push("prompt_learning.online_pool must be an array".to_string()),
            }

            if let Some(threshold) = mipro_map.get("few_shot_score_threshold") {
                match parse_float(threshold) {
                    Some(fval) => {
                        if !(0.0..=1.0).contains(&fval) {
                            errors.push("prompt_learning.mipro.few_shot_score_threshold must be between 0.0 and 1.0".to_string());
                        }
                    }
                    None => errors.push(
                        "prompt_learning.mipro.few_shot_score_threshold must be a number"
                            .to_string(),
                    ),
                }
            }

            if let Some(val) = mipro_map.get("min_bootstrap_demos") {
                match parse_int(val) {
                    Some(ival) => {
                        if ival < 0 {
                            errors.push(
                                "prompt_learning.mipro.min_bootstrap_demos must be >= 0"
                                    .to_string(),
                            );
                        } else if let Some(Value::Array(arr)) = bootstrap_seeds {
                            if ival as usize > arr.len() {
                                errors.push(format!(
                                    "prompt_learning.mipro.min_bootstrap_demos ({}) exceeds bootstrap_train_seeds count ({}). You can never have more demos than bootstrap seeds.",
                                    ival,
                                    arr.len()
                                ));
                            }
                        }
                    }
                    None => errors.push(
                        "prompt_learning.mipro.min_bootstrap_demos must be an integer".to_string(),
                    ),
                }
            }

            if let Some(reference_pool) = mipro_map
                .get("reference_pool")
                .or_else(|| pl_section.get("reference_pool"))
            {
                match reference_pool.as_array() {
                    Some(ref_list) => {
                        let mut all_train_test = HashSet::new();
                        if let Some(Value::Array(arr)) = bootstrap_seeds {
                            for item in arr {
                                if let Some(val) = value_to_string(item) {
                                    all_train_test.insert(val);
                                }
                            }
                        }
                        if let Some(Value::Array(arr)) = online_pool {
                            for item in arr {
                                if let Some(val) = value_to_string(item) {
                                    all_train_test.insert(val);
                                }
                            }
                        }
                        let test_pool = mipro_map
                            .get("test_pool")
                            .or_else(|| pl_section.get("test_pool"));
                        if let Some(Value::Array(arr)) = test_pool {
                            for item in arr {
                                if let Some(val) = value_to_string(item) {
                                    all_train_test.insert(val);
                                }
                            }
                        }
                        let mut overlapping = Vec::new();
                        for item in ref_list {
                            if let Some(val) = value_to_string(item) {
                                if all_train_test.contains(&val) {
                                    overlapping.push(val);
                                }
                            }
                        }
                        if !overlapping.is_empty() {
                            errors.push(format!(
                                "reference_pool seeds must not overlap with bootstrap/online/test pools. Found overlapping seeds: {:?}",
                                overlapping
                            ));
                        }
                    }
                    None => errors.push(
                        "prompt_learning.mipro.reference_pool (or prompt_learning.reference_pool) must be an array"
                            .to_string(),
                    ),
                }
            }
        }
        _ => {}
    }

    if let Some(Value::Object(gepa)) = pl_section.get("gepa") {
        if let Some(adaptive_pool) = gepa.get("adaptive_pool") {
            validate_adaptive_pool_config(adaptive_pool, "gepa.adaptive_pool", &mut errors);
        }
    }
    if let Some(Value::Object(mipro)) = pl_section.get("mipro") {
        if let Some(adaptive_pool) = mipro.get("adaptive_pool") {
            validate_adaptive_pool_config(adaptive_pool, "mipro.adaptive_pool", &mut errors);
        }
    }

    errors
}
