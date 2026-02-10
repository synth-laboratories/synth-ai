//! Progress tracking for optimization jobs.
//!
//! This module provides progress tracking and aggregation for GEPA/MIPRO
//! optimization jobs.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Instant;

use super::events::{EventCategory, EventParser, ParsedEvent};

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input/prompt tokens
    pub prompt_tokens: i64,
    /// Output/completion tokens
    pub completion_tokens: i64,
    /// Total tokens
    pub total_tokens: i64,
    /// Reasoning tokens (for o1-style models)
    #[serde(default)]
    pub reasoning_tokens: i64,
    /// Cached tokens
    #[serde(default)]
    pub cached_tokens: i64,
}

impl TokenUsage {
    /// Create from prompt and completion counts.
    pub fn new(prompt: i64, completion: i64) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            reasoning_tokens: 0,
            cached_tokens: 0,
        }
    }
}

/// Stage information for multi-stage prompts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageInfo {
    /// Instruction text
    pub instruction: String,
    /// Optional rules/constraints
    #[serde(default)]
    pub rules: HashMap<String, Value>,
    /// Optional temperature override
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Optional prompt variants
    #[serde(default)]
    pub prompts: Option<Vec<String>>,
}

/// Seed metadata for evaluated seeds.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SeedInfo {
    pub seed: i64,
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub expected: String,
    #[serde(default)]
    pub predicted: Option<String>,
    #[serde(default)]
    pub correct: Option<bool>,
    #[serde(default, alias = "score")]
    pub reward: Option<f64>,
}

/// Rollout sample details.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RolloutSample {
    pub seed: i64,
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub expected: String,
    #[serde(default)]
    pub predicted: String,
    #[serde(default)]
    pub correct: bool,
}

/// Information about a single candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateInfo {
    /// Unique candidate ID
    pub candidate_id: String,
    /// Reward on training set
    #[serde(default, alias = "accuracy")]
    pub reward: Option<f64>,
    /// Multi-objective scores
    #[serde(default)]
    pub objectives: Option<HashMap<String, f64>>,
    /// Validation reward (if validation phase completed)
    #[serde(default, alias = "val_accuracy")]
    pub val_reward: Option<f64>,
    /// Training reward
    #[serde(default, alias = "train_accuracy")]
    pub train_reward: Option<f64>,
    /// Generation number
    #[serde(default)]
    pub generation: Option<i32>,
    /// Parent candidate ID (for mutations)
    #[serde(default)]
    pub parent_id: Option<String>,
    /// Whether on Pareto frontier
    #[serde(default)]
    pub is_pareto: bool,
    /// Whether accepted into population
    #[serde(default)]
    pub accepted: bool,
    /// Type of mutation used
    #[serde(default)]
    pub mutation_type: Option<String>,
    /// Token usage for this candidate
    #[serde(default)]
    pub token_usage: Option<TokenUsage>,
    /// Cost in USD
    #[serde(default)]
    pub cost_usd: Option<f64>,
    /// Unix timestamp when evaluated
    #[serde(default)]
    pub timestamp: f64,
    /// Timestamp in milliseconds
    #[serde(default)]
    pub timestamp_ms: Option<i64>,
    /// First-class program stages
    #[serde(default)]
    pub stages: HashMap<String, StageInfo>,
    /// Prompt summary for compatibility
    #[serde(default)]
    pub prompt_summary: Option<String>,
    /// Mutation params
    #[serde(default)]
    pub mutation_params: Option<HashMap<String, Value>>,
    /// Transformation details
    #[serde(default)]
    pub transformation: Option<HashMap<String, Value>>,
    /// Seed rewards
    #[serde(default, alias = "seed_scores")]
    pub seed_rewards: Vec<Value>,
    /// Seeds evaluated
    #[serde(default)]
    pub seeds_evaluated: Vec<i64>,
    /// Seed metadata
    #[serde(default)]
    pub seed_info: Vec<SeedInfo>,
    /// Rollout samples
    #[serde(default)]
    pub rollout_sample: Vec<RolloutSample>,
    /// Evaluation duration in ms
    #[serde(default)]
    pub evaluation_duration_ms: Option<i64>,
    /// Minibatch rewards
    #[serde(default, alias = "minibatch_scores")]
    pub minibatch_rewards: Vec<f64>,
    /// Skip reason
    #[serde(default)]
    pub skip_reason: Option<String>,
    /// Raw event data for debugging
    #[serde(default)]
    pub raw_data: HashMap<String, Value>,
}

impl Default for CandidateInfo {
    fn default() -> Self {
        Self {
            candidate_id: String::new(),
            reward: None,
            objectives: None,
            val_reward: None,
            train_reward: None,
            generation: None,
            parent_id: None,
            is_pareto: false,
            accepted: false,
            mutation_type: None,
            token_usage: None,
            cost_usd: None,
            timestamp: 0.0,
            timestamp_ms: None,
            stages: HashMap::new(),
            prompt_summary: None,
            mutation_params: None,
            transformation: None,
            seed_rewards: Vec::new(),
            seeds_evaluated: Vec::new(),
            seed_info: Vec::new(),
            rollout_sample: Vec::new(),
            evaluation_duration_ms: None,
            minibatch_rewards: Vec::new(),
            skip_reason: None,
            raw_data: HashMap::new(),
        }
    }
}

/// Baseline information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BaselineInfo {
    /// Baseline reward
    #[serde(alias = "accuracy")]
    pub reward: Option<f64>,
    /// Multi-objective scores
    #[serde(default)]
    pub objectives: Option<HashMap<String, f64>>,
    /// Validation reward
    #[serde(default, alias = "val_accuracy")]
    pub val_reward: Option<f64>,
    /// Per-instance rewards
    #[serde(default, alias = "instance_scores")]
    pub instance_rewards: Vec<f64>,
    /// Per-instance objectives
    #[serde(default)]
    pub instance_objectives: Option<Vec<HashMap<String, f64>>>,
    /// Seeds evaluated
    #[serde(default)]
    pub seeds_evaluated: Vec<i64>,
    /// Prompt configuration (if provided)
    #[serde(default)]
    pub prompt: Option<Value>,
    /// Rollout samples (if provided)
    #[serde(default)]
    pub rollout_sample: Vec<RolloutSample>,
}

/// Frontier update record.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrontierUpdate {
    /// Update timestamp
    pub timestamp: f64,
    /// Candidates added
    #[serde(default)]
    pub added: Vec<String>,
    /// Candidates removed
    #[serde(default)]
    pub removed: Vec<String>,
    /// Current frontier
    #[serde(default)]
    pub frontier: Vec<String>,
    /// Rewards by candidate
    #[serde(default, alias = "frontier_scores")]
    pub frontier_rewards: HashMap<String, f64>,
    /// Objective scores by candidate (if provided)
    #[serde(default)]
    pub frontier_objectives: Option<Vec<HashMap<String, f64>>>,
    /// Frontier size
    #[serde(default)]
    pub frontier_size: i32,
    /// Best optimistic reward
    #[serde(default, alias = "optimistic_score")]
    pub optimistic_reward: Option<f64>,
    /// Generation number
    #[serde(default)]
    pub generation: Option<i32>,
    /// Baseline reward (if provided)
    #[serde(default, alias = "baseline_score")]
    pub baseline_reward: Option<f64>,
    /// Timestamp in milliseconds (if provided)
    #[serde(default)]
    pub timestamp_ms: Option<i64>,
}

/// Overall GEPA progress state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GEPAProgress {
    /// Current phase: "init", "optimization", "validation", "complete", "failed"
    pub phase: String,
    /// Rollouts completed
    pub rollouts_completed: i32,
    /// Total rollouts planned
    pub rollouts_total: i32,
    /// Generations completed
    pub generations_completed: i32,
    /// Candidates evaluated
    pub candidates_evaluated: i32,
    /// Current best reward
    #[serde(alias = "best_score")]
    pub best_reward: f64,
    /// Baseline reward for lift calculation
    #[serde(alias = "baseline_score")]
    pub baseline_reward: Option<f64>,
    /// Elapsed time in seconds
    pub elapsed_seconds: f64,
    /// Estimated time remaining
    pub eta_seconds: Option<f64>,
    /// Finish reason if complete
    pub finish_reason: Option<String>,
}

impl Default for GEPAProgress {
    fn default() -> Self {
        Self {
            phase: "init".to_string(),
            rollouts_completed: 0,
            rollouts_total: 0,
            generations_completed: 0,
            candidates_evaluated: 0,
            best_reward: 0.0,
            baseline_reward: None,
            elapsed_seconds: 0.0,
            eta_seconds: None,
            finish_reason: None,
        }
    }
}

impl GEPAProgress {
    /// Calculate progress percentage.
    pub fn progress_pct(&self) -> f64 {
        if self.rollouts_total > 0 {
            (self.rollouts_completed as f64 / self.rollouts_total as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate lift over baseline.
    pub fn lift(&self) -> Option<f64> {
        self.baseline_reward.map(|b| {
            if b > 0.0 {
                (self.best_reward - b) / b
            } else {
                0.0
            }
        })
    }
}

/// Progress tracker that aggregates events into state.
pub struct ProgressTracker {
    /// Overall progress
    pub progress: GEPAProgress,
    /// All evaluated candidates
    pub candidates: Vec<CandidateInfo>,
    /// Candidates indexed by ID
    candidates_by_id: HashMap<String, usize>,
    /// Baseline information
    pub baseline: Option<BaselineInfo>,
    /// Current Pareto frontier
    pub frontier: Vec<String>,
    /// Frontier update history
    pub frontier_history: Vec<FrontierUpdate>,
    /// Generation history
    pub generation_history: Vec<GenerationInfo>,
    /// Start time for elapsed calculation
    start_time: Option<Instant>,
    /// Last event sequence number
    pub last_seq: i64,
}

/// Generation summary info.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationInfo {
    /// Generation number
    pub generation: i32,
    /// Best reward in generation
    #[serde(alias = "best_accuracy")]
    pub best_reward: f64,
    /// Candidates proposed
    pub candidates_proposed: i32,
    /// Candidates accepted
    pub candidates_accepted: i32,
    /// Frontier size
    #[serde(default)]
    pub frontier_size: i32,
    /// Child candidates
    #[serde(default)]
    pub children: Vec<Value>,
    /// Generation duration ms
    #[serde(default)]
    pub duration_ms: Option<f64>,
    /// Timestamp seconds
    #[serde(default)]
    pub timestamp: f64,
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressTracker {
    fn extract_instance_rewards(value: &Value) -> Option<Vec<f64>> {
        let instance_objectives = value.get("instance_objectives")?.as_array()?;
        if instance_objectives.is_empty() {
            return None;
        }
        let mut values = Vec::with_capacity(instance_objectives.len());
        for item in instance_objectives {
            let reward_val = if let Some(obj) = item.as_object() {
                if let Some(objectives) = obj.get("objectives").and_then(|v| v.as_object()) {
                    objectives.get("reward").and_then(|v| {
                        v.as_f64()
                            .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
                    })
                } else {
                    obj.get("reward").and_then(|v| {
                        v.as_f64()
                            .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
                    })
                }
            } else {
                None
            };
            let reward_val = reward_val?;
            values.push(reward_val);
        }
        if values.is_empty() {
            None
        } else {
            Some(values)
        }
    }

    /// Create a new progress tracker.
    pub fn new() -> Self {
        Self {
            progress: GEPAProgress::default(),
            candidates: Vec::new(),
            candidates_by_id: HashMap::new(),
            baseline: None,
            frontier: Vec::new(),
            frontier_history: Vec::new(),
            generation_history: Vec::new(),
            start_time: None,
            last_seq: -1,
        }
    }

    /// Get the current best reward.
    pub fn best_reward(&self) -> f64 {
        self.progress.best_reward
    }

    /// Get baseline reward.
    pub fn baseline_reward(&self) -> Option<f64> {
        self.progress.baseline_reward
    }

    /// Get lift over baseline.
    pub fn lift(&self) -> Option<f64> {
        self.progress.lift()
    }

    /// Get current frontier candidates.
    pub fn current_frontier(&self) -> &[String] {
        &self.frontier
    }

    /// Update tracker with an event.
    pub fn update(&mut self, event: &ParsedEvent) {
        // Start timer on first event
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        // Update elapsed time
        if let Some(start) = self.start_time {
            self.progress.elapsed_seconds = start.elapsed().as_secs_f64();
        }

        // Track sequence
        if let Some(seq) = event.seq {
            if seq > self.last_seq {
                self.last_seq = seq;
            }
        }

        // Handle by category
        match event.category {
            EventCategory::Baseline => self.handle_baseline(event),
            EventCategory::Candidate => self.handle_candidate(event),
            EventCategory::Frontier => self.handle_frontier(event),
            EventCategory::Progress => self.handle_progress(event),
            EventCategory::Generation => self.handle_generation(event),
            EventCategory::Complete => self.handle_complete(event),
            EventCategory::Termination => self.handle_termination(event),
            EventCategory::Validation => self.handle_validation(event),
            _ => {}
        }
    }

    fn handle_baseline(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_baseline(event);

        self.baseline = Some(BaselineInfo {
            reward: data.reward,
            objectives: data.objectives,
            val_reward: None,
            instance_rewards: data.instance_rewards.unwrap_or_default(),
            instance_objectives: data.instance_objectives,
            seeds_evaluated: event
                .data
                .get("seeds_evaluated")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
                .unwrap_or_default(),
            prompt: event.data.get("prompt").cloned(),
            rollout_sample: event
                .data
                .get("rollout_sample")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| {
                            serde_json::from_value::<RolloutSample>(item.clone()).ok()
                        })
                        .collect()
                })
                .unwrap_or_default(),
        });

        if let Some(acc) = data.reward {
            self.progress.baseline_reward = Some(acc);
            // Initialize best reward to baseline
            if self.progress.best_reward == 0.0 {
                self.progress.best_reward = acc;
            }
        }

        self.progress.phase = "optimization".to_string();
    }

    fn handle_candidate(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_candidate(event);

        let is_baseline = event
            .data
            .get("is_baseline")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            || data.parent_id.is_none();

        let mut merged_data = event.data.as_object().cloned().unwrap_or_default();
        if let Some(program_candidate) = event
            .data
            .get("program_candidate")
            .and_then(|v| v.as_object())
        {
            for (k, v) in program_candidate {
                merged_data.insert(k.clone(), v.clone());
            }
        }
        let merged_value = Value::Object(merged_data.clone());

        if is_baseline && self.baseline.is_none() {
            let candidate_view = merged_data.clone();
            let candidate_value = Value::Object(candidate_view.clone());

            let parse_f64_map = |val: Option<&Value>| -> Option<HashMap<String, f64>> {
                let map = val?.as_object()?;
                let mut out = HashMap::new();
                for (k, v) in map {
                    let val = v
                        .as_f64()
                        .or_else(|| v.as_str().and_then(|s| s.parse().ok()));
                    let val = match val {
                        Some(val) => val,
                        None => return None,
                    };
                    out.insert(k.clone(), val);
                }
                Some(out)
            };

            // Try top-level objectives, then score.objectives
            let objectives = parse_f64_map(candidate_view.get("objectives")).or_else(|| {
                candidate_view
                    .get("score")
                    .and_then(|v| v.as_object())
                    .and_then(|score| parse_f64_map(score.get("objectives")))
            });

            let accuracy = objectives
                .as_ref()
                .and_then(|m| m.get("reward").copied())
                .or_else(|| candidate_view.get("reward").and_then(|v| v.as_f64()))
                .or_else(|| candidate_view.get("accuracy").and_then(|v| v.as_f64()))
                .or_else(|| candidate_view.get("score").and_then(|v| v.as_f64()))
                .or_else(|| {
                    candidate_view
                        .get("score")
                        .and_then(|v| v.as_object())
                        .and_then(|score| {
                            score
                                .get("reward")
                                .and_then(|v| v.as_f64())
                                .or_else(|| score.get("mean_reward").and_then(|v| v.as_f64()))
                        })
                });

            // Auto-derive objectives from accuracy if not found
            let objectives = objectives.or_else(|| {
                accuracy.map(|r| {
                    let mut m = HashMap::new();
                    m.insert("reward".to_string(), r);
                    m
                })
            });

            let instance_scores = candidate_view
                .get("instance_scores")
                .and_then(|v| v.as_array())
                .and_then(|arr| {
                    let mut out = Vec::with_capacity(arr.len());
                    for item in arr {
                        let val = item
                            .as_f64()
                            .or_else(|| item.as_str().and_then(|s| s.parse().ok()))?;
                        out.push(val);
                    }
                    Some(out)
                })
                .or_else(|| Self::extract_instance_rewards(&candidate_value))
                .unwrap_or_default();

            let instance_objectives = candidate_view
                .get("instance_objectives")
                .and_then(|v| v.as_array())
                .and_then(|arr| {
                    let mut out = Vec::with_capacity(arr.len());
                    for item in arr {
                        let obj = item.as_object()?;
                        let mut map = HashMap::new();
                        for (k, v) in obj {
                            let val = v
                                .as_f64()
                                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))?;
                            map.insert(k.clone(), val);
                        }
                        out.push(map);
                    }
                    Some(out)
                });

            self.baseline = Some(BaselineInfo {
                reward: accuracy,
                objectives,
                val_reward: None,
                instance_rewards: instance_scores,
                instance_objectives,
                seeds_evaluated: merged_data
                    .get("seeds_evaluated")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
                    .unwrap_or_default(),
                prompt: candidate_view.get("prompt").cloned(),
                rollout_sample: candidate_view
                    .get("rollout_sample")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|item| {
                                serde_json::from_value::<RolloutSample>(item.clone()).ok()
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            });

            if let Some(acc) = accuracy {
                self.progress.baseline_reward = Some(acc);
            }
        }

        if self.candidates_by_id.contains_key(&data.candidate_id) {
            return;
        }

        let mut candidate = CandidateInfo {
            candidate_id: data.candidate_id.clone(),
            reward: data.reward,
            objectives: data.objectives,
            val_reward: None,
            train_reward: data.reward,
            generation: data.generation,
            parent_id: data.parent_id,
            is_pareto: data.is_pareto,
            accepted: data.accepted,
            mutation_type: data.mutation_type,
            token_usage: None,
            cost_usd: None,
            timestamp: self.progress.elapsed_seconds,
            timestamp_ms: event.timestamp_ms,
            stages: HashMap::new(),
            prompt_summary: None,
            mutation_params: None,
            transformation: None,
            seed_rewards: Vec::new(),
            seeds_evaluated: Vec::new(),
            seed_info: Vec::new(),
            rollout_sample: Vec::new(),
            evaluation_duration_ms: None,
            minibatch_rewards: Vec::new(),
            skip_reason: None,
            raw_data: HashMap::new(),
        };

        candidate.seeds_evaluated = merged_data
            .get("seeds_evaluated")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
            .unwrap_or_default();

        if let Some(val) = merged_data.get("val_accuracy").and_then(|v| v.as_f64()) {
            candidate.val_reward = Some(val);
        } else if let Some(val) = merged_data.get("full_score").and_then(|v| v.as_f64()) {
            candidate.val_reward = Some(val);
        }

        if let Some(val) = merged_data.get("train_accuracy").and_then(|v| v.as_f64()) {
            candidate.train_reward = Some(val);
        } else if let Some(val) = merged_data.get("minibatch_score").and_then(|v| v.as_f64()) {
            candidate.train_reward = Some(val);
        }

        if let Some(cost) = merged_data.get("cost_usd").and_then(|v| v.as_f64()) {
            candidate.cost_usd = Some(cost);
        }

        if let Some(duration) = merged_data
            .get("evaluation_duration_ms")
            .and_then(|v| v.as_i64())
        {
            candidate.evaluation_duration_ms = Some(duration);
        }

        if let Some(scores) = merged_data
            .get("minibatch_scores")
            .and_then(|v| v.as_array())
        {
            candidate.minibatch_rewards = scores
                .iter()
                .filter_map(|v| {
                    v.as_f64()
                        .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
                })
                .collect();
        } else if let Some(score) = merged_data.get("minibatch_score").and_then(|v| v.as_f64()) {
            candidate.minibatch_rewards = vec![score];
        }

        if let Some(reason) = merged_data.get("skip_reason").and_then(|v| v.as_str()) {
            candidate.skip_reason = Some(reason.to_string());
        }

        if let Some(scores) = merged_data.get("seed_scores").and_then(|v| v.as_array()) {
            candidate.seed_rewards = scores.clone();
        }

        if let Some(info) = merged_data.get("seed_info").and_then(|v| v.as_array()) {
            let mut seed_info = Vec::with_capacity(info.len());
            for item in info {
                if let Ok(parsed) = serde_json::from_value::<SeedInfo>(item.clone()) {
                    seed_info.push(parsed);
                }
            }
            candidate.seed_info = seed_info;
        }

        if let Some(samples) = merged_data.get("rollout_sample").and_then(|v| v.as_array()) {
            let mut rollout_sample = Vec::with_capacity(samples.len());
            for item in samples {
                if let Ok(parsed) = serde_json::from_value::<RolloutSample>(item.clone()) {
                    rollout_sample.push(parsed);
                }
            }
            candidate.rollout_sample = rollout_sample;
        }

        if let Some(token_usage) = merged_data.get("token_usage") {
            if let Ok(parsed) = serde_json::from_value::<TokenUsage>(token_usage.clone()) {
                candidate.token_usage = Some(parsed);
            }
        }

        if let Some(mutation_params) = merged_data
            .get("mutation_params")
            .and_then(|v| v.as_object())
        {
            candidate.mutation_params = Some(mutation_params.clone().into_iter().collect());
        }

        if let Some(transformation) = merged_data
            .get("transformation")
            .and_then(|v| v.as_object())
        {
            candidate.transformation = Some(transformation.clone().into_iter().collect());
        }

        if let Some(stages) = merged_data.get("stages").and_then(|v| v.as_object()) {
            let mut stage_map = HashMap::new();
            for (key, value) in stages {
                if let Ok(stage) = serde_json::from_value::<StageInfo>(value.clone()) {
                    stage_map.insert(key.clone(), stage);
                }
            }
            candidate.stages = stage_map;
        }

        if candidate.prompt_summary.is_none() {
            if let Some(summary) = merged_data
                .get("prompt_summary")
                .and_then(|v| v.as_str())
                .or_else(|| merged_data.get("prompt_text").and_then(|v| v.as_str()))
            {
                candidate.prompt_summary = Some(summary.to_string());
            } else if !candidate.stages.is_empty() {
                let mut parts = Vec::new();
                let mut stage_ids: Vec<_> = candidate.stages.keys().collect();
                stage_ids.sort();
                for stage_id in stage_ids {
                    if let Some(stage) = candidate.stages.get(stage_id) {
                        if !stage.instruction.is_empty() {
                            parts.push(format!(
                                "[{}]: {}",
                                stage_id.to_uppercase(),
                                stage.instruction
                            ));
                        }
                    }
                }
                if !parts.is_empty() {
                    candidate.prompt_summary = Some(parts.join("\n"));
                }
            }
        }

        if let Some(raw) = merged_value.as_object() {
            candidate.raw_data = raw.clone().into_iter().collect();
        }

        // Update best reward
        if let Some(acc) = data.reward {
            if acc > self.progress.best_reward {
                self.progress.best_reward = acc;
            }
        }

        // Store candidate
        let idx = self.candidates.len();
        self.candidates.push(candidate);
        self.candidates_by_id.insert(data.candidate_id, idx);
        self.progress.candidates_evaluated += 1;
    }

    fn handle_frontier(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_frontier(event);

        self.frontier = data.frontier.clone();

        if let Some(best) = data.best_reward {
            if best > self.progress.best_reward {
                self.progress.best_reward = best;
            }
        }

        let update = FrontierUpdate {
            timestamp: self.progress.elapsed_seconds,
            added: data.added,
            removed: data.removed,
            frontier: data.frontier,
            frontier_rewards: data.frontier_rewards.unwrap_or_default(),
            frontier_objectives: data.frontier_objectives,
            frontier_size: data.frontier_size,
            optimistic_reward: data.best_reward,
            generation: event
                .data
                .get("generation")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32),
            baseline_reward: event.data.get("baseline_score").and_then(|v| v.as_f64()),
            timestamp_ms: event.timestamp_ms,
        };
        self.frontier_history.push(update);
    }

    fn handle_progress(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_progress(event);

        self.progress.rollouts_completed = data.rollouts_completed;
        if let Some(total) = data.rollouts_total {
            self.progress.rollouts_total = total;
        }

        if let Some(best) = data.best_reward {
            if best > self.progress.best_reward {
                self.progress.best_reward = best;
            }
        }

        if let Some(baseline) = data.baseline_reward {
            if self.progress.baseline_reward.is_none() {
                self.progress.baseline_reward = Some(baseline);
            }
        }

        // Estimate ETA
        if self.progress.rollouts_total > 0 && self.progress.rollouts_completed > 0 {
            let remaining = self.progress.rollouts_total - self.progress.rollouts_completed;
            let rate = self.progress.elapsed_seconds / self.progress.rollouts_completed as f64;
            self.progress.eta_seconds = Some(remaining as f64 * rate);
        }
    }

    fn handle_generation(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_generation(event);

        self.progress.generations_completed = data.generation;

        let info = GenerationInfo {
            generation: data.generation,
            best_reward: data.best_reward,
            candidates_proposed: data.candidates_proposed,
            candidates_accepted: data.candidates_accepted,
            frontier_size: event
                .data
                .get("frontier_size")
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(0),
            children: event
                .data
                .get("children")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default(),
            duration_ms: event.data.get("duration_ms").and_then(|v| v.as_f64()),
            timestamp: event
                .data
                .get("timestamp")
                .and_then(|v| v.as_f64())
                .unwrap_or(self.progress.elapsed_seconds),
        };
        self.generation_history.push(info);
    }

    fn handle_complete(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_complete(event);

        self.progress.phase = "complete".to_string();
        self.progress.finish_reason = data.finish_reason;

        if let Some(best) = data.best_reward {
            self.progress.best_reward = best;
        }

        if let Some(baseline) = data.baseline_reward {
            self.progress.baseline_reward = Some(baseline);
        }
    }

    fn handle_termination(&mut self, event: &ParsedEvent) {
        let data = EventParser::parse_termination(event);

        self.progress.phase = "complete".to_string();
        self.progress.finish_reason = Some(data.reason);
    }

    fn handle_validation(&mut self, event: &ParsedEvent) {
        self.progress.phase = "validation".to_string();

        // Update candidate validation scores if provided
        if let Some(candidate_id) = event.data.get("candidate_id").and_then(|v| v.as_str()) {
            if let Some(val_score) = event.data.get("val_accuracy").and_then(|v| v.as_f64()) {
                if let Some(&idx) = self.candidates_by_id.get(candidate_id) {
                    self.candidates[idx].val_reward = Some(val_score);
                }
            }
        }
    }

    /// Get a summary dict for serialization.
    pub fn to_summary(&self) -> serde_json::Value {
        serde_json::json!({
            "phase": self.progress.phase,
            "rollouts_completed": self.progress.rollouts_completed,
            "rollouts_total": self.progress.rollouts_total,
            "candidates_evaluated": self.progress.candidates_evaluated,
            "generations_completed": self.progress.generations_completed,
            "best_reward": self.progress.best_reward,
            "baseline_reward": self.progress.baseline_reward,
            "lift": self.lift(),
            "elapsed_seconds": self.progress.elapsed_seconds,
            "frontier_size": self.frontier.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_progress_default() {
        let progress = GEPAProgress::default();
        assert_eq!(progress.phase, "init");
        assert_eq!(progress.progress_pct(), 0.0);
        assert!(progress.lift().is_none());
    }

    #[test]
    fn test_progress_lift() {
        let mut progress = GEPAProgress::default();
        progress.baseline_reward = Some(0.5);
        progress.best_reward = 0.75;

        let lift = progress.lift().unwrap();
        assert!((lift - 0.5).abs() < 0.001); // 50% lift
    }

    #[test]
    fn test_tracker_baseline() {
        let mut tracker = ProgressTracker::new();

        let event = EventParser::parse(&json!({
            "type": "learning.policy.gepa.baseline",
            "seq": 1,
            "data": { "accuracy": 0.72 }
        }));

        tracker.update(&event);

        assert!(tracker.baseline.is_some());
        assert_eq!(tracker.baseline_reward(), Some(0.72));
        assert_eq!(tracker.progress.phase, "optimization");
    }

    #[test]
    fn test_tracker_candidate() {
        let mut tracker = ProgressTracker::new();

        // First baseline
        tracker.update(&EventParser::parse(&json!({
            "type": "learning.policy.gepa.baseline",
            "data": { "accuracy": 0.72 }
        })));

        // Then candidate
        tracker.update(&EventParser::parse(&json!({
            "type": "learning.policy.gepa.candidate.evaluated",
            "seq": 2,
            "data": {
                "candidate_id": "cand_1",
                "accuracy": 0.85,
                "accepted": true,
                "generation": 1
            }
        })));

        assert_eq!(tracker.candidates.len(), 1);
        assert_eq!(tracker.best_reward(), 0.85);
        assert_eq!(tracker.progress.candidates_evaluated, 1);
    }

    #[test]
    fn test_tracker_frontier() {
        let mut tracker = ProgressTracker::new();

        tracker.update(&EventParser::parse(&json!({
            "type": "learning.policy.gepa.frontier_updated",
            "data": {
                "frontier": ["cand_1", "cand_2"],
                "best_score": 0.88
            }
        })));

        assert_eq!(tracker.frontier.len(), 2);
        assert_eq!(tracker.frontier_history.len(), 1);
        assert_eq!(tracker.best_reward(), 0.88);
    }

    #[test]
    fn test_tracker_complete() {
        let mut tracker = ProgressTracker::new();

        tracker.update(&EventParser::parse(&json!({
            "type": "learning.policy.gepa.job.completed",
            "data": {
                "best_score": 0.92,
                "baseline_score": 0.72,
                "finish_reason": "budget_exhausted"
            }
        })));

        assert_eq!(tracker.progress.phase, "complete");
        assert_eq!(
            tracker.progress.finish_reason,
            Some("budget_exhausted".to_string())
        );
        assert_eq!(tracker.best_reward(), 0.92);
    }
}
