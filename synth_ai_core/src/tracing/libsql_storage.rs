//! libsql-based trace storage implementation.
//!
//! Supports local SQLite and remote Turso databases.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use libsql::{params, Builder, Connection, Database, Value as LibsqlValue};
use serde_json::Value;
use tokio::sync::Mutex;

use super::error::TracingError;
use super::models::{
    EventReward, MarkovBlanketMessage, OutcomeReward, SessionTimeStep, SessionTrace, TracingEvent,
};
use super::storage::{QueryParams, StorageConfig, TraceStorage};

/// SQL statements for schema creation.
const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS session_traces (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    num_timesteps INTEGER NOT NULL DEFAULT 0,
    num_events INTEGER NOT NULL DEFAULT 0,
    num_messages INTEGER NOT NULL DEFAULT 0,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS session_timesteps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    turn_number INTEGER,
    started_at TEXT,
    completed_at TEXT,
    step_metadata TEXT,
    UNIQUE(session_id, step_id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestep_id INTEGER,
    event_type TEXT NOT NULL,
    system_instance_id TEXT,
    event_time REAL,
    message_time INTEGER,
    model_name TEXT,
    provider TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    cost_cents INTEGER,
    latency_ms INTEGER,
    call_records TEXT,
    reward REAL,
    terminated INTEGER,
    truncated INTEGER,
    system_state_before TEXT,
    system_state_after TEXT,
    metadata TEXT,
    event_metadata TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestep_id INTEGER,
    message_type TEXT NOT NULL,
    content TEXT NOT NULL,
    event_time REAL,
    message_time INTEGER,
    metadata TEXT
);

CREATE TABLE IF NOT EXISTS outcome_rewards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    objective_key TEXT DEFAULT 'reward',
    total_reward REAL NOT NULL,
    achievements_count INTEGER NOT NULL,
    total_steps INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    reward_metadata TEXT,
    annotation TEXT
);

CREATE TABLE IF NOT EXISTS event_rewards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    message_id INTEGER,
    turn_number INTEGER,
    objective_key TEXT DEFAULT 'reward',
    reward_value REAL NOT NULL,
    reward_type TEXT,
    key TEXT,
    annotation TEXT,
    source TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_timesteps_session ON session_timesteps(session_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_timestep ON events(timestep_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_outcome_rewards_session ON outcome_rewards(session_id);
CREATE INDEX IF NOT EXISTS idx_event_rewards_session ON event_rewards(session_id);
"#;

/// libsql-based trace storage.
pub struct LibsqlTraceStorage {
    _db: Database,
    conn: Mutex<Connection>,
    config: StorageConfig,
    initialized: Mutex<bool>,
}

impl LibsqlTraceStorage {
    /// Create storage from a configuration.
    pub async fn new(config: StorageConfig) -> Result<Self, TracingError> {
        let db = if config.db_url == ":memory:" {
            Builder::new_local(":memory:").build().await?
        } else if config.db_url.starts_with("libsql://") || config.db_url.starts_with("https://") {
            // Remote Turso database
            let builder = Builder::new_remote(
                config.db_url.clone(),
                config.auth_token.clone().unwrap_or_default(),
            );
            builder.build().await?
        } else {
            // Local file database
            Builder::new_local(&config.db_url).build().await?
        };

        let conn = db.connect()?;

        let storage = Self {
            _db: db,
            conn: Mutex::new(conn),
            config,
            initialized: Mutex::new(false),
        };

        if storage.config.auto_init {
            storage.initialize().await?;
        }

        Ok(storage)
    }

    /// Create an in-memory storage.
    pub async fn new_memory() -> Result<Self, TracingError> {
        Self::new(StorageConfig::memory()).await
    }

    /// Create a file-based storage.
    pub async fn new_file(path: &str) -> Result<Self, TracingError> {
        Self::new(StorageConfig::file(path)).await
    }

    /// Helper to serialize a value to JSON string.
    fn json_str(value: &Value) -> Option<String> {
        if value.is_null() {
            None
        } else {
            Some(value.to_string())
        }
    }

    /// Helper to convert cost_usd to cost_cents for storage.
    fn cost_to_cents(cost_usd: Option<f64>) -> Option<i64> {
        cost_usd.map(|c| (c * 100.0).round() as i64)
    }
}

fn json_to_libsql(value: &Value) -> LibsqlValue {
    match value {
        Value::Null => LibsqlValue::Null,
        Value::Bool(v) => LibsqlValue::Integer(if *v { 1 } else { 0 }),
        Value::Number(num) => {
            if let Some(i) = num.as_i64() {
                LibsqlValue::Integer(i)
            } else if let Some(f) = num.as_f64() {
                LibsqlValue::Real(f)
            } else {
                LibsqlValue::Null
            }
        }
        Value::String(s) => LibsqlValue::Text(s.clone()),
        Value::Array(_) | Value::Object(_) => LibsqlValue::Text(value.to_string()),
    }
}

fn libsql_to_json(value: LibsqlValue) -> Value {
    match value {
        LibsqlValue::Null => Value::Null,
        LibsqlValue::Integer(i) => Value::Number(i.into()),
        LibsqlValue::Real(f) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        LibsqlValue::Text(s) => Value::String(s),
        LibsqlValue::Blob(blob) => {
            let arr = blob.into_iter().map(|b| Value::Number(b.into())).collect();
            Value::Array(arr)
        }
    }
}

#[async_trait]
impl TraceStorage for LibsqlTraceStorage {
    async fn initialize(&self) -> Result<(), TracingError> {
        let mut initialized = self.initialized.lock().await;
        if *initialized {
            return Ok(());
        }

        let conn = self.conn.lock().await;

        // Execute schema SQL statements one by one
        for statement in SCHEMA_SQL.split(';') {
            let trimmed = statement.trim();
            if !trimmed.is_empty() {
                conn.execute(trimmed, ()).await?;
            }
        }

        *initialized = true;
        Ok(())
    }

    async fn close(&self) -> Result<(), TracingError> {
        // libsql connections are closed when dropped
        Ok(())
    }

    async fn ensure_session(
        &self,
        session_id: &str,
        created_at: DateTime<Utc>,
        metadata: &Value,
    ) -> Result<(), TracingError> {
        let conn = self.conn.lock().await;

        conn.execute(
            "INSERT INTO session_traces (session_id, created_at, metadata)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(session_id) DO NOTHING",
            params![
                session_id,
                created_at.to_rfc3339(),
                Self::json_str(metadata)
            ],
        )
        .await?;

        Ok(())
    }

    async fn get_session(&self, session_id: &str) -> Result<Option<SessionTrace>, TracingError> {
        let conn = self.conn.lock().await;

        // Get session
        let mut rows = conn
            .query(
                "SELECT session_id, created_at, metadata FROM session_traces WHERE session_id = ?1",
                params![session_id],
            )
            .await?;

        let row = match rows.next().await? {
            Some(r) => r,
            None => return Ok(None),
        };

        let session_id: String = row.get(0)?;
        let created_at_str: String = row.get(1)?;
        let metadata_str: Option<String> = row.get(2)?;

        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| TracingError::Internal(format!("invalid datetime: {}", e)))?;

        let metadata: std::collections::HashMap<String, Value> = metadata_str
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        // Get timesteps
        let mut timestep_rows = conn
            .query(
                "SELECT id, step_id, step_index, turn_number, started_at, completed_at, step_metadata
                 FROM session_timesteps WHERE session_id = ?1 ORDER BY step_index",
                params![session_id.clone()],
            )
            .await?;

        let mut timesteps = Vec::new();
        while let Some(row) = timestep_rows.next().await? {
            let step_id: String = row.get(1)?;
            let step_index: i32 = row.get(2)?;
            let turn_number: Option<i32> = row.get(3)?;
            let started_at_str: Option<String> = row.get(4)?;
            let completed_at_str: Option<String> = row.get(5)?;
            let step_metadata_str: Option<String> = row.get(6)?;

            let timestamp = started_at_str
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            let completed_at = completed_at_str
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc));

            let step_metadata = step_metadata_str
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();

            timesteps.push(SessionTimeStep {
                step_id,
                step_index,
                timestamp,
                turn_number,
                events: Vec::new(), // Events loaded separately if needed
                markov_blanket_messages: Vec::new(),
                step_metadata,
                completed_at,
            });
        }

        Ok(Some(SessionTrace {
            session_id,
            created_at,
            session_time_steps: timesteps,
            event_history: Vec::new(), // Events loaded separately if needed
            markov_blanket_message_history: Vec::new(),
            metadata,
        }))
    }

    async fn delete_session(&self, session_id: &str) -> Result<bool, TracingError> {
        let conn = self.conn.lock().await;

        // Delete in order to respect foreign key-like relationships
        conn.execute(
            "DELETE FROM event_rewards WHERE session_id = ?1",
            params![session_id],
        )
        .await?;
        conn.execute(
            "DELETE FROM outcome_rewards WHERE session_id = ?1",
            params![session_id],
        )
        .await?;
        conn.execute(
            "DELETE FROM messages WHERE session_id = ?1",
            params![session_id],
        )
        .await?;
        conn.execute(
            "DELETE FROM events WHERE session_id = ?1",
            params![session_id],
        )
        .await?;
        conn.execute(
            "DELETE FROM session_timesteps WHERE session_id = ?1",
            params![session_id],
        )
        .await?;
        let result = conn
            .execute(
                "DELETE FROM session_traces WHERE session_id = ?1",
                params![session_id],
            )
            .await?;

        Ok(result > 0)
    }

    async fn ensure_timestep(
        &self,
        session_id: &str,
        step: &SessionTimeStep,
    ) -> Result<i64, TracingError> {
        let conn = self.conn.lock().await;

        // Try to insert, get ID
        conn.execute(
            "INSERT INTO session_timesteps (session_id, step_id, step_index, turn_number, started_at, step_metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(session_id, step_id) DO NOTHING",
            params![
                session_id,
                step.step_id.clone(),
                step.step_index,
                step.turn_number,
                step.timestamp.to_rfc3339(),
                Self::json_str(&serde_json::to_value(&step.step_metadata).unwrap_or_default())
            ],
        )
        .await?;

        // Get the ID
        let mut rows = conn
            .query(
                "SELECT id FROM session_timesteps WHERE session_id = ?1 AND step_id = ?2",
                params![session_id, step.step_id.clone()],
            )
            .await?;

        let row = rows.next().await?.ok_or(TracingError::Internal(
            "failed to get timestep id".to_string(),
        ))?;

        Ok(row.get(0)?)
    }

    async fn update_timestep(
        &self,
        session_id: &str,
        step_id: &str,
        completed_at: Option<DateTime<Utc>>,
    ) -> Result<(), TracingError> {
        let conn = self.conn.lock().await;

        if let Some(dt) = completed_at {
            conn.execute(
                "UPDATE session_timesteps SET completed_at = ?1 WHERE session_id = ?2 AND step_id = ?3",
                params![dt.to_rfc3339(), session_id, step_id],
            )
            .await?;
        }

        Ok(())
    }

    async fn insert_event(
        &self,
        session_id: &str,
        timestep_db_id: Option<i64>,
        event: &TracingEvent,
    ) -> Result<i64, TracingError> {
        let conn = self.conn.lock().await;

        let base = event.base();
        let event_type = event.event_type().to_string();

        // Extract type-specific fields
        let (
            model_name,
            provider,
            input_tokens,
            output_tokens,
            total_tokens,
            cost_cents,
            latency_ms,
            call_records,
            reward,
            terminated,
            truncated,
            system_state_before,
            system_state_after,
        ) = match event {
            TracingEvent::Cais(e) => (
                Some(e.model_name.clone()),
                e.provider.clone(),
                e.input_tokens,
                e.output_tokens,
                e.total_tokens,
                Self::cost_to_cents(e.cost_usd),
                e.latency_ms,
                e.call_records
                    .as_ref()
                    .map(|r| serde_json::to_string(r).ok())
                    .flatten(),
                None,
                None,
                None,
                e.system_state_before.as_ref().map(|v| v.to_string()),
                e.system_state_after.as_ref().map(|v| v.to_string()),
            ),
            TracingEvent::Environment(e) => (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(e.reward),
                Some(e.terminated as i32),
                Some(e.truncated as i32),
                e.system_state_before.as_ref().map(|v| v.to_string()),
                e.system_state_after.as_ref().map(|v| v.to_string()),
            ),
            TracingEvent::Runtime(_) => (
                None, None, None, None, None, None, None, None, None, None, None, None, None,
            ),
        };

        conn.execute(
            "INSERT INTO events (session_id, timestep_id, event_type, system_instance_id, event_time, message_time,
                                 model_name, provider, input_tokens, output_tokens, total_tokens, cost_cents, latency_ms,
                                 call_records, reward, terminated, truncated, system_state_before, system_state_after,
                                 metadata, event_metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
            params![
                session_id,
                timestep_db_id,
                event_type,
                base.system_instance_id.clone(),
                base.time_record.event_time,
                base.time_record.message_time,
                model_name,
                provider,
                input_tokens,
                output_tokens,
                total_tokens,
                cost_cents,
                latency_ms,
                call_records,
                reward,
                terminated,
                truncated,
                system_state_before,
                system_state_after,
                Self::json_str(&serde_json::to_value(&base.metadata).unwrap_or_default()),
                base.event_metadata.as_ref().map(|v| serde_json::to_string(v).ok()).flatten()
            ],
        )
        .await?;

        // Get the last inserted ID
        let mut rows = conn.query("SELECT last_insert_rowid()", ()).await?;
        let row = rows.next().await?.ok_or(TracingError::Internal(
            "failed to get last insert id".to_string(),
        ))?;

        Ok(row.get(0)?)
    }

    async fn insert_message(
        &self,
        session_id: &str,
        timestep_db_id: Option<i64>,
        msg: &MarkovBlanketMessage,
    ) -> Result<i64, TracingError> {
        let conn = self.conn.lock().await;

        let content = msg.content.as_text().unwrap_or_default();

        conn.execute(
            "INSERT INTO messages (session_id, timestep_id, message_type, content, event_time, message_time, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id,
                timestep_db_id,
                msg.message_type.clone(),
                content,
                msg.time_record.event_time,
                msg.time_record.message_time,
                Self::json_str(&serde_json::to_value(&msg.metadata).unwrap_or_default())
            ],
        )
        .await?;

        let mut rows = conn.query("SELECT last_insert_rowid()", ()).await?;
        let row = rows.next().await?.ok_or(TracingError::Internal(
            "failed to get last insert id".to_string(),
        ))?;

        Ok(row.get(0)?)
    }

    async fn insert_outcome_reward(
        &self,
        session_id: &str,
        reward: &OutcomeReward,
    ) -> Result<i64, TracingError> {
        let conn = self.conn.lock().await;

        conn.execute(
            "INSERT INTO outcome_rewards (session_id, objective_key, total_reward, achievements_count,
                                          total_steps, created_at, reward_metadata, annotation)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                session_id,
                reward.objective_key.clone(),
                reward.total_reward,
                reward.achievements_count,
                reward.total_steps,
                Utc::now().to_rfc3339(),
                Self::json_str(&serde_json::to_value(&reward.reward_metadata).unwrap_or_default()),
                reward.annotation.as_ref().map(|v| v.to_string())
            ],
        )
        .await?;

        let mut rows = conn.query("SELECT last_insert_rowid()", ()).await?;
        let row = rows.next().await?.ok_or(TracingError::Internal(
            "failed to get last insert id".to_string(),
        ))?;

        Ok(row.get(0)?)
    }

    async fn insert_event_reward(
        &self,
        session_id: &str,
        event_id: i64,
        message_id: Option<i64>,
        turn_number: Option<i32>,
        reward: &EventReward,
    ) -> Result<i64, TracingError> {
        let conn = self.conn.lock().await;

        conn.execute(
            "INSERT INTO event_rewards (event_id, session_id, message_id, turn_number, objective_key,
                                        reward_value, reward_type, key, annotation, source, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                event_id,
                session_id,
                message_id,
                turn_number,
                reward.objective_key.clone(),
                reward.reward_value,
                reward.reward_type.clone(),
                reward.key.clone(),
                reward.annotation.as_ref().map(|v| v.to_string()),
                reward.source.clone(),
                Utc::now().to_rfc3339()
            ],
        )
        .await?;

        let mut rows = conn.query("SELECT last_insert_rowid()", ()).await?;
        let row = rows.next().await?.ok_or(TracingError::Internal(
            "failed to get last insert id".to_string(),
        ))?;

        Ok(row.get(0)?)
    }

    async fn query(&self, sql: &str, params: QueryParams) -> Result<Vec<Value>, TracingError> {
        let conn = self.conn.lock().await;

        let mut rows = match params {
            QueryParams::None => conn.query(sql, ()).await?,
            QueryParams::Positional(values) => {
                let params = values
                    .into_iter()
                    .map(|v| json_to_libsql(&v))
                    .collect::<Vec<_>>();
                conn.query(sql, params).await?
            }
            QueryParams::Named(values) => {
                let params = values
                    .into_iter()
                    .map(|(k, v)| (k, json_to_libsql(&v)))
                    .collect::<Vec<_>>();
                conn.query(sql, params).await?
            }
        };
        let mut results = Vec::new();

        while let Some(row) = rows.next().await? {
            let mut obj = serde_json::Map::new();
            let column_count = row.column_count();
            for idx in 0..column_count {
                let name = row
                    .column_name(idx)
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| format!("col_{idx}"));
                let value = row.get_value(idx)?;
                obj.insert(name, libsql_to_json(value));
            }
            results.push(Value::Object(obj));
        }

        Ok(results)
    }

    async fn update_session_counts(&self, session_id: &str) -> Result<(), TracingError> {
        let conn = self.conn.lock().await;

        conn.execute(
            "UPDATE session_traces SET
                num_timesteps = (SELECT COUNT(*) FROM session_timesteps WHERE session_id = ?1),
                num_events = (SELECT COUNT(*) FROM events WHERE session_id = ?1),
                num_messages = (SELECT COUNT(*) FROM messages WHERE session_id = ?1)
             WHERE session_id = ?1",
            params![session_id],
        )
        .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracing::models::{BaseEventFields, LMCAISEvent, MessageContent, TimeRecord};

    #[tokio::test]
    async fn test_memory_storage() {
        let storage = LibsqlTraceStorage::new_memory().await.unwrap();

        // Test session
        let session_id = "test-session-1";
        storage
            .ensure_session(session_id, Utc::now(), &Value::Null)
            .await
            .unwrap();

        let session = storage.get_session(session_id).await.unwrap();
        assert!(session.is_some());
        assert_eq!(session.unwrap().session_id, session_id);
    }

    #[tokio::test]
    async fn test_timestep_insertion() {
        let storage = LibsqlTraceStorage::new_memory().await.unwrap();

        let session_id = "test-session-2";
        storage
            .ensure_session(session_id, Utc::now(), &Value::Null)
            .await
            .unwrap();

        let step = SessionTimeStep::new("step-1", 0);
        let step_id = storage.ensure_timestep(session_id, &step).await.unwrap();
        assert!(step_id > 0);

        // Calling again should return the same ID (idempotent)
        let step_id_2 = storage.ensure_timestep(session_id, &step).await.unwrap();
        assert_eq!(step_id, step_id_2);
    }

    #[tokio::test]
    async fn test_event_insertion() {
        let storage = LibsqlTraceStorage::new_memory().await.unwrap();

        let session_id = "test-session-3";
        storage
            .ensure_session(session_id, Utc::now(), &Value::Null)
            .await
            .unwrap();

        let event = TracingEvent::Cais(LMCAISEvent {
            base: BaseEventFields::new("test-system"),
            model_name: "gpt-4".to_string(),
            provider: Some("openai".to_string()),
            input_tokens: Some(100),
            output_tokens: Some(50),
            cost_usd: Some(0.0045),
            latency_ms: Some(1200),
            ..Default::default()
        });

        let event_id = storage
            .insert_event(session_id, None, &event)
            .await
            .unwrap();
        assert!(event_id > 0);
    }

    #[tokio::test]
    async fn test_message_insertion() {
        let storage = LibsqlTraceStorage::new_memory().await.unwrap();

        let session_id = "test-session-4";
        storage
            .ensure_session(session_id, Utc::now(), &Value::Null)
            .await
            .unwrap();

        let msg = MarkovBlanketMessage {
            content: MessageContent::from_text("Hello, world!"),
            message_type: "user".to_string(),
            time_record: TimeRecord::now(),
            metadata: Default::default(),
        };

        let msg_id = storage
            .insert_message(session_id, None, &msg)
            .await
            .unwrap();
        assert!(msg_id > 0);
    }

    #[tokio::test]
    async fn test_session_deletion() {
        let storage = LibsqlTraceStorage::new_memory().await.unwrap();

        let session_id = "test-session-5";
        storage
            .ensure_session(session_id, Utc::now(), &Value::Null)
            .await
            .unwrap();

        let deleted = storage.delete_session(session_id).await.unwrap();
        assert!(deleted);

        let session = storage.get_session(session_id).await.unwrap();
        assert!(session.is_none());
    }
}
