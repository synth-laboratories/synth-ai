use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseJobEvent {
    pub job_id: String,
    pub seq: i64,
    pub ts: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub level: String,
    pub message: String,
    #[serde(default)]
    pub data: Value,
    #[serde(default)]
    pub run_id: Option<String>,
}

impl BaseJobEvent {
    pub fn to_dict_value(&self) -> Value {
        let mut map = Map::new();
        map.insert("job_id".to_string(), Value::String(self.job_id.clone()));
        map.insert("seq".to_string(), Value::Number(self.seq.into()));
        map.insert("ts".to_string(), Value::String(self.ts.clone()));
        map.insert("type".to_string(), Value::String(self.event_type.clone()));
        map.insert("level".to_string(), Value::String(self.level.clone()));
        map.insert("message".to_string(), Value::String(self.message.clone()));
        map.insert("data".to_string(), self.data.clone());
        if let Some(run_id) = &self.run_id {
            map.insert("run_id".to_string(), Value::String(run_id.clone()));
        }
        Value::Object(map)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobEvent {
    #[serde(flatten)]
    pub base: BaseJobEvent,
    #[serde(default)]
    pub status: Option<String>,
}

impl JobEvent {
    pub fn to_dict_value(&self) -> Value {
        let mut value = self.base.to_dict_value();
        if let Some(status) = &self.status {
            if let Value::Object(ref mut map) = value {
                let data = map
                    .entry("data".to_string())
                    .or_insert_with(|| Value::Object(Map::new()));
                if let Value::Object(ref mut data_map) = data {
                    data_map.insert("status".to_string(), Value::String(status.clone()));
                }
            }
        }
        value
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateEvent {
    #[serde(flatten)]
    pub base: BaseJobEvent,
    #[serde(default)]
    pub candidate_id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
}

impl CandidateEvent {
    pub fn to_dict_value(&self) -> Value {
        let mut value = self.base.to_dict_value();
        if let Value::Object(ref mut map) = value {
            let data = map
                .entry("data".to_string())
                .or_insert_with(|| Value::Object(Map::new()));
            if let Value::Object(ref mut data_map) = data {
                if let Some(candidate_id) = &self.candidate_id {
                    data_map.insert(
                        "candidate_id".to_string(),
                        Value::String(candidate_id.clone()),
                    );
                }
                if let Some(status) = &self.status {
                    data_map.insert("status".to_string(), Value::String(status.clone()));
                }
            }
        }
        value
    }
}
