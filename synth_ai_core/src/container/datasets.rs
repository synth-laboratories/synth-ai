use crate::errors::CoreError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDatasetSpec {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub splits: Vec<String>,
    #[serde(default)]
    pub default_split: Option<String>,
    #[serde(default)]
    pub cardinality: Option<i64>,
    #[serde(default)]
    pub description: Option<String>,
}

impl TaskDatasetSpec {
    pub fn validate(&self) -> Result<(), CoreError> {
        if let Some(default_split) = &self.default_split {
            if !self.splits.is_empty() && !self.splits.contains(default_split) {
                return Err(CoreError::InvalidInput(
                    "default_split must be one of splits when provided".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn merge_with(&self, other: &TaskDatasetSpec) -> TaskDatasetSpec {
        TaskDatasetSpec {
            id: if other.id.is_empty() {
                self.id.clone()
            } else {
                other.id.clone()
            },
            name: if other.name.is_empty() {
                self.name.clone()
            } else {
                other.name.clone()
            },
            version: other.version.clone().or_else(|| self.version.clone()),
            splits: if other.splits.is_empty() {
                self.splits.clone()
            } else {
                other.splits.clone()
            },
            default_split: other
                .default_split
                .clone()
                .or_else(|| self.default_split.clone()),
            cardinality: other.cardinality.or(self.cardinality),
            description: other
                .description
                .clone()
                .or_else(|| self.description.clone()),
        }
    }
}

pub fn ensure_split(spec: &TaskDatasetSpec, split: Option<&str>) -> Result<String, CoreError> {
    if spec.splits.is_empty() {
        return Ok(split
            .unwrap_or_else(|| spec.default_split.as_deref().unwrap_or("default"))
            .to_string());
    }
    match split {
        Some(value) => {
            if spec.splits.contains(&value.to_string()) {
                Ok(value.to_string())
            } else {
                Err(CoreError::InvalidInput(format!(
                    "Unknown split '{}' for dataset {}",
                    value, spec.id
                )))
            }
        }
        None => {
            if let Some(default_split) = &spec.default_split {
                Ok(default_split.clone())
            } else {
                Err(CoreError::InvalidInput(format!(
                    "split must be provided for dataset {}",
                    spec.id
                )))
            }
        }
    }
}

pub fn normalise_seed(seed: i64, cardinality: Option<i64>) -> i64 {
    let mut value = seed;
    if value < 0 {
        value = value.abs();
    }
    if let Some(card) = cardinality {
        if card > 0 {
            value = value % card;
        }
    }
    value
}
