use crate::{MiproError, Result};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    pub input: serde_json::Value,
    pub expected: serde_json::Value,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dataset {
    #[serde(default)]
    pub id: String,
    pub examples: Vec<Example>,
}

impl Dataset {
    #[tracing::instrument]
    pub fn new(id: String, examples: Vec<Example>) -> Result<Self> {
        if examples.is_empty() {
            return Err(MiproError::InvalidArgument(
                "dataset must contain at least one example".to_string(),
            ));
        }
        Ok(Self { id, examples })
    }

    #[tracing::instrument]
    pub fn from_json_str(s: &str) -> Result<Self> {
        let ds: Dataset = serde_json::from_str(s)?;
        if ds.examples.is_empty() {
            return Err(MiproError::InvalidArgument(
                "dataset must contain at least one example".to_string(),
            ));
        }
        Ok(ds)
    }

    #[tracing::instrument]
    pub fn to_json_pretty(&self) -> Result<String> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    #[tracing::instrument]
    pub fn split_train_holdout(&self, holdout_ratio: f32, seed: u64) -> Result<(Dataset, Dataset)> {
        if !(0.0 < holdout_ratio && holdout_ratio < 1.0) {
            return Err(MiproError::InvalidArgument(
                "holdout_ratio must be in (0,1)".to_string(),
            ));
        }

        let n = self.examples.len();
        if n < 2 {
            return Err(MiproError::InvalidArgument(
                "dataset must contain at least two examples to split".to_string(),
            ));
        }

        let holdout_n = ((n as f32) * holdout_ratio)
            .round()
            .clamp(1.0, (n - 1) as f32) as usize;

        let mut idxs: Vec<usize> = (0..n).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        idxs.shuffle(&mut rng);

        let (holdout_idxs, train_idxs) = idxs.split_at(holdout_n);
        let holdout = Dataset {
            id: format!("{}:holdout", self.id),
            examples: holdout_idxs
                .iter()
                .map(|&i| self.examples[i].clone())
                .collect(),
        };
        let train = Dataset {
            id: format!("{}:train", self.id),
            examples: train_idxs
                .iter()
                .map(|&i| self.examples[i].clone())
                .collect(),
        };
        Ok((train, holdout))
    }
}
