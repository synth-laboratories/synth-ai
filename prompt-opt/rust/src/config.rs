use crate::models::MiproConfig;
use crate::{MiproError, Result};

impl MiproConfig {
    #[tracing::instrument]
    pub fn validate(&self) -> Result<()> {
        if self.num_candidates == 0 {
            return Err(MiproError::InvalidConfig(
                "num_candidates must be > 0".to_string(),
            ));
        }
        if !(0.0 < self.holdout_ratio && self.holdout_ratio < 1.0) {
            return Err(MiproError::InvalidConfig(
                "holdout_ratio must be in (0,1)".to_string(),
            ));
        }
        if self.max_iterations == 0 {
            return Err(MiproError::InvalidConfig(
                "max_iterations must be > 0".to_string(),
            ));
        }
        if self.early_stop_rounds == 0 {
            return Err(MiproError::InvalidConfig(
                "early_stop_rounds must be > 0".to_string(),
            ));
        }
        if self.min_improvement < 0.0 {
            return Err(MiproError::InvalidConfig(
                "min_improvement must be >= 0".to_string(),
            ));
        }
        Ok(())
    }
}
