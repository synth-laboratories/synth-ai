//! Data types for the Synth SDK.
//!
//! This module contains core data structures used throughout the SDK:
//! - Enums for job types, statuses, and configuration options
//! - Rubrics and criteria for evaluation
//! - Objectives and reward observations
//! - Judgements and rubric assignments
//! - Artifacts for storing outputs
//! - Context overrides for unified optimization

pub mod artifacts;
pub mod context_override;
pub mod contracts;
pub mod enum_values;
pub mod enums;
pub mod judgements;
pub mod levers;
pub mod objectives;
pub mod rewards;
pub mod rubrics;
pub mod sensors;

// Re-export all public types
pub use artifacts::{Artifact, ArtifactBundle, ArtifactContent};
pub use context_override::{
    ApplicationErrorType, ApplicationStatus, ContextOverride, ContextOverrideStatus,
};
pub use contracts::lever_sensor_v1_contract_schema;
pub use enum_values::data_enum_values;
pub use enums::{
    AdaptiveBatchLevel, AdaptiveCurriculumLevel, GraphType, InferenceMode, JobStatus, JobType,
    ObjectiveDirection, ObjectiveKey, OptimizationMode, OutputMode, ProviderName, RewardScope,
    RewardSource, RewardType, SuccessStatus, SynthModelName, TrainingType, VerifierMode,
};
pub use judgements::{CriterionScoreData, Judgement, RubricAssignment};
pub use levers::{
    Lever, LeverActor, LeverConstraints, LeverFormat, LeverKind, LeverMutability, LeverMutation,
    LeverProvenance, LeverSnapshot, MiproLeverSummary, ScopeKey, ScopeKind,
};
pub use objectives::{
    EventObjectiveAssignment, InstanceObjectiveAssignment, ObjectiveSpec,
    OutcomeObjectiveAssignment, RewardObservation,
};
pub use rewards::{
    CalibrationExample, EventRewardRecord, GoldExample, OutcomeRewardRecord, RewardAggregates,
};
pub use rubrics::{Criterion, CriterionExample, Rubric};
pub use sensors::{Sensor, SensorFrame, SensorFrameSummary, SensorKind};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all types are accessible
        let _ = JobStatus::Running;
        let _ = Rubric::new("1.0");
        let _ = ObjectiveSpec::maximize_reward();
        let _ = Judgement::new();
        let _ = Artifact::text("test");
        let _ = ContextOverride::new();
    }
}
