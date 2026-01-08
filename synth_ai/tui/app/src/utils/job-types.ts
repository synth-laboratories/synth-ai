/**
 * Job type and source enums.
 */

/** Job source - which API endpoint the job comes from (backend values) */
export enum JobSource {
  PromptLearning = "prompt-learning",
  Learning = "learning",
  Eval = "eval",
}

/** Training type / algorithm */
export enum TrainingType {
  // Prompt learning algorithms
  GEPA = "gepa",
  MIPRO = "mipro",
  // Evaluation
  Eval = "eval",
  // Fine-tuning
  SFT = "sft",
  DPO = "dpo",
  RLHF = "rlhf",
}

/** Normalize job source string to enum */
export function normalizeJobSource(source: string | null | undefined): JobSource | null {
  const s = (source || "").toLowerCase()
  switch (s) {
    case "prompt-learning":
    case "prompt_learning":
      return JobSource.PromptLearning
    case "learning":
      return JobSource.Learning
    case "eval":
      return JobSource.Eval
    default:
      return null
  }
}

/** Display names for job sources */
export const JobSourceDisplay: Record<JobSource, string> = {
  [JobSource.PromptLearning]: "Prompt Optimization",
  [JobSource.Learning]: "Learning",
  [JobSource.Eval]: "Eval",
}

/** Display names for training types */
export const TrainingTypeDisplay: Record<string, string> = {
  [TrainingType.GEPA]: "GEPA",
  [TrainingType.MIPRO]: "MIPRO",
  [TrainingType.Eval]: "Eval",
  [TrainingType.SFT]: "SFT",
  [TrainingType.DPO]: "DPO",
  [TrainingType.RLHF]: "RLHF",
}

/** Get display name for training type */
export function getTrainingTypeDisplay(trainingType: string | null | undefined): string {
  if (!trainingType) return ""
  const key = trainingType.toLowerCase()
  return TrainingTypeDisplay[key] || trainingType
}

/** Get display name for job source */
export function getJobSourceDisplay(source: JobSource | string | null | undefined): string {
  if (!source) return "unknown"
  const normalized = normalizeJobSource(source)
  if (normalized) return JobSourceDisplay[normalized]
  return String(source)
}

/** Check if job is an eval job */
export function isEvalJob(
  source: string | null | undefined,
  trainingType: string | null | undefined,
  jobId: string,
): boolean {
  return (
    normalizeJobSource(source) === JobSource.Eval ||
    trainingType === TrainingType.Eval ||
    jobId.startsWith("eval_")
  )
}
