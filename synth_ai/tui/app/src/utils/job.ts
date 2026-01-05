/**
 * Job-related helpers shared across the TUI.
 */
import type { JobSummary } from "../tui_data"

// Helper to extract environment name from job metadata
export function extractEnvName(job: JobSummary | null): string | null {
  if (!job?.metadata) return null
  const meta: any = job.metadata as any
  return (
    meta.prompt_initial_snapshot?.raw_config?.prompt_learning?.env_name ||
    meta.prompt_initial_snapshot?.optimizer_config?.env_name ||
    meta.config?.env_name ||
    meta.env_name ||
    null
  )
}


