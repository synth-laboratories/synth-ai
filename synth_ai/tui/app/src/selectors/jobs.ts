/**
 * Job list selectors (pure-ish helpers).
 * Keep list labels stable; do not depend on metadata fetched later.
 */
import type { JobSummary } from "../tui_data"
import type { ListFilterMode } from "../types"

export function getJobTypeKey(job: JobSummary): string {
  const trainingType = job.training_type?.toLowerCase() || ""
  if (trainingType) return trainingType
  const source = job.job_source?.toLowerCase() || ""
  if (source) return source
  const id = job.job_id || ""
  if (id.startsWith("eval_")) return "eval"
  if (id.startsWith("pl_")) return "prompt_learning"
  if (id.startsWith("learning_")) return "learning"
  return "unknown"
}

export function getJobTypeLabel(job: JobSummary): string {
  const type = getJobTypeKey(job)
  switch (type) {
    case "gepa":
    case "gepa_v1":
    case "prompt_gepa":
      return "Prompt GEPA"
    case "prompt_mipro":
      return "Prompt MIPRO"
    case "prompt-learning":
    case "prompt_learning":
      return "Prompt Optimization"
    case "eval":
      return "Eval"
    case "eval_verifier":
      return "Eval (Verifier)"
    case "eval_graph":
      return "Eval (Graph)"
    case "learning":
      return "Learning"
    case "graph_evolve":
      return "Graph Evolve"
    case "verifier_evolve":
      return "Verifier Evolve"
    case "policy_evolve":
      return "Policy Evolve"
    case "rlm_evolve":
      return "RLM Evolve"
    default:
      return type
        .replace(/_/g, " ")
        .replace(/-/g, " ")
        .split(" ")
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ")
  }
}

export function buildJobTypeOptions(
  jobs: JobSummary[],
): Array<{ id: string; label: string; count: number }> {
  const counts = new Map<string, { label: string; count: number }>()
  for (const job of jobs) {
    const key = getJobTypeKey(job)
    const label = getJobTypeLabel(job)
    const current = counts.get(key)
    if (current) {
      current.count += 1
    } else {
      counts.set(key, { label, count: 1 })
    }
  }
  return Array.from(counts.entries())
    .sort((a, b) => a[1].label.localeCompare(b[1].label))
    .map(([id, data]) => ({ id, label: data.label, count: data.count }))
}

export function getFilteredJobsByType(
  jobs: JobSummary[],
  typeFilter: ReadonlySet<string>,
  mode: ListFilterMode,
): JobSummary[] {
  if (mode === "none") return []
  if (mode === "all") return jobs
  if (!typeFilter.size) return []
  return jobs.filter((job) => typeFilter.has(getJobTypeKey(job)))
}
