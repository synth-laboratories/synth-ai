/**
 * Job detail panel formatting.
 */
import type { Snapshot } from "../types"
import type { JobEvent, JobSummary } from "../tui_data"
import { isEvalJob, num } from "../tui_data"
import { extractEnvName } from "../utils/job"
import { formatTimestamp, formatValue } from "./time"

export function formatDetails(snapshot: Snapshot): string {
  const job = snapshot.selectedJob
  if (!job) return "No job selected."

  // Eval jobs get specialized rendering
  if (isEvalJob(job)) {
    return formatEvalDetails(snapshot, job)
  }

  // Learning jobs (graph_gepa, etc.) - but not eval jobs
  if (job.job_source === "learning") {
    return formatLearningDetails(job)
  }

  // Default: prompt-learning jobs
  return formatPromptLearningDetails(snapshot, job)
}

export function formatEvalDetails(snapshot: Snapshot, job: JobSummary): string {
  const summary: any = snapshot.evalSummary ?? {}
  const rows: any[] = snapshot.evalResultRows ?? []

  const lines = [
    `Job: ${job.job_id}`,
    `Status: ${job.status}`,
    `Type: eval`,
    "",
    "═══ Eval Summary ═══",
  ]

  // Extract key metrics from summary
  const meanReward = summary.mean_reward
  if (meanReward != null) {
    lines.push(`  Mean Reward: ${formatValue(meanReward)}`)
  }
  const reward = summary.reward ?? summary.objectives?.reward ?? summary.accuracy
  if (reward != null) {
    lines.push(`  Reward: ${(reward * 100).toFixed(1)}%`)
  }
  if (summary.pass_rate != null) {
    lines.push(`  Pass Rate: ${(summary.pass_rate * 100).toFixed(1)}%`)
  }
  if (summary.completed != null && summary.total != null) {
    lines.push(`  Progress: ${summary.completed}/${summary.total}`)
  } else if (summary.completed != null) {
    lines.push(`  Completed: ${summary.completed}`)
  }
  if (summary.failed != null && summary.failed > 0) {
    lines.push(`  Failed: ${summary.failed}`)
  }

  // Show row count
  if (rows.length > 0) {
    lines.push(`  Results: ${rows.length} rows`)
    // Calculate score distribution
    const rewards = rows
      .map((row) => num(row.reward ?? row.outcome_reward ?? row.reward_mean ?? row.passed))
      .filter((val) => typeof val === "number") as number[]
    if (rewards.length > 0) {
      const mean = rewards.reduce((sum, val) => sum + val, 0) / rewards.length
      const passed = rewards.filter((s) => s >= 0.5 || s === 1).length
      lines.push(`  Avg Reward: ${mean.toFixed(4)}`)
      lines.push(`  Pass Rate: ${((passed / rewards.length) * 100).toFixed(1)}%`)
    }
  }

  lines.push("")
  lines.push("═══ Timing ═══")
  lines.push(`  Created: ${formatTimestamp(job.created_at)}`)
  lines.push(`  Started: ${formatTimestamp(job.started_at)}`)
  lines.push(`  Finished: ${formatTimestamp(job.finished_at)}`)

  if (job.error) {
    lines.push("")
    lines.push("═══ Error ═══")
    lines.push(`  ${job.error}`)
  }

  return lines.join("\n")
}

export function formatLearningDetails(job: JobSummary): string {
  const envName = extractEnvName(job)
  const lines = [
    `Job: ${job.job_id}`,
    `Status: ${job.status}`,
    `Type: ${job.job_type || "learning"}`,
    `Env: ${envName || "-"}`,
    "",
    "═══ Progress ═══",
    `  Best Reward: ${job.best_reward != null ? job.best_reward.toFixed(4) : "-"}`,
    `  Best Snapshot: ${job.best_snapshot_id || "-"}`,
    "",
    "═══ Timing ═══",
    `  Created: ${formatTimestamp(job.created_at)}`,
    `  Started: ${formatTimestamp(job.started_at)}`,
    `  Finished: ${formatTimestamp(job.finished_at)}`,
  ]

  if (job.error) {
    lines.push("")
    lines.push("═══ Error ═══")
    lines.push(`  ${job.error}`)
  }

  return lines.join("\n")
}

export function formatPromptLearningDetails(snapshot: Snapshot, job: JobSummary): string {
  const lastEvent = snapshot.events.length
    ? snapshot.events
        .filter(
          (event): event is JobEvent & { timestamp: string } =>
            typeof event.timestamp === "string" && event.timestamp.length > 0,
        )
        .reduce((latest, event) => {
          if (!latest) return event
          return event.timestamp > latest.timestamp ? event : latest
        }, null as (JobEvent & { timestamp: string }) | null)
    : null

  const lastEventTs = formatTimestamp(lastEvent?.timestamp)
  const totalTokens = job.total_tokens ?? calculateTotalTokensFromEvents(snapshot.events)
  const tokensDisplay = totalTokens > 0 ? totalTokens.toLocaleString() : "-"
  const costDisplay = job.total_cost_usd != null ? `$${job.total_cost_usd.toFixed(4)}` : "-"
  const envName = extractEnvName(job)

  const lines = [
    `Job: ${job.job_id}`,
    `Status: ${job.status}`,
    `Type: ${job.job_type || "prompt-learning"}`,
    `Env: ${envName || "-"}`,
    `Started: ${formatTimestamp(job.started_at)}`,
    `Finished: ${formatTimestamp(job.finished_at)}`,
    `Last Event: ${lastEventTs}`,
    "",
    "═══ Progress ═══",
    `  Best Reward: ${job.best_reward != null ? job.best_reward.toFixed(4) : "-"}`,
    `  Events: ${snapshot.events.length}`,
    `  Tokens: ${tokensDisplay}`,
    `  Cost: ${costDisplay}`,
  ]

  if (job.error) {
    lines.push("")
    lines.push("═══ Error ═══")
    lines.push(`  ${job.error}`)
  }
  if (snapshot.artifacts.length) {
    lines.push("")
    lines.push(`Artifacts: ${snapshot.artifacts.length}`)
  }

  return lines.join("\n")
}

export function calculateTotalTokensFromEvents(events: JobEvent[]): number {
  let total = 0
  for (const event of events) {
    const data: any = event.data as any
    if (!data) continue
    // Sum up token fields from various event types
    if (typeof data.prompt_tokens === "number") total += data.prompt_tokens
    if (typeof data.completion_tokens === "number") total += data.completion_tokens
    if (typeof data.reasoning_tokens === "number") total += data.reasoning_tokens
    // Also check for total_tokens directly
    if (typeof data.total_tokens === "number") total += data.total_tokens
  }
  return total
}
