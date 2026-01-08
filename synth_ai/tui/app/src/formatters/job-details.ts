/**
 * Job detail panel formatting.
 */
import type { Snapshot } from "../types"
import type { JobEvent, JobSummary } from "../tui_data"
import { isEvalJob, num } from "../tui_data"
import { extractEnvName } from "../utils/job"
import { formatTimestamp } from "./time"

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

  // Calculate aggregates from rows
  const completedRows = rows.filter((r) => !r.error)
  const failedRows = rows.filter((r) => r.error)
  const rewards = rows
    .map((row) => num(row.score ?? row.outcome_reward ?? row.reward_mean))
    .filter((val) => typeof val === "number" && Number.isFinite(val)) as number[]
  const meanReward = rewards.length > 0
    ? rewards.reduce((sum, val) => sum + val, 0) / rewards.length
    : null
  const totalTokens = rows.reduce((sum, r) => sum + (r.tokens ?? 0), 0)
  const totalCost = rows.reduce((sum, r) => sum + (r.cost_usd ?? 0), 0)
  const avgLatency = rows.length > 0
    ? rows.reduce((sum, r) => sum + (r.latency_ms ?? 0), 0) / rows.length
    : 0

  // Progress tracking
  const totalSeeds = summary.total ?? (summary.seeds?.length ?? rows.length)
  const completedSeeds = summary.completed ?? completedRows.length

  // Status indicator
  const statusEmoji = job.status === "completed" ? "✓" :
                      job.status === "failed" ? "✗" :
                      job.status === "running" ? "◉" : "○"

  const lines = [
    `${statusEmoji} Job: ${job.job_id}`,
    `Status: ${job.status}`,
  ]

  // Progress bar for running jobs
  if (totalSeeds > 0) {
    const pct = Math.min(100, Math.round((completedSeeds / totalSeeds) * 100))
    const barWidth = 20
    const filled = Math.round((pct / 100) * barWidth)
    const progressBar = "█".repeat(filled) + "░".repeat(barWidth - filled)
    lines.push(`Progress: [${progressBar}] ${completedSeeds}/${totalSeeds} (${pct}%)`)
  }

  lines.push("")
  lines.push("═══ Metrics ═══")

  // Mean reward - the key metric
  const displayMeanReward = summary.mean_reward ?? meanReward
  if (displayMeanReward != null) {
    const rewardPct = (displayMeanReward * 100).toFixed(1)
    lines.push(`  Mean Reward: ${displayMeanReward.toFixed(4)} (${rewardPct}%)`)
  } else {
    lines.push(`  Mean Reward: -`)
  }

  // Success/fail counts
  if (failedRows.length > 0) {
    lines.push(`  Completed: ${completedRows.length}  Failed: ${failedRows.length}`)
  } else if (completedRows.length > 0) {
    lines.push(`  Completed: ${completedRows.length}`)
  }

  lines.push("")
  lines.push("═══ Resources ═══")
  lines.push(`  Tokens: ${totalTokens > 0 ? totalTokens.toLocaleString() : "-"}`)
  lines.push(`  Cost: ${totalCost > 0 ? "$" + totalCost.toFixed(4) : "-"}`)
  lines.push(`  Avg Latency: ${avgLatency > 0 ? (avgLatency / 1000).toFixed(2) + "s" : "-"}`)

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
    `Type: ${job.training_type || "learning"}`,
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
    `Type: ${job.training_type || "prompt-learning"}`,
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

