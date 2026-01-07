/**
 * Results panel formatting (best snapshot + eval results + expanded view).
 */
import type { Snapshot } from "../types"
import { num } from "../tui_data"
import { truncate } from "../utils/truncate"
import { formatValue } from "./time"

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === "object" && !Array.isArray(value)
}

export function extractBestCandidate(
  snapshotPayload: Record<string, any>,
): Record<string, any> | null {
  if (!snapshotPayload) return null
  return (
    (isRecord(snapshotPayload.best_candidate) && snapshotPayload.best_candidate) ||
    (isRecord(snapshotPayload.best_candidate_template) && snapshotPayload.best_candidate_template) ||
    (isRecord(snapshotPayload.best_candidate_pattern) && snapshotPayload.best_candidate_pattern) ||
    (isRecord(snapshotPayload.best_prompt) && snapshotPayload.best_prompt) ||
    (isRecord(snapshotPayload.best_prompt_template) && snapshotPayload.best_prompt_template) ||
    (isRecord(snapshotPayload.best_prompt_pattern) && snapshotPayload.best_prompt_pattern) ||
    null
  )
}

export function extractBestCandidateText(snapshotPayload: Record<string, any>): string | null {
  if (!snapshotPayload) return null
  const bestCandidateMessages =
    snapshotPayload.best_candidate_messages ?? snapshotPayload.best_prompt_messages
  if (Array.isArray(bestCandidateMessages) && bestCandidateMessages.length > 0) {
    return bestCandidateMessages
      .map((msg: any) => {
        const role = msg?.role || "unknown"
        const content = msg?.content || ""
        return `[${role}] ${content}`
      })
      .join("\n")
  }
  const rendered =
    snapshotPayload.best_candidate_text ||
    snapshotPayload.best_prompt_text ||
    snapshotPayload.rendered_candidate ||
    snapshotPayload.rendered_prompt
  if (typeof rendered === "string" && rendered.trim()) return rendered
  return null
}

export function extractCandidateStages(bestCandidate: Record<string, any>): Array<Record<string, any>> {
  if (!bestCandidate) return []
  const stages =
    bestCandidate.stages || bestCandidate.sections || bestCandidate.prompt_sections || []
  if (Array.isArray(stages)) return stages
  if (isRecord(stages)) {
    return Object.entries(stages).map(([id, value]) => {
      if (isRecord(value)) return { id, ...value }
      return { id, content: value }
    })
  }
  return []
}

export function formatResults(snapshot: Snapshot): string {
  const job: any = snapshot.selectedJob
  if (!job) return "Results: -"
  if (job.job_source === "eval" || job.training_type === "eval") {
    return formatEvalResults(snapshot)
  }

  const lines: string[] = []
  const bestId = snapshot.bestSnapshotId || "-"
  if (bestId === "-") {
    lines.push("Best snapshot: -")
  } else if (snapshot.bestSnapshot) {
    lines.push(`Best snapshot: ${bestId}`)
  } else {
    lines.push(`Best snapshot: ${bestId} (press p to load)`)
  }

  if (snapshot.bestSnapshot) {
    const bestCandidate = extractBestCandidate(snapshot.bestSnapshot as any)
    const bestCandidateText = extractBestCandidateText(snapshot.bestSnapshot as any)
    if (bestCandidate) {
      const candidateId = bestCandidate.id || bestCandidate.template_id
      const candidateName = bestCandidate.name
      const candidateLabel = [candidateName, candidateId].filter(Boolean).join(" ")
      if (candidateLabel) lines.push(`Best candidate: ${candidateLabel}`)
      const stages = extractCandidateStages(bestCandidate)
      if (stages.length > 0) {
        const summary = stages.slice(0, 3).map((stage) => {
          const role = stage.role || "stage"
          const name = stage.name || stage.id || ""
          return name ? `${role}:${name}` : role
        })
        const suffix = stages.length > 3 ? " …" : ""
        lines.push(`Stages: ${summary.join(", ")}${suffix}`)
      }
    }
    if (bestCandidateText) {
      lines.push(`Best candidate text: ${truncate(bestCandidateText, 90)}`)
    }
  }

  return ["Results:", ...lines].join("\n")
}

export function formatEvalResults(snapshot: Snapshot): string {
  const summary: any = snapshot.evalSummary ?? {}
  const rows: any[] = snapshot.evalResultRows ?? []
  const lines: string[] = []

  // Show overall summary if available
  if (Object.keys(summary).length > 0) {
    lines.push("═══ Summary ═══")
    const keyOrder = ["mean_reward", "reward", "pass_rate", "completed", "failed", "total"]
    const shown = new Set<string>()

    for (const key of keyOrder) {
      let val = summary[key]
      if (key === "reward") {
        val = summary.reward ?? summary.objectives?.reward ?? summary.accuracy
        if (val != null) {
          shown.add("accuracy")
        }
      } else if (key === "mean_reward") {
        val = summary.mean_reward ?? summary.mean_score
        if (val != null) {
          shown.add("mean_score")
        }
      }
      if (val == null) continue
      if (key === "reward" || key === "pass_rate") {
        lines.push(`  ${key}: ${(val * 100).toFixed(1)}%`)
      } else {
        lines.push(`  ${key}: ${formatValue(val)}`)
      }
      shown.add(key)
    }
    // Show remaining keys
    for (const [key, value] of Object.entries(summary)) {
      if (shown.has(key)) continue
      if (typeof value === "object") continue
      lines.push(`  ${key}: ${formatValue(value)}`)
    }
    lines.push("")
  }

  if (summary.mean_reward == null && summary.mean_score == null && rows.length > 0) {
    const rewards = rows
      .map((row) => row.reward ?? row.outcome_reward ?? row.reward_mean ?? row.events_score)
      .filter((val) => typeof val === "number" && Number.isFinite(val)) as number[]
    if (rewards.length > 0) {
      const mean = rewards.reduce((acc, val) => acc + val, 0) / rewards.length
      if (lines.length === 0 || lines[0] !== "═══ Summary ═══") {
        lines.unshift("═══ Summary ═══")
      }
      lines.splice(1, 0, `  mean_reward: ${formatValue(mean)}`)
      lines.push("")
    }
  }

  // Show per-task results
  if (rows.length > 0) {
    lines.push("═══ Results by Task ═══")
    const limit = 15
    const displayRows = rows.slice(0, limit)

    for (const row of displayRows) {
      const taskId = row.task_id || row.id || row.name || "?"
      const reward = num(row.reward ?? row.outcome_reward ?? row.reward_mean ?? row.passed)
      const passed = row.passed != null ? (row.passed ? "✓" : "✗") : ""
      const status = row.status || ""
      const rewardStr = reward != null ? reward.toFixed(3) : "-"

      if (passed) {
        lines.push(`  ${passed} ${taskId}: ${rewardStr}`)
      } else if (status) {
        lines.push(`  [${status}] ${taskId}: ${rewardStr}`)
      } else {
        lines.push(`  ${taskId}: ${rewardStr}`)
      }
    }

    if (rows.length > limit) {
      lines.push(`  ... +${rows.length - limit} more tasks`)
    }
  } else if (Object.keys(summary).length === 0) {
    lines.push("No eval results yet.")
    lines.push("")
    lines.push("Results will appear after the eval completes.")
  }

  return lines.length > 0 ? lines.join("\n") : "Results: -"
}

export function formatResultsExpanded(snapshot: Snapshot): string | null {
  const job: any = snapshot.selectedJob
  if (!job) return null
  if (!snapshot.bestSnapshot && !snapshot.bestSnapshotId) {
    return "No best snapshot available yet.\n\nPress 'p' to try loading the best snapshot."
  }
  const lines: string[] = []
  lines.push(`Job: ${job.job_id}`)
  lines.push(`Status: ${job.status}`)
  lines.push(`Best Reward: ${job.best_reward ?? "-"}`)
  lines.push(`Best Snapshot ID: ${snapshot.bestSnapshotId || "-"}`)
  lines.push("")

  if (snapshot.bestSnapshot) {
    // GEPA stores best_candidate and best_candidate_messages directly in the snapshot
    const bestCandidate = extractBestCandidate(snapshot.bestSnapshot as any)
    const bestCandidateMessages =
      (snapshot.bestSnapshot as any).best_candidate_messages ??
      (snapshot.bestSnapshot as any).best_prompt_messages

    if (bestCandidate && typeof bestCandidate === "object") {
      const candidateId = (bestCandidate as any).id || (bestCandidate as any).template_id
      const candidateName = (bestCandidate as any).name
      if (candidateName) lines.push(`Candidate Name: ${candidateName}`)
      if (candidateId) lines.push(`Candidate ID: ${candidateId}`)
      lines.push("")

      // Extract stages from best_candidate
      const stages = extractCandidateStages(bestCandidate as any)
      if (Array.isArray(stages) && stages.length > 0) {
        lines.push(
          `=== CANDIDATE STAGES (${stages.length} stage${stages.length > 1 ? "s" : ""}) ===`,
        )
        lines.push("")
        for (let i = 0; i < stages.length; i++) {
          const stage = stages[i]
          const role = stage.role || "stage"
          const name = stage.name || stage.id || ""
          const content = stage.content || ""
          const order = stage.order !== undefined ? stage.order : i
          lines.push(`┌─ Stage ${order + 1}: ${role}${name ? ` (${name})` : ""} ─┐`)
          lines.push("")
          if (content) {
            lines.push(content)
          } else {
            lines.push("(empty)")
          }
          lines.push("")
          lines.push(`└${"─".repeat(40)}┘`)
          lines.push("")
        }
      }
    }

    // Show rendered messages (best_candidate_messages)
    if (Array.isArray(bestCandidateMessages) && bestCandidateMessages.length > 0) {
      lines.push(
        `=== RENDERED CANDIDATE MESSAGES (${bestCandidateMessages.length} message${bestCandidateMessages.length > 1 ? "s" : ""}) ===`,
      )
      lines.push("")
      for (let i = 0; i < bestCandidateMessages.length; i++) {
        const msg = bestCandidateMessages[i]
        const role = msg.role || "unknown"
        const content = msg.content || ""
        lines.push(`┌─ Message ${i + 1}: [${role}] ─┐`)
        lines.push("")
        lines.push(content)
        lines.push("")
        lines.push(`└${"─".repeat(40)}┘`)
        lines.push("")
      }
    }

    // Fallback: check for legacy extractors if nothing found
    if (!bestCandidate && !bestCandidateMessages) {
      const legacyCandidate = extractBestCandidate(snapshot.bestSnapshot as any)
      const legacyText = extractBestCandidateText(snapshot.bestSnapshot as any)

      if (legacyCandidate) {
        const stages = extractCandidateStages(legacyCandidate)
        if (stages.length > 0) {
          lines.push(
            `=== CANDIDATE STAGES (${stages.length} stage${stages.length > 1 ? "s" : ""}) ===`,
          )
          lines.push("")
          for (let i = 0; i < stages.length; i++) {
            const stage = stages[i]
            const role = stage.role || "stage"
            const name = stage.name || stage.id || ""
            const content = stage.content || ""
            lines.push(`┌─ Stage ${i + 1}: ${role}${name ? ` (${name})` : ""} ─┐`)
            lines.push("")
            if (content) {
              lines.push(content)
            }
            lines.push("")
            lines.push(`└${"─".repeat(40)}┘`)
            lines.push("")
          }
        }
      }

      if (legacyText) {
        lines.push("=== RENDERED CANDIDATE ===")
        lines.push("")
        lines.push(legacyText)
      }

      // Last resort: show raw data
      if (!legacyCandidate && !legacyText) {
        lines.push("=== RAW SNAPSHOT DATA ===")
        lines.push("")
        try {
          lines.push(JSON.stringify(snapshot.bestSnapshot, null, 2))
        } catch {
          lines.push(String(snapshot.bestSnapshot))
        }
      }
    }
  } else {
    lines.push("Best snapshot data not loaded. Press 'p' to load.")
  }

  return lines.join("\n")
}
