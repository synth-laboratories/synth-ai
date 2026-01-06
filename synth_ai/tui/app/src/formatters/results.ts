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

export function extractBestPrompt(snapshotPayload: Record<string, any>): Record<string, any> | null {
  if (!snapshotPayload) return null
  return (
    (isRecord(snapshotPayload.best_prompt) && snapshotPayload.best_prompt) ||
    (isRecord(snapshotPayload.best_prompt_template) && snapshotPayload.best_prompt_template) ||
    (isRecord(snapshotPayload.best_prompt_pattern) && snapshotPayload.best_prompt_pattern) ||
    null
  )
}

export function extractBestPromptText(snapshotPayload: Record<string, any>): string | null {
  if (!snapshotPayload) return null
  const bestPromptMessages = snapshotPayload.best_prompt_messages
  if (Array.isArray(bestPromptMessages) && bestPromptMessages.length > 0) {
    return bestPromptMessages
      .map((msg: any) => {
        const role = msg?.role || "unknown"
        const content = msg?.content || ""
        return `[${role}] ${content}`
      })
      .join("\n")
  }
  const rendered = snapshotPayload.best_prompt_text || snapshotPayload.rendered_prompt
  if (typeof rendered === "string" && rendered.trim()) return rendered
  return null
}

export function extractPromptSections(bestPrompt: Record<string, any>): Array<Record<string, any>> {
  const sections = bestPrompt.sections || bestPrompt.prompt_sections || []
  return Array.isArray(sections) ? sections : []
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
    const bestPrompt = extractBestPrompt(snapshot.bestSnapshot as any)
    const bestPromptText = extractBestPromptText(snapshot.bestSnapshot as any)
    if (bestPrompt) {
      const promptId = bestPrompt.id || bestPrompt.template_id
      const promptName = bestPrompt.name
      const promptLabel = [promptName, promptId].filter(Boolean).join(" ")
      if (promptLabel) lines.push(`Best prompt: ${promptLabel}`)
      const sections = extractPromptSections(bestPrompt)
      if (sections.length > 0) {
        const summary = sections.slice(0, 3).map((section) => {
          const role = section.role || "stage"
          const name = section.name || section.id || ""
          return name ? `${role}:${name}` : role
        })
        const suffix = sections.length > 3 ? " …" : ""
        lines.push(`Stages: ${summary.join(", ")}${suffix}`)
      }
    }
    if (bestPromptText) {
      lines.push(`Best prompt text: ${truncate(bestPromptText, 90)}`)
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
    const keyOrder = ["mean_score", "accuracy", "pass_rate", "completed", "failed", "total"]
    const shown = new Set<string>()

    for (const key of keyOrder) {
      if (summary[key] != null) {
        const val = summary[key]
        if (key === "accuracy" || key === "pass_rate") {
          lines.push(`  ${key}: ${(val * 100).toFixed(1)}%`)
        } else {
          lines.push(`  ${key}: ${formatValue(val)}`)
        }
        shown.add(key)
      }
    }
    // Show remaining keys
    for (const [key, value] of Object.entries(summary)) {
      if (shown.has(key)) continue
      if (typeof value === "object") continue
      lines.push(`  ${key}: ${formatValue(value)}`)
    }
    lines.push("")
  }

  if (summary.mean_score == null && rows.length > 0) {
    const scores = rows
      .map((row) => row.outcome_reward ?? row.score ?? row.reward_mean ?? row.events_score)
      .filter((val) => typeof val === "number" && Number.isFinite(val)) as number[]
    if (scores.length > 0) {
      const mean = scores.reduce((acc, val) => acc + val, 0) / scores.length
      if (lines.length === 0 || lines[0] !== "═══ Summary ═══") {
        lines.unshift("═══ Summary ═══")
      }
      lines.splice(1, 0, `  mean_score: ${formatValue(mean)}`)
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
      const score = num(row.score ?? row.reward_mean ?? row.outcome_reward ?? row.passed)
      const passed = row.passed != null ? (row.passed ? "✓" : "✗") : ""
      const status = row.status || ""
      const scoreStr = score != null ? score.toFixed(3) : "-"

      if (passed) {
        lines.push(`  ${passed} ${taskId}: ${scoreStr}`)
      } else if (status) {
        lines.push(`  [${status}] ${taskId}: ${scoreStr}`)
      } else {
        lines.push(`  ${taskId}: ${scoreStr}`)
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
  lines.push(`Best Score: ${job.best_score ?? "-"}`)
  lines.push(`Best Snapshot ID: ${snapshot.bestSnapshotId || "-"}`)
  lines.push("")

  if (snapshot.bestSnapshot) {
    // GEPA stores best_prompt and best_prompt_messages directly in the snapshot
    const bestPrompt = (snapshot.bestSnapshot as any).best_prompt
    const bestPromptMessages = (snapshot.bestSnapshot as any).best_prompt_messages

    if (bestPrompt && typeof bestPrompt === "object") {
      const promptId = (bestPrompt as any).id || (bestPrompt as any).template_id
      const promptName = (bestPrompt as any).name
      if (promptName) lines.push(`Prompt Name: ${promptName}`)
      if (promptId) lines.push(`Prompt ID: ${promptId}`)
      lines.push("")

      // Extract sections from best_prompt (each section = a stage)
      const sections = (bestPrompt as any).sections || (bestPrompt as any).prompt_sections || []
      if (Array.isArray(sections) && sections.length > 0) {
        lines.push(
          `=== PROMPT TEMPLATE (${sections.length} stage${sections.length > 1 ? "s" : ""}) ===`,
        )
        lines.push("")
        for (let i = 0; i < sections.length; i++) {
          const section = sections[i]
          const role = section.role || "stage"
          const name = section.name || section.id || ""
          const content = section.content || ""
          const order = section.order !== undefined ? section.order : i
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

    // Show rendered messages (best_prompt_messages)
    if (Array.isArray(bestPromptMessages) && bestPromptMessages.length > 0) {
      lines.push(
        `=== RENDERED MESSAGES (${bestPromptMessages.length} message${bestPromptMessages.length > 1 ? "s" : ""}) ===`,
      )
      lines.push("")
      for (let i = 0; i < bestPromptMessages.length; i++) {
        const msg = bestPromptMessages[i]
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
    if (!bestPrompt && !bestPromptMessages) {
      const legacyPrompt = extractBestPrompt(snapshot.bestSnapshot as any)
      const legacyText = extractBestPromptText(snapshot.bestSnapshot as any)

      if (legacyPrompt) {
        const sections = extractPromptSections(legacyPrompt)
        if (sections.length > 0) {
          lines.push(
            `=== PROMPT SECTIONS (${sections.length} stage${sections.length > 1 ? "s" : ""}) ===`,
          )
          lines.push("")
          for (let i = 0; i < sections.length; i++) {
            const section = sections[i]
            const role = section.role || "stage"
            const name = section.name || section.id || ""
            const content = section.content || ""
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
        lines.push("=== RENDERED PROMPT ===")
        lines.push("")
        lines.push(legacyText)
      }

      // Last resort: show raw data
      if (!legacyPrompt && !legacyText) {
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


