/**
 * Results panel formatting (best snapshot + eval results + expanded view).
 */
import type { Snapshot } from "../types"
import { truncate } from "../utils/truncate"

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

  if (rows.length === 0 && Object.keys(summary).length === 0) {
    lines.push("═══ Eval Results ═══")
    lines.push("")
    lines.push("Waiting for results...")
    lines.push("Results will stream in as seeds complete.")
    return lines.join("\n")
  }

  // Show per-seed results table
  if (rows.length > 0) {
    lines.push("═══ Per-Seed Results ═══")
    lines.push("  Seed   Reward    Latency   Tokens   Cost")
    lines.push("  ────   ──────    ───────   ──────   ────")

    const limit = 12
    const sortedRows = [...rows].sort((a, b) => (a.seed ?? 0) - (b.seed ?? 0))
    const displayRows = sortedRows.slice(0, limit)

    for (const row of displayRows) {
      const seed = String(row.seed ?? "?").padStart(4)
      const score = row.score ?? row.outcome_reward ?? row.reward_mean
      const rewardStr = typeof score === "number" ? score.toFixed(3).padStart(6) : "     -"
      const latencyMs = row.latency_ms
      const latencyStr = typeof latencyMs === "number"
        ? (latencyMs < 1000 ? `${Math.round(latencyMs)}ms` : `${(latencyMs/1000).toFixed(1)}s`).padStart(7)
        : "      -"
      const tokens = row.tokens
      const tokensStr = typeof tokens === "number" ? String(tokens).padStart(6) : "     -"
      const cost = row.cost_usd
      const costStr = typeof cost === "number" ? `$${cost.toFixed(3)}`.padStart(7) : "      -"

      // Status indicator
      const statusIcon = row.error ? "✗" : (score != null ? "✓" : "◦")

      lines.push(`  ${statusIcon}${seed}   ${rewardStr}   ${latencyStr}   ${tokensStr}  ${costStr}`)
    }

    if (rows.length > limit) {
      lines.push(`  ... +${rows.length - limit} more seeds`)
    }

    // Show any errors
    const errorRows = rows.filter((r) => r.error)
    if (errorRows.length > 0 && errorRows.length <= 3) {
      lines.push("")
      lines.push("═══ Errors ═══")
      for (const row of errorRows.slice(0, 3)) {
        const errMsg = String(row.error || "unknown").slice(0, 50)
        lines.push(`  Seed ${row.seed}: ${errMsg}`)
      }
    } else if (errorRows.length > 3) {
      lines.push("")
      lines.push(`  ${errorRows.length} seeds failed (see events for details)`)
    }
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
