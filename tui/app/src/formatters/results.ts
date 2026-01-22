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

function asArray(value: unknown): Array<Record<string, any>> {
  if (!Array.isArray(value)) return []
  return value.filter((item) => isRecord(item)) as Array<Record<string, any>>
}

function mean(values: Array<unknown>): number | null {
  if (!Array.isArray(values)) return null
  const numeric = values.map((v) => num(v)).filter((v): v is number => typeof v === "number")
  if (numeric.length === 0) return null
  const total = numeric.reduce((acc, val) => acc + val, 0)
  return total / numeric.length
}

function formatReward(value: number | null): string {
  if (value == null) return "-"
  return value.toFixed(3)
}

type CandidateView = {
  id: string
  label: string
  reward: number | null
  meanReward: number | null
  isPareto: boolean
  source: "optimized" | "attempted" | "live"
  payload: Record<string, any>
}

function extractCandidateGroups(snapshot: Snapshot): {
  attempted: Array<Record<string, any>>
  optimized: Array<Record<string, any>>
} {
  // Prefer API candidates if loaded
  if (snapshot.apiCandidatesLoaded && snapshot.apiCandidates.length > 0) {
    const apiCandidates = snapshot.apiCandidates.map((c: any) => ({
      candidate_id: c.candidate_id,
      id: c.candidate_id,
      name: c.content?.name || c.content?.candidate_name,
      reward: c.reward,
      is_pareto: c.is_pareto,
      isPareto: c.is_pareto,
      generation: c.generation,
      parent_id: c.parent_id,
      status: c.status,
      stages: c.content?.stages,
      prompt_text: c.content?.prompt_text,
      messages: c.content?.messages,
      score: {
        reward: c.reward,
        instance_scores: c.seed_rewards?.map((sr: any) => sr.reward),
      },
      instance_scores: c.seed_rewards?.map((sr: any) => sr.reward),
      graph_text_export: c.graph_text_export,
      _content: c.content,
    }))
    const optimized = apiCandidates.filter((c) => c.is_pareto)
    return {
      attempted: apiCandidates,
      optimized,
    }
  }

  // Fallback to event-based candidates
  const job: any = snapshot.selectedJob
  const metadata = isRecord(job?.metadata) ? job.metadata : {}
  const attemptedPrimary = asArray(metadata?.attempted_candidates)
  const attemptedFallback = asArray((snapshot.bestSnapshot as any)?.attempted_candidates)
  const optimizedPrimary = asArray(metadata?.optimized_candidates)
  const optimizedFallback = asArray((snapshot.bestSnapshot as any)?.optimized_candidates)
  return {
    attempted: attemptedPrimary.length > 0 ? attemptedPrimary : attemptedFallback,
    optimized: optimizedPrimary.length > 0 ? optimizedPrimary : optimizedFallback,
  }
}

function extractCandidateId(candidate: Record<string, any>, fallback: string): string {
  return (
    candidate.candidate_id ||
    candidate.version_id ||
    candidate.id ||
    candidate.template_id ||
    fallback
  )
}

function extractCandidateLabel(candidate: Record<string, any>, fallback: string): string {
  const name = candidate.name || candidate.candidate_name
  const candidateId = extractCandidateId(candidate, fallback)
  return [name, candidateId].filter(Boolean).join(" ") || candidateId
}

function extractCandidateReward(candidate: Record<string, any>): number | null {
  const score = isRecord(candidate.score) ? candidate.score : null
  const scoreReward = num(score?.reward ?? score?.accuracy ?? score?.objectives?.reward)
  if (scoreReward != null) return scoreReward

  const direct = num(
    candidate.reward ??
      candidate.accuracy ??
      candidate.train_accuracy ??
      candidate.val_accuracy ??
      candidate.full_score ??
      candidate.minibatch_score,
  )
  if (direct != null) return direct

  const instanceScores =
    (Array.isArray(candidate.instance_scores) && candidate.instance_scores) ||
    (Array.isArray(candidate.instance_rewards) && candidate.instance_rewards) ||
    (Array.isArray(score?.instance_scores) && score?.instance_scores) ||
    (Array.isArray(score?.instance_rewards) && score?.instance_rewards) ||
    []
  return mean(instanceScores)
}

function extractCandidateMeanReward(candidate: Record<string, any>): number | null {
  const instanceScores =
    (Array.isArray(candidate.instance_scores) && candidate.instance_scores) ||
    (Array.isArray(candidate.instance_rewards) && candidate.instance_rewards) ||
    (Array.isArray(candidate.score?.instance_scores) && candidate.score?.instance_scores) ||
    (Array.isArray(candidate.score?.instance_rewards) && candidate.score?.instance_rewards) ||
    []
  const computed = mean(instanceScores)
  return computed ?? extractCandidateReward(candidate)
}

function extractPayloadMeanReward(payload: Record<string, any>): number | null {
  const instanceScores =
    (Array.isArray(payload.instance_scores) && payload.instance_scores) ||
    (Array.isArray(payload.instance_rewards) && payload.instance_rewards) ||
    (Array.isArray(payload.score?.instance_scores) && payload.score?.instance_scores) ||
    (Array.isArray(payload.score?.instance_rewards) && payload.score?.instance_rewards) ||
    []
  const computed = mean(instanceScores)
  if (computed != null) return computed
  const score = isRecord(payload.score) ? payload.score : null
  return (
    num(payload.reward ?? payload.accuracy ?? payload.full_score ?? payload.minibatch_score ?? score?.reward) ??
    null
  )
}

function collectCandidateViews(snapshot: Snapshot): CandidateView[] {
  const { attempted, optimized } = extractCandidateGroups(snapshot)
  const byId = new Map<string, CandidateView>()
  const ordered: CandidateView[] = []
  const seen = new Set<string>()

  const upsert = (
    candidate: Record<string, any>,
    fallbackId: string,
    source: CandidateView["source"],
    isPareto: boolean,
  ): void => {
    const id = String(extractCandidateId(candidate, fallbackId))
    const label = extractCandidateLabel(candidate, id)
    const reward = extractCandidateReward(candidate)
    const meanReward = extractCandidateMeanReward(candidate)
    const payload = candidate
    const existing = byId.get(id)
    const paretoFlag = isPareto || candidate.is_pareto === true || candidate.isPareto === true

    if (existing) {
      existing.label = label || existing.label
      if (reward != null) existing.reward = reward
      if (meanReward != null) existing.meanReward = meanReward
      existing.isPareto = existing.isPareto || paretoFlag
      existing.payload = payload
      existing.source = existing.source === "optimized" ? existing.source : source
      return
    }

    const view: CandidateView = {
      id,
      label: label || id,
      reward,
      meanReward,
      isPareto: paretoFlag,
      source,
      payload,
    }
    byId.set(id, view)
  }

  optimized.forEach((candidate, idx) => {
    upsert(candidate, `pareto_${idx + 1}`, "optimized", true)
  })
  attempted.forEach((candidate, idx) => {
    upsert(candidate, `cand_${idx + 1}`, "attempted", false)
  })

  for (const live of snapshot.allCandidates ?? []) {
    const payload = isRecord(live.payload) ? live.payload : {}
    const id = String(live.id)
    const label =
      payload.candidate_name ||
      payload.name ||
      payload.candidate_id ||
      payload.version_id ||
      id
    const reward = live.reward ?? extractPayloadMeanReward(payload)
    const meanReward = extractPayloadMeanReward(payload)
    const isPareto = payload.is_pareto === true || payload.isPareto === true
    const existing = byId.get(id)
    if (existing) {
      existing.label = existing.label || label
      if (reward != null) existing.reward = reward
      if (meanReward != null) existing.meanReward = meanReward
      existing.isPareto = existing.isPareto || isPareto
      existing.payload = { ...payload, ...existing.payload }
      continue
    }
    byId.set(id, {
      id,
      label,
      reward: reward ?? null,
      meanReward,
      isPareto,
      source: "live",
      payload,
    })
  }

  const pushById = (id: string) => {
    if (seen.has(id)) return
    const view = byId.get(id)
    if (!view) return
    ordered.push(view)
    seen.add(id)
  }

  optimized.forEach((candidate, idx) => {
    const id = String(extractCandidateId(candidate, `pareto_${idx + 1}`))
    pushById(id)
  })
  attempted.forEach((candidate, idx) => {
    const id = String(extractCandidateId(candidate, `cand_${idx + 1}`))
    pushById(id)
  })

  const remaining = Array.from(byId.values()).filter((view) => !seen.has(view.id))
  remaining.sort((a, b) => {
    const rewardA = a.meanReward ?? a.reward ?? -Infinity
    const rewardB = b.meanReward ?? b.reward ?? -Infinity
    if (rewardA !== rewardB) return rewardB - rewardA
    return a.id.localeCompare(b.id)
  })
  for (const view of remaining) {
    ordered.push(view)
  }

  return ordered
}

function formatCandidateContent(payload: Record<string, any>): string[] {
  const lines: string[] = []
  const stages =
    (isRecord(payload.stages) && payload.stages) ||
    (isRecord(payload.object?.stages) && payload.object?.stages) ||
    null

  if (stages) {
    const entries = Object.entries(stages)
    for (const [stageId, stageData] of entries) {
      const stage = isRecord(stageData) ? stageData : { instruction: stageData }
      const instruction = stage.instruction || stage.content || ""
      lines.push(`-- ${stageId} --`)
      if (instruction) {
        lines.push(String(instruction))
      } else {
        lines.push("(empty stage)")
      }
      lines.push("")
    }
    return lines
  }

  const promptText =
    payload.prompt_text ||
    payload.prompt_summary ||
    payload.rendered_prompt ||
    payload.rendered_candidate ||
    payload.text ||
    null
  if (typeof promptText === "string" && promptText.trim()) {
    lines.push(promptText)
    return lines
  }

  const messages = payload.messages || payload.best_candidate_messages || payload.best_prompt_messages
  if (Array.isArray(messages) && messages.length > 0) {
    for (const msg of messages) {
      const role = msg?.role || "unknown"
      const content = msg?.content || ""
      lines.push(`[${role}] ${content}`)
      lines.push("")
    }
    return lines
  }

  return lines
}

export function extractBestCandidate(
  snapshotPayload: Record<string, any>,
): Record<string, any> | null {
  if (!snapshotPayload) return null
  return (
    (isRecord(snapshotPayload.best_candidate) && snapshotPayload.best_candidate) ||
    (isRecord(snapshotPayload.best_candidate_pattern) && snapshotPayload.best_candidate_pattern) ||
    (isRecord(snapshotPayload.best_candidate_template) && snapshotPayload.best_candidate_template) ||
    (isRecord(snapshotPayload.best_prompt) && snapshotPayload.best_prompt) ||
    (isRecord(snapshotPayload.best_prompt_pattern) && snapshotPayload.best_prompt_pattern) ||
    (isRecord(snapshotPayload.best_prompt_template) && snapshotPayload.best_prompt_template) ||
    null
  )
}

export function extractBestCandidateText(snapshotPayload: Record<string, any>): string | null {
  if (!snapshotPayload) return null
  let bestCandidateMessages =
    snapshotPayload.best_candidate_messages ?? snapshotPayload.best_prompt_messages
  if (!bestCandidateMessages) {
    const pattern =
      snapshotPayload.best_candidate_pattern ??
      snapshotPayload.best_prompt_pattern ??
      snapshotPayload.best_candidate ??
      snapshotPayload.best_prompt
    if (isRecord(pattern) && Array.isArray(pattern.messages)) {
      bestCandidateMessages = pattern.messages
    }
  }
  if (Array.isArray(bestCandidateMessages) && bestCandidateMessages.length > 0) {
    return bestCandidateMessages
      .map((msg: any) => {
        const role = msg?.role || "unknown"
        const content = msg?.content || msg?.pattern || ""
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
  let stages =
    bestCandidate.stages || bestCandidate.sections || bestCandidate.prompt_sections || []
  if (!stages && isRecord(bestCandidate.pattern) && Array.isArray(bestCandidate.pattern.messages)) {
    stages = bestCandidate.pattern.messages.map((msg: any) => ({
      role: msg?.role || "system",
      content: msg?.content || msg?.pattern || "",
      name: msg?.name,
    }))
  }
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
  if (job.job_source === "eval" || job.job_type === "eval") {
    return formatEvalResults(snapshot)
  }

  const lines: string[] = []
  const { attempted, optimized } = extractCandidateGroups(snapshot)
  const paretoMean =
    optimized.length > 0
      ? mean(optimized.map((cand) => extractCandidateMeanReward(cand)))
      : null
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
      const candidateId = extractCandidateId(bestCandidate, bestCandidate.id || "-")
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

  if (attempted.length > 0 || optimized.length > 0) {
    const counts = `Candidates: ${attempted.length} | Pareto: ${optimized.length}`
    const paretoSuffix = paretoMean != null ? ` (mean=${formatReward(paretoMean)})` : ""
    lines.push(`${counts}${paretoSuffix}`)
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
        val = summary.mean_reward
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

  if (summary.mean_reward == null && rows.length > 0) {
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
  const lines: string[] = []
  lines.push(`Job: ${job.job_id}`)
  lines.push(`Status: ${job.status}`)
  lines.push(`Best Reward: ${job.best_reward ?? "-"}`)
  lines.push(`Best Snapshot ID: ${snapshot.bestSnapshotId || "-"}`)

  const { attempted, optimized } = extractCandidateGroups(snapshot)
  const paretoMean =
    optimized.length > 0
      ? mean(optimized.map((cand) => extractCandidateMeanReward(cand)))
      : null
  lines.push(`Total Candidates: ${attempted.length}`)
  lines.push(
    `Pareto Frontier: ${optimized.length}${
      paretoMean != null ? ` (mean reward=${formatReward(paretoMean)})` : ""
    }`,
  )
  lines.push("")

  if (optimized.length > 0) {
    lines.push(`=== PARETO FRONTIER (${optimized.length}) ===`)
    optimized.forEach((candidate, idx) => {
      const label = extractCandidateLabel(candidate, `pareto_${idx + 1}`)
      const meanReward = extractCandidateMeanReward(candidate)
      const reward = extractCandidateReward(candidate)
      const rank = candidate.rank != null ? `#${candidate.rank}` : `#${idx + 1}`
      const parts = [`${rank} ${label}`]
      if (meanReward != null) parts.push(`mean=${formatReward(meanReward)}`)
      if (reward != null && reward !== meanReward) parts.push(`reward=${formatReward(reward)}`)
      lines.push(`  ${parts.join(" | ")}`)
    })
    lines.push("")
  }

  if (attempted.length > 0) {
    const paretoIds = new Set(
      optimized.map((candidate, idx) => extractCandidateId(candidate, `pareto_${idx + 1}`)),
    )
    lines.push(`=== ALL CANDIDATES (${attempted.length}) ===`)
    attempted.forEach((candidate, idx) => {
      const fallbackId = `cand_${idx + 1}`
      const candidateId = extractCandidateId(candidate, fallbackId)
      const label = extractCandidateLabel(candidate, candidateId)
      const reward = extractCandidateReward(candidate)
      const meanReward = paretoIds.has(candidateId)
        ? extractCandidateMeanReward(candidate)
        : null
      const parts = [`#${idx + 1} ${label}`]
      if (reward != null) parts.push(`reward=${formatReward(reward)}`)
      if (meanReward != null) parts.push(`pareto_mean=${formatReward(meanReward)}`)
      if (paretoIds.has(candidateId)) parts.push("pareto")
      lines.push(`  ${parts.join(" | ")}`)
    })
    lines.push("")
  } else {
    lines.push("No candidates available yet.")
    lines.push("")
  }

  if (snapshot.bestSnapshot) {
    // GEPA stores best_candidate and best_candidate_messages directly in the snapshot
    const bestCandidate = extractBestCandidate(snapshot.bestSnapshot as any)
    const bestCandidateMessages =
      (snapshot.bestSnapshot as any).best_candidate_messages ??
      (snapshot.bestSnapshot as any).best_prompt_messages

    if (bestCandidate && typeof bestCandidate === "object") {
      const candidateId = extractCandidateId(bestCandidate as any, (bestCandidate as any).id || "-")
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

export function formatCandidatesModal(
  snapshot: Snapshot,
  selectedIndex: number,
): { raw: string; selectedIndex: number; total: number } | null {
  const job: any = snapshot.selectedJob
  if (!job) return null

  const candidates = collectCandidateViews(snapshot)
  if (candidates.length === 0) {
    return { raw: "No candidates available yet.", selectedIndex: 0, total: 0 }
  }

  const clampedIndex = Math.max(0, Math.min(selectedIndex, candidates.length - 1))
  const selected = candidates[clampedIndex]
  const paretoCount = candidates.filter((cand) => cand.isPareto).length
  const paretoMean =
    paretoCount > 0
      ? mean(
          candidates
            .filter((cand) => cand.isPareto)
            .map((cand) => cand.meanReward ?? cand.reward),
        )
      : null

  const lines: string[] = []
  lines.push(`Job: ${job.job_id}`)
  lines.push(`Status: ${job.status}`)
  lines.push(
    `Candidates: ${candidates.length} | Pareto: ${paretoCount}${
      paretoMean != null ? ` (mean=${formatReward(paretoMean)})` : ""
    }`,
  )
  lines.push("")
  lines.push("=== CANDIDATES ===")
  candidates.forEach((candidate, idx) => {
    const cursor = idx === clampedIndex ? ">" : " "
    const reward = candidate.reward != null ? `reward=${formatReward(candidate.reward)}` : null
    const meanReward =
      candidate.meanReward != null ? `mean=${formatReward(candidate.meanReward)}` : null
    const tags = candidate.isPareto ? "pareto" : null
    const parts = [meanReward, reward, tags].filter(Boolean)
    const label = truncate(candidate.label, 48)
    lines.push(` ${cursor} #${idx + 1} ${label}${parts.length ? ` | ${parts.join(" | ")}` : ""}`)
  })

  lines.push("")
  lines.push(`=== SELECTED CANDIDATE (${clampedIndex + 1}/${candidates.length}) ===`)
  lines.push(`ID: ${selected.id}`)
  if (selected.reward != null) lines.push(`Reward: ${formatReward(selected.reward)}`)
  if (selected.meanReward != null) lines.push(`Mean Reward: ${formatReward(selected.meanReward)}`)
  lines.push(`Pareto: ${selected.isPareto ? "yes" : "no"}`)
  lines.push(`Source: ${selected.source}`)
  lines.push("")

  const contentLines = formatCandidateContent(selected.payload)
  if (contentLines.length > 0) {
    lines.push("=== CONTENT ===")
    lines.push("")
    lines.push(...contentLines)
  } else {
    lines.push("No candidate content available yet.")
  }

  return { raw: lines.join("\n"), selectedIndex: clampedIndex, total: candidates.length }
}
