import type { AppData } from "../types"
import { num } from "../tui_data"

export const REWARD_MAX = 1.0

export type GraphEvolveCandidate = {
  id: string
  reward: number
  generation: number
  payload: Record<string, any>
}

export type GenerationGroup = {
  generation: number
  candidates: GraphEvolveCandidate[]
}

export type GenerationSummary = {
  generation: number
  reward: number
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

export function extractGraphEvolveCandidates(data: AppData): GraphEvolveCandidate[] {
  const out: GraphEvolveCandidate[] = []
  for (const candidate of data.allCandidates ?? []) {
    const payload = candidate.payload && typeof candidate.payload === "object" ? candidate.payload : {}
    const reward = num(candidate.reward ?? payload.reward ?? payload.score) ?? null
    const generation = num(payload.generation)
    if (reward == null || generation == null) continue
    out.push({
      id: candidate.id,
      reward,
      generation,
      payload,
    })
  }
  return out
}

export function groupCandidatesByGeneration(
  candidates: GraphEvolveCandidate[],
): GenerationGroup[] {
  const byGen = new Map<number, GraphEvolveCandidate[]>()
  for (const candidate of candidates) {
    const list = byGen.get(candidate.generation) ?? []
    list.push(candidate)
    byGen.set(candidate.generation, list)
  }
  const generations = Array.from(byGen.keys()).sort((a, b) => b - a)
  return generations.map((generation) => ({
    generation,
    candidates: byGen.get(generation) ?? [],
  }))
}

export function summarizeBestCandidatesByGeneration(
  candidates: GraphEvolveCandidate[],
): GenerationSummary[] {
  const groups = groupCandidatesByGeneration(candidates)
  const summaries: GenerationSummary[] = []
  for (const group of groups) {
    if (!group.candidates.length) continue
    const best = group.candidates.reduce((current, candidate) =>
      candidate.reward > current.reward ? candidate : current,
    )
    summaries.push({
      generation: group.generation,
      reward: best.reward,
    })
  }
  return summaries
}

function formatRewardTrack(
  reward: number,
  trackWidth: number,
  precision: number,
  rewardMax: number,
): string {
  if (trackWidth <= 0) return ""
  const maxValue = rewardMax > 0 ? rewardMax : 1
  const normalized = maxValue > 0 ? reward / maxValue : 0
  const clamped = clamp(normalized, 0, 1)
  let label = reward.toFixed(precision)
  if (trackWidth >= label.length + 2) {
    label = `[${label}]`
  }
  if (trackWidth <= label.length) {
    return label.slice(0, Math.max(1, trackWidth))
  }
  const maxPosition = trackWidth - label.length
  const position = Math.round(clamped * maxPosition)
  const left = "-".repeat(position)
  const right = "-".repeat(trackWidth - label.length - position)
  return `${left}${label}${right}`
}

export function formatRaceLine(options: {
  label: string
  reward: number
  trackWidth?: number
  labelWidth?: number
  scorePrecision?: number
  rewardMax?: number
}): string {
  const trackWidth = options.trackWidth ?? 18
  const labelWidth = options.labelWidth ?? 4
  const scorePrecision = options.scorePrecision ?? 2
  const rewardMax = options.rewardMax ?? REWARD_MAX
  const label = options.label.padEnd(labelWidth)
  const track = formatRewardTrack(options.reward, trackWidth, scorePrecision, rewardMax)
  return `${label} ${track}`
}

export function formatRacePreview(options: {
  candidates: GraphEvolveCandidate[]
  maxCandidates: number
  trackWidth?: number
  labelWidth?: number
  scorePrecision?: number
  rewardMax?: number
}): { lines: string[]; bestReward: number | null } {
  const ranked = [...options.candidates].sort((a, b) => b.reward - a.reward)
  if (ranked.length === 0) {
    return { lines: ["No candidates yet."], bestReward: null }
  }
  const top = ranked.slice(0, options.maxCandidates)
  const lines = top.map((candidate, idx) =>
    formatRaceLine({
      label: `#${idx + 1}`,
      reward: candidate.reward,
      trackWidth: options.trackWidth,
      labelWidth: options.labelWidth,
      scorePrecision: options.scorePrecision,
      rewardMax: options.rewardMax,
    }),
  )
  return { lines, bestReward: ranked[0]?.reward ?? null }
}
