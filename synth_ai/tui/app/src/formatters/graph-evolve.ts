import type { AppData } from "../types"
import { num } from "../tui_data"

export const GOLD_TARGET = 1.0

export type GraphEvolveCandidate = {
  id: string
  reward: number
  generation: number
  payload: Record<string, any>
}

export type CandidateWithDelta = GraphEvolveCandidate & {
  delta: number
}

export type GenerationGroup = {
  generation: number
  candidates: GraphEvolveCandidate[]
}

export type GenerationSummary = {
  generation: number
  reward: number
  delta: number
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

export function withDistance(
  candidates: GraphEvolveCandidate[],
  goldTarget: number = GOLD_TARGET,
): CandidateWithDelta[] {
  return candidates
    .map((candidate) => ({
      ...candidate,
      delta: Math.abs(goldTarget - candidate.reward),
    }))
    .sort((a, b) => {
      if (a.delta !== b.delta) return a.delta - b.delta
      return b.reward - a.reward
    })
}

export function summarizeBestCandidatesByGeneration(
  candidates: GraphEvolveCandidate[],
  goldTarget: number = GOLD_TARGET,
): GenerationSummary[] {
  const groups = groupCandidatesByGeneration(candidates)
  const summaries: GenerationSummary[] = []
  for (const group of groups) {
    const ranked = withDistance(group.candidates, goldTarget)
    if (!ranked.length) continue
    const best = ranked[0]
    summaries.push({
      generation: group.generation,
      reward: best.reward,
      delta: best.delta,
    })
  }
  return summaries
}

function buildGapFill(gapLen: number, label: string): string {
  if (gapLen <= 0) return ""
  if (gapLen <= label.length) return label.slice(0, gapLen)
  const padding = gapLen - label.length
  return `${"-".repeat(padding)}${label}`
}

export function formatRaceLine(options: {
  label: string
  reward: number
  delta: number
  maxDelta: number
  trackWidth?: number
  labelWidth?: number
  scorePrecision?: number
  deltaPrecision?: number
}): string {
  const trackWidth = options.trackWidth ?? 18
  const labelWidth = options.labelWidth ?? 12
  const scorePrecision = options.scorePrecision ?? 2
  const deltaPrecision = options.deltaPrecision ?? 2
  const label = options.label.padEnd(labelWidth)
  const scoreLabel = options.reward.toFixed(scorePrecision)
  const box = `[${scoreLabel}]`
  const deltaLabel = options.delta.toFixed(deltaPrecision)
  const minGap = clamp(deltaLabel.length + 2, 3, trackWidth)
  const scaled = options.maxDelta > 0
    ? Math.round((options.delta / options.maxDelta) * trackWidth)
    : minGap
  const gapLen = clamp(scaled, minGap, trackWidth)
  const gap = `|${buildGapFill(gapLen, deltaLabel)}|`
  const gapField = gap.padStart(trackWidth + 2, " ")
  return `${label} ${gapField}${box}`
}

export function formatRacePreview(options: {
  candidates: GraphEvolveCandidate[]
  maxCandidates: number
  trackWidth?: number
  labelWidth?: number
  scorePrecision?: number
  goldTarget?: number
}): { lines: string[]; bestDelta: number | null } {
  const goldTarget = options.goldTarget ?? GOLD_TARGET
  const ranked = withDistance(options.candidates, goldTarget)
  if (!ranked.length) {
    return { lines: ["No candidates yet."], bestDelta: null }
  }
  const maxDelta = Math.max(...ranked.map((candidate) => candidate.delta), 0.0001)
  const top = ranked.slice(0, options.maxCandidates)
  const lines = top.map((candidate, idx) =>
    formatRaceLine({
      label: `Candidate ${idx + 1}`,
      reward: candidate.reward,
      delta: candidate.delta,
      maxDelta,
      trackWidth: options.trackWidth,
      labelWidth: options.labelWidth,
      scorePrecision: options.scorePrecision,
    }),
  )
  return { lines, bestDelta: ranked[0]?.delta ?? null }
}
