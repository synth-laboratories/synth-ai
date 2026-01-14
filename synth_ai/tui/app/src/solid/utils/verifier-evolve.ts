import type { AppContext } from "../../context"
import type { AppData } from "../../types"
import { extractGraphEvolveCandidates, groupCandidatesByGeneration } from "../../formatters/graph-evolve"
import { clampIndex, moveSelectionIndex } from "./list"

function getVerifierEvolveGenerations(data: AppData) {
  const candidates = extractGraphEvolveCandidates(data)
  return groupCandidatesByGeneration(candidates)
}

export function moveVerifierEvolveGenerationSelection(ctx: AppContext, delta: number): void {
  const generations = getVerifierEvolveGenerations(ctx.state.data)
  if (generations.length === 0) return
  const next = moveSelectionIndex(
    ctx.state.ui.verifierEvolveGenerationIndex,
    delta,
    generations.length,
  )
  if (next !== ctx.state.ui.verifierEvolveGenerationIndex) {
    ctx.setUi("verifierEvolveGenerationIndex", next)
  }
}

export function getSelectedVerifierEvolveGeneration(data: AppData, index: number): number | null {
  const generations = getVerifierEvolveGenerations(data)
  if (generations.length === 0) return null
  const clamped = clampIndex(index, generations.length)
  return generations[clamped]?.generation ?? null
}
