import type { AppState } from "../../state/app-state"
import { ListPane, type ListFilterMode } from "../../types"

export function getListFilterCount(ui: AppState, pane: ListPane): number {
  const mode = ui.listFilterMode[pane]
  if (mode !== "subset") return 0
  const selections = ui.listFilterSelections[pane]
  return selections ? selections.size : 0
}

function formatIndexPart(selectedIndex: number, totalCount: number): string {
  if (totalCount <= 0) return ""
  return `[${selectedIndex + 1}/${totalCount}]`
}

function formatFilterPart(mode: ListFilterMode, filterCount: number): string {
  if (mode === "all") {
    return "Filter (f)"
  }
  return `Filter: ${filterCount} (f)`
}

export function formatListTitle(
  label: string,
  mode: ListFilterMode,
  filterCount: number,
  selectedIndex?: number,
  totalCount?: number,
): string {
  const parts = [label]

  if (totalCount != null && totalCount > 0) {
    parts.push(formatIndexPart(selectedIndex ?? 0, totalCount))
    parts.push(formatFilterPart(mode, filterCount))
  }

  return parts.join("  ")
}
