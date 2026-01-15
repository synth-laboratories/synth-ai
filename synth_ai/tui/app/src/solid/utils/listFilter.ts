import type { AppState } from "../../state/app-state"
import { ListPane, type ListFilterMode } from "../../types"

export function getListFilterCount(ui: AppState, pane: ListPane): number {
  const mode = ui.listFilterMode[pane]
  if (mode !== "subset") return 0
  const selections = ui.listFilterSelections[pane]
  return selections ? selections.size : 0
}

export function formatListFilterTitle(
  label: string,
  mode: ListFilterMode,
  count: number,
): string {
  if (mode === "all") {
    return `${label} [ Filter (f) ]`
  }
  return `${label} [Filter: ${count} (f)]`
}
