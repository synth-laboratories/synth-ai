import type { AppState } from "../../state/app-state"
import { ListPane } from "../../types"

export function getListFilterCount(ui: AppState, pane: ListPane): number {
  const selections = ui.listFilterSelections[pane]
  return selections ? selections.size : 0
}

export function formatListFilterTitle(
  label: string,
  count: number,
  totalOptions: number,
): string {
  if (totalOptions > 0 && count >= totalOptions) {
    return `${label} [ Filter (f) ]`
  }
  return `${label} [Filter: ${count} (f)]`
}
