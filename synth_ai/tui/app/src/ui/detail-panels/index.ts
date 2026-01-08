/**
 * Detail panel configurations registry.
 * Routes to the correct detail panel renderer based on active view.
 */
import type { AppContext } from "../../context"
import { renderJobsDetail } from "./jobs-detail"
import { renderLogsDetail } from "./logs-detail"

export { renderJobsDetail } from "./jobs-detail"
export { renderLogsDetail } from "./logs-detail"

/**
 * Render the active detail panel (right side) based on current view mode.
 */
export function renderActiveDetailPanel(ctx: AppContext): void {
  const { activeListPanel } = ctx.state.appState

  switch (activeListPanel) {
    case "jobs":
      renderJobsDetail(ctx)
      break
    case "logs":
      renderLogsDetail(ctx)
      break
  }
}
