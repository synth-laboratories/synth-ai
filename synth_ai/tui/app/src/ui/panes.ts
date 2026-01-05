/**
 * Pane focus + visual indicators (jobs vs events).
 */
import type { AppContext } from "../context"

export function setActivePane(ctx: AppContext, pane: "jobs" | "events"): void {
  const { ui } = ctx
  const { appState } = ctx.state
  if (appState.activePane === pane) return

  appState.activePane = pane
  if (pane === "jobs") {
    ui.jobsSelect.focus()
  } else {
    ui.jobsSelect.blur()
  }

  updatePaneIndicators(ctx)
  ctx.requestRender()
}

export function updatePaneIndicators(ctx: AppContext): void {
  const { ui } = ctx
  const { appState } = ctx.state
  ui.jobsTabText.fg = appState.activePane === "jobs" ? "#f8fafc" : "#94a3b8"
  ui.eventsTabText.fg = appState.activePane === "events" ? "#f8fafc" : "#94a3b8"
  ui.jobsBox.borderColor = appState.activePane === "jobs" ? "#60a5fa" : "#334155"
  ui.eventsBox.borderColor = appState.activePane === "events" ? "#60a5fa" : "#334155"
}




