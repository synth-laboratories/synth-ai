/**
 * Pane focus + visual indicators (jobs vs events, and principal pane switching).
 */
import type { AppContext } from "../context"

/**
 * Set the principal pane (top-level view: jobs monitor vs opencode agent).
 */
export function setPrincipalPane(ctx: AppContext, pane: "jobs" | "opencode"): void {
  const { ui } = ctx
  const { appState } = ctx.state
  if (appState.principalPane === pane) return

  appState.principalPane = pane

  if (pane === "jobs") {
    // Show jobs view, hide opencode view
    ui.detailColumn.visible = true
    ui.openCodePane.visible = false
    ui.jobsSelect.focus()
  } else {
    // Show opencode view, hide jobs view
    ui.detailColumn.visible = false
    ui.openCodePane.visible = true
    ui.jobsSelect.blur()
    ui.openCodeInput.focus()
  }

  updatePrincipalIndicators(ctx)
  ctx.requestRender()
}

/**
 * Update visual indicators for principal pane.
 */
export function updatePrincipalIndicators(ctx: AppContext): void {
  const { ui } = ctx
  const { appState } = ctx.state

  // Update tab colors based on principal pane
  const isJobsView = appState.principalPane === "jobs"
  const isOpenCodeView = appState.principalPane === "opencode"

  // OpenCode tab highlight
  ui.openCodeTabText.fg = isOpenCodeView ? "#a855f7" : "#94a3b8"
  ui.sessionsTabText.fg = "#64748b"

  // If in jobs view, also update sub-pane indicators
  if (isJobsView) {
    updatePaneIndicators(ctx)
  } else {
    // In opencode view, dim the jobs/events tabs
    ui.jobsTabText.fg = "#64748b"
    ui.eventsTabText.fg = "#64748b"
  }
}

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

  // Only highlight if we're in jobs principal pane
  const isJobsView = appState.principalPane === "jobs"

  ui.jobsTabText.fg = isJobsView && appState.activePane === "jobs" ? "#f8fafc" : "#94a3b8"
  ui.eventsTabText.fg = isJobsView && appState.activePane === "events" ? "#f8fafc" : "#94a3b8"
  ui.jobsBox.borderColor = isJobsView && appState.activePane === "jobs" ? "#60a5fa" : "#334155"
  ui.eventsBox.borderColor = isJobsView && appState.activePane === "events" ? "#60a5fa" : "#334155"
}

export function blurForModal(ctx: AppContext): void {
  ctx.ui.jobsSelect.blur()
  ctx.ui.openCodeInput.blur()
}

export function restoreFocusFromModal(ctx: AppContext): void {
  const { appState } = ctx.state
  if (appState.principalPane === "jobs" && appState.activePane === "jobs") {
    ctx.ui.jobsSelect.focus()
  } else if (appState.principalPane === "opencode") {
    ctx.ui.openCodeInput.focus()
  }
}




