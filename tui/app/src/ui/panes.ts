/**
 * Pane focus + visual indicators (jobs, events, logs, opencode).
 */
import type { AppContext } from "../context"
import type { ActivePane, PrincipalPane } from "../types"
import { focusManager } from "../focus"
import { moveLogSelection, pageLogSelection, getSelectedLogFile } from "./logs"
import { moveEventSelection, toggleSelectedEventExpanded } from "./events"

/** Create a focusable handler for the logs pane */
function createLogsPaneFocusable(ctx: AppContext, openLogFileModal: (filePath: string) => void) {
  return {
    id: "logs-pane",
    handleKey: (key: any): boolean => {
      if (key.name === "up" || key.name === "k") {
        moveLogSelection(ctx, -1)
        ctx.render()
        return true
      }
      if (key.name === "down" || key.name === "j") {
        moveLogSelection(ctx, 1)
        ctx.render()
        return true
      }
      if (key.name === "pageup") {
        pageLogSelection(ctx, "up")
        ctx.render()
        return true
      }
      if (key.name === "pagedown") {
        pageLogSelection(ctx, "down")
        ctx.render()
        return true
      }
      if (key.name === "return" || key.name === "enter") {
        const file = getSelectedLogFile(ctx)
        if (file) {
          openLogFileModal(file.path)
        }
        return true
      }
      return false
    },
  }
}

/** Create a focusable handler for the events pane */
function createEventsPaneFocusable(ctx: AppContext, openEventModal: () => void) {
  return {
    id: "events-pane",
    handleKey: (key: any): boolean => {
      if (key.name === "up" || key.name === "k") {
        moveEventSelection(ctx, -1)
        ctx.render()
        return true
      }
      if (key.name === "down" || key.name === "j") {
        moveEventSelection(ctx, 1)
        ctx.render()
        return true
      }
      if (key.name === "return" || key.name === "enter") {
        openEventModal()
        return true
      }
      if (key.name === "x") {
        toggleSelectedEventExpanded(ctx)
        ctx.render()
        return true
      }
      return false
    },
  }
}

let logsFocusable: ReturnType<typeof createLogsPaneFocusable> | null = null
let eventsFocusable: ReturnType<typeof createEventsPaneFocusable> | null = null

/** Initialize pane focusables (call once after modals are set up) */
export function initPaneFocusables(ctx: AppContext, openEventModal: () => void, openLogFileModal: (filePath: string) => void): void {
  logsFocusable = createLogsPaneFocusable(ctx, openLogFileModal)
  eventsFocusable = createEventsPaneFocusable(ctx, openEventModal)
}

export function setActivePane(ctx: AppContext, pane: ActivePane): void {
  const { ui } = ctx
  const { appState } = ctx.state
  if (appState.activePane === pane) return

  // Pop current pane focusable if any
  if (appState.activePane === "logs" && logsFocusable) {
    focusManager.pop("logs-pane")
  }
  if (appState.activePane === "events" && eventsFocusable) {
    focusManager.pop("events-pane")
  }

  appState.activePane = pane

  // Push new pane focusable or focus jobs select
  if (pane === "jobs") {
    ui.jobsSelect.focus()
  } else {
    ui.jobsSelect.blur()
    if (pane === "logs" && logsFocusable) {
      focusManager.push(logsFocusable)
    }
    if (pane === "events" && eventsFocusable) {
      focusManager.push(eventsFocusable)
    }
  }

  updatePaneIndicators(ctx)
  ctx.requestRender()
}

export function cycleActivePane(ctx: AppContext): void {
  const { appState } = ctx.state
  const panes: ActivePane[] = ["jobs", "events", "logs"]
  const currentIdx = panes.indexOf(appState.activePane)
  const nextIdx = (currentIdx + 1) % panes.length
  setActivePane(ctx, panes[nextIdx])
}

export function updatePaneIndicators(ctx: AppContext): void {
  const { ui } = ctx
  const { appState } = ctx.state

  // Update tab text colors
  ui.jobsTabText.fg = appState.activePane === "jobs" ? "#f8fafc" : "#94a3b8"
  ui.eventsTabText.fg = appState.activePane === "events" ? "#f8fafc" : "#94a3b8"
  ui.logsTabText.fg = appState.activePane === "logs" ? "#f8fafc" : "#94a3b8"

  // Update box border colors
  ui.jobsBox.borderColor = appState.activePane === "jobs" ? "#60a5fa" : "#334155"
  ui.eventsBox.borderColor = appState.activePane === "events" ? "#60a5fa" : "#334155"
  ui.logsBox.borderColor = appState.activePane === "logs" ? "#60a5fa" : "#334155"

  // Show/hide panels based on active pane
  // When logs pane is active, hide other panels to give logs full space
  const inLogsMode = appState.activePane === "logs"

  // Hide detail panels when in logs mode
  ui.detailBox.visible = !inLogsMode
  ui.resultsBox.visible = !inLogsMode
  ui.metricsBox.visible = !inLogsMode
  ui.taskAppsBox.visible = !inLogsMode

  // Toggle events/logs visibility
  ui.eventsBox.visible = !inLogsMode
  ui.logsBox.visible = inLogsMode
}

/** Track previous focus state for modal restoration */
let previousPaneBeforeModal: ActivePane | null = null

/** Blur all panes when opening a modal */
export function blurForModal(ctx: AppContext): void {
  const { ui } = ctx
  const { appState } = ctx.state

  previousPaneBeforeModal = appState.activePane

  // Blur jobs select
  ui.jobsSelect.blur()

  // Pop any active pane focusables
  if (appState.activePane === "logs" && logsFocusable) {
    focusManager.pop("logs-pane")
  }
  if (appState.activePane === "events" && eventsFocusable) {
    focusManager.pop("events-pane")
  }
}

/** Restore focus after closing a modal */
export function restoreFocusFromModal(ctx: AppContext): void {
  const { ui } = ctx
  const { appState } = ctx.state

  const paneToRestore = previousPaneBeforeModal || appState.activePane
  previousPaneBeforeModal = null

  // If in OpenCode mode, focus the OpenCode pane
  if (appState.principalPane === "opencode") {
    return
  }

  if (paneToRestore === "jobs") {
    ui.jobsSelect.focus()
  } else if (paneToRestore === "logs" && logsFocusable) {
    focusManager.push(logsFocusable)
  } else if (paneToRestore === "events" && eventsFocusable) {
    focusManager.push(eventsFocusable)
  }
}

/** Set the principal pane (jobs view vs opencode view) */
export function setPrincipalPane(ctx: AppContext, pane: PrincipalPane): void {
  const { ui } = ctx
  const { appState } = ctx.state

  if (appState.principalPane === pane) return

  // Pop current focusables
  if (appState.activePane === "logs" && logsFocusable) {
    focusManager.pop("logs-pane")
  }
  if (appState.activePane === "events" && eventsFocusable) {
    focusManager.pop("events-pane")
  }
  ui.jobsSelect.blur()

  appState.principalPane = pane

  if (pane === "jobs") {
    ui.jobsSelect.focus()
  }
  ctx.requestRender()
}

/** Toggle between jobs and opencode principal panes */
export function togglePrincipalPane(ctx: AppContext): void {
  const { appState } = ctx.state
  const newPane = appState.principalPane === "jobs" ? "opencode" : "jobs"
  setPrincipalPane(ctx, newPane)
}

/** Update visual indicators for principal pane */
