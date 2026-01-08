/**
 * Pane focus + visual indicators (jobs, events, logs, opencode).
 */
import type { AppContext } from "../context"
import type { ActivePane, ListPanelId, PrincipalPane } from "../types"
import { focusManager } from "../focus"
import { BOX } from "../theme"
import { setActiveListPanel } from "../state/app-state"
import { scrollLogContent, pageLogContent, toggleLogContentTailMode } from "./logs"
import { moveEventSelection, toggleSelectedEventExpanded } from "./events"
import { scrollOpenCode, handleOpenCodeInput, handleOpenCodeBackspace, sendOpenCodeMessage } from "./opencode"

/** Create a focusable handler for the logs pane (scrolls content, not file list) */
function createLogsPaneFocusable(ctx: AppContext) {
  return {
    id: "logs-pane",
    handleKey: (key: any): boolean => {
      if (key.name === "up" || key.name === "k") {
        scrollLogContent(ctx, -1)
        ctx.render()
        return true
      }
      if (key.name === "down" || key.name === "j") {
        scrollLogContent(ctx, 1)
        ctx.render()
        return true
      }
      if (key.name === "pageup") {
        pageLogContent(ctx, "up")
        ctx.render()
        return true
      }
      if (key.name === "pagedown") {
        pageLogContent(ctx, "down")
        ctx.render()
        return true
      }
      if (key.name === "t") {
        toggleLogContentTailMode(ctx)
        ctx.render()
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

/** Create a focusable handler for the OpenCode pane */
function createOpenCodePaneFocusable(ctx: AppContext) {
  return {
    id: "opencode-pane",
    handleKey: (key: any): boolean => {
      if (key.name === "up" || key.name === "k") {
        scrollOpenCode(ctx, -1)
        ctx.render()
        return true
      }
      if (key.name === "down" || key.name === "j") {
        scrollOpenCode(ctx, 1)
        ctx.render()
        return true
      }
      if (key.name === "pageup") {
        scrollOpenCode(ctx, -10)
        ctx.render()
        return true
      }
      if (key.name === "pagedown") {
        scrollOpenCode(ctx, 10)
        ctx.render()
        return true
      }
      if (key.name === "return" || key.name === "enter") {
        void sendOpenCodeMessage(ctx)
        return true
      }
      if (key.name === "backspace") {
        handleOpenCodeBackspace(ctx)
        return true
      }
      // Let global shortcuts through (Shift+O for sessions, Shift+G for toggle, q for quit)
      if (key.shift && (key.name === "o" || key.name === "g")) {
        return false
      }
      if (key.name === "q" || key.name === "escape") {
        return false
      }
      // Handle character input
      if (key.sequence && key.sequence.length === 1 && !key.ctrl && !key.meta) {
        handleOpenCodeInput(ctx, key.sequence)
        return true
      }
      return false
    },
  }
}

let logsFocusable: ReturnType<typeof createLogsPaneFocusable> | null = null
let eventsFocusable: ReturnType<typeof createEventsPaneFocusable> | null = null
let openCodeFocusable: ReturnType<typeof createOpenCodePaneFocusable> | null = null

/** Initialize pane focusables (call once after modals are set up) */
export function initPaneFocusables(ctx: AppContext, openEventModal: () => void): void {
  logsFocusable = createLogsPaneFocusable(ctx)
  eventsFocusable = createEventsPaneFocusable(ctx, openEventModal)
  openCodeFocusable = createOpenCodePaneFocusable(ctx)
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

/**
 * Set both the active pane and the list panel in the left sidebar.
 * Use this when switching between major views (jobs vs logs).
 */
export function setActivePaneWithListPanel(
  ctx: AppContext,
  pane: ActivePane,
  listPanel: ListPanelId
): void {
  setActiveListPanel(listPanel)
  setActivePane(ctx, pane)
}

export function cycleActivePane(ctx: AppContext): void {
  const { appState } = ctx.state

  // Cycle depends on the current view mode
  if (appState.activeListPanel === "logs") {
    // In logs view: cycle between left panel (jobs) and content (logs)
    const nextPane = appState.activePane === "jobs" ? "logs" : "jobs"
    setActivePane(ctx, nextPane)
  } else {
    // In jobs view: cycle between left panel, events, and logs pane (old behavior)
    const panes: ActivePane[] = ["jobs", "events", "logs"]
    const currentIdx = panes.indexOf(appState.activePane)
    const nextIdx = (currentIdx + 1) % panes.length
    setActivePane(ctx, panes[nextIdx])
  }
}

export function updatePaneIndicators(ctx: AppContext): void {
  const { ui } = ctx
  const { appState } = ctx.state

  // View mode is determined by activeListPanel (what the left panel shows)
  const inLogsView = appState.activeListPanel === "logs"

  // Update tab text colors based on view mode
  ui.jobsTabText.fg = !inLogsView ? "#f8fafc" : "#94a3b8"
  ui.eventsTabText.fg = appState.activePane === "events" ? "#f8fafc" : "#94a3b8"
  ui.logsTabText.fg = inLogsView ? "#f8fafc" : "#94a3b8"

  // Update box border colors based on focus
  // Left panel (jobsBox) is focused when activePane is "jobs"
  ui.jobsBox.borderColor = appState.activePane === "jobs" ? BOX.borderColorFocused : BOX.borderColor
  ui.eventsBox.borderColor = appState.activePane === "events" ? BOX.borderColorFocused : BOX.borderColor
  // Right panel (logsBox) is focused when activePane is "logs" AND in logs view
  ui.logsBox.borderColor = (appState.activePane === "logs" && inLogsView) ? BOX.borderColorFocused : BOX.borderColor

  // Show/hide panels based on view mode (activeListPanel), not focus
  // Hide detail panels when in logs view
  ui.detailBox.visible = !inLogsView
  ui.resultsBox.visible = !inLogsView
  ui.metricsBox.visible = !inLogsView
  ui.taskAppsBox.visible = !inLogsView

  // Toggle events/logs visibility based on view mode
  ui.eventsBox.visible = !inLogsView
  ui.logsBox.visible = inLogsView
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
    if (openCodeFocusable) {
      focusManager.push(openCodeFocusable)
    }
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
  if (appState.principalPane === "opencode" && openCodeFocusable) {
    focusManager.pop("opencode-pane")
  }
  if (appState.activePane === "logs" && logsFocusable) {
    focusManager.pop("logs-pane")
  }
  if (appState.activePane === "events" && eventsFocusable) {
    focusManager.pop("events-pane")
  }
  ui.jobsSelect.blur()

  appState.principalPane = pane

  // Update visibility
  if (pane === "jobs") {
    ui.detailColumn.visible = true
    ui.openCodeBox.visible = false
    ui.jobsSelect.focus()
  } else {
    ui.detailColumn.visible = false
    ui.openCodeBox.visible = true
    if (openCodeFocusable) {
      focusManager.push(openCodeFocusable)
    }
  }

  updatePrincipalIndicators(ctx)
  ctx.requestRender()
}

/** Toggle between jobs and opencode principal panes */
export function togglePrincipalPane(ctx: AppContext): void {
  const { appState } = ctx.state
  const newPane = appState.principalPane === "jobs" ? "opencode" : "jobs"
  setPrincipalPane(ctx, newPane)
}

/** Update visual indicators for principal pane */
export function updatePrincipalIndicators(ctx: AppContext): void {
  const { ui } = ctx
  const { appState } = ctx.state

  // Update OpenCode box border to show when active
  ui.openCodeBox.borderColor = appState.principalPane === "opencode" ? BOX.borderColorFocused : BOX.borderColor
}
