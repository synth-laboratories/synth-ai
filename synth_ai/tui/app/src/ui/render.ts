/**
 * Central rendering: syncs OpenTUI UI tree from the current state.
 */
import type { AppContext } from "../context"

import { renderListPanel } from "../components/list-panel"
import { getListPanelConfig } from "./list-panels"
import { renderActiveDetailPanel } from "./detail-panels"
import { updatePaneIndicators } from "./panes"
import { formatStatus } from "./status"
import { footerText } from "./footer"

/**
 * Render the active list panel in the left sidebar.
 */
function renderActiveListPanel(ctx: AppContext): void {
  const { ui } = ctx
  const { appState } = ctx.state
  const config = getListPanelConfig(ctx, appState.activeListPanel)
  renderListPanel(
    { box: ui.jobsBox, select: ui.jobsSelect, emptyText: ui.jobsEmptyText },
    config
  )
}

export function renderApp(ctx: AppContext): void {
  const { ui, renderer } = ctx
  const { appState } = ctx.state

  // Render left sidebar (list panel)
  renderActiveListPanel(ctx)

  // Render right side (detail panels) based on current view
  renderActiveDetailPanel(ctx)

  // Update visual indicators for pane focus
  updatePaneIndicators(ctx)

  // Common elements
  ui.statusText.content = formatStatus(ctx)
  ui.footerText.content = footerText(ctx)
  ui.eventsBox.title = appState.eventFilter ? `Events (filter: ${appState.eventFilter})` : "Events"

  renderer.requestRender()
}

