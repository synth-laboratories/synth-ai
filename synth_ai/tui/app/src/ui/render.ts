/**
 * Central rendering: syncs OpenTUI UI tree from the current state.
 */
import type { AppContext } from "../context"

import { formatDetails, formatMetrics, formatResults } from "../formatters"
import { renderEventCards } from "./events"
import { renderJobsList } from "./jobs"
import { renderLogs } from "./logs"
import { updatePaneIndicators } from "./panes"
import { formatStatus } from "./status"
import { footerText } from "./footer"

export function renderApp(ctx: AppContext): void {
  const { ui, renderer } = ctx
  const { appState, snapshot } = ctx.state

  renderJobsList(ctx)

  const detailContent = formatDetails(snapshot)
  ui.detailText.content = detailContent
  // Dynamic height based on content lines
  ui.detailBox.height = detailContent.split("\n").length + 2  // +2 for border
  ui.resultsText.content = formatResults(snapshot)
  ui.metricsText.content = formatMetrics(snapshot.metrics)
  // Task Apps are only shown in the modal (press 'u'), not in the main view
  ui.taskAppsBox.visible = false
  renderEventCards(ctx)
  renderLogs(ctx)
  updatePaneIndicators(ctx)
  ui.statusText.content = formatStatus(ctx)
  ui.footerText.content = footerText(ctx)
  ui.eventsBox.title = appState.eventFilter ? `Events (filter: ${appState.eventFilter})` : "Events"
  renderer.requestRender()
}

