/**
 * Jobs detail panel renderer.
 * Renders the right side panels when viewing jobs: details, results, metrics, events.
 */
import type { AppContext } from "../../context"
import { formatDetails, formatMetrics, formatResults } from "../../formatters"
import { renderEventCards } from "../events"

/**
 * Render the jobs detail panels (right side).
 */
export function renderJobsDetail(ctx: AppContext): void {
  const { ui } = ctx
  const { snapshot } = ctx.state

  // Detail box: job info
  const detailContent = formatDetails(snapshot)
  ui.detailText.content = detailContent
  // Dynamic height based on content lines
  ui.detailBox.height = detailContent.split("\n").length + 2 // +2 for border

  // Results box: job result rows
  ui.resultsText.content = formatResults(snapshot)

  // Metrics box: job metrics
  ui.metricsText.content = formatMetrics(snapshot.metrics)

  // Task Apps box: hidden in main view (only shown in modal via 'u')
  ui.taskAppsBox.visible = false

  // Events box: event cards for selected job
  renderEventCards(ctx)
}
