/**
 * Central rendering: syncs OpenTUI UI tree from the current state.
 */
import type { AppContext } from "../context"
import type { TunnelRecord, TunnelHealthResult } from "../types"

import { formatDetails, formatMetrics, formatResults } from "../formatters"
import { extractEnvName } from "../utils/job"
import { getFilteredJobs } from "../selectors/jobs"
import { renderEventCards } from "./events"
import { updatePaneIndicators } from "./panes"
import { footerText, formatHeaderMeta, formatStatus } from "./text"

/**
 * Format the Task Apps panel content showing tunnel status.
 */
function formatTaskApps(
  tunnels: TunnelRecord[],
  healthResults: Map<string, TunnelHealthResult>,
  loading: boolean
): string {
  if (loading) {
    return "Loading task apps..."
  }

  if (tunnels.length === 0) {
    return "No active task apps"
  }

  const lines: string[] = []
  for (const tunnel of tunnels.slice(0, 4)) {
    const health = healthResults.get(tunnel.id)
    const hostname = tunnel.hostname.replace(/^https?:\/\//, "")
    const shortHost = hostname.length > 35 ? hostname.slice(0, 32) + "..." : hostname

    let statusIcon: string
    let statusText: string

    if (!health) {
      statusIcon = "?"
      statusText = "checking"
    } else if (health.healthy) {
      statusIcon = "\u2713"
      statusText = health.response_time_ms != null ? `${health.response_time_ms}ms` : "healthy"
    } else {
      statusIcon = "\u2717"
      statusText = health.error?.slice(0, 20) || "unhealthy"
    }

    lines.push(`[${statusIcon}] ${shortHost} (${statusText})`)
  }

  if (tunnels.length > 4) {
    lines.push(`  ... +${tunnels.length - 4} more`)
  }

  return lines.join("\n")
}

export function renderApp(ctx: AppContext): void {
  const { ui, renderer } = ctx
  const { appState, snapshot } = ctx.state

  const filteredJobs = getFilteredJobs(snapshot.jobs, appState.jobStatusFilter)
  ui.jobsBox.title = appState.jobStatusFilter.size
    ? `Jobs (status: ${Array.from(appState.jobStatusFilter).join(", ")})`
    : "Jobs"

  ui.jobsSelect.options = filteredJobs.length
    ? filteredJobs.map((job) => {
        const shortId = job.job_id.slice(-8)
        const score = job.best_score == null ? "-" : job.best_score.toFixed(4)
        const label =
          job.training_type || (job.job_source === "learning" ? "eval" : "prompt")
        const envName = extractEnvName(job)
        const desc = envName
          ? `${job.status} | ${label} | ${envName} | ${score}`
          : `${job.status} | ${label} | ${score}`
        return { name: shortId, description: desc, value: job.job_id }
      })
    : [
        {
          name: "no jobs",
          description: appState.jobStatusFilter.size
            ? `no jobs with selected status`
            : "no prompt-learning jobs found",
          value: "",
        },
      ]

  ui.detailText.content = formatDetails(snapshot)
  ui.resultsText.content = formatResults(snapshot)
  ui.metricsText.content = formatMetrics(snapshot.metrics)
  ui.taskAppsText.content = formatTaskApps(
    snapshot.tunnels,
    snapshot.tunnelHealthResults,
    snapshot.tunnelsLoading
  )
  ui.taskAppsBox.title = snapshot.tunnels.length > 0
    ? `Task Apps (${snapshot.tunnels.length})`
    : "Task Apps"
  renderEventCards(ctx)
  updatePaneIndicators(ctx)
  ui.headerMetaText.content = formatHeaderMeta(ctx)
  ui.statusText.content = formatStatus(ctx)
  ui.footerText.content = footerText(ctx)
  ui.eventsBox.title = appState.eventFilter ? `Events (filter: ${appState.eventFilter})` : "Events"
  renderer.requestRender()
}


