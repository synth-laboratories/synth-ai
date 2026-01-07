/**
 * Central rendering: syncs OpenTUI UI tree from the current state.
 */
import type { AppContext } from "../context"

import { formatDetails, formatMetrics, formatResults } from "../formatters"
import { extractEnvName } from "../utils/job"
import { getFilteredJobs } from "../selectors/jobs"
import { renderEventCards } from "./events"
import { updatePaneIndicators } from "./panes"
import { footerText, formatHeaderMeta, formatStatus } from "./text"

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
        const reward = job.best_reward == null ? "-" : job.best_reward.toFixed(4)
        const label =
          job.training_type || (job.job_source === "learning" ? "eval" : "prompt")
        const envName = extractEnvName(job)
        const desc = envName
          ? `${job.status} | ${label} | ${envName} | ${reward}`
          : `${job.status} | ${label} | ${reward}`
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
  // Task Apps are only shown in the modal (press 'u'), not in the main view
  ui.taskAppsBox.visible = false
  renderEventCards(ctx)
  updatePaneIndicators(ctx)
  ui.headerMetaText.content = formatHeaderMeta(ctx)
  ui.statusText.content = formatStatus(ctx)
  ui.footerText.content = footerText(ctx)
  ui.eventsBox.title = appState.eventFilter ? `Events (filter: ${appState.eventFilter})` : "Events"
  renderer.requestRender()
}

