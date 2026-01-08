/**
 * Jobs list rendering.
 */
import type { AppContext } from "../context"
import { formatJobCard } from "../formatters"
import { getFilteredJobs } from "../selectors/jobs"

export function renderJobsList(ctx: AppContext): void {
  const { ui } = ctx
  const { appState, snapshot } = ctx.state

  const filteredJobs = getFilteredJobs(snapshot.jobs, appState.jobStatusFilter)

  ui.jobsBox.title = appState.jobStatusFilter.size
    ? `Jobs (status: ${Array.from(appState.jobStatusFilter).join(", ")})`
    : "Jobs"

  if (filteredJobs.length) {
    ui.jobsSelect.visible = true
    ui.jobsEmptyText.visible = false
    ui.jobsSelect.options = filteredJobs.map((job) => {
      const card = formatJobCard(job)
      return { name: card.name, description: card.description, value: job.job_id }
    })
  } else {
    ui.jobsSelect.visible = false
    ui.jobsEmptyText.visible = true
  }
}
