/**
 * Jobs list panel configuration.
 */
import type { AppContext } from "../../context"
import type { ListPanelConfig, ListPanelItem } from "../../components/list-panel"
import type { JobSummary } from "../../tui_data"
import { formatJobCard } from "../../formatters/job-card"
import { getFilteredJobs } from "../../selectors/jobs"
import { selectJob } from "../../api/jobs"

export function createJobsListConfig(ctx: AppContext): ListPanelConfig<JobSummary> {
  return {
    id: "jobs",
    title: "Jobs",
    emptyMessage: "No jobs found",
    formatItem: (job: JobSummary): ListPanelItem => {
      const card = formatJobCard(job)
      return {
        id: job.job_id,
        name: card.name,
        description: card.description,
      }
    },
    getItems: () => {
      const { snapshot, appState } = ctx.state
      return getFilteredJobs(snapshot.jobs, appState.jobStatusFilter)
    },
    onSelect: async (job: JobSummary) => {
      // Only select if job changed
      if (ctx.state.snapshot.selectedJob?.job_id !== job.job_id) {
        await selectJob(ctx, job.job_id)
      }
    },
    getTitleSuffix: () => {
      const { appState } = ctx.state
      if (appState.jobStatusFilter.size === 0) return ""
      return `status: ${Array.from(appState.jobStatusFilter).join(", ")}`
    },
  }
}
