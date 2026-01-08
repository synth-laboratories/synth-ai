/**
 * Job card formatting for the jobs list.
 */
import type { JobSummary } from "../tui_data"
import { formatDate } from "../utils/date"
import { normalizeJobStatus, JobStatus } from "../utils/job-status"
import { getTrainingTypeDisplay, getJobSourceDisplay } from "../utils/job-types"

export type JobCardOption = {
  name: string
  description: string
}

function getRelevantDate(job: JobSummary, status: JobStatus): string {
  switch (status) {
    case JobStatus.Queued:
      return formatDate(job.created_at)
    case JobStatus.Running:
      return formatDate(job.started_at) || formatDate(job.created_at)
    default:
      return formatDate(job.finished_at) || formatDate(job.started_at) || formatDate(job.created_at)
  }
}

export function formatJobCard(job: JobSummary): JobCardOption {
  const jobType = job.training_type
    ? getTrainingTypeDisplay(job.training_type)
    : getJobSourceDisplay(job.job_source)
  const status = normalizeJobStatus(job.status)
  const dateStr = getRelevantDate(job, status)

  return {
    name: jobType,
    description: `${status} | ${dateStr}`,
  }
}
