/**
 * Job card formatting for the SolidJS jobs list.
 */
import type { JobSummary } from "../../tui_data"
import { formatTimestamp } from "./time"

export type JobCardOption = {
  name: string
  description: string
}

function getRelevantDate(job: JobSummary): string {
  // Use started_at if running, finished_at if terminal, else created_at
  const dateStr = job.finished_at || job.started_at || job.created_at
  return formatTimestamp(dateStr)
}

/**
 * Formats a job for the ListPanel component.
 */
export function formatJobCard(job: JobSummary): JobCardOption {
  const jobType = job.job_type || job.job_source || "job"
  const status = job.status || "-"
  const dateStr = getRelevantDate(job)
  const shortId = job.job_id.slice(-8)

  return {
    name: jobType,
    description: `${status} | ${shortId} | ${dateStr}`,
  }
}

