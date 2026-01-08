/**
 * Job list selectors (pure-ish helpers).
 */
import type { JobSummary } from "../tui_data"
import { normalizeJobStatus, JobStatus } from "../utils/job-status"

export function getFilteredJobs(
  jobs: JobSummary[],
  jobStatusFilter: ReadonlySet<string>,
): JobSummary[] {
  if (!jobStatusFilter.size) return jobs
  return jobs.filter((job) => jobStatusFilter.has(normalizeJobStatus(job.status)))
}

export function buildJobStatusOptions(
  jobs: JobSummary[],
): Array<{ status: string; count: number }> {
  const counts = new Map<string, number>()
  for (const job of jobs) {
    const status = normalizeJobStatus(job.status)
    counts.set(status, (counts.get(status) || 0) + 1)
  }

  const order = [
    JobStatus.Running,
    JobStatus.Queued,
    JobStatus.Completed,
    JobStatus.Error,
    JobStatus.Canceled,
    JobStatus.Unknown,
  ]
  const statuses = Array.from(counts.keys()).sort((a, b) => {
    const ai = order.indexOf(a as JobStatus)
    const bi = order.indexOf(b as JobStatus)
    if (ai === -1 && bi === -1) return a.localeCompare(b)
    if (ai === -1) return 1
    if (bi === -1) return -1
    return ai - bi
  })

  return statuses.map((status) => ({ status, count: counts.get(status) || 0 }))
}




