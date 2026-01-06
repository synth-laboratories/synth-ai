/**
 * Job list selectors (pure-ish helpers).
 */
import type { JobSummary } from "../tui_data"

export function getFilteredJobs(
  jobs: JobSummary[],
  jobStatusFilter: ReadonlySet<string>,
): JobSummary[] {
  if (!jobStatusFilter.size) return jobs
  return jobs.filter((job) => jobStatusFilter.has(String(job.status || "unknown").toLowerCase()))
}

export function buildJobStatusOptions(
  jobs: JobSummary[],
): Array<{ status: string; count: number }> {
  const counts = new Map<string, number>()
  for (const job of jobs) {
    const status = String(job.status || "unknown").toLowerCase()
    counts.set(status, (counts.get(status) || 0) + 1)
  }

  const order = ["running", "queued", "succeeded", "failed", "canceled", "cancelled", "unknown"]
  const statuses = Array.from(counts.keys()).sort((a, b) => {
    const ai = order.indexOf(a)
    const bi = order.indexOf(b)
    if (ai === -1 && bi === -1) return a.localeCompare(b)
    if (ai === -1) return 1
    if (bi === -1) return -1
    return ai - bi
  })

  return statuses.map((status) => ({ status, count: counts.get(status) || 0 }))
}




