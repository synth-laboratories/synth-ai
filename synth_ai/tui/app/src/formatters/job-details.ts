/**
 * Job detail panel formatting.
 */
import type { Snapshot } from "../types"
import { formatTimestamp } from "./time"
import { normalizeJobStatus, JobStatus, isTerminalStatus } from "../utils/job-status"

export function formatDetails(snapshot: Snapshot): string {
  const job = snapshot.selectedJob
  if (!job) return "No job selected."

  const status = normalizeJobStatus(job.status)
  const lines = [`Status: ${status}`]

  // Only show timestamps based on job status
  if (status !== JobStatus.Queued) {
    lines.push(`Started: ${formatTimestamp(job.started_at)}`)
  }
  if (status === JobStatus.Running && job.updated_at) {
    lines.push(`Last Update: ${formatTimestamp(job.updated_at)}`)
  }
  if (isTerminalStatus(job.status)) {
    const finishedDate = job.finished_at || job.updated_at
    if (finishedDate) {
      lines.push(`Finished: ${formatTimestamp(finishedDate)}`)
    }
  }

  return lines.join("\n")
}
