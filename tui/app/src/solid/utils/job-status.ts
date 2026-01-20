/**
 * Job status normalization utilities.
 *
 * Maps backend status values to canonical display values.
 */

/** Canonical job status values for display */
export enum JobStatus {
  Queued = "Queued",
  Running = "Running",
  Completed = "Completed",
  Error = "Error",
  Canceled = "Canceled",
  Unknown = "Unknown",
}

/** Map backend status to display status */
export function normalizeJobStatus(status: string | null | undefined): JobStatus {
  const s = (status || "").toLowerCase()
  switch (s) {
    case "queued":
      return JobStatus.Queued
    case "running":
      return JobStatus.Running
    case "completed":
    case "succeeded":
      return JobStatus.Completed
    case "failed":
      return JobStatus.Error
    case "canceled":
    case "cancelled":
      return JobStatus.Canceled
    default:
      return JobStatus.Unknown
  }
}

/** Check if job is in a terminal state */
export function isTerminalStatus(status: string | null | undefined): boolean {
  const normalized = normalizeJobStatus(status)
  return normalized === JobStatus.Completed ||
         normalized === JobStatus.Error ||
         normalized === JobStatus.Canceled
}
