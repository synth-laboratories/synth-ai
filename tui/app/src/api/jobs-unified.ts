/**
 * Unified Jobs API
 *
 * Fetches jobs from multiple endpoints and combines them for status tracking.
 */

import { apiGetV1 } from "./client"

export interface JobRecord {
  id: string
  type: "graphgen" | "eval" | "gepa" | "unknown"
  status: string
  previous_status?: string
  updated_at: string
  created_at: string
}

export interface JobStatusChange {
  id: string
  type: string
  previousStatus: string
  currentStatus: string
  updatedAt: Date
}

// Track previous job statuses for change detection
const previousStatuses = new Map<string, string>()

/**
 * Fetch jobs from all endpoints and return those updated since a given time.
 */
export async function fetchRecentJobs(
  since: Date,
  options: { signal?: AbortSignal } = {}
): Promise<JobStatusChange[]> {
  const results: JobStatusChange[] = []

  // Define endpoints to query
  const endpoints = [
    { path: "/graphgen/jobs", type: "graphgen" as const },
    { path: "/prompt-learning/jobs", type: "gepa" as const },
  ]

  // Fetch from all endpoints in parallel
  const fetchPromises = endpoints.map(async (endpoint) => {
    try {
      const jobs = await apiGetV1(endpoint.path, options)
      if (!Array.isArray(jobs)) return []

      return jobs
        .filter((job: any) => {
          const updatedAt = new Date(job.updated_at || job.created_at)
          return updatedAt > since
        })
        .map((job: any) => ({
          id: job.job_id || job.id,
          type: endpoint.type,
          status: job.status || "unknown",
          updated_at: job.updated_at || job.created_at,
        }))
    } catch {
      // Endpoint might not exist or be unavailable
      return []
    }
  })

  const allResults = await Promise.all(fetchPromises)
  const flatResults = allResults.flat()

  // Sort by update time (most recent first)
  flatResults.sort((a, b) =>
    new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
  )

  // Convert to JobStatusChange with change detection
  for (const job of flatResults) {
    const previousStatus = previousStatuses.get(job.id) || "unknown"
    const currentStatus = job.status

    // Only include if status actually changed
    if (previousStatus !== currentStatus) {
      results.push({
        id: job.id,
        type: job.type,
        previousStatus,
        currentStatus,
        updatedAt: new Date(job.updated_at),
      })

      // Update tracked status
      previousStatuses.set(job.id, currentStatus)
    }
  }

  return results
}

/**
 * Clear the previous status cache (e.g., when starting a new session).
 */
export function clearJobStatusCache(): void {
  previousStatuses.clear()
}

/**
 * Get the top N most recent job changes.
 */
export async function getRecentJobChanges(
  since: Date,
  limit: number = 3,
  options: { signal?: AbortSignal } = {}
): Promise<JobStatusChange[]> {
  const changes = await fetchRecentJobs(since, options)
  return changes.slice(0, limit)
}
