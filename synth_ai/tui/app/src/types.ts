/**
 * Shared type definitions for the TUI app.
 */

import type { JobEvent, JobSummary } from "./tui_data"

/** Type of job that can be run in the TUI */
export enum JobType {
  Eval = "eval",
  Huh = "huh"
}

/** A prompt candidate (snapshot) with score */
export type PromptCandidate = {
  id: string
  isBaseline: boolean
  score: number | null
  payload: Record<string, any>
  createdAt: string | null
  tag: string | null
}

/** Tunnel record from backend */
export type TunnelRecord = {
  id: string
  hostname: string
  tunnel_token: string
  status: string
  local_host: string
  local_port: number
  org_id: string
  org_name?: string
  created_at: string
  deleted_at?: string
  metadata: Record<string, any>
  dns_verified: boolean
  last_verified_at?: string
  health_status?: string
  health_check_error?: string
}

/** Client-side health check result */
export type TunnelHealthResult = {
  healthy: boolean
  status_code?: number
  error?: string
  response_time_ms?: number
  checked_at: Date
}

export type Snapshot = {
  jobs: JobSummary[]
  selectedJob: JobSummary | null
  events: JobEvent[]
  metrics: Record<string, unknown>
  bestSnapshotId: string | null
  bestSnapshot: Record<string, any> | null
  evalSummary: Record<string, any> | null
  evalResultRows: Array<Record<string, any>>
  artifacts: Array<Record<string, unknown>>
  orgId: string | null
  userId: string | null
  balanceDollars: number | null
  status: string
  lastError: string | null
  lastRefresh: number | null
  /** All prompt candidates (baseline + optimized) */
  allCandidates: PromptCandidate[]
  /** Active tunnels (task apps) */
  tunnels: TunnelRecord[]
  /** Client-side health results keyed by tunnel ID */
  tunnelHealthResults: Map<string, TunnelHealthResult>
  /** Whether tunnels are currently being refreshed */
  tunnelsLoading: boolean
}
