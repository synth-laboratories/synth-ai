/**
 * Shared type definitions for the TUI app.
 */

import type { JobEvent, JobSummary } from "./tui_data"

export type EnvKeyOption = {
  key: string
  sources: string[]
  varNames: string[]
}

/** A prompt candidate (snapshot) with reward */
export type PromptCandidate = {
  id: string
  isBaseline: boolean
  reward: number | null
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
  /** Interactive sessions (OpenCode) */
  sessions: SessionRecord[]
  /** Client-side health results keyed by session ID */
  sessionHealthResults: Map<string, SessionHealthResult>
  /** Whether sessions are currently being refreshed */
  sessionsLoading: boolean
}

export type BackendId = "prod" | "dev" | "local"

export type BackendConfig = {
  id: BackendId
  label: string
  baseUrl: string
}

export type BackendKeySource = {
  sourcePath: string | null
  varName: string | null
}

/** Interactive session record from backend */
export type SessionRecord = {
  session_id: string
  container_id: string
  state: "disconnected" | "connecting" | "connected" | "reconnecting" | "error"
  mode: "async" | "interactive" | "hybrid"
  model: string | null
  access_url: string | null
  tunnel_url: string | null
  opencode_url: string | null
  health_url: string | null
  created_at: string
  connected_at: string | null
  last_activity: string | null
  error_message: string | null
  metadata: Record<string, any>
  is_local: boolean
}

/** Response from connect-local endpoint */
export type ConnectLocalResponse = {
  session_id: string
  state: string
  access_url: string | null
  tunnel_url: string | null
  opencode_url: string | null
  connected_at: string | null
  error: string | null
}

/** Session health check result */
export type SessionHealthResult = {
  healthy: boolean
  status_code?: number
  error?: string
  response_time_ms?: number
  checked_at: Date
}

/** OpenCode message in conversation */
export type OpenCodeMessage = {
  id: string
  role: "user" | "assistant" | "tool"
  content: string
  timestamp: Date
  toolName?: string
  toolStatus?: "pending" | "running" | "completed" | "failed"
}
