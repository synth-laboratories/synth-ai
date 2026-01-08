/**
 * Shared type definitions for the TUI app.
 */

import type { ChildProcess } from "child_process"
import type { JobEvent, JobSummary } from "./tui_data"

/** Type of job that can be run in the TUI */
export enum JobType {
  Eval = "eval",
}

/** Active pane in the TUI (jobs list, events, or logs) */
export type ActivePane = "jobs" | "events" | "logs"

/** Principal pane - top-level view mode */
export type PrincipalPane = "jobs" | "opencode"

/** Environment key option discovered from .env files */
export type EnvKeyOption = {
  key: string
  sources: string[]
  varNames: string[]
}

/** Source of a deployment log entry */
export type LogSource = "uvicorn" | "cloudflare" | "app"

/** A log entry from the deployment process */
export type DeploymentLog = {
  type: "log"
  source: LogSource
  message: string
  timestamp: number
  level?: string
  deployment_id?: string
}

/** A status message from the deployment process */
export type DeploymentStatus = {
  type: "status"
  status: "starting" | "ready" | "error" | "stopped"
  url?: string
  port?: number
  error?: string
  message?: string
  deployment_id?: string
  timestamp?: number
}

/** Union type for all deployment log entries */
export type DeploymentLogEntry = DeploymentLog | DeploymentStatus

/** A deployed LocalAPI instance */
export type Deployment = {
  id: string
  localApiPath: string
  url: string | null
  status: "deploying" | "ready" | "error" | "stopped"
  logs: DeploymentLogEntry[]
  proc: ChildProcess | null
  startedAt: Date
  error?: string
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

/** OpenCode session record */
export type SessionRecord = {
  session_id: string
  container_id?: string
  state: "connected" | "connecting" | "reconnecting" | "disconnected" | "error"
  opencode_url?: string | null
  access_url?: string | null
  tunnel_url?: string | null
  health_url?: string | null
  is_local: boolean
  mode: string
  model?: string | null
  created_at?: string | null
  connected_at?: string | null
  last_activity?: string | null
  error_message?: string | null
  metadata?: Record<string, unknown>
}

/** Response from connect-local endpoint */
export type ConnectLocalResponse = {
  session_id?: string
  error?: string
}

/** Session health check result */
export type SessionHealthResult = {
  healthy: boolean
  response_time_ms?: number
  error?: string
  checked_at: Date
}

/** OpenCode message in the chat pane */
export type OpenCodeMessage = {
  id: string
  role: "user" | "assistant" | "tool"
  content: string
  timestamp: Date
  toolName?: string
  toolStatus?: "pending" | "running" | "completed" | "failed"
}

/** Backend ID for multi-backend support */
export type BackendId = "prod" | "dev" | "local"

/** Frontend URL identifier for key storage (keys are shared by frontend URL) */
export type FrontendUrlId = "usesynth.ai" | "localhost:3000"

/** Backend configuration */
export type BackendConfig = {
  id: BackendId
  label: string
  baseUrl: string
}

/** Source information for an API key */
export type BackendKeySource = {
  sourcePath: string | null
  varName: string | null
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
  /** Active deployments with their streaming logs */
  deployments: Map<string, Deployment>
  /** OpenCode sessions */
  sessions: SessionRecord[]
  /** Session health results keyed by session_id */
  sessionHealthResults: Map<string, SessionHealthResult>
  /** Whether sessions are currently being refreshed */
  sessionsLoading: boolean
}
