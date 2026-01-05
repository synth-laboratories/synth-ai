/**
 * Shared type definitions for the TUI app.
 */

import type { JobEvent, JobSummary } from "./tui_data"

export type EnvKeyOption = {
  key: string
  sources: string[]
  varNames: string[]
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
