/**
 * Snapshot state - the main data model for the TUI.
 */

import type { Snapshot } from "../types"

export const snapshot: Snapshot = {
  jobs: [],
  selectedJob: null,
  events: [],
  metrics: {},
  bestSnapshotId: null,
  bestSnapshot: null,
  evalSummary: null,
  evalResultRows: [],
  artifacts: [],
  orgId: null,
  userId: null,
  balanceDollars: null,
  status: "Loading jobs...",
  lastError: null,
  lastRefresh: null,
  allCandidates: [],
  apiCandidates: [],
  apiCandidatesLoaded: false,
  tunnels: [],
  tunnelHealthResults: new Map(),
  tunnelsLoading: false,
  deployments: new Map(),
  sessions: [],
  sessionHealthResults: new Map(),
  sessionsLoading: false,
}

export function resetSnapshot(): void {
  snapshot.jobs = []
  snapshot.selectedJob = null
  snapshot.events = []
  snapshot.metrics = {}
  snapshot.bestSnapshotId = null
  snapshot.bestSnapshot = null
  snapshot.evalSummary = null
  snapshot.evalResultRows = []
  snapshot.artifacts = []
  snapshot.allCandidates = []
  snapshot.apiCandidates = []
  snapshot.apiCandidatesLoaded = false
  snapshot.lastError = null
  snapshot.lastRefresh = null
  snapshot.tunnels = []
  snapshot.tunnelHealthResults = new Map()
  snapshot.tunnelsLoading = false
  snapshot.deployments = new Map()
  snapshot.sessions = []
  snapshot.sessionHealthResults = new Map()
  snapshot.sessionsLoading = false
}
