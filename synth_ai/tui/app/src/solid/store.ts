import { createStore, type SetStoreFunction } from "solid-js/store"

import type { AppData } from "../types"
import type { AppState } from "../state/app-state"
import { createInitialAppState } from "../state/app-state"

export type AppStore = {
  data: AppData
  setData: SetStoreFunction<AppData>
  ui: AppState
  setUi: SetStoreFunction<AppState>
}

export function createInitialData(): AppData {
  return {
    jobs: [],
    jobsCache: [],
    jobsCacheAppended: [],
    jobsCacheKey: null,
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
    orgName: null,
    userEmail: null,
    balanceDollars: null,
    status: "Loading jobs...",
    lastError: null,
    lastRefresh: null,
    allCandidates: [],
    tunnels: [],
    tunnelHealthResults: new Map(),
    tunnelsLoading: false,
    deployments: new Map(),
    sessions: [],
    sessionHealthResults: new Map(),
    sessionsLoading: false,
  }
}

export function createAppStore(): AppStore {
  const [data, setData] = createStore<AppData>(createInitialData())
  const [ui, setUi] = createStore<AppState>(createInitialAppState())
  return { data, setData, ui, setUi }
}
