import { createSignal, type Accessor } from "solid-js"
import type { SetStoreFunction } from "solid-js/store"

import type { AppContext } from "../../context"
import type { AppData, SessionHealthResult, SessionRecord, UsageData } from "../../types"
import type { AppState } from "../../state/app-state"
import type { ActiveModal } from "../modals/types"
import { modeUrls, switchMode } from "../../state/app-state"
import { buildJobTypeOptions } from "../../selectors/jobs"
import { buildLogTypeOptions } from "./useLogsListState"
import { fetchMetrics } from "../../api/jobs"
import { apiGet } from "../../api/client"
import { fetchSessions, disconnectSession, checkSessionHealth } from "../../api/sessions"
import { refreshTunnels, refreshTunnelHealth } from "../../api/tunnels"
import { connectLocalOpenCodeSession } from "../../api/opencode"
import { openBrowser } from "../../auth"
import { copyToClipboard } from "../../utils/clipboard"
import {
  persistModeSelection,
  persistListFilters,
  readPersistedSettings,
} from "../../persistence/settings"
import { moveSelectionIndex, resolveSelectionWindow } from "../utils/list"
import { isAbortError } from "../../utils/abort"
import { isAborted } from "../../utils/request"
import { log } from "../../utils/log"
import type { LogFileInfo } from "../utils/logs"
import { ListPane, type Mode } from "../../types"
import { extractGraphEvolveCandidates, groupCandidatesByGeneration } from "../../formatters/graph-evolve"

type UseOverlayModalsOptions = {
  ctx: AppContext
  data: AppData
  ui: AppState
  setData: SetStoreFunction<AppData>
  setUi: SetStoreFunction<AppState>
  modalInputValue: Accessor<string>
  setModalInputValue: (value: string) => void
  openOverlayModal: (kind: ActiveModal) => void
  closeActiveModal: () => void
  runAction: (key: string, task: () => Promise<void>) => void
  ensureCandidatesModal: () => void
  ensureGraphEvolveGenerationsModal: () => void
  ensureTraceViewerModal: () => void
  promptLogin: () => void
  refreshData: () => Promise<boolean>
  logFiles: Accessor<LogFileInfo[]>
}

export type OverlayModalState = {
  usageData: Accessor<UsageData | null>
  settingsCursor: Accessor<number>
  sessionsCache: Accessor<SessionRecord[]>
  sessionsHealthCache: Accessor<Map<string, SessionHealthResult>>
  sessionsSelectedIndex: Accessor<number>
  sessionsScrollOffset: Accessor<number>
  openFilterModal: () => void
  applyFilterModal: () => void
  openSnapshotModal: () => void
  applySnapshotModal: () => Promise<void>
  openConfigModal: () => void
  openResultsModal: () => void
  openCandidatesForGeneration: (generation: number) => void
  openProfileModal: () => void
  openListFilterModal: () => void
  moveListFilter: (delta: number) => void
  toggleListFilterSelection: () => void
  selectAllListFilterSelection: () => void
  openSettingsModal: () => void
  moveSettingsCursor: (delta: number) => void
  selectSettingsBackend: () => Promise<void>
  openUsageModal: () => void
  openMetricsModal: () => void
  openMetricsAndFetch: () => void
  openUsageBilling: () => void
  openTaskAppsModal: () => void
  moveTaskAppsSelection: (delta: number) => void
  copySelectedTunnelUrl: () => Promise<void>
  openSessionsModal: () => void
  moveSessionsSelection: (delta: number) => void
  refreshSessionsModal: () => Promise<void>
  connectLocalSession: () => Promise<void>
  disconnectSelectedSession: () => Promise<void>
  copySelectedSessionUrl: () => Promise<void>
  selectSession: () => void
  openTracesModal: () => void
}

export function useOverlayModals(options: UseOverlayModalsOptions): OverlayModalState {
  const [usageData, setUsageData] = createSignal<UsageData | null>(null)
  const [settingsCursor, setSettingsCursor] = createSignal(0)
  const [sessionsSelectedIndex, setSessionsSelectedIndex] = createSignal(0)
  const [sessionsScrollOffset, setSessionsScrollOffset] = createSignal(0)
  const [sessionsCache, setSessionsCache] = createSignal<SessionRecord[]>([])
  const [sessionsHealthCache, setSessionsHealthCache] = createSignal<Map<string, SessionHealthResult>>(new Map())

  const openFilterModal = (): void => {
    options.setModalInputValue(options.ui.eventFilter)
    options.openOverlayModal("filter")
  }

  const applyFilterModal = (): void => {
    options.setUi("eventFilter", options.modalInputValue().trim())
    options.closeActiveModal()
  }

  const openSnapshotModal = (): void => {
    options.setModalInputValue("")
    options.openOverlayModal("snapshot")
  }

  const applySnapshotModal = async (): Promise<void> => {
    const trimmed = options.modalInputValue().trim()
    if (!trimmed) {
      options.closeActiveModal()
      return
    }
    const job = options.data.selectedJob
    if (!job) {
      options.closeActiveModal()
      return
    }
    options.closeActiveModal()
    try {
      await apiGet(`/prompt-learning/online/jobs/${job.job_id}/snapshots/${trimmed}`)
      options.setData("status", `Snapshot ${trimmed} fetched`)
    } catch (err: any) {
      if (isAbortError(err)) return
      options.setData("lastError", err?.message || "Snapshot fetch failed")
    }
  }

  const openConfigModal = (): void => {
    options.setUi("configModalOffset", 0)
    options.openOverlayModal("config")
  }

  const openResultsModal = (): void => {
    const job = options.data.selectedJob
    const graphType = job?.metadata && typeof job.metadata === "object" ? (job.metadata as any).graph_type : null
    if (job?.training_type === "graph_evolve" && graphType === "verifier") {
      options.setUi("candidatesGenerationFilter", null)
      options.openOverlayModal("generations")
      void options.ensureGraphEvolveGenerationsModal()
      return
    }
    options.setUi("candidatesGenerationFilter", null)
    options.openOverlayModal("results")
    void options.ensureCandidatesModal()
  }

  const openCandidatesForGeneration = (generation: number): void => {
    options.setUi("candidatesGenerationFilter", generation)
    const candidates = extractGraphEvolveCandidates(options.data)
    const generations = groupCandidatesByGeneration(candidates)
    const index = generations.findIndex((group) => group.generation === generation)
    if (index >= 0) {
      options.setUi("verifierEvolveGenerationIndex", index)
    }
    options.openOverlayModal("results")
    void options.ensureCandidatesModal()
  }

  const openProfileModal = (): void => {
    options.openOverlayModal("profile")
  }

  const persistListFilterState = (): void => {
    void persistListFilters(
      options.ui.currentMode,
      {
        [ListPane.Jobs]: {
          mode: options.ui.listFilterMode[ListPane.Jobs],
          selections: Array.from(options.ui.listFilterSelections[ListPane.Jobs]),
        },
        [ListPane.Logs]: {
          mode: options.ui.listFilterMode[ListPane.Logs],
          selections: Array.from(options.ui.listFilterSelections[ListPane.Logs]),
        },
      },
      (message) => options.setData("status", message),
    )
  }

  const openListFilterModal = (): void => {
    options.setUi("listFilterPane", options.ui.activePane)
    if (options.ui.activePane === ListPane.Jobs) {
      options.setUi("listFilterOptions", buildJobTypeOptions(options.data.jobs))
    } else if (options.ui.activePane === ListPane.Logs) {
      options.setUi("listFilterOptions", buildLogTypeOptions(options.logFiles()))
    } else {
      options.setUi("listFilterOptions", [])
    }
    const pane = options.ui.activePane as ListPane
    const mode = options.ui.listFilterMode[pane]
    if ((mode === "all" || mode === "none") && options.ui.listFilterSelections[pane].size > 0) {
      options.setUi("listFilterSelections", pane, new Set<string>())
    }
    options.setUi("listFilterVisibleCount", options.ctx.state.config.listFilterVisibleCount)
    options.setUi("listFilterCursor", 0)
    options.setUi("listFilterWindowStart", 0)
    options.openOverlayModal("list-filter")
  }

  const moveListFilter = (delta: number): void => {
    log("state", `moveListFilter delta=${delta} cursor=${options.ui.listFilterCursor} total=${options.ui.listFilterOptions.length}`)
    const totalOptions = options.ui.listFilterOptions.length
    const total = totalOptions > 0 ? totalOptions + 1 : 0
    const visibleCount = options.ui.listFilterVisibleCount
    const nextSelected = moveSelectionIndex(options.ui.listFilterCursor, delta, total)
    const window = resolveSelectionWindow(
      total,
      nextSelected,
      options.ui.listFilterWindowStart,
      visibleCount,
    )
    options.setUi("listFilterCursor", window.selectedIndex)
    options.setUi("listFilterWindowStart", window.windowStart)
    log("state", `moveListFilter result cursor=${window.selectedIndex} windowStart=${window.windowStart}`)
  }

  const toggleListFilterSelection = (): void => {
    log("state", `toggleListFilterSelection cursor=${options.ui.listFilterCursor}`)
    const totalOptions = options.ui.listFilterOptions.length
    if (!totalOptions) {
      log("state", "toggleListFilterSelection: no options available")
      return
    }
    if (options.ui.listFilterCursor === 0) {
      selectAllListFilterSelection()
      return
    }
    const option = options.ui.listFilterOptions[options.ui.listFilterCursor - 1]
    if (!option) {
      log("state", "toggleListFilterSelection: no option at cursor")
      return
    }
    const pane = options.ui.listFilterPane as ListPane
    const mode = options.ui.listFilterMode[pane]
    const allIds = options.ui.listFilterOptions.map((entry) => entry.id)
    let nextMode = mode
    let next = new Set(options.ui.listFilterSelections[pane])
    if (mode === "all") {
      nextMode = "subset"
      next = new Set(allIds)
    } else if (mode === "none") {
      nextMode = "subset"
      next = new Set()
    }
    const wasSelected = next.has(option.id)
    if (wasSelected) {
      next.delete(option.id)
    } else {
      next.add(option.id)
    }
    if (next.size === 0) {
      nextMode = "none"
    } else if (allIds.length > 0 && allIds.every((id) => next.has(id))) {
      nextMode = "all"
      next = new Set()
    } else {
      nextMode = "subset"
    }
    log("state", `toggleListFilterSelection: option=${option.id} ${wasSelected ? "deselected" : "selected"} total=${next.size}`)
    options.setUi("listFilterMode", pane, nextMode)
    options.setUi("listFilterSelections", pane, next)
    persistListFilterState()
  }

  const selectAllListFilterSelection = (): void => {
    const pane = options.ui.listFilterPane as ListPane
    const isAll = options.ui.listFilterMode[pane] === "all"
    const nextMode = isAll ? "none" : "all"
    options.setUi("listFilterMode", pane, nextMode)
    options.setUi("listFilterSelections", pane, new Set<string>())
    log("state", `selectAllListFilterSelection: mode=${nextMode}`)
    persistListFilterState()
  }

  const openSettingsModal = (): void => {
    const settingsOptions: Array<Mode | "reset"> = ["prod", "dev", "local", "reset"]
    options.setUi("settingsOptions", settingsOptions)
    const selected = options.ui.settingsMode ?? "reset"
    setSettingsCursor(Math.max(
      0,
      settingsOptions.indexOf(selected),
    ))
    options.openOverlayModal("settings")
    void readPersistedSettings()
      .then((settings) => {
        options.setUi("settingsKeys", settings.keys)
      })
      .catch(() => undefined)
  }

  const moveSettingsCursor = (delta: number): void => {
    const total = options.ui.settingsOptions.length
    setSettingsCursor((cur) => moveSelectionIndex(cur, delta, total))
  }

  const selectSettingsBackend = async (): Promise<void> => {
    const selectedMode = options.ui.settingsOptions[settingsCursor()]
    if (!selectedMode) return
    if (selectedMode === "reset") {
      options.setUi("settingsMode", null)
      await persistModeSelection(null)
      options.closeActiveModal()
      options.setData("status", "Mode selection cleared.")
      return
    }

    const urls = modeUrls[selectedMode]
    if (!urls.backendUrl || !urls.frontendUrl) {
      options.setData("status", `Missing URLs for ${selectedMode}.`)
      return
    }

    options.setUi("settingsMode", selectedMode)
    options.setUi("currentMode", selectedMode)
    switchMode(selectedMode)
    await persistModeSelection(selectedMode)
    const settings = await readPersistedSettings()
    const listFilters = settings.listFilters?.[selectedMode]
    if (listFilters) {
      options.setUi("listFilterMode", ListPane.Jobs, listFilters[ListPane.Jobs].mode)
      options.setUi("listFilterSelections", ListPane.Jobs, new Set(listFilters[ListPane.Jobs].selections))
      options.setUi("listFilterMode", ListPane.Logs, listFilters[ListPane.Logs].mode)
      options.setUi("listFilterSelections", ListPane.Logs, new Set(listFilters[ListPane.Logs].selections))
    }
    process.env.SYNTH_API_KEY = settings.keys[selectedMode] || ""

    if (!process.env.SYNTH_API_KEY) {
      options.closeActiveModal()
      options.setData("status", "Sign in required")
      options.promptLogin()
      return
    }

    options.closeActiveModal()
    options.setData("status", `Switching to ${selectedMode}...`)
    await options.refreshData()
  }

  const openUsageModal = (): void => {
    options.setUi("usageModalOffset", 0)
    setUsageData(null)
    options.openOverlayModal("usage")
    options.runAction("usage", () => fetchUsageData())
  }

  const openMetricsModal = (): void => {
    options.setUi("metricsModalOffset", 0)
    options.openOverlayModal("metrics")
  }

  const openMetricsAndFetch = (): void => {
    openMetricsModal()
    options.runAction("metrics", async () => {
      await fetchMetrics(options.ctx)
    })
  }

  const fetchUsageData = async (): Promise<void> => {
    try {
      const response = await apiGet("/usage-plan", { version: "v1" })
      const data: UsageData = {
        plan_type: response.plan_type as UsageData["plan_type"],
        status: response.status as UsageData["status"],
        access_tier: response.access_tier ?? "alpha",
        rollout_credits_balance_usd: response.rollout_credits_balance_usd ?? null,
        rollout_credits_used_this_period_usd: response.rollout_credits_used_this_period_usd ?? null,
        byok_providers: response.byok_providers || [],
        limits: {
          monthly_rollout_credits_usd: response.limits?.monthly_rollout_credits_usd ?? 0,
          max_overdraft_usd: response.limits?.max_overdraft_usd ?? 0,
          unlimited_non_rollout: response.limits?.unlimited_non_rollout ?? false,
          team_features_enabled: response.limits?.team_features_enabled ?? false,
          byok_enabled: response.limits?.byok_enabled ?? false,
        },
        usage_summary: response.usage_summary
          ? {
              total_cost_usd: response.usage_summary.total_cost_usd ?? 0,
              total_charged_usd: response.usage_summary.total_charged_usd ?? 0,
              total_uncharged_usd: response.usage_summary.total_uncharged_usd ?? 0,
              by_type: response.usage_summary.by_type || [],
            }
          : undefined,
      }
      setUsageData(data)
    } catch (err: any) {
      if (isAbortError(err)) return
      setUsageData({
        plan_type: "free",
        status: "active",
        rollout_credits_balance_usd: null,
        rollout_credits_used_this_period_usd: null,
        byok_providers: [],
        limits: {
          monthly_rollout_credits_usd: 0,
          max_overdraft_usd: 0,
          unlimited_non_rollout: false,
          team_features_enabled: false,
          byok_enabled: false,
        },
      })
      options.setData("lastError", `Usage fetch failed: ${err?.message || "Unknown"}`)
    }
  }

  const openUsageBilling = (): void => {
    try {
      const frontendUrl = process.env.SYNTH_FRONTEND_URL || ""
      const usageUrl = `${frontendUrl}/usage`
      openBrowser(usageUrl)
      options.setData("status", `Opened: ${usageUrl}`)
    } catch (err: any) {
      options.setData("status", `Failed to open browser: ${err?.message || "Unknown"}`)
    }
  }

  const openTaskAppsModal = (): void => {
    options.setUi("taskAppsModalOffset", 0)
    options.setUi("taskAppsModalSelectedIndex", 0)
    options.openOverlayModal("task-apps")
    options.runAction("task-apps-refresh", () => refreshTaskApps())
  }

  const refreshTaskApps = async (): Promise<void> => {
    options.setData("status", "Loading task apps...")
    const ok = await refreshTunnels(options.ctx)
    if (!ok || isAborted()) return
    await refreshTunnelHealth(options.ctx)
    if (!isAborted()) {
      options.setData("status", "Task apps updated")
    }
  }

  const moveTaskAppsSelection = (delta: number): void => {
    const activeTunnels = options.data.tunnels.filter((t) => t.status === "active" && !t.deleted_at)
    const total = activeTunnels.length
    const nextIndex = moveSelectionIndex(
      options.ui.taskAppsModalSelectedIndex || 0,
      delta,
      total,
    )
    options.setUi("taskAppsModalSelectedIndex", nextIndex)
  }

  const copySelectedTunnelUrl = async (): Promise<void> => {
    const activeTunnels = options.data.tunnels.filter((t) => t.status === "active" && !t.deleted_at)
    const tunnel = activeTunnels[options.ui.taskAppsModalSelectedIndex || 0]
    if (!tunnel) return
    const hostname = tunnel.hostname.replace(/^https?:\/\//, "")
    const url = `https://${hostname}`
    await copyToClipboard(url)
    options.setData("status", `Copied: ${url}`)
  }

  const openSessionsModal = (): void => {
    setSessionsSelectedIndex(0)
    setSessionsScrollOffset(0)
    options.openOverlayModal("sessions")
    options.runAction("sessions-refresh", () => refreshSessionsModal())
  }

  const moveSessionsSelection = (delta: number): void => {
    const sessions = sessionsCache()
    const active = sessions.filter(
      (s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting",
    )
    const total = active.length
    setSessionsSelectedIndex((current) => moveSelectionIndex(current, delta, total))
  }

  const refreshSessionsModal = async (): Promise<void> => {
    options.setData("status", "Loading sessions...")
    try {
      const sessions = await fetchSessions()
      if (isAborted()) return
      options.setData("sessions", sessions)
      setSessionsCache(sessions)
      await refreshSessionHealth(sessions)
    } catch (err: any) {
      if (isAbortError(err)) return
      options.setData("lastError", err?.message || "Failed to load sessions")
    }
  }

  const refreshSessionHealth = async (sessions: SessionRecord[]): Promise<void> => {
    const next = new Map(sessionsHealthCache())
    const activeSessions = sessions.filter(
      (s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting",
    )
    for (const session of activeSessions) {
      if (isAborted()) return
      const result = await checkSessionHealth(session, 5000)
      next.set(session.session_id, result)
      const nextMap = new Map(next)
      setSessionsHealthCache(nextMap)
      options.setData("sessionHealthResults", nextMap)
    }
  }

  const connectLocalSession = async (): Promise<void> => {
    const opencodeUrl = options.ui.openCodeUrl
    if (!opencodeUrl) {
      options.setData("lastError", "OpenCode server not started")
      options.setData("status", "No OpenCode server URL available - server may not be running")
      return
    }

    options.setData("status", `Connecting to OpenCode at ${opencodeUrl}...`)

    const result = await connectLocalOpenCodeSession(opencodeUrl, 5000)
    if (result.aborted) return
    if (!result.ok) {
      options.setData("lastError", result.error)
      options.setData(
        "status",
        result.health?.healthy
          ? "Session creation failed"
          : `Connection failed - is OpenCode running at ${opencodeUrl}?`,
      )
      return
    }

    const sessionId = result.session.session_id
    const nextSessions = [result.session, ...sessionsCache().filter((s) => s.session_id !== sessionId)]
    setSessionsCache(nextSessions)
    options.setData("sessions", nextSessions)
    const nextHealth = new Map(sessionsHealthCache())
    nextHealth.set(sessionId, result.health)
    options.setData("sessionHealthResults", new Map(nextHealth))
    setSessionsHealthCache(nextHealth)

    options.setUi("openCodeSessionId", sessionId)
    options.setData("status", `Connected to OpenCode at ${opencodeUrl} | Session: ${sessionId}`)
  }

  const disconnectSelectedSession = async (): Promise<void> => {
    const sessions = sessionsCache()
    const active = sessions.filter(
      (s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting",
    )
    const session = active[sessionsSelectedIndex()]
    if (!session) return

    options.setData("status", `Disconnecting ${session.session_id}...`)

    try {
      const result = await disconnectSession(session.session_id)
      if (result.disconnected) {
        options.setData("status", `Disconnected from ${session.session_id}`)
        if (options.ui.openCodeSessionId === session.session_id) {
          options.setUi("openCodeSessionId", null)
        }
        await refreshSessionsModal()
      } else {
        options.setData("status", "Disconnect failed")
      }
    } catch (err: any) {
      options.setData("lastError", err?.message || "Failed to disconnect")
      options.setData("status", "Disconnect failed")
    }
  }

  const copySelectedSessionUrl = async (): Promise<void> => {
    const sessions = sessionsCache()
    const active = sessions.filter(
      (s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting",
    )
    const session = active[sessionsSelectedIndex()]
    if (!session) return
    const url = session.opencode_url || session.access_url || ""
    if (!url) return
    await copyToClipboard(url)
    options.setData("status", `Copied: ${url}`)
  }

  const selectSession = (): void => {
    const sessions = sessionsCache()
    const active = sessions.filter(
      (s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting",
    )
    const session = active[sessionsSelectedIndex()]
    if (!session) return
    options.setUi("openCodeSessionId", session.session_id)
    options.setData("sessions", (existing) => {
      if (existing.find((s) => s.session_id === session.session_id)) {
        return existing
      }
      return [...existing, session]
    })
    options.setData("status", `Selected session: ${session.session_id}`)
    options.closeActiveModal()
  }

  const openTracesModal = (): void => {
    options.openOverlayModal("traces")
    void options.ensureTraceViewerModal()
  }

  return {
    usageData,
    settingsCursor,
    sessionsCache,
    sessionsHealthCache,
    sessionsSelectedIndex,
    sessionsScrollOffset,
    openFilterModal,
    applyFilterModal,
    openSnapshotModal,
    applySnapshotModal,
    openConfigModal,
    openResultsModal,
    openCandidatesForGeneration,
    openProfileModal,
    openListFilterModal,
    moveListFilter,
    toggleListFilterSelection,
    selectAllListFilterSelection,
    openSettingsModal,
    moveSettingsCursor,
    selectSettingsBackend,
    openUsageModal,
    openMetricsModal,
    openMetricsAndFetch,
    openUsageBilling,
    openTaskAppsModal,
    moveTaskAppsSelection,
    copySelectedTunnelUrl,
    openSessionsModal,
    moveSessionsSelection,
    refreshSessionsModal,
    connectLocalSession,
    disconnectSelectedSession,
    copySelectedSessionUrl,
    selectSession,
    openTracesModal,
  }
}
