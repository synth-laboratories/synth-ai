import { render, useKeyboard, useRenderer, useTerminalDimensions } from "@opentui/solid"
import { ChatPane } from "./opencode"
import { ErrorBoundary, Show, createEffect, createMemo, createSignal, onCleanup } from "solid-js"
import fs from "node:fs"
import path from "node:path"

import { computeLayoutMetrics, defaultLayoutSpec } from "./layout"
import { useSolidData } from "./data"
import { COLORS } from "./theme"
import { KeyHint } from "./components/KeyHint"
import { JobsList } from "./ui/list-panels/JobsList"
import { LogsList } from "./ui/list-panels/LogsList"
import { JobsDetail } from "./ui/detail-panels/JobsDetail"
import { LogsDetail } from "./ui/detail-panels/LogsDetail"
import { useJobDetailsStream } from "./api/useJobDetailsStream"
import type { JobDetailsStreamEvent } from "./api/job-details-stream"
import { CreateJobModal, type JobCreatedInfo } from "./modals/CreateJobModal"
import { CandidatesModal } from "./modals/CandidatesModal"
import { TraceViewerModal } from "./modals/TraceViewerModal"
import { scanMultipleDirectories, type ScannedLocalAPI } from "./utils/localapi-scanner"
import { toDisplayPath } from "./utils/files"

import { getFilteredEvents } from "../formatters"
import { formatMetricsCharts } from "../formatters/metrics"
import { buildJobStatusOptions, getFilteredJobs } from "../selectors/jobs"
import { cancelSelected, fetchArtifacts, fetchMetrics, selectJob } from "../api/jobs"
import { apiGet, apiGetV1 } from "../api/client"
import { fetchSessions, disconnectSession, checkSessionHealth } from "../api/sessions"
import { openBrowser, runDeviceCodeAuth, type AuthStatus } from "../auth"
import { copyToClipboard } from "../utils/clipboard"
import { scanEnvKeys } from "../utils/env"
import { clearLoggedOutMarker, deleteSavedApiKey, saveApiKey, setLoggedOutMarker } from "../utils/logout-marker"
import { persistSettings } from "../persistence/settings"
import { listLogFiles, moveLogSelection } from "../ui/logs"
import { moveEventSelection } from "../ui/events"
import { refreshEvents } from "../api/events"
import type { JobEvent } from "../tui_data"
import type { SessionHealthResult, SessionRecord, Snapshot, TunnelHealthResult, TunnelRecord } from "../types"
import { focusManager } from "../focus"
import {
  backendConfigs,
  frontendKeys,
  frontendKeySources,
  getFrontendUrl,
  getFrontendUrlId,
  getKeyForBackend,
} from "../state/app-state"
import { pollingState, clearEventsTimer, clearJobsTimer } from "../state/polling"
import { shutdown } from "../lifecycle"

type ModalState =
  | {
      type: "event"
      title: string
      raw: string
      offset: number
      fullscreen?: boolean
    }
  | {
      type: "log"
      title: string
      raw: string
      offset: number
      tail: boolean
      path: string
      fullscreen?: boolean
    }

type ActiveModal =
  | "filter"
  | "job-filter"
  | "snapshot"
  | "key"
  | "settings"
  | "env-key"
  | "usage"
  | "task-apps"
  | "sessions"
  | "config"
  | "results"
  | "profile"
  | "urls"
  | "login"
  | "metrics"
  | "traces"

type UsageData = {
  plan_type: "free" | "pro" | "team" | "byok"
  status: "active" | "cancelled" | "past_due" | "trialing" | "inactive"
  access_tier?: string | null
  rollout_credits_balance_usd?: number | null
  rollout_credits_used_this_period_usd?: number | null
  byok_providers?: string[]
  limits: {
    monthly_rollout_credits_usd: number
    max_overdraft_usd: number
    unlimited_non_rollout: boolean
    team_features_enabled: boolean
    byok_enabled: boolean
  }
  usage_summary?: {
    total_cost_usd: number
    total_charged_usd: number
    total_uncharged_usd: number
    by_type: Array<{
      usage_type: string
      total_cost_usd: number
      charged_cost_usd: number
      uncharged_cost_usd: number
      event_count: number
      byok_event_count: number
    }>
  }
}

export async function runSolidApp(): Promise<void> {
  return new Promise<void>((resolve) => {
    render(
      () => <SolidShell onExit={resolve} />,
      {
        targetFps: 30,
        exitOnCtrlC: true,
        useKittyKeyboard: {},
      },
    )
  })
}

function SolidShell(props: { onExit?: () => void }) {
  const { onExit } = props
  const dimensions = useTerminalDimensions()
  const renderer = useRenderer()

  // Set global renderer for OpenCode embed to find
  ;(globalThis as any).__OPENCODE_EMBED_RENDERER__ = renderer
  const layout = createMemo(() =>
    computeLayoutMetrics(dimensions().width, dimensions().height),
  )
  const data = useSolidData()
  const appState = data.ctx.state.appState
  const snapshot = data.ctx.state.snapshot
  const snapshotMemo = createMemo(() => {
    data.version()
    // Important: return a new reference so downstream memos (e.g. JobsDetail)
    // recompute when snapshot fields are mutated in-place.
    return { ...data.ctx.state.snapshot }
  })
  const [selectedIndex, setSelectedIndex] = createSignal(0)
  const jobs = createMemo(() => {
    data.version()
    return data.ctx.state.snapshot.jobs
  })
  const activePane = createMemo(() => {
    data.version()
    return data.ctx.state.appState.activePane
  })
  const principalPane = createMemo(() => {
    data.version()
    return data.ctx.state.appState.principalPane
  })
  const opencodeUrl = createMemo(() => {
    data.version()
    return (
      data.ctx.state.appState.openCodeUrl ||
      process.env.OPENCODE_URL ||
      "http://localhost:3000"
    )
  })
  const opencodeSessionId = createMemo(() => {
    data.version()
    return data.ctx.state.appState.openCodeSessionId ?? undefined
  })
  createEffect(() => {
    data.version()
    const sessionID = opencodeSessionId()
    process.env.OPENCODE_ROUTE = JSON.stringify(
      sessionID ? { type: "session", sessionID } : { type: "home" },
    )
  })
  const opencodeDimensions = createMemo(() => ({
    width: Math.max(1, layout().detailWidth),
    height: Math.max(1, layout().contentHeight),
  }))
  const events = createMemo(() => {
    data.version()
    return getFilteredEvents(
      data.ctx.state.snapshot.events,
      data.ctx.state.appState.eventFilter,
    )
  })
  const eventWindow = createMemo(() => {
    data.version()
    const list = events()
    const total = list.length
    const visibleTarget = Math.max(1, data.ctx.state.config.eventVisibleCount)
    const reserved = 16
    const available = Math.max(1, layout().contentHeight - reserved)
    const visible = Math.max(1, Math.min(visibleTarget, available))
    const selected = clamp(
      data.ctx.state.appState.selectedEventIndex,
      0,
      Math.max(0, total - 1),
    )
    let windowStart = clamp(
      data.ctx.state.appState.eventWindowStart,
      0,
      Math.max(0, total - visible),
    )
    if (selected < windowStart) {
      windowStart = selected
    } else if (selected >= windowStart + visible) {
      windowStart = selected - visible + 1
    }
    return {
      total,
      visible,
      selected,
      windowStart,
      slice: list.slice(windowStart, windowStart + visible),
    }
  })

  // Selected job ID for SSE streaming
  const selectedJobId = createMemo(() => {
    data.version()
    return data.ctx.state.snapshot.selectedJob?.job_id ?? null
  })

  // Track highest seen event seq for incremental SSE updates
  const [lastSeenSeq, setLastSeenSeq] = createSignal(0)

  // Subscribe to real-time job details updates via SSE
  useJobDetailsStream({
    jobId: selectedJobId,
    sinceSeq: lastSeenSeq,
    enabled: () => principalPane() === "jobs" && activePane() !== "logs",
    onEvent: (event: JobDetailsStreamEvent) => {
      // Update highest seen seq
      if (event.seq > lastSeenSeq()) {
        setLastSeenSeq(event.seq)
      }
      // Keep polling cursor in sync with SSE cursor so refreshEvents() doesn't refetch from 0.
      appState.lastSeq = Math.max(appState.lastSeq || 0, event.seq)

      // Convert SSE event to JobEvent format and add to snapshot
      const jobEvent: JobEvent = {
        seq: event.seq,
        type: event.type,
        message: event.message,
        data: event.data as JobEvent["data"],
        timestamp: new Date(event.ts).toISOString(),
      }

      // Add event to snapshot (avoiding duplicates by seq)
      const existingSeqs = new Set(snapshot.events.map(e => e.seq))
      if (!existingSeqs.has(jobEvent.seq)) {
        snapshot.events = [...snapshot.events, jobEvent].sort((a, b) => a.seq - b.seq)
        data.ctx.render()
      }
    },
    onError: (error) => {
      // Log but don't show to user - polling will still work as fallback
      console.error("Job details SSE error:", error.message)
    },
  })

  // Poll/backfill events to avoid gaps when SSE drops or when selecting an older job.
  // We keep this lightweight (small interval + bounded backfill loop).
  createEffect(() => {
    const jobId = selectedJobId()
    const enabled = principalPane() === "jobs" && activePane() !== "logs"
    if (!jobId || !enabled) return

    let cancelled = false

    async function backfillOnce(): Promise<void> {
      // Pull up to ~10 pages (2000 events) max per selection, but stop early if no progress.
      for (let i = 0; i < 10; i++) {
        const beforeSeq = appState.lastSeq
        const beforeLen = snapshot.events.length
        const ok = await refreshEvents(data.ctx)
        if (!ok) break
        if (cancelled) return
        if (snapshot.events.length === beforeLen && appState.lastSeq === beforeSeq) break
      }
      if (!cancelled) data.ctx.render()
    }

    void backfillOnce()

    const interval = setInterval(() => {
      void refreshEvents(data.ctx).then((ok) => {
        if (ok && !cancelled) data.ctx.render()
      })
    }, 3000)
    onCleanup(() => {
      cancelled = true
      clearInterval(interval)
    })
  })
  const logFiles = createMemo(() => {
    data.version()
    return listLogFiles()
  })
  const logsWindow = createMemo(() => {
    data.version()
    const files = logFiles()
    const total = files.length
    const visible = Math.max(1, layout().contentHeight - 4)
    const selected = clamp(
      data.ctx.state.appState.logsSelectedIndex,
      0,
      Math.max(0, total - 1),
    )
    let windowStart = clamp(
      data.ctx.state.appState.logsWindowStart,
      0,
      Math.max(0, total - visible),
    )
    if (selected < windowStart) {
      windowStart = selected
    } else if (selected >= windowStart + visible) {
      windowStart = selected - visible + 1
    }
    return {
      total,
      visible,
      selected,
      windowStart,
      slice: files.slice(windowStart, windowStart + visible),
    }
  })
  const logsTitle = createMemo(() => {
    const window = logsWindow()
    if (window.total > window.visible) {
      const end = Math.min(window.windowStart + window.visible, window.total)
      return `Logs (files) [${window.windowStart + 1}-${end}/${window.total}]`
    }
    return "Logs (files)"
  })
  const logsView = createMemo(() => {
    data.version()
    const files = logFiles()
    const selected = appState.logsSelectedIndex
    if (selected < 0 || selected >= files.length) {
      return { lines: [], visible: [] }
    }
    const file = files[selected]
    let content = ""
    try {
      content = fs.readFileSync(file.path, "utf8")
    } catch (err: any) {
      content = `Failed to read ${file.path}: ${err?.message || String(err)}`
    }
    const lines = content.split("\n")
    const visibleHeight = Math.max(1, layout().contentHeight - 4)
    const offset = appState.logsWindowStart ?? 0
    const clampedOffset = Math.max(0, Math.min(offset, Math.max(0, lines.length - visibleHeight)))
    const visible = lines.slice(clampedOffset, clampedOffset + visibleHeight)
    return { lines, visible }
  })
  const openCodeStatus = createMemo(() => {
    data.version()
    return data.ctx.state.appState.openCodeStatus
  })
  const statusText = createMemo(() => {
    data.version()
    const status = data.ctx.state.snapshot.status || "Ready"
    const health = data.ctx.state.appState.healthStatus || "unknown"
    const openCode = openCodeStatus()
    const base = `${status} | health=${health} | pane=${data.ctx.state.appState.activePane}`
    const session = opencodeSessionId()
    if (!openCode) return base
    return session ? `${base} | opencode=${openCode} (session ${session.slice(-6)})` : `${base} | opencode=${openCode}`
  })
  const lastError = createMemo(() => {
    data.version()
    return data.ctx.state.snapshot.lastError
  })
  const [modal, setModal] = createSignal<ModalState | null>(null)
  const [activeModal, setActiveModal] = createSignal<ActiveModal | null>(null)
  const [modalInputValue, setModalInputValue] = createSignal("")
  const [usageData, setUsageData] = createSignal<UsageData | null>(null)
  const [sessionsSelectedIndex, setSessionsSelectedIndex] = createSignal(0)
  const [sessionsScrollOffset, setSessionsScrollOffset] = createSignal(0)
  const [sessionsCache, setSessionsCache] = createSignal<SessionRecord[]>([])
  const [sessionsHealthCache, setSessionsHealthCache] = createSignal<Map<string, SessionHealthResult>>(new Map())
  const [loginStatus, setLoginStatus] = createSignal<AuthStatus>({ state: "idle" })
  const [loginInProgress, setLoginInProgress] = createSignal(false)
  const [settingsCursor, setSettingsCursor] = createSignal(0)
  const [showCreateJobModal, setShowCreateJobModal] = createSignal(false)
  const [scannedLocalAPIs, setScannedLocalAPIs] = createSignal<ScannedLocalAPI[]>([])
  
  // Scan for LocalAPI files when modal opens
  createEffect(() => {
    if (showCreateJobModal()) {
      // Scan CWD and common locations
      const dirsToScan = [
        process.cwd(),
        // Add more directories as needed
      ]
      const found = scanMultipleDirectories(dirsToScan)
      setScannedLocalAPIs(found)
    }
  })
  
  const localApiFiles = createMemo(() => {
    return scannedLocalAPIs().map(api => toDisplayPath(api.filepath))
  })
  const modalLayout = createMemo(() => {
    const state = modal()
    if (state?.fullscreen) {
      return {
        width: Math.max(1, layout().totalWidth),
        height: Math.max(1, layout().totalHeight),
        left: 0,
        top: 0,
      }
    }
    const width = Math.min(100, Math.max(40, layout().totalWidth - 4))
    const height = Math.min(26, Math.max(12, layout().totalHeight - 6))
    const left = Math.max(0, Math.floor((layout().totalWidth - width) / 2))
    const top = Math.max(1, Math.floor((layout().totalHeight - height) / 2))
    return { width, height, left, top }
  })
  const modalBodyHeight = createMemo(() => Math.max(1, modalLayout().height - 4))
  const modalLines = createMemo(() => {
    const state = modal()
    if (!state) return []
    const maxWidth = Math.max(10, modalLayout().width - 4)
    return wrapText(state.raw, maxWidth)
  })
  const modalView = createMemo(() => {
    const state = modal()
    if (!state) return null
    const lines = modalLines()
    const maxOffset = Math.max(0, lines.length - modalBodyHeight())
    const offset = clamp(state.offset, 0, maxOffset)
    const resolvedOffset = state.type === "log" && state.tail ? maxOffset : offset
    const visible = lines.slice(resolvedOffset, resolvedOffset + modalBodyHeight())
    return {
      total: lines.length,
      offset: resolvedOffset,
      maxOffset,
      visible,
      visibleCount: modalBodyHeight(),
    }
  })
  const modalHint = createMemo(() => {
    const state = modal()
    const view = modalView()
    if (!state || !view) return ""
    const range = view.total > view.visibleCount
      ? `[${view.offset + 1}-${Math.min(view.offset + view.visible.length, view.total)}/${view.total}] `
      : ""
    const fullscreenHint = "Shift+F fullscreen | "
    if (state.type === "log") {
      const tail = state.tail ? " [TAIL]" : ""
      return `${range}${fullscreenHint}j/k scroll | t tail${tail} | y copy | q close`
    }
    return `${range}${fullscreenHint}j/k scroll | q close`
  })

  function buildScrollableModal(raw: string, width: number, height: number, offset: number) {
    // Account for borders (2) + padding left/right (4) = 6 chars of horizontal chrome
    const maxWidth = Math.max(10, width - 6)
    const lines = wrapText(raw, maxWidth)
    // Account for: 2 (borders) + 2 (padding top/bottom) + 1 (title) + 1 (hint) = 6 lines of chrome
    const bodyHeight = Math.max(1, height - 6)
    const maxOffset = Math.max(0, lines.length - bodyHeight)
    const clamped = clamp(offset, 0, maxOffset)
    const visible = lines.slice(clamped, clamped + bodyHeight)
    return { lines, visible, offset: clamped, maxOffset, bodyHeight }
  }

  createEffect(() => {
    const count = jobs().length
    if (count === 0) {
      setSelectedIndex(0)
      return
    }
    if (selectedIndex() >= count) {
      setSelectedIndex(count - 1)
    }
  })

  // Auto-select job when highlighted index changes (only in jobs pane)
  createEffect(() => {
    const pane = activePane()
    if (pane !== "jobs") return
    
    const index = selectedIndex()
    const jobsList = jobs()
    const currentSnapshot = snapshotMemo()
    const currentSelected = currentSnapshot.selectedJob
    
    // Only proceed if we have jobs and a valid index
    if (jobsList.length === 0 || index < 0 || index >= jobsList.length) {
      return
    }
    
    const job = jobsList[index]
    if (!job?.job_id) {
      return
    }
    
    // Auto-select if no job is currently selected, or if it's a different job
    if (!currentSelected || currentSelected.job_id !== job.job_id) {
      void data.select(job.job_id)
    }
  })

  createEffect(() => {
    const current = modal()
    if (!current || current.type !== "log") return
    const filePath = current.path
    const timer = setInterval(() => {
      const raw = readLogFile(filePath)
      setModal((prev) => {
        if (!prev || prev.type !== "log") return prev
        return { ...prev, raw }
      })
    }, 1000)
    onCleanup(() => clearInterval(timer))
  })

  createEffect(() => {
    const state = modal()
    if (!state) return
    const lines = modalLines()
    const maxOffset = Math.max(0, lines.length - modalBodyHeight())
    let nextOffset = state.offset
    if (state.type === "log" && state.tail) {
      nextOffset = maxOffset
    }
    nextOffset = clamp(nextOffset, 0, maxOffset)
    if (nextOffset !== state.offset) {
      setModal({ ...state, offset: nextOffset })
    }
  })

  createEffect(() => {
    data.version()
    if (!process.env.SYNTH_API_KEY && snapshot.status === "Sign in required" && activeModal() !== "login") {
      setLoginStatus({ state: "idle" })
      setLoginInProgress(false)
      setActiveModal("login")
    }
  })

  function openEventModal(event: JobEvent): void {
    const detail = event.message ?? formatEventDetail(event.data)
    const header = `${event.type} (seq ${event.seq})`
    const raw = detail ? `${header}\n\n${detail}` : header
    setModal({
      type: "event",
      title: "Event Detail",
      raw,
      offset: 0,
    })
  }

  function readLogFile(filePath: string): string {
    try {
      return fs.readFileSync(filePath, "utf8")
    } catch (err: any) {
      return `Failed to read ${filePath}: ${err?.message || String(err)}`
    }
  }

  function openLogModal(filePath: string): void {
    const raw = readLogFile(filePath)
    setModal({
      type: "log",
      title: `Log: ${path.basename(filePath)}`,
      raw,
      offset: 0,
      tail: true,
      path: filePath,
    })
  }

  function closeActiveModal(): void {
    // Some modals use an <input>. If it keeps focus after closing, it can swallow
    // subsequent keypresses (e.g. Settings navigation) depending on the runtime/terminal.
    try {
      if (modalInputRef && typeof modalInputRef.blur === "function") {
        modalInputRef.blur()
      }
    } catch {
      // Best-effort blur only.
    }
    setActiveModal(null)
    setModalInputValue("")
  }

  function openFilterModal(): void {
    setModalInputValue(appState.eventFilter)
    setActiveModal("filter")
  }

  function applyFilterModal(): void {
    appState.eventFilter = modalInputValue().trim()
    closeActiveModal()
    data.ctx.render()
  }

  function openSnapshotModal(): void {
    setModalInputValue("")
    setActiveModal("snapshot")
  }

  async function applySnapshotModal(): Promise<void> {
    const trimmed = modalInputValue().trim()
    if (!trimmed) {
      closeActiveModal()
      return
    }
    const job = snapshot.selectedJob
    if (!job) {
      closeActiveModal()
      return
    }
    closeActiveModal()
    try {
      await apiGet(`/prompt-learning/online/jobs/${job.job_id}/snapshots/${trimmed}`)
      snapshot.status = `Snapshot ${trimmed} fetched`
    } catch (err: any) {
      snapshot.lastError = err?.message || "Snapshot fetch failed"
    }
    data.ctx.render()
  }

  function openKeyModal(): void {
    setModalInputValue("")
    setActiveModal("key")
  }

  function applyKeyModal(): void {
    const trimmed = modalInputValue().trim()
    if (!trimmed) {
      closeActiveModal()
      return
    }
    process.env.SYNTH_API_KEY = trimmed
    snapshot.status = "API key updated"
    closeActiveModal()
    data.ctx.render()
  }

  function pasteKeyModal(): void {
    try {
      if (process.platform !== "darwin") return
      const result = require("child_process").spawnSync("pbpaste", [], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      })
      if (result.status !== 0) return
      const text = result.stdout ? String(result.stdout).replace(/\s+/g, "") : ""
      if (!text) return
      setModalInputValue((current) => `${current}${text}`)
      if (modalInputRef) {
        modalInputRef.value = `${modalInputValue()}${text}`
      }
    } catch {
      // ignore
    }
  }

  function openConfigModal(): void {
    appState.configModalOffset = 0
    setActiveModal("config")
  }

  function openResultsModal(): void {
    setModal(null)
    setActiveModal("results")
  }

  function openProfileModal(): void {
    setActiveModal("profile")
  }

  function openUrlsModal(): void {
    setActiveModal("urls")
  }

  function openJobFilterModal(): void {
    appState.jobFilterOptions = buildJobStatusOptions(snapshot.jobs)
    appState.jobFilterCursor = 0
    appState.jobFilterWindowStart = 0
    setActiveModal("job-filter")
  }

  function moveJobFilter(delta: number): void {
    const max = Math.max(0, appState.jobFilterOptions.length - 1)
    appState.jobFilterCursor = clamp(appState.jobFilterCursor + delta, 0, max)
    if (appState.jobFilterCursor < appState.jobFilterWindowStart) {
      appState.jobFilterWindowStart = appState.jobFilterCursor
    } else if (appState.jobFilterCursor >= appState.jobFilterWindowStart + data.ctx.state.config.jobFilterVisibleCount) {
      appState.jobFilterWindowStart = appState.jobFilterCursor - data.ctx.state.config.jobFilterVisibleCount + 1
    }
    data.ctx.render()
  }

  function toggleJobFilterSelection(): void {
    const option = appState.jobFilterOptions[appState.jobFilterCursor]
    if (!option) return
    if (appState.jobStatusFilter.has(option.status)) {
      appState.jobStatusFilter.delete(option.status)
    } else {
      appState.jobStatusFilter.add(option.status)
    }
    applyJobFilterSelection()
  }

  function clearJobFilterSelection(): void {
    appState.jobStatusFilter.clear()
    applyJobFilterSelection()
  }

  function applyJobFilterSelection(): void {
    const filteredJobs = getFilteredJobs(snapshot.jobs, appState.jobStatusFilter)
    if (!filteredJobs.length) {
      snapshot.selectedJob = null
      snapshot.events = []
      snapshot.metrics = {}
      snapshot.bestSnapshotId = null
      snapshot.bestSnapshot = null
      snapshot.allCandidates = []
      appState.selectedEventIndex = 0
      appState.eventWindowStart = 0
      snapshot.status = appState.jobStatusFilter.size
        ? "No jobs with selected status"
        : "No prompt-learning jobs found"
      data.ctx.render()
      return
    }
    if (!snapshot.selectedJob || !filteredJobs.some((job) => job.job_id === snapshot.selectedJob?.job_id)) {
      void selectJob(data.ctx, filteredJobs[0].job_id).then(() => data.ctx.render()).catch(() => {})
      return
    }
    data.ctx.render()
  }

  function openSettingsModal(): void {
    // Ensure any previously-focused modal input doesn't swallow navigation keys.
    try {
      if (modalInputRef && typeof modalInputRef.blur === "function") {
        modalInputRef.blur()
      }
    } catch {
      // Best-effort blur only.
    }
    appState.settingsOptions = [backendConfigs.prod, backendConfigs.dev, backendConfigs.local]
    setSettingsCursor(Math.max(
      0,
      appState.settingsOptions.findIndex((opt) => opt.id === appState.currentBackend),
    ))
    setActiveModal("settings")
  }

  function isUpKey(evt: any): boolean {
    const name = typeof evt?.name === "string" ? evt.name : ""
    return name === "up" || name === "arrowup" || name === "k"
  }

  function isDownKey(evt: any): boolean {
    const name = typeof evt?.name === "string" ? evt.name : ""
    return name === "down" || name === "arrowdown" || name === "j"
  }

  function moveSettingsCursor(delta: number): void {
    const max = Math.max(0, appState.settingsOptions.length - 1)
    setSettingsCursor((cur) => clamp(cur + delta, 0, max))
  }

  async function selectSettingsBackend(): Promise<void> {
    const selected = appState.settingsOptions[settingsCursor()]
    if (!selected) return
    appState.currentBackend = selected.id
    const baseUrl = selected.baseUrl.replace(/\/api$/, "")
    process.env.SYNTH_BACKEND_URL = baseUrl
    process.env.SYNTH_API_KEY = getKeyForBackend(selected.id) || ""

    closeActiveModal()
    snapshot.status = `Switching to ${selected.label}...`
    data.ctx.render()

    await persistSettings({
      settingsFilePath: data.ctx.state.config.settingsFilePath,
      getCurrentBackend: () => appState.currentBackend,
      getFrontendKey: (id) => frontendKeys[id],
      getFrontendKeySource: (id) => frontendKeySources[id],
    })
    await data.refresh()
  }

  async function rescanEnvKeys(): Promise<void> {
    appState.envKeyScanInProgress = true
    appState.envKeyError = null
    data.ctx.render()
    try {
      appState.envKeyOptions = await scanEnvKeys(data.ctx.state.config.envKeyScanRoot)
      appState.envKeyCursor = 0
      appState.envKeyWindowStart = 0
    } catch (err: any) {
      appState.envKeyError = err?.message || "Scan failed"
    } finally {
      appState.envKeyScanInProgress = false
      data.ctx.render()
    }
  }

  function openEnvKeyModal(): void {
    setActiveModal("env-key")
    void rescanEnvKeys()
  }

  function moveEnvKeyCursor(delta: number): void {
    const max = Math.max(0, appState.envKeyOptions.length - 1)
    appState.envKeyCursor = clamp(appState.envKeyCursor + delta, 0, max)
    if (appState.envKeyCursor < appState.envKeyWindowStart) {
      appState.envKeyWindowStart = appState.envKeyCursor
    } else if (appState.envKeyCursor >= appState.envKeyWindowStart + data.ctx.state.config.envKeyVisibleCount) {
      appState.envKeyWindowStart = appState.envKeyCursor - data.ctx.state.config.envKeyVisibleCount + 1
    }
    data.ctx.render()
  }

  async function selectEnvKey(): Promise<void> {
    const selected = appState.envKeyOptions[appState.envKeyCursor]
    if (!selected) {
      // Close modal when no keys available (pressing enter should dismiss)
      closeActiveModal()
      return
    }
    const frontendUrlId = getFrontendUrlId(appState.currentBackend)
    frontendKeys[frontendUrlId] = selected.key
    frontendKeySources[frontendUrlId] = {
      sourcePath: selected.sources[0] || null,
      varName: selected.varNames[0] || null,
    }
    process.env.SYNTH_API_KEY = selected.key
    closeActiveModal()
    await persistSettings({
      settingsFilePath: data.ctx.state.config.settingsFilePath,
      getCurrentBackend: () => appState.currentBackend,
      getFrontendKey: (id) => frontendKeys[id],
      getFrontendKeySource: (id) => frontendKeySources[id],
    })
    snapshot.status = "API key loaded from env file"
    data.ctx.render()
  }

  function openUsageModal(): void {
    appState.usageModalOffset = 0
    setUsageData(null)
    setActiveModal("usage")
    void fetchUsageData()
  }

  function openMetricsModal(): void {
    appState.metricsModalOffset = 0
    setActiveModal("metrics")
  }

  async function fetchUsageData(): Promise<void> {
    try {
      const response = await apiGetV1("/usage-plan")
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
      snapshot.lastError = `Usage fetch failed: ${err?.message || "Unknown"}`
      data.ctx.render()
    }
  }

  function openUsageBilling(): void {
    try {
      const frontendUrl = getFrontendUrl(appState.currentBackend)
      const usageUrl = `${frontendUrl}/usage`
      openBrowser(usageUrl)
      snapshot.status = `Opened: ${usageUrl}`
    } catch (err: any) {
      snapshot.status = `Failed to open browser: ${err?.message || "Unknown"}`
    }
    data.ctx.render()
  }

  function openTaskAppsModal(): void {
    appState.taskAppsModalOffset = 0
    appState.taskAppsModalSelectedIndex = 0
    setActiveModal("task-apps")
  }

  function moveTaskAppsSelection(delta: number): void {
    const activeTunnels = snapshot.tunnels.filter((t) => t.status === "active" && !t.deleted_at)
    const maxIndex = Math.max(0, activeTunnels.length - 1)
    appState.taskAppsModalSelectedIndex = clamp(
      (appState.taskAppsModalSelectedIndex || 0) + delta,
      0,
      maxIndex,
    )
    data.ctx.render()
  }

  async function copySelectedTunnelUrl(): Promise<void> {
    const activeTunnels = snapshot.tunnels.filter((t) => t.status === "active" && !t.deleted_at)
    const tunnel = activeTunnels[appState.taskAppsModalSelectedIndex || 0]
    if (!tunnel) return
    const hostname = tunnel.hostname.replace(/^https?:\/\//, "")
    const url = `https://${hostname}`
    await copyToClipboard(url)
    snapshot.status = `Copied: ${url}`
    data.ctx.render()
  }

  function openSessionsModal(): void {
    setSessionsSelectedIndex(0)
    setSessionsScrollOffset(0)
    setActiveModal("sessions")
    void refreshSessionsModal()
  }

  function moveSessionsSelection(delta: number): void {
    const sessions = sessionsCache()
    const active = sessions.filter(
      (s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting",
    )
    const maxIndex = Math.max(0, active.length - 1)
    setSessionsSelectedIndex((current) => clamp(current + delta, 0, maxIndex))
  }

  async function refreshSessionsModal(): Promise<void> {
    snapshot.status = "Loading sessions..."
    data.ctx.render()
    try {
      const sessions = await fetchSessions()
      snapshot.sessions = sessions
      setSessionsCache(sessions)
      await refreshSessionHealth(sessions)
    } catch (err: any) {
      snapshot.lastError = err?.message || "Failed to load sessions"
      data.ctx.render()
    }
  }

  async function refreshSessionHealth(sessions: SessionRecord[]): Promise<void> {
    const next = new Map(sessionsHealthCache())
    const activeSessions = sessions.filter(
      (s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting",
    )
    for (const session of activeSessions) {
      const result = await checkSessionHealth(session)
      next.set(session.session_id, result)
      snapshot.sessionHealthResults.set(session.session_id, result)
      setSessionsHealthCache(new Map(next))
    }
    data.ctx.render()
  }

  async function connectLocalSession(): Promise<void> {
    const opencodeUrl = appState.openCodeUrl
    if (!opencodeUrl) {
      snapshot.lastError = "OpenCode server not started"
      snapshot.status = "No OpenCode server URL available - server may not be running"
      data.ctx.render()
      return
    }

    snapshot.status = `Connecting to OpenCode at ${opencodeUrl}...`
    data.ctx.render()

    const healthCheck = await checkSessionHealth({
      session_id: "local",
      container_id: "",
      state: "connecting",
      mode: "interactive",
      model: "gpt-4o-mini",
      access_url: opencodeUrl,
      tunnel_url: null,
      opencode_url: opencodeUrl,
      health_url: `${opencodeUrl}/health`,
      created_at: new Date().toISOString(),
      connected_at: null,
      last_activity: null,
      error_message: null,
      metadata: {},
      is_local: true,
    })

    if (!healthCheck.healthy) {
      snapshot.lastError = healthCheck.error || "OpenCode server not reachable"
      snapshot.status = `Connection failed - is OpenCode running at ${opencodeUrl}?`
      data.ctx.render()
      return
    }

    snapshot.status = "Creating session on OpenCode..."
    data.ctx.render()

    const createResponse = await fetch(`${opencodeUrl}/session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    })

    if (!createResponse.ok) {
      const errorText = await createResponse.text().catch(() => "")
      snapshot.lastError = `Failed to create session: ${createResponse.status} ${errorText}`
      snapshot.status = "Session creation failed"
      data.ctx.render()
      return
    }

    const sessionData = await createResponse.json() as { id: string; title?: string }
    const sessionId = sessionData.id

    const localSession: SessionRecord = {
      session_id: sessionId,
      container_id: "",
      state: "connected",
      mode: "interactive",
      model: "gpt-4o-mini",
      access_url: opencodeUrl,
      tunnel_url: null,
      opencode_url: opencodeUrl,
      health_url: `${opencodeUrl}/health`,
      created_at: new Date().toISOString(),
      connected_at: new Date().toISOString(),
      last_activity: new Date().toISOString(),
      error_message: null,
      metadata: {},
      is_local: true,
    }

    const nextSessions = [localSession, ...sessionsCache().filter((s) => s.session_id !== sessionId)]
    setSessionsCache(nextSessions)
    snapshot.sessions = nextSessions
    const nextHealth = new Map(sessionsHealthCache())
    nextHealth.set(sessionId, healthCheck)
    snapshot.sessionHealthResults.set(sessionId, healthCheck)
    setSessionsHealthCache(nextHealth)

    appState.openCodeSessionId = sessionId
    snapshot.status = `Connected to OpenCode at ${opencodeUrl} | Session: ${sessionId}`
    data.ctx.render()
  }

  createEffect(() => {
    data.version()
    const opencodeUrl = appState.openCodeUrl
    if (!opencodeUrl) {
      appState.openCodeAutoConnectAttempted = false
      return
    }
    if (appState.openCodeSessionId) return
    if (appState.openCodeAutoConnectAttempted) return
    appState.openCodeAutoConnectAttempted = true
    void connectLocalSession()
  })

  async function disconnectSelectedSession(): Promise<void> {
    const sessions = sessionsCache()
    const active = sessions.filter(
      (s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting",
    )
    const session = active[sessionsSelectedIndex()]
    if (!session) return

    snapshot.status = `Disconnecting ${session.session_id}...`
    data.ctx.render()

    try {
      const result = await disconnectSession(session.session_id)
      if (result.disconnected) {
        snapshot.status = `Disconnected from ${session.session_id}`
        if (appState.openCodeSessionId === session.session_id) {
          appState.openCodeSessionId = null
        }
        await refreshSessionsModal()
      } else {
        snapshot.status = "Disconnect failed"
      }
      data.ctx.render()
    } catch (err: any) {
      snapshot.lastError = err?.message || "Failed to disconnect"
      snapshot.status = "Disconnect failed"
      data.ctx.render()
    }
  }

  async function copySelectedSessionUrl(): Promise<void> {
    const sessions = sessionsCache()
    const active = sessions.filter(
      (s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting",
    )
    const session = active[sessionsSelectedIndex()]
    if (!session) return
    const url = session.opencode_url || session.access_url || ""
    if (!url) return
    await copyToClipboard(url)
    snapshot.status = `Copied: ${url}`
    data.ctx.render()
  }

  function selectSession(): void {
    const sessions = sessionsCache()
    const active = sessions.filter(
      (s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting",
    )
    const session = active[sessionsSelectedIndex()]
    if (!session) return
    appState.openCodeSessionId = session.session_id
    if (!snapshot.sessions.find((s) => s.session_id === session.session_id)) {
      snapshot.sessions.push(session)
    }
    snapshot.status = `Selected session: ${session.session_id}`
    closeActiveModal()
    data.ctx.render()
  }

  async function startLoginAuth(): Promise<void> {
    if (loginInProgress()) return
    setLoginInProgress(true)
    const result = await runDeviceCodeAuth((status) => {
      setLoginStatus(status)
    })
    setLoginInProgress(false)

    if (result.success && result.apiKey) {
      const frontendUrlId = getFrontendUrlId(appState.currentBackend)
      frontendKeys[frontendUrlId] = result.apiKey
      frontendKeySources[frontendUrlId] = { sourcePath: null, varName: "device_code_auth" }
      process.env.SYNTH_API_KEY = result.apiKey
      await saveApiKey(result.apiKey)
      await clearLoggedOutMarker()
      await persistSettings({
        settingsFilePath: data.ctx.state.config.settingsFilePath,
        getCurrentBackend: () => appState.currentBackend,
        getFrontendKey: (id) => frontendKeys[id],
        getFrontendKeySource: (id) => frontendKeySources[id],
      })
      closeActiveModal()
      snapshot.lastError = null
      snapshot.status = "Authenticated! Loading..."
      data.ctx.render()
      await data.refresh()
    }
  }

  async function logout(): Promise<void> {
    await setLoggedOutMarker()
    await deleteSavedApiKey()
    process.env.SYNTH_API_KEY = ""

    if (pollingState.sseDisconnect) {
      pollingState.sseDisconnect()
      pollingState.sseDisconnect = null
    }
    pollingState.sseConnected = false
    clearJobsTimer()
    clearEventsTimer()

    snapshot.jobs = []
    snapshot.selectedJob = null
    snapshot.events = []
    snapshot.metrics = {}
    snapshot.bestSnapshotId = null
    snapshot.bestSnapshot = null
    snapshot.evalSummary = null
    snapshot.evalResultRows = []
    snapshot.artifacts = []
    snapshot.orgId = null
    snapshot.userId = null
    snapshot.balanceDollars = null
    snapshot.lastRefresh = null
    snapshot.allCandidates = []
    snapshot.lastError = "Logged out"
    snapshot.status = "Sign in required"
    data.ctx.render()

    setLoginStatus({ state: "idle" })
    setLoginInProgress(false)
    setActiveModal("login")
  }

  useKeyboard((evt) => {
    const detailModal = modal()
    if (detailModal) {
      if ((evt.name === "f" && evt.shift) || evt.name === "F") {
        evt.preventDefault()
        setModal({ ...detailModal, fullscreen: !detailModal.fullscreen })
        return
      }
      if (evt.name === "q" || evt.name === "escape" || evt.name === "return" || evt.name === "enter") {
        evt.preventDefault()
        setModal(null)
        return
      }
      if (evt.name === "j" || evt.name === "down") {
        evt.preventDefault()
        if (detailModal.type === "log") {
          setModal({ ...detailModal, offset: detailModal.offset + 1, tail: false })
        } else {
          setModal({ ...detailModal, offset: detailModal.offset + 1 })
        }
        return
      }
      if (evt.name === "k" || evt.name === "up") {
        evt.preventDefault()
        if (detailModal.type === "log") {
          setModal({ ...detailModal, offset: detailModal.offset - 1, tail: false })
        } else {
          setModal({ ...detailModal, offset: detailModal.offset - 1 })
        }
        return
      }
      if (evt.name === "t" && detailModal.type === "log") {
        evt.preventDefault()
        setModal({ ...detailModal, tail: true })
        return
      }
      if (evt.name === "y" && detailModal.type === "log") {
        evt.preventDefault()
        void copyToClipboard(readLogFile(detailModal.path)).then(() => {
          snapshot.status = `Copied: ${path.basename(detailModal.path)}`
          data.ctx.render()
        })
        return
      }
      return
    }

    if (focusManager.handleKey(evt)) {
      return
    }

    const overlayModal = activeModal()
    if (overlayModal) {
      if (overlayModal === "filter") {
        if (evt.name === "return" || evt.name === "enter") {
          evt.preventDefault()
          applyFilterModal()
          return
        }
        if (evt.name === "q" || evt.name === "escape") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      if (overlayModal === "snapshot") {
        if (evt.name === "return" || evt.name === "enter") {
          evt.preventDefault()
          void applySnapshotModal()
          return
        }
        if (evt.name === "q" || evt.name === "escape") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      if (overlayModal === "key") {
        if (evt.name === "return" || evt.name === "enter") {
          evt.preventDefault()
          applyKeyModal()
          return
        }
        if (evt.name === "v" && (evt.ctrl || evt.meta)) {
          evt.preventDefault()
          pasteKeyModal()
          return
        }
        if (evt.name === "q" || evt.name === "escape") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      if (overlayModal === "settings") {
        if (isUpKey(evt)) {
          evt.preventDefault()
          moveSettingsCursor(-1)
          return
        }
        if (isDownKey(evt)) {
          evt.preventDefault()
          moveSettingsCursor(1)
          return
        }
        if (evt.name === "return" || evt.name === "enter") {
          evt.preventDefault()
          void selectSettingsBackend()
          return
        }
        if (evt.name === "k" && evt.shift) {
          evt.preventDefault()
          closeActiveModal()
          openKeyModal()
          return
        }
        if (evt.name === "e" && evt.shift) {
          evt.preventDefault()
          closeActiveModal()
          openEnvKeyModal()
          return
        }
        if (evt.name === "q" || evt.name === "escape") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      if (overlayModal === "env-key") {
        if (evt.name === "up" || evt.name === "k") {
          evt.preventDefault()
          moveEnvKeyCursor(-1)
          return
        }
        if (evt.name === "down" || evt.name === "j") {
          evt.preventDefault()
          moveEnvKeyCursor(1)
          return
        }
        if (evt.name === "r") {
          evt.preventDefault()
          void rescanEnvKeys()
          return
        }
        if (evt.name === "return" || evt.name === "enter") {
          evt.preventDefault()
          void selectEnvKey()
          return
        }
        if (evt.name === "q" || evt.name === "escape") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      if (overlayModal === "usage") {
        if (evt.name === "b") {
          evt.preventDefault()
          openUsageBilling()
          return
        }
        if (evt.name === "up" || evt.name === "k") {
          evt.preventDefault()
          appState.usageModalOffset = Math.max(0, (appState.usageModalOffset || 0) - 1)
          data.ctx.render()
          return
        }
        if (evt.name === "down" || evt.name === "j") {
          evt.preventDefault()
          appState.usageModalOffset = (appState.usageModalOffset || 0) + 1
          data.ctx.render()
          return
        }
        if (evt.name === "q" || evt.name === "escape" || evt.name === "return" || evt.name === "enter") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      if (overlayModal === "metrics") {
        if (evt.name === "up" || evt.name === "k") {
          evt.preventDefault()
          appState.metricsModalOffset = Math.max(0, (appState.metricsModalOffset || 0) - 1)
          data.ctx.render()
          return
        }
        if (evt.name === "down" || evt.name === "j") {
          evt.preventDefault()
          appState.metricsModalOffset = (appState.metricsModalOffset || 0) + 1
          data.ctx.render()
          return
        }
        if (evt.name === "m") {
          evt.preventDefault()
          void fetchMetrics(data.ctx).then(() => data.ctx.render())
          return
        }
        if (evt.name === "q" || evt.name === "escape" || evt.name === "return" || evt.name === "enter") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      if (overlayModal === "task-apps") {
        if (evt.name === "up" || evt.name === "k") {
          evt.preventDefault()
          moveTaskAppsSelection(-1)
          return
        }
        if (evt.name === "down" || evt.name === "j") {
          evt.preventDefault()
          moveTaskAppsSelection(1)
          return
        }
        if (evt.name === "y") {
          evt.preventDefault()
          void copySelectedTunnelUrl()
          return
        }
        if (evt.name === "q" || evt.name === "escape" || evt.name === "return" || evt.name === "enter") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      if (overlayModal === "sessions") {
        if (evt.name === "up" || evt.name === "k") {
          evt.preventDefault()
          moveSessionsSelection(-1)
          return
        }
        if (evt.name === "down" || evt.name === "j") {
          evt.preventDefault()
          moveSessionsSelection(1)
          return
        }
        if (evt.name === "y") {
          evt.preventDefault()
          void copySelectedSessionUrl()
          return
        }
        if (evt.name === "c" && !evt.shift) {
          evt.preventDefault()
          void connectLocalSession()
          return
        }
        if (evt.name === "d") {
          evt.preventDefault()
          void disconnectSelectedSession()
          return
        }
        if (evt.name === "r") {
          evt.preventDefault()
          void refreshSessionsModal()
          return
        }
        if (evt.name === "return" || evt.name === "enter") {
          evt.preventDefault()
          selectSession()
          return
        }
        if (evt.name === "q" || evt.name === "escape") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      if (overlayModal === "job-filter") {
        if (evt.name === "up" || evt.name === "k") {
          evt.preventDefault()
          moveJobFilter(-1)
          return
        }
        if (evt.name === "down" || evt.name === "j") {
          evt.preventDefault()
          moveJobFilter(1)
          return
        }
        if (evt.name === "space" || evt.name === "return" || evt.name === "enter") {
          evt.preventDefault()
          toggleJobFilterSelection()
          return
        }
        if (evt.name === "c") {
          evt.preventDefault()
          clearJobFilterSelection()
          return
        }
        if (evt.name === "q" || evt.name === "escape") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      if (overlayModal === "config") {
        if (evt.name === "up" || evt.name === "k") {
          evt.preventDefault()
          appState.configModalOffset = Math.max(0, appState.configModalOffset - 1)
          data.ctx.render()
          return
        }
        if (evt.name === "down" || evt.name === "j") {
          evt.preventDefault()
          appState.configModalOffset = appState.configModalOffset + 1
          data.ctx.render()
          return
        }
        if (evt.name === "return" || evt.name === "enter" || evt.name === "i" || evt.name === "q" || evt.name === "escape") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      if (overlayModal === "profile" || overlayModal === "urls") {
        if (evt.name === "return" || evt.name === "enter" || evt.name === "q" || evt.name === "escape") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      if (overlayModal === "login") {
        if (evt.name === "return" || evt.name === "enter") {
          evt.preventDefault()
          void startLoginAuth()
          return
        }
        if (evt.name === "q" || evt.name === "escape") {
          evt.preventDefault()
          closeActiveModal()
          return
        }
        return
      }
      return
    }

    if (evt.ctrl && evt.name === "c") {
      evt.preventDefault()
      onExit?.()
      void shutdown(0)
      return
    }

    if (appState.principalPane === "opencode") {
      if (evt.name === "g" && evt.shift) {
        evt.preventDefault()
        appState.principalPane = "jobs"
        data.ctx.render()
      }
      if (evt.name === "o" && evt.shift) {
        evt.preventDefault()
        openSessionsModal()
      }
      return
    }

    if (evt.name === "q" || evt.name === "escape") {
      evt.preventDefault()
      onExit?.()
      void shutdown(0)
      return
    }
    if (evt.name === "tab") {
      evt.preventDefault()
      const order = ["jobs", "events", "logs"] as const
      const current = appState.activePane
      const idx = Math.max(0, order.indexOf(current))
      appState.activePane = order[(idx + 1) % order.length]
      data.ctx.render()
      return
    }
    if (evt.name === "r") {
      evt.preventDefault()
      void data.refresh()
      return
    }
    if (evt.name === "b") {
      evt.preventDefault()
      appState.activePane = "jobs"
      data.ctx.render()
      return
    }
    if (evt.name === "e") {
      evt.preventDefault()
      appState.activePane = "events"
      data.ctx.render()
      return
    }
    if (evt.name === "l" && evt.shift) {
      evt.preventDefault()
      appState.activePane = "logs"
      data.ctx.render()
      return
    }
    if (evt.name === "g" && evt.shift) {
      evt.preventDefault()
      appState.principalPane = appState.principalPane === "jobs" ? "opencode" : "jobs"
      data.ctx.render()
      return
    }
    if (evt.name === "l") {
      evt.preventDefault()
      void logout()
      return
    }
    if (evt.name === "f") {
      evt.preventDefault()
      openFilterModal()
      return
    }
    if (evt.name === "i" && !evt.shift) {
      evt.preventDefault()
      openConfigModal()
      return
    }
    if (evt.name === "i" && evt.shift) {
      evt.preventDefault()
      // Install OpenCode if not available
      const status = appState.openCodeStatus || ""
      if (status.includes("not available") || status.includes("install")) {
        void data.installOpenCode()
      }
      return
    }
    if (evt.name === "p") {
      evt.preventDefault()
      if (!process.env.SYNTH_API_KEY) return
      openProfileModal()
      return
    }
    // Candidates viewer: use "v" so Shift+O remains for OpenCode sessions.
    if (evt.name === "v") {
      evt.preventDefault()
      openResultsModal()
      return
    }
    if (evt.name === "j" && evt.shift) {
      evt.preventDefault()
      openJobFilterModal()
      return
    }
    if (evt.name === "s" && !evt.shift) {
      evt.preventDefault()
      openSnapshotModal()
      return
    }
    if (evt.name === "s" && evt.shift) {
      evt.preventDefault()
      openUrlsModal()
      return
    }
    if (evt.name === "t") {
      evt.preventDefault()
      openSettingsModal()
      return
    }
    if (evt.name === "x") {
      evt.preventDefault()
      if (snapshot.selectedJob) {
        setActiveModal("traces")
      }
      return
    }
    if (evt.name === "d") {
      evt.preventDefault()
      openUsageModal()
      return
    }
    if (evt.name === "u") {
      evt.preventDefault()
      openTaskAppsModal()
      return
    }
    if (evt.name === "n") {
      evt.preventDefault()
      setShowCreateJobModal(true)
      return
    }

    // Handle create job modal keys when open
    if (showCreateJobModal()) {
      const handled = (CreateJobModal as any).handleKeyPress?.(evt)
      if (handled) {
        data.ctx.render()
        return
      }
    }
    // Sessions modal is only reachable from the Agent (OpenCode) pane via Shift+O.
    // Avoid binding Shift+O globally in the jobs pane; it makes `o` feel unreliable.
    if (evt.name === "c") {
      evt.preventDefault()
      void cancelSelected(data.ctx).then(() => data.ctx.render())
      return
    }
    if (evt.name === "a") {
      evt.preventDefault()
      void fetchArtifacts(data.ctx).then(() => data.ctx.render())
      return
    }
    if (evt.name === "M" || (evt.name === "m" && evt.shift)) {
      // Shift+M = fullscreen metrics modal
      evt.preventDefault()
      openMetricsModal()
      return
    }
    if (evt.name === "m") {
      evt.preventDefault()
      void fetchMetrics(data.ctx).then(() => data.ctx.render())
      return
    }
    if (evt.name === "j") {
      evt.preventDefault()
      const pane = appState.activePane
      if (pane === "jobs") {
        const count = jobs().length
        if (count === 0) return
        setSelectedIndex((current) => (current + 1) % count)
        return
      }
      if (pane === "events") {
        moveEventSelection(data.ctx, 1)
        data.ctx.render()
        return
      }
      if (pane === "logs") {
        moveLogSelection(data.ctx, 1)
        data.ctx.render()
        return
      }
    }
    if (evt.name === "k") {
      evt.preventDefault()
      const pane = appState.activePane
      if (pane === "jobs") {
        const count = jobs().length
        if (count === 0) return
        setSelectedIndex((current) => (current - 1 + count) % count)
        return
      }
      if (pane === "events") {
        moveEventSelection(data.ctx, -1)
        data.ctx.render()
        return
      }
      if (pane === "logs") {
        moveLogSelection(data.ctx, -1)
        data.ctx.render()
        return
      }
    }
    if (evt.name === "down" || evt.name === "arrowdown") {
      evt.preventDefault()
      const pane = appState.activePane
      if (pane === "jobs") {
        const count = jobs().length
        if (count === 0) return
        setSelectedIndex((current) => (current + 1) % count)
        return
      }
      if (pane === "events") {
        moveEventSelection(data.ctx, 1)
        data.ctx.render()
        return
      }
      if (pane === "logs") {
        moveLogSelection(data.ctx, 1)
        data.ctx.render()
        return
      }
    }
    if (evt.name === "up" || evt.name === "arrowup") {
      evt.preventDefault()
      const pane = appState.activePane
      if (pane === "jobs") {
        const count = jobs().length
        if (count === 0) return
        setSelectedIndex((current) => (current - 1 + count) % count)
        return
      }
      if (pane === "events") {
        moveEventSelection(data.ctx, -1)
        data.ctx.render()
        return
      }
      if (pane === "logs") {
        moveLogSelection(data.ctx, -1)
        data.ctx.render()
        return
      }
    }
    if (evt.name === "return" || evt.name === "enter") {
      evt.preventDefault()
      const pane = appState.activePane
      if (pane === "jobs") {
        const job = jobs()[selectedIndex()]
        if (job?.job_id) {
          void data.select(job.job_id)
        }
        return
      }
      if (pane === "events") {
        const event = events()[eventWindow().selected]
        if (event) {
          openEventModal(event)
        }
        return
      }
      if (pane === "logs") {
        const selected = appState.logsSelectedIndex
        const file = logFiles()[selected]
        if (file) {
          openLogModal(file.path)
        }
        return
      }
    }
  })

  let modalInputRef: any
  let lastModalKind: ActiveModal | null = null
  createEffect(() => {
    const kind = activeModal()
    if (kind !== lastModalKind) {
      lastModalKind = kind
      if (kind === "filter" || kind === "snapshot" || kind === "key") {
        if (modalInputRef) {
          modalInputRef.value = modalInputValue()
          setTimeout(() => modalInputRef.focus(), 1)
        }
      }
    }
  })

  function ModalFrame(props: {
    title: string
    width: number
    height: number
    borderColor: string
    titleColor?: string
    hint?: string
    children: any
  }) {
    const frameWidth = Math.min(props.width, Math.max(20, dimensions().width - 4))
    const frameHeight = Math.min(props.height, Math.max(6, dimensions().height - 4))
    const left = Math.max(0, Math.floor((dimensions().width - frameWidth) / 2))
    const top = Math.max(1, Math.floor((dimensions().height - frameHeight) / 2))
    // Calculate content height: frame - borders(2) - padding(2) - title(1) - hint(1) = height - 6
    const contentHeight = Math.max(1, frameHeight - 6)

    return (
      <box
        position="absolute"
        left={left}
        top={top}
        width={frameWidth}
        height={frameHeight}
        backgroundColor="#0b1220"
        border
        borderStyle="single"
        borderColor={props.borderColor}
        zIndex={30}
        flexDirection="column"
        paddingLeft={2}
        paddingRight={2}
        paddingTop={1}
        paddingBottom={1}
      >
        <text fg={props.titleColor ?? props.borderColor}>
          <b>{props.title}</b>
        </text>
        <box height={contentHeight} overflow="hidden">
          {props.children}
        </box>
        <Show when={props.hint}>
          <text fg="#94a3b8">{props.hint}</text>
        </Show>
      </box>
    )
  }

  function renderActiveModal(kind: ActiveModal) {
    // Modal content is mostly derived from non-reactive state objects (appState/snapshot).
    // Make modal rendering depend on the reactive version signal so calls to
    // `data.ctx.render()` (which bumps version) repaint the modal (e.g. settings cursor).
    data.version()

    if (kind === "filter") {
      return (
        <ModalFrame
          title="Event Filter"
          width={52}
          height={7}
          borderColor="#60a5fa"
          titleColor="#60a5fa"
          hint="Enter apply | q close"
        >
          <box flexDirection="column" gap={1}>
            <text fg="#e2e8f0">Event filter:</text>
            <input
              placeholder="Type to filter events"
              onInput={(value) => setModalInputValue(value)}
              ref={(ref) => {
                modalInputRef = ref
              }}
            />
          </box>
        </ModalFrame>
      )
    }

    if (kind === "snapshot") {
      return (
        <ModalFrame
          title="Snapshot ID"
          width={50}
          height={7}
          borderColor="#60a5fa"
          titleColor="#60a5fa"
          hint="Enter apply | q close"
        >
          <box flexDirection="column" gap={1}>
            <text fg="#e2e8f0">Snapshot ID:</text>
            <input
              placeholder="Enter snapshot id"
              onInput={(value) => setModalInputValue(value)}
              ref={(ref) => {
                modalInputRef = ref
              }}
            />
          </box>
        </ModalFrame>
      )
    }

    if (kind === "key") {
      return (
        <ModalFrame
          title="API Key"
          width={70}
          height={7}
          borderColor="#7dd3fc"
          titleColor="#7dd3fc"
          hint="Paste or type key | Enter to apply | q close"
        >
          <box flexDirection="column" gap={1}>
            <text fg="#e2e8f0">API Key:</text>
            <input
              placeholder=""
              onInput={(value) => setModalInputValue(value)}
              ref={(ref) => {
                modalInputRef = ref
              }}
            />
          </box>
        </ModalFrame>
      )
    }

    if (kind === "settings") {
      // Use a function for the text content to ensure reactivity to settingsCursor() changes
      const settingsContent = () => {
        const cursorIdx = settingsCursor()
        const lines: string[] = []
        for (let idx = 0; idx < appState.settingsOptions.length; idx++) {
          const opt = appState.settingsOptions[idx]
          const active = appState.currentBackend === opt.id
          const cursor = idx === cursorIdx ? ">" : " "
          lines.push(`${cursor} [${active ? "x" : " "}] ${opt.label} (${opt.id})`)
        }
        const selected = appState.settingsOptions[cursorIdx]
        if (selected) {
          const key = getKeyForBackend(selected.id)
          const keyPreview = key.trim() ? `...${key.slice(-8)}` : "(no key)"
          const frontendUrl = getFrontendUrl(selected.id)
          lines.push("")
          lines.push(`Backend: ${selected.baseUrl}`)
          lines.push(`Frontend: ${frontendUrl}`)
          lines.push(`Key: ${keyPreview}`)
        }
        return lines.join("\n")
      }

      return (
        <ModalFrame
          title="Settings - Backend"
          width={64}
          height={14}
          borderColor="#38bdf8"
          titleColor="#38bdf8"
          hint="j/k navigate | enter select | shift+e env keys | shift+k key | q close"
        >
          <text fg="#e2e8f0">{settingsContent()}</text>
        </ModalFrame>
      )
    }

    if (kind === "env-key") {
      const lines: string[] = []
      if (appState.envKeyScanInProgress) {
        lines.push("Scanning...")
      } else if (appState.envKeyError) {
        lines.push(`Error: ${appState.envKeyError}`)
      } else if (!appState.envKeyOptions.length) {
        const scanRoot = data.ctx.state.config.envKeyScanRoot
        lines.push("No API keys found in .env files")
        lines.push("")
        lines.push(`Scanned: ${scanRoot}`)
        lines.push("")
        lines.push("Looking for vars:")
        lines.push("  SYNTH_API_KEY")
        lines.push("  SYNTH_TUI_API_KEY_PROD")
        lines.push("  SYNTH_TUI_API_KEY_DEV")
        lines.push("  SYNTH_TUI_API_KEY_LOCAL")
      } else {
        const max = Math.max(0, appState.envKeyOptions.length - 1)
        const start = clamp(appState.envKeyWindowStart, 0, Math.max(0, max))
        const end = Math.min(appState.envKeyOptions.length, start + data.ctx.state.config.envKeyVisibleCount)
        for (let idx = start; idx < end; idx++) {
          const option = appState.envKeyOptions[idx]
          const cursor = idx === appState.envKeyCursor ? ">" : " "
          const preview = option.key ? `${option.key.slice(0, 8)}...` : "(empty)"
          lines.push(`${cursor} ${preview}`)
        }
        const selected = appState.envKeyOptions[appState.envKeyCursor]
        if (selected) {
          const sources = selected.sources.slice(0, 2).join(", ")
          const suffix = selected.sources.length > 2 ? ` +${selected.sources.length - 2}` : ""
          lines.push("")
          lines.push(`Source: ${sources}${suffix}`)
          lines.push(`Vars: ${selected.varNames.join(", ")}`)
        }
      }

      return (
        <ModalFrame
          title="Scan .env Files for API Keys"
          width={64}
          height={16}
          borderColor="#a78bfa"
          titleColor="#a78bfa"
          hint="j/k navigate | enter select | r rescan | q close"
        >
          <text fg="#e2e8f0">{lines.join("\n")}</text>
        </ModalFrame>
      )
    }

    if (kind === "usage") {
      const raw = formatUsageDetails(usageData())
      const view = buildScrollableModal(raw, 72, 28, appState.usageModalOffset || 0)
      const range = view.lines.length > view.bodyHeight
        ? `[${view.offset + 1}-${view.offset + view.visible.length}/${view.lines.length}] `
        : ""
      return (
        <ModalFrame
          title={`Usage & Plan - ${formatPlanName(usageData()?.plan_type || "free")} ${range}`.trim()}
          width={72}
          height={28}
          borderColor="#10b981"
          titleColor="#10b981"
          hint="j/k scroll | b billing | q close"
        >
          <text fg="#e2e8f0">{view.visible.join("\n")}</text>
        </ModalFrame>
      )
    }

    if (kind === "metrics") {
      const m: any = snapshot.metrics || {}
      const pts = Array.isArray(m?.points) ? m.points : []
      const job = snapshot.selectedJob
      const isGepa = job?.training_type === "gepa" || job?.training_type === "graph_gepa"
      
      // Build fullscreen metrics content
      const raw = formatMetricsCharts(snapshot.metrics, {
        width: dimensions().width - 6,
        height: dimensions().height - 8,
        isGepa,
      })
      const view = buildScrollableModal(raw, dimensions().width - 4, dimensions().height - 6, appState.metricsModalOffset || 0)
      const hint = view.lines.length > view.bodyHeight
        ? `[${view.offset + 1}-${view.offset + view.visible.length}/${view.lines.length}] j/k scroll | m refresh | q close`
        : "m refresh | q close"
      return (
        <ModalFrame
          title={`Metrics (${pts.length} points)`}
          width={dimensions().width - 4}
          height={dimensions().height - 6}
          borderColor="#8b5cf6"
          titleColor="#8b5cf6"
          hint={hint}
        >
          <text fg="#e2e8f0">{view.visible.join("\n")}</text>
        </ModalFrame>
      )
    }

    if (kind === "task-apps") {
      const raw = formatTunnelDetails(snapshot.tunnels, snapshot.tunnelHealthResults, appState.taskAppsModalSelectedIndex || 0)
      const view = buildScrollableModal(raw, 90, 20, appState.taskAppsModalOffset || 0)
      const hint = view.lines.length > view.bodyHeight
        ? `[${view.offset + 1}-${view.offset + view.visible.length}/${view.lines.length}] j/k select | y copy hostname | q close`
        : "j/k select | y copy hostname | q close"
      return (
        <ModalFrame
          title={`Task Apps (${snapshot.tunnels.length} tunnel${snapshot.tunnels.length !== 1 ? "s" : ""})`}
          width={90}
          height={20}
          borderColor="#06b6d4"
          titleColor="#06b6d4"
          hint={hint}
        >
          <text fg="#e2e8f0">{view.visible.join("\n")}</text>
        </ModalFrame>
      )
    }

    if (kind === "sessions") {
      const sessions = sessionsCache()
      const raw = formatSessionDetails(sessions, sessionsHealthCache(), sessionsSelectedIndex(), appState.openCodeUrl)
      const view = buildScrollableModal(raw, 70, 20, sessionsScrollOffset())
      const hint = view.lines.length > view.bodyHeight
        ? `[${view.offset + 1}-${view.offset + view.visible.length}/${view.lines.length}] j/k select | c connect local | d disconnect | y copy URL | enter select | q close`
        : "j/k select | c connect local | d disconnect | y copy URL | enter select | q close"
      return (
        <ModalFrame
          title={`OpenCode Sessions (${sessions.filter((s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting").length} active)`}
          width={70}
          height={20}
          borderColor="#60a5fa"
          titleColor="#60a5fa"
          hint={hint}
        >
          <text fg="#e2e8f0">{view.visible.join("\n")}</text>
        </ModalFrame>
      )
    }

    if (kind === "config") {
      const raw = formatConfigMetadata(snapshot)
      const view = buildScrollableModal(raw, 100, 24, appState.configModalOffset)
      const hint = view.lines.length > view.bodyHeight
        ? `[${view.offset + 1}-${view.offset + view.visible.length}/${view.lines.length}] j/k scroll | q close`
        : "q close"
      return (
        <ModalFrame
          title="Job Configuration"
          width={100}
          height={24}
          borderColor="#f59e0b"
          titleColor="#f59e0b"
          hint={hint}
        >
          <text fg="#e2e8f0">{view.visible.join("\n")}</text>
        </ModalFrame>
      )
    }

    if (kind === "results") {
      return (
        <CandidatesModal
          visible={true}
          snapshot={snapshot}
          width={dimensions().width}
          height={dimensions().height}
          onClose={closeActiveModal}
          onStatus={(message) => {
            snapshot.status = message
            data.ctx.render()
          }}
        />
      )
    }

    if (kind === "traces") {
      return (
        <TraceViewerModal
          visible={true}
          snapshot={snapshot}
          width={dimensions().width}
          height={dimensions().height}
          onClose={closeActiveModal}
          onStatus={(message) => {
            snapshot.status = message
            data.ctx.render()
          }}
        />
      )
    }

    if (kind === "profile") {
      const org = snapshot.orgId || "-"
      const user = snapshot.userId || "-"
      const apiKey = process.env.SYNTH_API_KEY || "-"
      return (
        <ModalFrame
          title="Profile"
          width={72}
          height={15}
          borderColor="#818cf8"
          titleColor="#818cf8"
          hint="q close"
        >
          <text fg="#e2e8f0">{`Organization:\n${org}\n\nUser:\n${user}\n\nAPI Key:\n${apiKey}`}</text>
        </ModalFrame>
      )
    }

    if (kind === "urls") {
      const backend = process.env.SYNTH_BACKEND_URL || "-"
      const frontend = process.env.SYNTH_FRONTEND_URL || "-"
      return (
        <ModalFrame
          title="URLs"
          width={60}
          height={10}
          borderColor="#f59e0b"
          titleColor="#f59e0b"
          hint="q close"
        >
          <text fg="#e2e8f0">{`Backend:\n${backend}\n\nFrontend:\n${frontend}`}</text>
        </ModalFrame>
      )
    }

    if (kind === "job-filter") {
      const max = Math.max(0, appState.jobFilterOptions.length - 1)
      const start = clamp(appState.jobFilterWindowStart, 0, Math.max(0, max))
      const end = Math.min(appState.jobFilterOptions.length, start + data.ctx.state.config.jobFilterVisibleCount)
      const lines: string[] = []
      for (let idx = start; idx < end; idx++) {
        const option = appState.jobFilterOptions[idx]
        const active = appState.jobStatusFilter.has(option.status)
        const cursor = idx === appState.jobFilterCursor ? ">" : " "
        lines.push(`${cursor} [${active ? "x" : " "}] ${option.status} (${option.count})`)
      }
      if (!lines.length) {
        lines.push("  (no statuses available)")
      }
      return (
        <ModalFrame
          title="Job filter (status)"
          width={52}
          height={11}
          borderColor="#60a5fa"
          titleColor="#60a5fa"
          hint="j/k move | space select | c clear | q close"
        >
          <text fg="#e2e8f0">{lines.join("\n")}</text>
        </ModalFrame>
      )
    }

    if (kind === "login") {
      const status = loginStatus()
      let content = ""
      let hint = "Enter start | q cancel"
      if (status.state === "idle") {
        content = "Press Enter to open browser and sign in..."
      } else if (status.state === "initializing") {
        content = "Initializing..."
        hint = "Please wait..."
      } else if (status.state === "waiting") {
        content = `Browser opened. Complete sign-in there.\n\nURL: ${status.verificationUri}`
        hint = "Waiting for browser auth... | q cancel"
      } else if (status.state === "polling") {
        content = "Browser opened. Complete sign-in there.\n\nChecking for completion..."
        hint = "Waiting for browser auth... | q cancel"
      } else if (status.state === "success") {
        content = "Authentication successful!"
        hint = "Loading..."
      } else if (status.state === "error") {
        content = `Error: ${status.message}`
        hint = "Enter retry | q close"
      }
      return (
        <ModalFrame
          title="Sign In / Sign Up"
          width={60}
          height={10}
          borderColor="#22c55e"
          titleColor="#22c55e"
          hint={hint}
        >
          <text fg="#e2e8f0">{content}</text>
        </ModalFrame>
      )
    }

    return null
  }

  return (
    <box
      width={layout().totalWidth}
      height={layout().totalHeight}
      flexDirection="column"
      backgroundColor="#0b1120"
    >
      <box
        height={defaultLayoutSpec.headerHeight}
        backgroundColor={COLORS.bgHeader}
        border
        borderStyle="single"
        borderColor={COLORS.border}
        alignItems="center"
      >
        <text fg={COLORS.text}>Synth AI</text>
      </box>

      <box
        height={defaultLayoutSpec.tabsHeight}
        backgroundColor={COLORS.bgTabs}
        border
        borderStyle="single"
        borderColor={COLORS.borderDim}
        alignItems="center"
        flexDirection="row"
        gap={2}
      >
        <KeyHint description="Create New Job" keyLabel="n" />
        <KeyHint description="View Jobs" keyLabel="b" active={activePane() === "jobs"} />
        <KeyHint description="View Job's Events" keyLabel="e" active={activePane() === "events"} />
        <KeyHint description="View Logs" keyLabel="shift+l" active={activePane() === "logs"} />
        <KeyHint description="Agent" keyLabel="shift+g" active={principalPane() === "opencode"} />
      </box>

      <box
        flexDirection="row"
        height={layout().contentHeight}
        flexGrow={1}
        border={false}
      >
        <Show
          when={activePane() === "jobs"}
          fallback={
            <LogsList
              logs={logFiles()}
              selectedIndex={appState.logsSelectedIndex}
              focused={activePane() === "logs"}
              width={layout().jobsWidth}
              height={layout().contentHeight}
            />
          }
        >
          <JobsList
            jobs={jobs()}
            selectedIndex={selectedIndex()}
            focused={activePane() === "jobs"}
            width={layout().jobsWidth}
            height={layout().contentHeight}
          />
        </Show>

        <Show
          when={principalPane() === "jobs"}
          fallback={
            <box flexDirection="column" flexGrow={1} border={false}>
              <ErrorBoundary
                fallback={(err) => (
                  <box flexDirection="column" paddingLeft={2} paddingTop={1} gap={1}>
                    <text fg={COLORS.error}>OpenCode embed failed to render.</text>
                    <text fg={COLORS.textDim}>{String(err)}</text>
                    <text fg={COLORS.textDim}>Try restarting the TUI or running opencode-synth tui standalone.</text>
                  </box>
                )}
              >
                <ChatPane
                  url={opencodeUrl()}
                  sessionId={opencodeSessionId()}
                  width={opencodeDimensions().width}
                  height={opencodeDimensions().height}
                  onExit={() => {
                    data.ctx.state.appState.principalPane = "jobs"
                    data.ctx.render()
                  }}
                />
              </ErrorBoundary>
            </box>
          }
        >
          <Show
            when={activePane() !== "logs"}
            fallback={
              <LogsDetail
                title={logsTitle()}
                lines={logsView().lines}
                visibleLines={logsView().visible}
              />
            }
          >
            <JobsDetail
              snapshot={snapshotMemo()}
              events={events()}
              eventWindow={eventWindow()}
              lastError={lastError()}
              detailWidth={layout().detailWidth}
              detailHeight={layout().contentHeight}
              eventsFocused={activePane() === "events"}
              metricsView={data.ctx.state.appState.metricsView}
            />
          </Show>
        </Show>
      </box>

      <Show when={modal()}>
        <box
          position="absolute"
          left={modalLayout().left}
          top={modalLayout().top}
          width={modalLayout().width}
          height={modalLayout().height}
          backgroundColor="#0b1220"
          border
          borderStyle="single"
          borderColor="#60a5fa"
          zIndex={20}
          flexDirection="column"
          paddingLeft={2}
          paddingRight={2}
          paddingTop={1}
          paddingBottom={1}
        >
          <text fg="#60a5fa">
            {modal()!.title}
          </text>
          <box flexGrow={1}>
            <text fg="#e2e8f0">{modalView()?.visible.join("\n") ?? ""}</text>
          </box>
          <text fg="#94a3b8">{modalHint()}</text>
        </box>
      </Show>

      <Show when={activeModal()}>
        {(kind) => renderActiveModal(kind())}
      </Show>

      <CreateJobModal
        visible={showCreateJobModal()}
        onClose={() => setShowCreateJobModal(false)}
        onJobCreated={(info: JobCreatedInfo) => {
          if (info.jobSubmitted) {
            snapshot.status = `${info.trainingType} job submitted for ${toDisplayPath(info.localApiPath)}`
          } else if (info.deployedUrl) {
            snapshot.status = `Deployed: ${info.deployedUrl}`
          } else {
            snapshot.status = `Ready to deploy: ${toDisplayPath(info.localApiPath)}`
          }
          snapshot.lastError = null
          data.ctx.render()
          // Refresh jobs list to show new job
          void data.refresh()
        }}
        onStatusUpdate={(status: string) => {
          snapshot.status = status
          data.ctx.render()
        }}
        onError={(error: string) => {
          snapshot.lastError = error
          data.ctx.render()
        }}
        localApiFiles={localApiFiles()}
        width={Math.min(70, layout().totalWidth - 4)}
        height={Math.min(24, layout().totalHeight - 4)}
      />

      <box
        height={defaultLayoutSpec.statusHeight}
        backgroundColor="#0f172a"
        border
        borderStyle="single"
        borderColor="#334155"
        paddingLeft={1}
        alignItems="center"
      >
        <text fg="#e2e8f0">{statusText()}</text>
      </box>

      <box
        height={defaultLayoutSpec.footerHeight}
        backgroundColor={COLORS.bgTabs}
        paddingLeft={1}
        alignItems="center"
        flexDirection="row"
        gap={2}
      >
        <text fg={COLORS.textDim}>Keys: </text>
        <Show 
          when={principalPane() === "opencode"}
          fallback={
            <box flexDirection="row" gap={1}>
              <KeyHint description="select" keyLabel="j/k" />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="view" keyLabel="enter" />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="refresh" keyLabel="r" />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="candidates" keyLabel="v" />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="metrics" keyLabel="m" />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="fullscreen" keyLabel="M" />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="new" keyLabel="n" />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="switch" keyLabel="tab" />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="agent" keyLabel="shift+g" />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="quit" keyLabel="q" />
            </box>
          }
        >
          <box flexDirection="row" gap={1}>
            <KeyHint description="back" keyLabel="shift+g" />
            <text fg={COLORS.textDim}>|</text>
            <KeyHint description="sessions" keyLabel="shift+o" />
            <text fg={COLORS.textDim}>|</text>
            <KeyHint description="quit" keyLabel="q" />
          </box>
        </Show>
      </box>
    </box>
  )
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function formatEventDetail(data: JobEvent["data"]): string {
  if (data == null) return ""
  if (typeof data === "string") return data
  if (typeof data === "number" || typeof data === "boolean") return String(data)
  try {
    return JSON.stringify(data, null, 2)
  } catch {
    return String(data)
  }
}

function formatPlanName(planType: string): string {
  switch (planType) {
    case "pro": return "Pro"
    case "team": return "Team"
    case "byok": return "BYOK"
    case "free":
    default: return "Free"
  }
}

function formatStatus(status: string): string {
  switch (status) {
    case "active": return "Active"
    case "trialing": return "Trial"
    case "past_due": return "Past Due"
    case "cancelled": return "Cancelled"
    default: return status
  }
}

function formatUSD(amount: number | null | undefined): string {
  if (amount == null) return "-"
  return `$${amount.toFixed(2)}`
}

function formatUsageDetails(data: UsageData | null): string {
  if (!data) {
    return "Loading usage data..."
  }

  const lines: string[] = []
  lines.push("=== PLAN INFO ===")
  lines.push("")
  lines.push(`Plan:     ${formatPlanName(data.plan_type)}`)
  lines.push(`Status:   ${formatStatus(data.status)}`)

  const accessTier = data.access_tier || "alpha"
  lines.push(`Access:   ${accessTier.charAt(0).toUpperCase() + accessTier.slice(1)}`)

  if (data.byok_providers && data.byok_providers.length > 0) {
    const providers = data.byok_providers.map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join(", ")
    lines.push(`BYOK:     ${providers}`)
  }
  lines.push("")

  lines.push("Features:")
  if (data.limits.unlimited_non_rollout) {
    lines.push("  [*] Unlimited non-rollout usage")
  }
  if (data.limits.byok_enabled) {
    lines.push("  [*] BYOK enabled")
  }
  if (data.limits.team_features_enabled) {
    lines.push("  [*] Team features")
  }
  lines.push("")

  if (data.plan_type === "pro" || data.plan_type === "team") {
    lines.push("=== ROLLOUT CREDITS ===")
    lines.push("")
    lines.push(`Monthly:   ${formatUSD(data.limits.monthly_rollout_credits_usd)}`)
    lines.push(`Remaining: ${formatUSD(data.rollout_credits_balance_usd)}`)
    lines.push(`Used:      ${formatUSD(data.rollout_credits_used_this_period_usd)}`)
    lines.push("")
  }

  lines.push("=== USAGE (30 DAYS) ===")
  lines.push("")

  if (data.usage_summary) {
    const summary = data.usage_summary
    lines.push(`Total:   ${formatUSD(summary.total_cost_usd)}`)
    lines.push(`Charged: ${formatUSD(summary.total_charged_usd)}`)
    if (summary.total_uncharged_usd > 0) {
      lines.push(`Savings: ${formatUSD(summary.total_uncharged_usd)}`)
    }
    lines.push("")

    if (summary.by_type && summary.by_type.length > 0) {
      lines.push("By type:")
      for (const item of summary.by_type) {
        const byok = item.byok_event_count > 0 ? ` (${item.byok_event_count} BYOK)` : ""
        lines.push(
          `  ${item.usage_type.padEnd(12)} ${formatUSD(item.total_cost_usd).padStart(10)} (${item.event_count} events${byok})`,
        )
      }
    } else {
      lines.push("No usage in last 30 days.")
    }
  } else {
    lines.push("No usage data available.")
  }

  return lines.join("\n")
}

function formatTunnelDetails(
  tunnels: TunnelRecord[],
  healthResults: Map<string, TunnelHealthResult>,
  selectedIndex: number,
): string {
  const activeTunnels = tunnels.filter((t) => t.status === "active" && !t.deleted_at)
  if (activeTunnels.length === 0) {
    return "No active task apps (tunnels).\n\nTask apps are Cloudflare managed tunnels that expose\nlocal APIs to the internet for remote execution.\n\nPress 'q' to close."
  }

  const lines: string[] = []
  activeTunnels.forEach((tunnel, idx) => {
    const health = healthResults.get(tunnel.id)
    const isSelected = idx === selectedIndex

    let healthIcon = "?"
    let healthText = "checking..."
    if (health) {
      if (health.healthy) {
        healthIcon = "\u2713"
        healthText = health.response_time_ms != null
          ? `Healthy (${health.response_time_ms}ms)`
          : "Healthy"
      } else {
        healthIcon = "\u2717"
        healthText = health.error?.slice(0, 40) || "Unhealthy"
      }
    }

    const portMatch = tunnel.hostname.match(/task-(\d+)-\d+/)
    const displayPort = portMatch ? portMatch[1] : tunnel.local_port?.toString() || "?"

    const prefix = isSelected ? "> " : "  "
    const hostname = tunnel.hostname.replace(/^https?:\/\//, "")
    const shortHost = hostname.length > 50 ? hostname.slice(0, 47) + "..." : hostname

    lines.push(`${prefix}[${healthIcon}] ${shortHost}`)
    lines.push(`    Port: ${displayPort} | Status: ${healthText}`)
    lines.push(`    Local: ${tunnel.local_host}:${tunnel.local_port}`)
    if (tunnel.created_at) {
      const created = new Date(tunnel.created_at)
      lines.push(`    Created: ${created.toLocaleString()}`)
    }
    if (tunnel.org_name) {
      lines.push(`    Org: ${tunnel.org_name}`)
    }
    lines.push("")
  })

  return lines.join("\n")
}

function formatSessionDetails(
  sessions: SessionRecord[],
  healthResults: Map<string, SessionHealthResult>,
  selectedIndex: number,
  openCodeUrl: string | null,
): string {
  const activeSessions = sessions.filter(
    (s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting",
  )

  const serverUrl = openCodeUrl || "(not started)"

  if (activeSessions.length === 0) {
    return `No active OpenCode sessions.

Interactive sessions connect to local or remote OpenCode servers
for real-time agent interaction.

OpenCode server: ${serverUrl}

Quick connect:
  Press 'c' to connect to the local OpenCode server
  Press 'C' to connect with custom URL

Press 'q' to close.`
  }

  const lines: string[] = []

  activeSessions.forEach((session, idx) => {
    const health = healthResults.get(session.session_id)
    const isSelected = idx === selectedIndex

    let stateIcon = "?"
    let stateText: string = session.state
    if (session.state === "connected") {
      if (health) {
        if (health.healthy) {
          stateIcon = "\u2713"
          stateText = health.response_time_ms != null
            ? `Connected (${health.response_time_ms}ms)`
            : "Connected"
        } else {
          stateIcon = "\u2717"
          stateText = health.error?.slice(0, 30) || "Unhealthy"
        }
      } else {
        stateIcon = "\u2713"
        stateText = "Connected"
      }
    } else if (session.state === "connecting" || session.state === "reconnecting") {
      stateIcon = "\u21BB"
      stateText = session.state
    } else if (session.state === "error") {
      stateIcon = "\u2717"
      stateText = session.error_message?.slice(0, 30) || "Error"
    }

    const prefix = isSelected ? "> " : "  "
    const localTag = session.is_local ? " [local]" : ""

    lines.push(`${prefix}[${stateIcon}] ${session.session_id}${localTag}`)
    lines.push(`    State: ${stateText}`)
    lines.push(`    Mode: ${session.mode} | Model: ${session.model || "default"}`)

    if (session.opencode_url) {
      const shortUrl = session.opencode_url.length > 50
        ? session.opencode_url.slice(0, 47) + "..."
        : session.opencode_url
      lines.push(`    URL: ${shortUrl}`)
    }
    if (session.tunnel_url && session.tunnel_url !== session.opencode_url) {
      lines.push(`    Tunnel: ${session.tunnel_url}`)
    }

    if (session.connected_at) {
      const connectedAt = new Date(session.connected_at)
      lines.push(`    Connected: ${connectedAt.toLocaleString()}`)
    }
    if (session.last_activity) {
      const lastActivity = new Date(session.last_activity)
      lines.push(`    Last activity: ${lastActivity.toLocaleString()}`)
    }

    lines.push("")
  })

  return lines.join("\n")
}

function formatConfigMetadata(snapshot: Snapshot): string {
  const job = snapshot.selectedJob
  if (!job) return "(no metadata)"

  const lines: string[] = []
  lines.push(`Job: ${job.job_id}`)
  lines.push(`Status: ${job.status}`)
  lines.push(`Type: ${job.training_type || "-"}`)
  lines.push(`Source: ${job.job_source || "unknown"}`)
  lines.push("")

  if (snapshot.lastError && snapshot.status?.includes("Error")) {
    lines.push("═══ Error Loading Metadata ═══")
    lines.push(snapshot.lastError)
    lines.push("")
    lines.push("The job details could not be loaded.")
    return lines.join("\n")
  }

  const meta: any = job.metadata
  if (!meta || Object.keys(meta).length === 0) {
    if (snapshot.status?.includes("Loading")) {
      lines.push("Loading job configuration...")
      lines.push("")
      lines.push("Modal will auto-update when loaded.")
    } else if (!job.training_type) {
      lines.push("Loading job configuration...")
      lines.push("")
      lines.push("Press 'i' again after job details finish loading.")
    } else {
      lines.push("No metadata available for this job.")
      lines.push("")
      lines.push(`(job_source: ${job.job_source}, training_type: ${job.training_type})`)
    }
    return lines.join("\n")
  }

  const desc = meta.request_metadata?.description || meta.description
  if (desc) {
    lines.push(`Description: ${desc}`)
    lines.push("")
  }

  const rawConfig =
    meta.prompt_initial_snapshot?.raw_config?.prompt_learning
    || meta.config?.prompt_learning
    || meta.job_config?.prompt_learning
    || meta.prompt_learning
    || meta.config
    || meta.job_config
    || null

  const optimizerConfig = meta.prompt_initial_snapshot?.optimizer_config || meta.optimizer_config || null

  const policy = rawConfig?.policy || optimizerConfig?.policy_config
  if (policy) {
    lines.push("═══ Model Configuration ═══")
    if (policy.model) lines.push(`  Model: ${policy.model}`)
    if (policy.provider) lines.push(`  Provider: ${policy.provider}`)
    if (policy.temperature != null) lines.push(`  Temperature: ${policy.temperature}`)
    if (policy.max_completion_tokens) lines.push(`  Max Tokens: ${policy.max_completion_tokens}`)
    lines.push("")
  }

  try {
    const metaJson = JSON.stringify(meta, null, 2)
    if (metaJson.length < 2000) {
      lines.push("═══ Raw Metadata ═══")
      lines.push(metaJson)
    }
  } catch {
    // ignore
  }

  return lines.join("\n")
}

function wrapText(text: string, width: number): string[] {
  const lines: string[] = []
  for (const raw of text.split("\n")) {
    if (raw.length <= width) {
      lines.push(raw)
      continue
    }
    if (raw.trim() === "") {
      lines.push("")
      continue
    }
    let start = 0
    while (start < raw.length) {
      lines.push(raw.slice(start, start + width))
      start += width
    }
  }
  return lines
}
