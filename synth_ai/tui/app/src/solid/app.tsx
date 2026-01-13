import { render, useRenderer, useTerminalDimensions } from "@opentui/solid"
import { ErrorBoundary, Show, createEffect, createMemo, createSignal, onCleanup, onMount, type Component } from "solid-js"
import { Dynamic } from "solid-js/web"
import path from "node:path"

import { computeLayoutMetrics, defaultLayoutSpec } from "./layout"
import { useSolidData } from "./data"
import { COLORS } from "./theme"
import { KeyHint } from "./components/KeyHint"
import { ModalFrame } from "./components/ModalFrame"
import { JobsList } from "./ui/list-panels/JobsList"
import { LogsList } from "./ui/list-panels/LogsList"
import { JobsDetail } from "./ui/detail-panels/JobsDetail"
import { LogsDetail } from "./ui/detail-panels/LogsDetail"
import { ActiveModalRenderer } from "./modals/ActiveModalRenderer"
import type { ActiveModal, ModalState, UsageData } from "./modals/types"
import type { JobCreatedInfo } from "./modals/CreateJobModal"
import { toDisplayPath } from "./utils/files"
import { useLocalApiScanner } from "./utils/localapi-watch"
import { useLiveLogs } from "./utils/live-logs"
import { useJobEvents } from "./hooks/useJobEvents"
import { useJobSelection } from "./hooks/useJobSelection"
import { useAppKeybindings } from "./hooks/useAppKeybindings"

import { formatEventDetail, getFilteredEvents } from "../formatters"
import { buildJobStatusOptions, getFilteredJobs } from "../selectors/jobs"
import { cancelSelected, fetchArtifacts, fetchMetrics } from "../api/jobs"
import { apiGet } from "../api/client"
import { fetchSessions, disconnectSession, checkSessionHealth } from "../api/sessions"
import { refreshTunnels, refreshTunnelHealth } from "../api/tunnels"
import { connectLocalOpenCodeSession } from "../api/opencode"
import { openBrowser, runDeviceCodeAuth, type AuthStatus } from "../auth"
import { copyToClipboard } from "../utils/clipboard"
import { clearLoggedOutMarker, deleteSavedApiKey, saveApiKey, setLoggedOutMarker } from "../utils/logout-marker"
import { persistSettings } from "../persistence/settings"
import { moveEventSelection } from "./utils/events"
import { readLogFile } from "./utils/logs"
import type { JobEvent } from "../tui_data"
import type { SessionHealthResult, SessionRecord } from "../types"
import { focusManager } from "../focus"
import { modeKeys, modeUrls, switchMode } from "../state/app-state"
import { pollingState, clearEventsTimer, clearJobsTimer } from "../state/polling"
import { installSignalHandlers, registerCleanup, unregisterCleanup, registerRenderer, shutdown } from "../lifecycle"
import { createAbortControllerRegistry, isAbortError } from "../utils/abort"
import { isAborted } from "../utils/request"
import { clamp, wrapModalText } from "../utils/truncate"
import { formatActionKeys } from "../input/keymap"

function wireShutdown(renderer: { stop: () => void; destroy: () => void }): void {
  registerRenderer(renderer)
  installSignalHandlers() // Safe to call multiple times
}

export async function runSolidApp(): Promise<void> {
  return new Promise<void>((resolve) => {
    render(
      () => <SolidShell onExit={resolve} />,
      {
        targetFps: 30,
        exitOnCtrlC: false,
        useKittyKeyboard: {},
      },
    )
  })
}

function SolidShell(props: { onExit?: () => void }) {
  if (process.env.SYNTH_TUI_BENCH) {
    const start = (globalThis as any).__TUI_BENCH_START
    const elapsed = typeof start === "number" ? Date.now() - start : 0
    process.stderr.write(`tui_solid_shell_init ${elapsed}ms\n`)
  }
  const { onExit } = props
  const dimensions = useTerminalDimensions()
  const renderer = useRenderer()
  wireShutdown(renderer)

  // Set global renderer for OpenCode embed to find
  ;(globalThis as any).__OPENCODE_EMBED_RENDERER__ = renderer
  const layout = createMemo(() =>
    computeLayoutMetrics(dimensions().width, dimensions().height),
  )
  const data = useSolidData()
  const [chatPaneComponent, setChatPaneComponent] = createSignal<Component<any> | null>(null)
  const [candidatesModalComponent, setCandidatesModalComponent] = createSignal<Component<any> | null>(null)
  const [traceViewerModalComponent, setTraceViewerModalComponent] = createSignal<Component<any> | null>(null)
  const [createJobModalComponent, setCreateJobModalComponent] = createSignal<Component<any> | null>(null)
  let chatPaneLoading = false
  let candidatesModalLoading = false
  let traceViewerModalLoading = false
  let createJobModalLoading = false

  async function ensureChatPane(): Promise<void> {
    if (chatPaneComponent() || chatPaneLoading) return
    chatPaneLoading = true
    const mod = await import("./opencode")
    setChatPaneComponent(() => mod.ChatPane)
    chatPaneLoading = false
  }

  async function ensureCandidatesModal(): Promise<void> {
    if (candidatesModalComponent() || candidatesModalLoading) return
    candidatesModalLoading = true
    const mod = await import("./modals/CandidatesModal")
    setCandidatesModalComponent(() => mod.CandidatesModal)
    candidatesModalLoading = false
  }

  async function ensureTraceViewerModal(): Promise<void> {
    if (traceViewerModalComponent() || traceViewerModalLoading) return
    traceViewerModalLoading = true
    const mod = await import("./modals/TraceViewerModal")
    setTraceViewerModalComponent(() => mod.TraceViewerModal)
    traceViewerModalLoading = false
  }

  async function ensureCreateJobModal(): Promise<void> {
    if (createJobModalComponent() || createJobModalLoading) return
    createJobModalLoading = true
    const mod = await import("./modals/CreateJobModal")
    setCreateJobModalComponent(() => mod.CreateJobModal)
    createJobModalLoading = false
  }
  const actions = createAbortControllerRegistry()
  const actionsCleanupName = "solid-actions-abort"
  registerCleanup(actionsCleanupName, () => actions.abortAll())
  onCleanup(() => {
    actions.abortAll()
    unregisterCleanup(actionsCleanupName)
  })
  const appState = data.ctx.state.appState
  const snapshot = data.ctx.state.snapshot
  onMount(() => {
    if (!process.env.SYNTH_TUI_BENCH) return
    const start = (globalThis as any).__TUI_BENCH_START
    const elapsed = typeof start === "number" ? Date.now() - start : null
    setTimeout(() => {
      const suffix = elapsed == null ? "" : ` ${elapsed}ms`
      process.stderr.write(`tui_first_render${suffix}\n`)
      onExit?.()
      void shutdown(0)
    }, 0)
  })
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
  const focusTarget = createMemo(() => {
    data.version()
    return data.ctx.state.appState.focusTarget
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
  createEffect(() => {
    if (principalPane() === "opencode") {
      void ensureChatPane()
    }
  })
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

  useJobEvents({
    ctx: data.ctx,
    selectedJobId,
    activePane,
    principalPane,
  })
  const liveLogs = useLiveLogs({
    listActive: () => activePane() === "logs",
    detailActive: () => activePane() === "logs" && principalPane() === "jobs",
    onSelectionAdjusted: () => data.ctx.render(),
  })

  const logFiles = createMemo(() => (
    activePane() === "logs" ? liveLogs.files() : []
  ))
  const logsTitle = createMemo(() => {
    const files = logFiles()
    const total = files.length
    const visible = Math.max(1, layout().contentHeight - 2)
    const selected = clamp(liveLogs.selectedIndex(), 0, Math.max(0, total - 1))
    let windowStart = clamp(0, 0, Math.max(0, total - visible))
    if (selected < windowStart) {
      windowStart = selected
    } else if (selected >= windowStart + visible) {
      windowStart = selected - visible + 1
    }
    if (total > visible) {
      const end = Math.min(windowStart + visible, total)
      return `Logs (files) [${windowStart + 1}-${end}/${total}]`
    }
    return "Logs (files)"
  })
  const logsView = createMemo(() => {
    if (activePane() !== "logs" || principalPane() !== "jobs") {
      return { lines: [], visible: [] }
    }
    const lines = liveLogs.lines()
    if (lines.length === 0) {
      return { lines: [], visible: [] }
    }
    const visibleHeight = Math.max(1, layout().contentHeight - 4)
    const offset = Math.max(0, lines.length - visibleHeight)
    const visible = lines.slice(offset, offset + visibleHeight)
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
  const localApiScanner = useLocalApiScanner({
    enabled: showCreateJobModal,
    directories: () => [process.cwd()],
  })
  const localApiFiles = createMemo(() => (
    localApiScanner.files().map((api) => toDisplayPath(api.filepath))
  ))
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
    return wrapModalText(state.raw, maxWidth)
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
    const fullscreenHint = `${formatActionKeys("detail.toggleFullscreen", { primaryOnly: true })} fullscreen | `
    const scrollHint = `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} scroll`
    if (state.type === "log") {
      const tail = state.tail ? " [TAIL]" : ""
      return `${range}${fullscreenHint}${scrollHint} | ${formatActionKeys("detail.tail", { primaryOnly: true })} tail${tail} | ${formatActionKeys("modal.copy", { primaryOnly: true })} copy | ${formatActionKeys("app.back")} close`
    }
    return `${range}${fullscreenHint}${scrollHint} | ${formatActionKeys("app.back")} close`
  })

  function runAction(key: string, task: () => Promise<void>): void {
    void actions.run(key, () => task()).catch((err) => {
      if (isAbortError(err)) return
      snapshot.lastError = err?.message || "Action failed"
      data.ctx.render()
    })
  }

  useJobSelection({
    jobs,
    selectedIndex,
    setSelectedIndex,
    activePane,
    snapshot: snapshotMemo,
    onSelectJob: (jobId) => runAction("select-job", () => data.select(jobId)),
  })

  createEffect(() => {
    const current = modal()
    if (!current || current.type !== "log") return
    const filePath = current.path
    let disposed = false
    const timer = setInterval(() => {
      void readLogFile(filePath).then((raw) => {
        if (disposed) return
        setModal((prev) => {
          if (!prev || prev.type !== "log") return prev
          return { ...prev, raw }
        })
      })
    }, 1000)
    const cleanupName = "log-modal-refresh-interval"
    const cleanup = () => {
      clearInterval(timer)
    }
    registerCleanup(cleanupName, cleanup)
    onCleanup(() => {
      disposed = true
      cleanup()
      unregisterCleanup(cleanupName)
    })
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

  async function openLogModal(filePath: string): Promise<void> {
    const raw = await readLogFile(filePath)
    setModal({
      type: "log",
      title: `Log: ${path.basename(filePath)}`,
      raw,
      offset: 0,
      tail: true,
      path: filePath,
    })
  }

  function openMetricsAndFetch(): void {
    openMetricsModal()
    runAction("metrics", async () => {
      await fetchMetrics(data.ctx)
      data.ctx.render()
    })
  }

  focusManager.register({
    id: "list",
    order: 0,
    enabled: () => true,
    handleAction: (action) => {
      if (action === "nav.down" || action === "nav.up") {
        const delta = action === "nav.down" ? 1 : -1
        if (appState.activePane === "jobs") {
          const count = jobs().length
          if (count > 0) {
            setSelectedIndex((current) => (current + delta + count) % count)
          }
        } else if (appState.activePane === "logs") {
          if (liveLogs.moveSelection(delta)) {
            data.ctx.render()
          }
        }
        return true
      }
      if (action === "pane.select") {
        if (appState.activePane === "jobs") {
          const job = jobs()[selectedIndex()]
          if (job?.job_id) {
            runAction("select-job", () => data.select(job.job_id))
          }
        } else if (appState.activePane === "logs") {
          const selected = liveLogs.selectedIndex()
          const file = logFiles()[selected]
          if (file) {
            void openLogModal(file.path)
          }
        }
        return true
      }
      return false
    },
  })

  focusManager.register({
    id: "metrics",
    order: 1,
    enabled: () => appState.principalPane === "jobs" && appState.activePane === "jobs",
    handleAction: (action) => {
      if (action === "pane.select") {
        openMetricsAndFetch()
        return true
      }
      return false
    },
  })

  focusManager.register({
    id: "events",
    order: 2,
    enabled: () => appState.principalPane === "jobs" && appState.activePane === "jobs",
    handleAction: (action) => {
      if (action === "nav.down" || action === "nav.up") {
        const delta = action === "nav.down" ? 1 : -1
        moveEventSelection(data.ctx, delta)
        data.ctx.render()
        return true
      }
      if (action === "pane.select") {
        const event = events()[eventWindow().selected]
        if (event) {
          openEventModal(event)
        }
        return true
      }
      return false
    },
  })

  focusManager.register({
    id: "agent",
    order: 3,
    enabled: () => appState.principalPane === "opencode",
  })

  function closeActiveModal(): void {
    const kind = activeModal()
    if (kind === "usage") {
      actions.abort("usage")
    } else if (kind === "sessions") {
      actions.abort("sessions-refresh")
      actions.abort("sessions-connect")
      actions.abort("sessions-disconnect")
    } else if (kind === "task-apps") {
      actions.abort("task-apps-refresh")
    } else if (kind === "metrics") {
      actions.abort("metrics")
    }
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
      if (isAbortError(err)) return
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

  async function pasteKeyModal(): Promise<void> {
    try {
      if (process.platform !== "darwin") return
      const { spawn } = require("child_process")
      const proc = spawn("pbpaste", [], { stdio: ["ignore", "pipe", "ignore"] })
      let output = ""
      proc.stdout?.on("data", (data: Buffer) => {
        output += data.toString("utf8")
      })
      await new Promise<void>((resolve, reject) => {
        proc.on("error", reject)
        proc.on("close", () => resolve())
      })
      const text = output.replace(/\s+/g, "")
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
    void ensureCandidatesModal()
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
      runAction("select-job", async () => {
        await data.select(filteredJobs[0].job_id)
        data.ctx.render()
      })
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
    appState.settingsOptions = ["prod", "dev", "local"]
    setSettingsCursor(Math.max(
      0,
      appState.settingsOptions.indexOf(appState.currentMode),
    ))
    setActiveModal("settings")
  }

  function moveSettingsCursor(delta: number): void {
    const max = Math.max(0, appState.settingsOptions.length - 1)
    setSettingsCursor((cur) => clamp(cur + delta, 0, max))
  }

  async function selectSettingsBackend(): Promise<void> {
    const selectedMode = appState.settingsOptions[settingsCursor()]
    if (!selectedMode) return
    const urls = modeUrls[selectedMode]
    if (!urls.backendUrl || !urls.frontendUrl) {
      snapshot.status = `Missing URLs for ${selectedMode}.`
      data.ctx.render()
      return
    }

    // Switch mode (updates env vars and state)
    switchMode(selectedMode)

    closeActiveModal()
    snapshot.status = `Switching to ${selectedMode}...`
    data.ctx.render()

    await persistSettings({
      settingsFilePath: data.ctx.state.config.settingsFilePath,
      getCurrentMode: () => appState.currentMode,
      getModeKeys: () => modeKeys,
    })
    await data.refresh()
  }

  function openUsageModal(): void {
    appState.usageModalOffset = 0
    setUsageData(null)
    setActiveModal("usage")
    runAction("usage", () => fetchUsageData())
  }

  function openMetricsModal(): void {
    appState.metricsModalOffset = 0
    setActiveModal("metrics")
  }

  async function fetchUsageData(): Promise<void> {
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
      snapshot.lastError = `Usage fetch failed: ${err?.message || "Unknown"}`
      data.ctx.render()
    }
  }

  function openUsageBilling(): void {
    try {
      const frontendUrl = process.env.SYNTH_FRONTEND_URL || ""
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
    runAction("task-apps-refresh", () => refreshTaskApps())
  }

  async function refreshTaskApps(): Promise<void> {
    snapshot.status = "Loading task apps..."
    data.ctx.render()
    const ok = await refreshTunnels(data.ctx)
    if (!ok || isAborted()) return
    data.ctx.render()
    await refreshTunnelHealth(data.ctx)
    if (!isAborted()) {
      snapshot.status = "Task apps updated"
      data.ctx.render()
    }
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
    runAction("sessions-refresh", () => refreshSessionsModal())
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
      if (isAborted()) return
      snapshot.sessions = sessions
      setSessionsCache(sessions)
      await refreshSessionHealth(sessions)
    } catch (err: any) {
      if (isAbortError(err)) return
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
      if (isAborted()) return
      const result = await checkSessionHealth(session, 5000)
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

    const result = await connectLocalOpenCodeSession(opencodeUrl, 5000)
    if (result.aborted) return
    if (!result.ok) {
      snapshot.lastError = result.error
      snapshot.status = result.health?.healthy
        ? "Session creation failed"
        : `Connection failed - is OpenCode running at ${opencodeUrl}?`
      data.ctx.render()
      return
    }

    const sessionId = result.session.session_id
    const nextSessions = [result.session, ...sessionsCache().filter((s) => s.session_id !== sessionId)]
    setSessionsCache(nextSessions)
    snapshot.sessions = nextSessions
    const nextHealth = new Map(sessionsHealthCache())
    nextHealth.set(sessionId, result.health)
    snapshot.sessionHealthResults.set(sessionId, result.health)
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
    runAction("opencode-connect", () => connectLocalSession())
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
      // Store key for current mode
      modeKeys[appState.currentMode] = result.apiKey
      process.env.SYNTH_API_KEY = result.apiKey
      await saveApiKey(result.apiKey)
      await clearLoggedOutMarker()
      await persistSettings({
        settingsFilePath: data.ctx.state.config.settingsFilePath,
        getCurrentMode: () => appState.currentMode,
        getModeKeys: () => modeKeys,
      })
      closeActiveModal()
      snapshot.lastError = null
      snapshot.status = "Authenticated! Loading..."
      data.ctx.render()
      await data.refresh()
    }
  }

  async function logout(): Promise<void> {
    actions.abortAll()
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

  useAppKeybindings({
    onExit,
    render: data.ctx.render,
    snapshot,
    modal,
    setModal,
    activeModal,
    setActiveModal,
    showCreateJobModal,
    setShowCreateJobModal,
    createJobModalComponent,
    chatPaneComponent,
    closeActiveModal,
    applyFilterModal,
    applySnapshotModal,
    applyKeyModal,
    pasteKeyModal,
    moveSettingsCursor,
    selectSettingsBackend,
    openKeyModal,
    openUsageBilling,
    moveTaskAppsSelection,
    copySelectedTunnelUrl,
    moveSessionsSelection,
    copySelectedSessionUrl,
    connectLocalSession,
    disconnectSelectedSession,
    refreshSessionsModal,
    selectSession,
    moveJobFilter,
    toggleJobFilterSelection,
    clearJobFilterSelection,
    startLoginAuth,
    openFilterModal,
    openConfigModal,
    openProfileModal,
    openResultsModal,
    openJobFilterModal,
    openSnapshotModal,
    openUrlsModal,
    openSettingsModal,
    openUsageModal,
    openTaskAppsModal,
    openSessionsModal,
    openMetricsAndFetch,
    ensureCreateJobModal,
    ensureTraceViewerModal,
    logout,
    runAction,
    refreshData: data.refresh,
    ensureOpenCodeServer: data.ensureOpenCodeServer,
    cancelSelectedJob: async () => {
      await cancelSelected(data.ctx)
      data.ctx.render()
    },
    fetchArtifacts: async () => {
      await fetchArtifacts(data.ctx)
      data.ctx.render()
    },
    refreshMetrics: async () => {
      await fetchMetrics(data.ctx)
      data.ctx.render()
    },
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
        <KeyHint description="Create New Job" keyLabel={formatActionKeys("modal.open.createJob", { primaryOnly: true })} />
        <KeyHint description="View Jobs" keyLabel={formatActionKeys("pane.jobs", { primaryOnly: true })} active={activePane() === "jobs"} />
        <KeyHint description="View Logs" keyLabel={formatActionKeys("pane.logs", { primaryOnly: true })} active={activePane() === "logs"} />
        <KeyHint description="Agent" keyLabel={formatActionKeys("pane.togglePrincipal", { primaryOnly: true })} active={principalPane() === "opencode"} />
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
              selectedIndex={liveLogs.selectedIndex()}
              focused={focusTarget() === "list"}
              width={layout().jobsWidth}
              height={layout().contentHeight}
            />
          }
        >
          <JobsList
            jobs={jobs()}
            selectedIndex={selectedIndex()}
            focused={focusTarget() === "list"}
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
                <Show
                  when={chatPaneComponent()}
                  fallback={
                    <box flexDirection="column" paddingLeft={2} paddingTop={1}>
                      <text fg={COLORS.textDim}>Loading agent...</text>
                    </box>
                  }
                >
                  <Dynamic
                    component={chatPaneComponent() as Component<any>}
                    url={opencodeUrl()}
                    sessionId={opencodeSessionId()}
                    width={opencodeDimensions().width}
                    height={opencodeDimensions().height}
                    onExit={() => {
                      data.ctx.state.appState.principalPane = "jobs"
                      data.ctx.render()
                    }}
                  />
                </Show>
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
              eventsFocused={focusTarget() === "events"}
              metricsFocused={focusTarget() === "metrics"}
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
        {(kind) => (
          <ActiveModalRenderer
            kind={kind()}
            dataVersion={data.version}
            dimensions={dimensions}
            setModalInputValue={setModalInputValue}
            setModalInputRef={(ref) => {
              modalInputRef = ref
            }}
            settingsCursor={settingsCursor}
            usageData={usageData}
            sessionsCache={sessionsCache}
            sessionsHealthCache={sessionsHealthCache}
            sessionsSelectedIndex={sessionsSelectedIndex}
            sessionsScrollOffset={sessionsScrollOffset}
            loginStatus={loginStatus}
            candidatesModalComponent={candidatesModalComponent}
            traceViewerModalComponent={traceViewerModalComponent}
            closeActiveModal={closeActiveModal}
            onStatusUpdate={(message: string) => {
              snapshot.status = message
              data.ctx.render()
            }}
            snapshot={snapshot}
          />
        )}
      </Show>

      <Show
        when={createJobModalComponent()}
        fallback={
          <Show when={showCreateJobModal()}>
            <ModalFrame
              title="Create Job"
              width={50}
              height={8}
              borderColor="#60a5fa"
              titleColor="#60a5fa"
              hint="Loading..."
              dimensions={dimensions}
            >
              <text fg="#e2e8f0">Loading create job flow...</text>
            </ModalFrame>
          </Show>
        }
      >
        <Dynamic
          component={createJobModalComponent() as Component<any>}
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
            runAction("refresh", () => data.refresh())
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
      </Show>

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
              <KeyHint
                description="select"
                keyLabel={`${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })}`}
              />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="view" keyLabel={formatActionKeys("pane.select")} />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="refresh" keyLabel={formatActionKeys("app.refresh")} />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="candidates" keyLabel={formatActionKeys("modal.open.results", { primaryOnly: true })} />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="metrics" keyLabel={formatActionKeys("modal.open.metrics", { primaryOnly: true })} />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="new" keyLabel={formatActionKeys("modal.open.createJob", { primaryOnly: true })} />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint
                description="focus"
                keyLabel={`${formatActionKeys("focus.next", { primaryOnly: true })}/${formatActionKeys("focus.prev", { primaryOnly: true })}`}
              />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="agent" keyLabel={formatActionKeys("pane.togglePrincipal", { primaryOnly: true })} />
              <text fg={COLORS.textDim}>|</text>
              <KeyHint description="quit" keyLabel={formatActionKeys("app.quit", { primaryOnly: true })} />
            </box>
          }
        >
          <box flexDirection="row" gap={1}>
            <KeyHint description="back" keyLabel={formatActionKeys("app.back", { primaryOnly: true })} />
            <text fg={COLORS.textDim}>|</text>
            <KeyHint description="sessions" keyLabel={formatActionKeys("pane.openSessions", { primaryOnly: true })} />
          </box>
        </Show>
      </box>
    </box>
  )
}
