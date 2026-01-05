import {
  createCliRenderer,
  BoxRenderable,
  TextRenderable,
  SelectRenderable,
  InputRenderable,
  SelectRenderableEvents,
  InputRenderableEvents,
} from "@opentui/core"
import path from "node:path"
import { promises as fs } from "node:fs"
import {
  coerceJob,
  extractEvents,
  extractJobs,
  isEvalJob,
  mergeJobs,
  num,
  type JobEvent,
  type JobSummary,
} from "./tui_data"
import { createLoginModal, type LoginModalController } from "./login_modal"

type EnvKeyOption = {
  key: string
  sources: string[]
  varNames: string[]
}

/** A prompt candidate (snapshot) with score */
type PromptCandidate = {
  id: string
  isBaseline: boolean
  score: number | null
  payload: Record<string, any>
  createdAt: string | null
  tag: string | null
}

type Snapshot = {
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

type BackendId = "prod" | "dev" | "local"

type BackendConfig = {
  id: BackendId
  label: string
  baseUrl: string
}

type BackendKeySource = {
  sourcePath: string | null
  varName: string | null
}

const backendConfigs: Record<BackendId, BackendConfig> = {
  prod: {
    id: "prod",
    label: "Prod",
    baseUrl: ensureApiBase(
      process.env.SYNTH_TUI_PROD_API_BASE || "https://api.usesynth.ai/api",
    ),
  },
  dev: {
    id: "dev",
    label: "Dev",
    baseUrl: ensureApiBase(
      process.env.SYNTH_TUI_DEV_API_BASE || "https://agent-learning.onrender.com/api",
    ),
  },
  local: {
    id: "local",
    label: "Local",
    baseUrl: ensureApiBase(
      process.env.SYNTH_TUI_LOCAL_API_BASE || "http://localhost:8000/api",
    ),
  },
}

const backendKeys: Record<BackendId, string> = {
  prod: process.env.SYNTH_TUI_API_KEY_PROD || process.env.SYNTH_API_KEY || "",
  dev: process.env.SYNTH_TUI_API_KEY_DEV || "",
  local: process.env.SYNTH_TUI_API_KEY_LOCAL || "",
}

const backendKeySources: Record<BackendId, BackendKeySource> = {
  prod: { sourcePath: null, varName: null },
  dev: { sourcePath: null, varName: null },
  local: { sourcePath: null, varName: null },
}

let currentBackend: BackendId = normalizeBackendId(
  process.env.SYNTH_TUI_BACKEND || "prod",
)
const initialJobId = process.env.SYNTH_TUI_JOB_ID || ""
const refreshInterval = parseFloat(process.env.SYNTH_TUI_REFRESH_INTERVAL || "5")
const eventInterval = parseFloat(process.env.SYNTH_TUI_EVENT_INTERVAL || "2")
const maxRefreshInterval = parseFloat(process.env.SYNTH_TUI_REFRESH_MAX || "60")
const maxEventInterval = parseFloat(process.env.SYNTH_TUI_EVENT_MAX || "15")
const eventHistoryLimit = parseInt(process.env.SYNTH_TUI_EVENT_CARDS || "200", 10)
const eventCollapseLimit = parseInt(process.env.SYNTH_TUI_EVENT_COLLAPSE || "160", 10)
const eventVisibleCount = parseInt(process.env.SYNTH_TUI_EVENT_VISIBLE || "6", 10)
const jobLimit = parseInt(process.env.SYNTH_TUI_LIMIT || "50", 10)
const envKeyVisibleCount = parseInt(process.env.SYNTH_TUI_ENV_KEYS_VISIBLE || "8", 10)
const envKeyScanRoot = process.env.SYNTH_TUI_ENV_SCAN_ROOT || process.cwd()
const settingsFilePath =
  process.env.SYNTH_TUI_SETTINGS_FILE || path.join(process.cwd(), ".env.synth")

const snapshot: Snapshot = {
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
}

let lastSeq = 0
let autoSelected = false
let healthStatus = "unknown"
let jobsPollMs = Math.max(1, refreshInterval) * 1000
let eventsPollMs = Math.max(0.5, eventInterval) * 1000
let jobsInFlight = false
let eventsInFlight = false
let jobsTimer: ReturnType<typeof setTimeout> | null = null
let eventsTimer: ReturnType<typeof setTimeout> | null = null
let selectedEventIndex = 0
let activePane: "jobs" | "events" = "jobs"
let eventWindowStart = 0
let eventModalOffset = 0
let resultsModalOffset = 0
let promptBrowserIndex = 0
let promptBrowserOffset = 0
let jobSelectToken = 0
let eventsToken = 0
let eventFilter = ""
let jobStatusFilter = new Set<string>()
let jobFilterOptions: Array<{ status: string; count: number }> = []
let jobFilterCursor = 0
let jobFilterWindowStart = 0
const jobFilterVisibleCount = 6
let settingsCursor = 0
let settingsOptions: BackendConfig[] = []
let keyModalBackend: BackendId = "prod"
let keyPasteActive = false
let keyPasteBuffer = ""
let envKeyOptions: EnvKeyOption[] = []
let envKeyCursor = 0
let envKeyWindowStart = 0
let envKeyScanInProgress = false
let envKeyError: string | null = null

await loadPersistedSettings()

const renderer = await createCliRenderer({
  useConsole: false,
  useAlternateScreen: true,
  openConsoleOnError: false,
  backgroundColor: "#0b1120",
})
const ui = buildLayout(renderer)

// Login modal controller
const loginModal = createLoginModal({
  ui,
  renderer,
  getCurrentBackend: () => currentBackend,
  getBackendConfig: () => getBackendConfig(),
  getBackendKeys: () => backendKeys,
  setBackendKey: (backend, key, source) => {
    backendKeys[backend] = key
    backendKeySources[backend] = source
  },
  persistSettings,
  bootstrap,
  getSnapshot: () => snapshot,
  renderSnapshot,
  getActivePane: () => activePane,
})

renderer.start()

renderer.keyInput.on("keypress", (key: any) => {
  if (key.ctrl && key.name === "c") {
    renderer.stop()
    renderer.destroy()
    process.exit(0)
  }
  if (key.name === "q" || key.name === "escape") {
    if (loginModal.isVisible) {
      loginModal.toggle(false)
      return
    }
    if (ui.keyModalVisible) {
      toggleKeyModal(false)
      return
    }
    if (ui.envKeyModalVisible) {
      toggleEnvKeyModal(false)
      return
    }
    if (ui.jobFilterModalVisible) {
      toggleJobFilterModal(false)
      return
    }
    if (ui.settingsModalVisible) {
      toggleSettingsModal(false)
      return
    }
    if (ui.eventModalVisible) {
      toggleEventModal(false)
      return
    }
    if (ui.resultsModalVisible) {
      toggleResultsModal(false)
      return
    }
    if (ui.configModalVisible) {
      toggleConfigModal(false)
      return
    }
    if (ui.promptBrowserVisible) {
      togglePromptBrowserModal(false)
      return
    }
    if (ui.modalVisible) {
      toggleModal(false)
    } else {
      renderer.stop()
      renderer.destroy()
      process.exit(0)
    }
  }
  // Login modal handlers
  if (loginModal.isVisible) {
    if (key.name === "q") {
      loginModal.toggle(false)
      return
    }
    if (key.name === "return" || key.name === "enter") {
      if (!loginModal.isInProgress) {
        void loginModal.startAuth()
      }
      return
    }
    return
  }
  if (ui.keyModalVisible) {
    if (key.name === "q") {
      toggleKeyModal(false)
      return
    }
    if (key.name === "return" || key.name === "enter") {
      applyKeyModal(ui.keyModalInput.value || "")
      return
    }
    if (key.name === "v" && (key.ctrl || key.meta)) {
      pasteKeyFromClipboard()
      return
    }
    if (key.name === "backspace" || key.name === "delete") {
      const current = ui.keyModalInput.value || ""
      ui.keyModalInput.value = current.slice(0, Math.max(0, current.length - 1))
      renderer.requestRender()
      return
    }
    const seq = key.sequence || ""
    if (seq) {
      if (seq.includes("\u001b[200~")) {
        beginPasteCapture(seq)
        return
      }
      if (keyPasteActive) {
        continuePasteCapture(seq)
        return
      }
      if (!seq.startsWith("\u001b") && !key.ctrl && !key.meta) {
        ui.keyModalInput.value = (ui.keyModalInput.value || "") + seq
        renderer.requestRender()
        return
      }
    }
    return
  }
  if (ui.envKeyModalVisible) {
    if (key.name === "q") {
      toggleEnvKeyModal(false)
      return
    }
    if (key.name === "return" || key.name === "enter") {
      applyEnvKeySelection()
      return
    }
    if (key.name === "m") {
      toggleEnvKeyModal(false)
      openKeyModal()
      return
    }
    if (key.name === "r") {
      void rescanEnvKeys()
      return
    }
    if (key.name === "up" || key.name === "k") {
      moveEnvKeySelection(-1)
      return
    }
    if (key.name === "down" || key.name === "j") {
      moveEnvKeySelection(1)
      return
    }
    return
  }
  if (ui.eventModalVisible) {
    if (key.name === "up" || key.name === "k") {
      moveEventModal(-1)
    }
    if (key.name === "down" || key.name === "j") {
      moveEventModal(1)
    }
    if (key.name === "return" || key.name === "enter") {
      toggleEventModal(false)
    }
    return
  }
  if (ui.resultsModalVisible) {
    if (key.name === "up" || key.name === "k") {
      moveResultsModal(-1)
    }
    if (key.name === "down" || key.name === "j") {
      moveResultsModal(1)
    }
    if (key.name === "return" || key.name === "enter") {
      toggleResultsModal(false)
    }
    if (key.name === "y") {
      void copyPromptToClipboard()
    }
    return
  }
  if (ui.configModalVisible) {
    if (key.name === "up" || key.name === "k") {
      moveConfigModal(-1)
    }
    if (key.name === "down" || key.name === "j") {
      moveConfigModal(1)
    }
    if (key.name === "return" || key.name === "enter" || key.name === "i") {
      toggleConfigModal(false)
    }
    return
  }
  if (ui.promptBrowserVisible) {
    // Scroll within current candidate
    if (key.name === "up" || key.name === "k") {
      movePromptBrowserScroll(-1)
    }
    if (key.name === "down" || key.name === "j") {
      movePromptBrowserScroll(1)
    }
    // Navigate between candidates
    if (key.name === "left" || key.name === "h") {
      movePromptBrowserCandidate(-1)
    }
    if (key.name === "right" || key.name === "l") {
      movePromptBrowserCandidate(1)
    }
    // Copy to clipboard
    if (key.name === "y") {
      const content = getPromptBrowserClipboardContent()
      if (content) {
        void copyToClipboard(content).then(() => {
          snapshot.status = "Prompt copied to clipboard"
          renderSnapshot()
        })
      }
    }
    // Close modal
    if (key.name === "return" || key.name === "enter") {
      togglePromptBrowserModal(false)
    }
    return
  }
  if (ui.settingsModalVisible) {
    if (key.name === "up" || key.name === "k") {
      moveSettingsSelection(-1)
      return
    }
    if (key.name === "down" || key.name === "j") {
      moveSettingsSelection(1)
      return
    }
    if (key.name === "a") {
      void openEnvKeyModal()
      return
    }
    if (key.name === "m") {
      openKeyModal()
      return
    }
    if (key.name === "return" || key.name === "enter") {
      void applySettingsSelection()
      return
    }
    if (key.name === "q") {
      toggleSettingsModal(false)
      return
    }
    return
  }
  if (ui.filterModalVisible) {
    if (key.name === "q" || key.name === "escape") {
      toggleFilterModal(false)
    }
    return
  }
  if (ui.jobFilterModalVisible) {
    if (key.name === "q") {
      toggleJobFilterModal(false)
      return
    }
    if (key.name === "return" || key.name === "enter" || key.name === "space") {
      toggleSelectedJobFilter()
      return
    }
    if (key.name === "up" || key.name === "k") {
      moveJobFilterSelection(-1)
      return
    }
    if (key.name === "down" || key.name === "j") {
      moveJobFilterSelection(1)
      return
    }
    if (key.name === "a") {
      jobStatusFilter.clear()
      const statuses = collectJobStatuses(snapshot.jobs)
      for (const status of statuses) {
        jobStatusFilter.add(status)
      }
      refreshJobFilterOptions()
      applyJobFilterSelection()
      return
    }
    if (key.name === "x") {
      jobStatusFilter.clear()
      refreshJobFilterOptions()
      applyJobFilterSelection()
      return
    }
    return
  }
  if (ui.modalVisible) return
  if (key.name === "e") setActivePane("events")
  if (key.name === "tab") setActivePane(activePane === "jobs" ? "events" : "jobs")
  if (key.name === "b") setActivePane("jobs")
  if (key.name === "r") refreshJobs()
  if (key.name === "m") fetchMetrics()
  if (key.name === "p") fetchBestSnapshot()
  if (key.name === "f") toggleFilterModal(true)
  if (key.name === "t") toggleSettingsModal(true)
  if (key.name === "l" && !key.shift) loginModal.toggle(true)
  if (key.shift && key.name === "l") void loginModal.logout()
  if (key.shift && key.name === "j") toggleJobFilterModal(true)
  if (key.name === "c") cancelSelected()
  if (key.name === "a") fetchArtifacts()
  if (key.name === "s") {
    if (snapshot.selectedJob) toggleModal(true)
  }
  if (key.name === "o") {
    openResultsModal()
  }
  if (key.name === "i") {
    openConfigModal()
  }
  if (key.shift && key.name === "p") {
    void openPromptBrowserModal()
  }
  if (activePane === "events" && (key.name === "up" || key.name === "k")) {
    moveEventSelection(-1)
  }
  if (activePane === "events" && (key.name === "down" || key.name === "j")) {
    moveEventSelection(1)
  }
  if (activePane === "events" && (key.name === "return" || key.name === "enter")) {
    openSelectedEventModal()
  }
})

renderer.keyInput.on("paste", (key: any) => {
  if (!ui.keyModalVisible) return
  const seq = typeof key?.sequence === "string" ? key.sequence : ""
  if (!seq) return
  const cleaned = seq
    .replace("\u001b[200~", "")
    .replace("\u001b[201~", "")
    .replace(/\s+/g, "")
  if (!cleaned) return
  ui.keyModalInput.value = (ui.keyModalInput.value || "") + cleaned
  renderer.requestRender()
})

ui.jobsSelect.on(SelectRenderableEvents.SELECTION_CHANGED, (_idx: number, option: any) => {
  if (!option?.value) return
  if (snapshot.selectedJob?.job_id !== option.value) {
    selectJob(option.value)
  }
})

ui.modalInput.on(InputRenderableEvents.CHANGE, (value: string) => {
  if (!value.trim()) {
    toggleModal(false)
    return
  }
  fetchSnapshot(value.trim())
  toggleModal(false)
})

ui.filterInput.on(InputRenderableEvents.CHANGE, (value: string) => {
  eventFilter = value.trim()
  toggleFilterModal(false)
  renderSnapshot()
})

// job filter list is rendered manually

ui.keyModalInput.on(InputRenderableEvents.ENTER, (value: string) => {
  applyKeyModal(value)
})

ui.modalInput.on(InputRenderableEvents.ENTER, (value: string) => {
  if (!value.trim()) {
    toggleModal(false)
    return
  }
  fetchSnapshot(value.trim())
  toggleModal(false)
})

ui.jobsSelect.focus()
renderSnapshot()

if (!getActiveApiKey()) {
  snapshot.lastError = `Missing API key for ${getBackendConfig().label}`
  snapshot.status = "Sign in required"
  renderSnapshot()
  // Auto-show login modal when no API key
  loginModal.toggle(true)
} else {
  bootstrap().catch((err) => {
    snapshot.lastError = err?.message || "Bootstrap failed"
    snapshot.status = "Startup error"
    renderSnapshot()
  })
}

async function bootstrap(): Promise<void> {
  void refreshHealth()
  await refreshIdentity()
  await refreshJobs()
  if (initialJobId) {
    await selectJob(initialJobId)
  } else if (snapshot.jobs.length > 0) {
    await selectJob(snapshot.jobs[0].job_id)
  }
  scheduleJobsPoll(0)
  scheduleEventsPoll(0)
  setInterval(refreshHealth, 30_000)
  setInterval(refreshIdentity, 60_000)
  renderSnapshot()
}

async function refreshIdentity(): Promise<void> {
  try {
    const me = await apiGetV1("/me")
    snapshot.orgId = typeof me?.org_id === "string" ? me.org_id : null
    snapshot.userId = typeof me?.user_id === "string" ? me.user_id : null
  } catch (err: any) {
    snapshot.orgId = snapshot.orgId || null
    snapshot.userId = snapshot.userId || null
  }
  try {
    const balance = await apiGetV1("/balance/autumn-normalized")
    const cents = balance?.remaining_credits_cents
    const dollars = typeof cents === "number" && Number.isFinite(cents) ? cents / 100 : null
    snapshot.balanceDollars = dollars
  } catch (err: any) {
    snapshot.balanceDollars = snapshot.balanceDollars || null
  }
  renderSnapshot()
}

async function refreshJobs(): Promise<boolean> {
  try {
    snapshot.status = "Refreshing jobs..."
    const promptPayload = await apiGet(`/prompt-learning/online/jobs?limit=${jobLimit}&offset=0`)
    const promptJobs = extractJobs(promptPayload, "prompt-learning")

    // Fetch learning jobs (includes both eval and graph_gepa/GEPA jobs)
    let learningJobs: JobSummary[] = []
    let learningError: string | null = null
    try {
      const learningPayload = await apiGet(`/learning/jobs?limit=${jobLimit}`)
      learningJobs = extractJobs(learningPayload, "learning")
    } catch (err: any) {
      learningError = err?.message || "Failed to load learning jobs"
    }

    const jobs = mergeJobs(promptJobs, learningJobs)
    snapshot.jobs = jobs
    snapshot.lastRefresh = Date.now()
    snapshot.lastError = learningError
    if (!snapshot.selectedJob && jobs.length > 0 && !autoSelected) {
      autoSelected = true
      await selectJob(jobs[0].job_id)
      return
    }
    if (snapshot.selectedJob) {
      const match = jobs.find((j) => j.job_id === snapshot.selectedJob?.job_id)
      // Only update if match has more data than current (preserve metadata)
      if (match && !snapshot.selectedJob.metadata) {
        snapshot.selectedJob = match
      }
    }
    if (jobs.length === 0) {
      snapshot.status = "No jobs found"
    } else {
      const promptCount = promptJobs.length
      const learningCount = learningJobs.length
      snapshot.status =
        learningCount > 0
          ? `Loaded ${promptCount} prompt-learning, ${learningCount} learning job(s)`
          : `Loaded ${promptCount} prompt-learning job(s)`
    }
    return true
  } catch (err: any) {
    snapshot.lastError = err?.message || "Failed to load jobs"
    snapshot.status = "Failed to load jobs"
    return false
  } finally {
    renderSnapshot()
  }
}

async function selectJob(jobId: string): Promise<void> {
  const token = ++jobSelectToken
  eventsToken++
  lastSeq = 0
  snapshot.events = []
  snapshot.metrics = {}
  snapshot.bestSnapshotId = null
  snapshot.bestSnapshot = null
  snapshot.evalSummary = null
  snapshot.evalResultRows = []
  snapshot.allCandidates = []
  selectedEventIndex = 0
  eventWindowStart = 0
  const immediate = snapshot.jobs.find((job) => job.job_id === jobId)
  snapshot.selectedJob =
    immediate ??
    ({
      job_id: jobId,
      status: "loading",
      training_type: null,
      created_at: null,
      started_at: null,
      finished_at: null,
      best_score: null,
      best_snapshot_id: null,
      total_tokens: null,
      total_cost_usd: null,
      error: null,
      job_source: null,
    } as JobSummary)
  snapshot.status = `Loading job ${jobId}...`
  renderSnapshot()
  const jobSource = immediate?.job_source ?? null
  try {
    const path =
      jobSource === "eval"
        ? `/eval/jobs/${jobId}`
        : jobSource === "learning"
          ? `/learning/jobs/${jobId}?include_metadata=true`
        : `/prompt-learning/online/jobs/${jobId}?include_events=false&include_snapshot=false&include_metadata=true`
    const job = await apiGet(path)
    if (token !== jobSelectToken || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    const coerced = coerceJob(job, jobSource ?? "prompt-learning")
    // Extract metadata for learning and prompt-learning jobs
    if (jobSource !== "eval") {
      // Extract metadata - prompt_initial_snapshot is already nested inside metadata
      const jobMeta = job?.metadata ?? {}
      // If prompt_initial_snapshot exists at top level, merge it (fallback)
      if (job?.prompt_initial_snapshot && !jobMeta.prompt_initial_snapshot) {
        coerced.metadata = { ...jobMeta, prompt_initial_snapshot: job.prompt_initial_snapshot }
      } else {
        coerced.metadata = jobMeta
      }
      snapshot.bestSnapshotId = extractBestSnapshotId(job)
    }
    if (jobSource === "eval" || isEvalJob(coerced)) {
      snapshot.evalSummary =
        job?.results && typeof job.results === "object" ? job.results : null
    }
    snapshot.selectedJob = coerced
    snapshot.status = `Selected job ${jobId}`
  } catch (err: any) {
    if (token !== jobSelectToken || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    const errMsg = err?.message || `Failed to load job ${jobId}`
    snapshot.lastError = errMsg
    snapshot.status = `Error: ${errMsg}`
  }
  if (jobSource !== "learning" && jobSource !== "eval" && !isEvalJob(snapshot.selectedJob)) {
    await fetchBestSnapshot(token)
  }
  if (jobSource === "eval" || isEvalJob(snapshot.selectedJob)) {
    await fetchEvalResults(token)
  }
  renderSnapshot()
}

async function fetchMetrics(): Promise<void> {
  const job = snapshot.selectedJob
  if (!job) return
  const jobId = job.job_id
  try {
    if (isEvalJob(job)) {
      await fetchEvalResults()
      return
    }
    snapshot.status = "Loading metrics..."
    const path =
      job.job_source === "learning"
        ? `/learning/jobs/${job.job_id}/metrics`
        : `/prompt-learning/online/jobs/${job.job_id}/metrics`
    const payload = await apiGet(path)
    if (snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.metrics = payload
    snapshot.status = `Loaded metrics for ${job.job_id}`
  } catch (err: any) {
    if (snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.lastError = err?.message || "Failed to load metrics"
    snapshot.status = "Failed to load metrics"
  }
  renderSnapshot()
}

async function fetchEvalResults(token?: number): Promise<void> {
  const job = snapshot.selectedJob
  if (!job || !isEvalJob(job)) return
  const jobId = job.job_id
  try {
    snapshot.status = "Loading eval results..."
    const payload = await apiGet(`/eval/jobs/${job.job_id}/results`)
    if ((token != null && token !== jobSelectToken) || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.evalSummary =
      payload?.summary && typeof payload.summary === "object" ? payload.summary : null
    snapshot.evalResultRows = Array.isArray(payload?.results) ? payload.results : []
    snapshot.status = `Loaded eval results for ${job.job_id}`
  } catch (err: any) {
    if ((token != null && token !== jobSelectToken) || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.lastError = err?.message || "Failed to load eval results"
    snapshot.status = "Failed to load eval results"
  }
  renderSnapshot()
}

async function fetchBestSnapshot(token?: number): Promise<void> {
  const job = snapshot.selectedJob
  if (!job || job.job_source === "learning" || isEvalJob(job)) return
  const jobId = job.job_id
  try {
    snapshot.status = "Loading best snapshot..."
    const payload = await apiGet(`/prompt-learning/online/jobs/${job.job_id}/best-snapshot`)
    if ((token != null && token !== jobSelectToken) || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.bestSnapshot = isRecord(payload?.best_snapshot) ? payload.best_snapshot : null
    snapshot.bestSnapshotId = extractBestSnapshotId(payload)
    if (snapshot.bestSnapshotId || snapshot.bestSnapshot) {
      snapshot.status = `Loaded best snapshot for ${job.job_id}`
    } else {
      snapshot.status = "Best snapshot not available yet"
    }
  } catch (err: any) {
    if ((token != null && token !== jobSelectToken) || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.lastError = err?.message || "Failed to load best snapshot"
    snapshot.status = "Failed to load best snapshot"
  }
  renderSnapshot()
}

/**
 * Fetch all prompt candidates (baseline + evaluated candidates from events) for the current job.
 * This populates snapshot.allCandidates for the prompt browser.
 */
async function fetchAllCandidates(token?: number): Promise<void> {
  const job = snapshot.selectedJob
  if (!job || job.job_source === "learning" || isEvalJob(job)) return
  const jobId = job.job_id
  const candidates: PromptCandidate[] = []

  // 1. Extract baseline from prompt_initial_snapshot in job metadata
  const meta = job.metadata || {}
  const initialSnapshot = meta.prompt_initial_snapshot
  if (initialSnapshot && typeof initialSnapshot === "object") {
    // The actual prompt is in initial_prompt field (serialized prompt)
    // This contains the actual prompt text, sections, etc.
    const initialPrompt = initialSnapshot.initial_prompt
    const rawConfig = initialSnapshot.raw_config

    candidates.push({
      id: "baseline",
      isBaseline: true,
      score: null, // Baseline score will be extracted from events if available
      payload: {
        source: "initial_config",
        initial_prompt: initialPrompt, // The actual serialized prompt
        raw_config: rawConfig,
        ...initialSnapshot,
      },
      createdAt: job.created_at || null,
      tag: "baseline",
    })
  }

  // 2. Extract candidates from events (contains all evaluated candidates with prompt_text)
  // Look for gepa.candidate.evaluated and proposal.scored events
  const relevantEvents = snapshot.events.filter(ev => {
    const t = ev.type || ""
    return t.includes("candidate.evaluated") ||
           t.includes("proposal.scored") ||
           t.includes("optimized.scored")
  })

  // Track seen version_ids to avoid duplicates
  // Include baseline_transformation to skip it (we already have the real baseline from prompt_initial_snapshot)
  const seenVersionIds = new Set<string>(["baseline", "baseline_transformation"])

  for (const ev of relevantEvents) {
    const payload = ev.payload || ev.data || {}
    // Get version_id from event payload
    const versionId = payload.version_id || payload.program_candidate?.candidate_id
    if (!versionId || seenVersionIds.has(versionId)) continue
    seenVersionIds.add(versionId)

    // Extract data from program_candidate block if available (preferred)
    const pc = payload.program_candidate || {}

    // Get score - prefer reward, then accuracy, then full_score
    const score = pc.reward ?? pc.accuracy ?? payload.accuracy ?? payload.full_score ?? null

    // Get prompt text from multiple sources
    const promptText = pc.prompt_text || payload.prompt_text || ""
    const stages = pc.stages || payload.stages || {}

    // Skip candidates without useful prompt content (empty prompt_text AND empty stages)
    const hasPromptContent = (promptText && promptText.length > 0) || Object.keys(stages).length > 0
    if (!hasPromptContent) continue

    const status = pc.status || payload.status || "evaluated"
    const generation = pc.generation ?? payload.generation ?? null
    const mutationType = pc.mutation_type || payload.mutation_type || ""
    const parentId = pc.parent_id || payload.parent_id || null
    const seedScores = pc.seed_scores || payload.seed_scores || []

    candidates.push({
      id: versionId,
      isBaseline: false, // Real baseline comes from prompt_initial_snapshot
      score: typeof score === "number" ? score : null,
      payload: {
        prompt_text: promptText,
        stages,
        status,
        generation,
        mutation_type: mutationType,
        parent_id: parentId,
        seed_scores: seedScores,
        program_candidate: pc,
        ...payload,
      },
      createdAt: ev.created_at || null,
      tag: status === "accepted" ? "accepted" : (status === "rejected" ? "rejected" : null),
    })
  }

  // 3. If we have a bestSnapshot but it's not in the list, add it
  if (snapshot.bestSnapshot && snapshot.bestSnapshotId) {
    const hasBest = candidates.some(c => c.id === snapshot.bestSnapshotId)
    if (!hasBest) {
      candidates.push({
        id: snapshot.bestSnapshotId,
        isBaseline: false,
        score: job.best_score ?? null,
        payload: snapshot.bestSnapshot,
        createdAt: null,
        tag: "best",
      })
    }
  }

  // 4. Also fetch best snapshot data if available for richer content
  try {
    const artifacts = await apiGet(`/prompt-learning/online/jobs/${jobId}/artifacts`)
    if ((token != null && token !== jobSelectToken) || snapshot.selectedJob?.job_id !== jobId) {
      return
    }

    if (Array.isArray(artifacts)) {
      for (const art of artifacts) {
        const snapId = art.snapshot_id
        if (!snapId || seenVersionIds.has(snapId)) continue

        try {
          const snapData = await apiGet(`/prompt-learning/online/jobs/${jobId}/snapshots/${snapId}`)
          if ((token != null && token !== jobSelectToken) || snapshot.selectedJob?.job_id !== jobId) {
            return
          }
          if (snapData && typeof snapData === "object") {
            seenVersionIds.add(snapId)
            const isBest = snapId === snapshot.bestSnapshotId
            candidates.push({
              id: snapId,
              isBaseline: false,
              score: snapData.score ?? null,
              payload: snapData.payload || snapData,
              createdAt: snapData.created_at || null,
              tag: isBest ? "best" : (snapData.tag || null),
            })
          }
        } catch {
          // Skip snapshots that fail to load
        }
      }
    }
  } catch {
    // If artifacts fail, we still have candidates from events
  }

  // Sort: baseline first, then by score descending
  candidates.sort((a, b) => {
    if (a.isBaseline) return -1
    if (b.isBaseline) return 1
    const aScore = a.score ?? -Infinity
    const bScore = b.score ?? -Infinity
    return bScore - aScore
  })

  snapshot.allCandidates = candidates
}

async function refreshEvents(): Promise<boolean> {
  const job = snapshot.selectedJob
  if (!job) return true
  const jobId = job.job_id
  const token = eventsToken
  try {
    // Build list of event endpoints to try (with fallbacks)
    const isGepa = job.training_type === "gepa" || job.training_type === "graph_gepa"
    const paths =
      isEvalJob(job)
        ? [
            `/eval/jobs/${job.job_id}/events?since_seq=${lastSeq}&limit=200`,
            `/learning/jobs/${job.job_id}/events?since_seq=${lastSeq}&limit=200`,
          ]
        : job.job_source === "learning"
          ? [`/learning/jobs/${job.job_id}/events?since_seq=${lastSeq}&limit=200`]
          : isGepa
            ? [
                `/prompt-learning/online/jobs/${job.job_id}/events?since_seq=${lastSeq}&limit=200`,
                `/learning/jobs/${job.job_id}/events?since_seq=${lastSeq}&limit=200`,
              ]
            : [`/prompt-learning/online/jobs/${job.job_id}/events?since_seq=${lastSeq}&limit=200`]
    let payload: any = null
    let lastErr: any = null
    for (const path of paths) {
      try {
        payload = await apiGet(path)
        lastErr = null
        break
      } catch (err: any) {
        lastErr = err
      }
    }
    if (lastErr) {
      if (token !== eventsToken || snapshot.selectedJob?.job_id !== jobId) {
        return true
      }
      snapshot.lastError = lastErr?.message || "Failed to load events"
      return false
    }
    if (token !== eventsToken || snapshot.selectedJob?.job_id !== jobId) {
      return true
    }
    const { events, nextSeq } = extractEvents(payload)
    if (events.length > 0) {
      snapshot.events.push(...events)
      const filter = eventFilter.trim().toLowerCase()
      const newMatchCount =
        filter.length === 0 ? events.length : events.filter((event) => eventMatchesFilter(event, filter)).length
      if (activePane === "events" && newMatchCount > 0) {
        if (selectedEventIndex > 0) {
          selectedEventIndex += newMatchCount
        }
        if (eventWindowStart > 0) {
          eventWindowStart += newMatchCount
        }
      }
      if (eventHistoryLimit > 0 && snapshot.events.length > eventHistoryLimit) {
        snapshot.events = snapshot.events.slice(-eventHistoryLimit)
        selectedEventIndex = clamp(
          selectedEventIndex,
          0,
          Math.max(0, snapshot.events.length - 1),
        )
        eventWindowStart = clamp(
          eventWindowStart,
          0,
          Math.max(0, snapshot.events.length - Math.max(1, eventVisibleCount)),
        )
      }
      lastSeq = Math.max(lastSeq, ...events.map((e) => e.seq))
    }
    if (typeof nextSeq === "number" && Number.isFinite(nextSeq)) {
      lastSeq = Math.max(lastSeq, nextSeq)
    }
    renderSnapshot()
    return true
  } catch {
    return false
  }
}

async function cancelSelected(): Promise<void> {
  const job = snapshot.selectedJob
  if (!job) return
  try {
    await apiPost(`/prompt-learning/online/jobs/${job.job_id}/cancel`, {})
    snapshot.status = "Cancel requested"
  } catch (err: any) {
    snapshot.lastError = err?.message || "Cancel failed"
  }
  renderSnapshot()
}

async function fetchArtifacts(): Promise<void> {
  const job = snapshot.selectedJob
  if (!job) return
  try {
    const payload = await apiGet(`/prompt-learning/online/jobs/${job.job_id}/artifacts`)
    snapshot.artifacts = Array.isArray(payload) ? payload : payload?.artifacts || []
    snapshot.status = "Artifacts fetched"
  } catch (err: any) {
    snapshot.lastError = err?.message || "Artifacts fetch failed"
  }
  renderSnapshot()
}

async function fetchSnapshot(snapshotId: string): Promise<void> {
  const job = snapshot.selectedJob
  if (!job) return
  try {
    await apiGet(`/prompt-learning/online/jobs/${job.job_id}/snapshots/${snapshotId}`)
    snapshot.status = `Snapshot ${snapshotId} fetched`
  } catch (err: any) {
    snapshot.lastError = err?.message || "Snapshot fetch failed"
  }
  renderSnapshot()
}

function renderSnapshot(): void {
  const filteredJobs = getFilteredJobs()
  ui.jobsBox.title = jobStatusFilter.size
    ? `Jobs (status: ${Array.from(jobStatusFilter).join(", ")})`
    : "Jobs"
  ui.jobsSelect.options = filteredJobs.length
    ? filteredJobs.map((job) => {
        const shortId = job.job_id.slice(-8)
        const score = job.best_score == null ? "-" : job.best_score.toFixed(4)
        const label =
          job.training_type || (job.job_source === "learning" ? "eval" : "prompt")
        const envName = extractEnvName(job)
        const desc = envName
          ? `${job.status} | ${label} | ${envName} | ${score}`
          : `${job.status} | ${label} | ${score}`
        return { name: shortId, description: desc, value: job.job_id }
      })
    : [
        {
          name: "no jobs",
          description: jobStatusFilter.size
            ? `no jobs with selected status`
            : "no prompt-learning jobs found",
          value: "",
        },
      ]

  ui.detailText.content = formatDetails()
  ui.resultsText.content = formatResults()
  ui.metricsText.content = formatMetrics()
  renderEventCards()
  updatePaneIndicators()
  ui.headerMetaText.content = formatHeaderMeta()
  ui.statusText.content = formatStatus()
  ui.footerText.content = footerText()
  ui.eventsBox.title = eventFilter ? `Events (filter: ${eventFilter})` : "Events"
  updateEventModalContent()
  if (ui.settingsModalVisible) {
    renderSettingsList()
  }
  // Auto-refresh config modal if it's visible
  if (ui.configModalVisible) {
    const newPayload = formatConfigMetadata()
    if (newPayload && newPayload !== ui.configModalPayload) {
      ui.configModalPayload = newPayload
      updateConfigModalContent()
    }
  }
  renderer.requestRender()
}

function formatDetails(): string {
  const job = snapshot.selectedJob
  if (!job) return "No job selected."

  // Eval jobs get specialized rendering
  if (isEvalJob(job)) {
    return formatEvalDetails(job)
  }

  // Learning jobs (graph_gepa, etc.) - but not eval jobs
  if (job.job_source === "learning") {
    return formatLearningDetails(job)
  }

  // Default: prompt-learning jobs
  return formatPromptLearningDetails(job)
}

function formatEvalDetails(job: JobSummary): string {
  const summary = snapshot.evalSummary ?? {}
  const rows = snapshot.evalResultRows ?? []

  const lines = [
    `Job: ${job.job_id}`,
    `Status: ${job.status}`,
    `Type: eval`,
    "",
    "═══ Eval Summary ═══",
  ]

  // Extract key metrics from summary
  if (summary.mean_score != null) {
    lines.push(`  Mean Score: ${formatValue(summary.mean_score)}`)
  }
  if (summary.accuracy != null) {
    lines.push(`  Accuracy: ${(summary.accuracy * 100).toFixed(1)}%`)
  }
  if (summary.pass_rate != null) {
    lines.push(`  Pass Rate: ${(summary.pass_rate * 100).toFixed(1)}%`)
  }
  if (summary.completed != null && summary.total != null) {
    lines.push(`  Progress: ${summary.completed}/${summary.total}`)
  } else if (summary.completed != null) {
    lines.push(`  Completed: ${summary.completed}`)
  }
  if (summary.failed != null && summary.failed > 0) {
    lines.push(`  Failed: ${summary.failed}`)
  }

  // Show row count
  if (rows.length > 0) {
    lines.push(`  Results: ${rows.length} rows`)
    // Calculate score distribution
    const scores = rows
      .map((row) => num(row.score ?? row.reward_mean ?? row.outcome_reward ?? row.passed))
      .filter((val) => typeof val === "number")
    if (scores.length > 0) {
      const mean = scores.reduce((sum, val) => sum + val, 0) / scores.length
      const passed = scores.filter((s) => s >= 0.5 || s === 1).length
      lines.push(`  Avg Score: ${mean.toFixed(4)}`)
      lines.push(`  Pass Rate: ${((passed / scores.length) * 100).toFixed(1)}%`)
    }
  }

  lines.push("")
  lines.push("═══ Timing ═══")
  lines.push(`  Created: ${formatTimestamp(job.created_at)}`)
  lines.push(`  Started: ${formatTimestamp(job.started_at)}`)
  lines.push(`  Finished: ${formatTimestamp(job.finished_at)}`)

  if (job.error) {
    lines.push("")
    lines.push("═══ Error ═══")
    lines.push(`  ${job.error}`)
  }

  return lines.join("\n")
}

function formatLearningDetails(job: JobSummary): string {
  const envName = extractEnvName(job)
  const lines = [
    `Job: ${job.job_id}`,
    `Status: ${job.status}`,
    `Type: ${job.training_type || "learning"}`,
    `Env: ${envName || "-"}`,
    "",
    "═══ Progress ═══",
    `  Best Score: ${job.best_score != null ? job.best_score.toFixed(4) : "-"}`,
    `  Best Snapshot: ${job.best_snapshot_id || "-"}`,
    "",
    "═══ Timing ═══",
    `  Created: ${formatTimestamp(job.created_at)}`,
    `  Started: ${formatTimestamp(job.started_at)}`,
    `  Finished: ${formatTimestamp(job.finished_at)}`,
  ]

  if (job.error) {
    lines.push("")
    lines.push("═══ Error ═══")
    lines.push(`  ${job.error}`)
  }

  return lines.join("\n")
}

function formatPromptLearningDetails(job: JobSummary): string {
  const lastEvent = snapshot.events.length
    ? snapshot.events
        .filter((event) => event.timestamp)
        .reduce((latest, event) => {
          if (!latest) return event
          if (!event.timestamp) return latest
          return event.timestamp > latest.timestamp ? event : latest
        }, null as JobEvent | null)
    : null
  const lastEventTs = formatTimestamp(lastEvent?.timestamp)
  const totalTokens = job.total_tokens ?? calculateTotalTokensFromEvents()
  const tokensDisplay = totalTokens > 0 ? totalTokens.toLocaleString() : "-"
  const costDisplay = job.total_cost_usd != null ? `$${job.total_cost_usd.toFixed(4)}` : "-"
  const envName = extractEnvName(job)

  const lines = [
    `Job: ${job.job_id}`,
    `Status: ${job.status}`,
    `Type: ${job.training_type || "prompt-learning"}`,
    `Env: ${envName || "-"}`,
    `Started: ${formatTimestamp(job.started_at)}`,
    `Finished: ${formatTimestamp(job.finished_at)}`,
    `Last Event: ${lastEventTs}`,
    "",
    "═══ Progress ═══",
    `  Best Score: ${job.best_score != null ? job.best_score.toFixed(4) : "-"}`,
    `  Events: ${snapshot.events.length}`,
    `  Tokens: ${tokensDisplay}`,
    `  Cost: ${costDisplay}`,
  ]

  if (job.error) {
    lines.push("")
    lines.push("═══ Error ═══")
    lines.push(`  ${job.error}`)
  }
  if (snapshot.artifacts.length) {
    lines.push("")
    lines.push(`Artifacts: ${snapshot.artifacts.length}`)
  }

  return lines.join("\n")
}

function formatMetrics(): string {
  const metrics = snapshot.metrics || {}
  const points = Array.isArray((metrics as any)?.points) ? (metrics as any).points : []
  if (points.length > 0) {
    const latestByName = new Map<string, any>()
    for (const point of points) {
      if (point?.name) {
        latestByName.set(String(point.name), point)
      }
    }
    const rows = Array.from(latestByName.values()).sort((a, b) =>
      String(a.name).localeCompare(String(b.name)),
    )
    if (rows.length === 0) return "Metrics: -"
    const limit = 12
    const lines = rows.slice(0, limit).map((point) => {
      const value = formatValue(point.value ?? point.data ?? "-")
      const step = point.step != null ? ` (step ${point.step})` : ""
      return `- ${point.name}: ${value}${step}`
    })
    if (rows.length > limit) {
      lines.push(`... +${rows.length - limit} more`)
    }
    return ["Metrics (latest):", ...lines].join("\n")
  }
  const keys = Object.keys(metrics).filter((k) => k !== "points" && k !== "job_id")
  if (keys.length === 0) return "Metrics: -"
  return ["Metrics:", ...keys.map((k) => `- ${k}: ${formatValue((metrics as any)[k])}`)].join("\n")
}

function formatResults(): string {
  const job = snapshot.selectedJob
  if (!job) return "Results: -"
  if (job.job_source === "eval" || job.training_type === "eval") {
    return formatEvalResults()
  }
  const lines: string[] = []
  const bestId = snapshot.bestSnapshotId || "-"
  if (bestId === "-") {
    lines.push("Best snapshot: -")
  } else if (snapshot.bestSnapshot) {
    lines.push(`Best snapshot: ${bestId}`)
  } else {
    lines.push(`Best snapshot: ${bestId} (press p to load)`)
  }
  if (snapshot.bestSnapshot) {
    const bestPrompt = extractBestPrompt(snapshot.bestSnapshot)
    const bestPromptText = extractBestPromptText(snapshot.bestSnapshot)
    if (bestPrompt) {
      const promptId = bestPrompt.id || bestPrompt.template_id
      const promptName = bestPrompt.name
      const promptLabel = [promptName, promptId].filter(Boolean).join(" ")
      if (promptLabel) lines.push(`Best prompt: ${promptLabel}`)
      const sections = extractPromptSections(bestPrompt)
      if (sections.length > 0) {
        const summary = sections.slice(0, 3).map((section) => {
          const role = section.role || "stage"
          const name = section.name || section.id || ""
          return name ? `${role}:${name}` : role
        })
        const suffix = sections.length > 3 ? " …" : ""
        lines.push(`Stages: ${summary.join(", ")}${suffix}`)
      }
    }
    if (bestPromptText) {
      lines.push(`Best prompt text: ${truncate(bestPromptText, 90)}`)
    }
  }
  return ["Results:", ...lines].join("\n")
}

function formatEvalResults(): string {
  const summary = snapshot.evalSummary ?? {}
  const rows = snapshot.evalResultRows ?? []
  const lines: string[] = []

  // Show overall summary if available
  if (Object.keys(summary).length > 0) {
    lines.push("═══ Summary ═══")
    const keyOrder = ["mean_score", "accuracy", "pass_rate", "completed", "failed", "total"]
    const shown = new Set<string>()

    for (const key of keyOrder) {
      if (summary[key] != null) {
        const val = summary[key]
        if (key === "accuracy" || key === "pass_rate") {
          lines.push(`  ${key}: ${(val * 100).toFixed(1)}%`)
        } else {
          lines.push(`  ${key}: ${formatValue(val)}`)
        }
        shown.add(key)
      }
    }
    // Show remaining keys
    for (const [key, value] of Object.entries(summary)) {
      if (shown.has(key)) continue
      if (typeof value === "object") continue
      lines.push(`  ${key}: ${formatValue(value)}`)
    }
    lines.push("")
  }
  if (summary.mean_score == null && rows.length > 0) {
    const scores = rows
      .map((row) => row.outcome_reward ?? row.score ?? row.reward_mean ?? row.events_score)
      .filter((val) => typeof val === "number" && Number.isFinite(val)) as number[]
    if (scores.length > 0) {
      const mean = scores.reduce((acc, val) => acc + val, 0) / scores.length
      if (lines.length === 0 || lines[0] !== "═══ Summary ═══") {
        lines.unshift("═══ Summary ═══")
      }
      lines.splice(1, 0, `  mean_score: ${formatValue(mean)}`)
      lines.push("")
    }
  }

  // Show per-task results
  if (rows.length > 0) {
    lines.push("═══ Results by Task ═══")
    const limit = 15
    const displayRows = rows.slice(0, limit)

    for (const row of displayRows) {
      const taskId = row.task_id || row.id || row.name || "?"
      const score = num(row.score ?? row.reward_mean ?? row.outcome_reward ?? row.passed)
      const passed = row.passed != null ? (row.passed ? "✓" : "✗") : ""
      const status = row.status || ""
      const scoreStr = score != null ? score.toFixed(3) : "-"

      if (passed) {
        lines.push(`  ${passed} ${taskId}: ${scoreStr}`)
      } else if (status) {
        lines.push(`  [${status}] ${taskId}: ${scoreStr}`)
      } else {
        lines.push(`  ${taskId}: ${scoreStr}`)
      }
    }

    if (rows.length > limit) {
      lines.push(`  ... +${rows.length - limit} more tasks`)
    }
  } else if (Object.keys(summary).length === 0) {
    lines.push("No eval results yet.")
    lines.push("")
    lines.push("Results will appear after the eval completes.")
  }

  return lines.length > 0 ? lines.join("\n") : "Results: -"
}

function formatHeaderMeta(): string {
  const org = snapshot.orgId || "-"
  const user = snapshot.userId || "-"
  const balance =
    snapshot.balanceDollars == null ? "-" : `$${snapshot.balanceDollars.toFixed(2)}`
  const backendLabel = getBackendConfig().label
  return `backend: ${backendLabel}  org: ${org}  user: ${user}  balance: ${balance}`
}

function formatTimestamp(value: any): string {
  if (value == null || value === "") return "-"
  if (value instanceof Date) {
    return value.toLocaleString()
  }
  if (typeof value === "object") {
    const seconds = (value as any).seconds
    const nanos = (value as any).nanoseconds ?? (value as any).nanos
    if (Number.isFinite(Number(seconds))) {
      const ms = Number(seconds) * 1000 + (Number(nanos) || 0) / 1e6
      return new Date(ms).toLocaleString()
    }
  }
  if (typeof value === "number") {
    const ms = value > 1e12 ? value : value * 1000
    return new Date(ms).toLocaleString()
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    const normalized = trimmed
      .replace(" ", "T")
      .replace(/(\.\d{3})\d+/, "$1")
    const parsed = Date.parse(normalized)
    if (Number.isFinite(parsed)) {
      return new Date(parsed).toLocaleString()
    }
    if (/^-?\d+(?:\.\d+)?$/.test(trimmed)) {
      const numeric = Number(trimmed)
      const ms = numeric > 1e12 ? numeric : numeric * 1000
      return new Date(ms).toLocaleString()
    }
    const numericMatch = trimmed.match(/-?\d+(?:\.\d+)?/)
    if (numericMatch) {
      const parsedNumber = Number(numericMatch[0])
      if (Number.isFinite(parsedNumber)) {
        const ms = parsedNumber > 1e12 ? parsedNumber : parsedNumber * 1000
        return new Date(ms).toLocaleString()
      }
    }
  }
  return String(value)
}

function renderEventCards(): void {
  const { collapsedHeight, expandedHeight, gap, visibleCount } = getEventLayoutMetrics()
  const recentAll = getFilteredEvents()
  if (recentAll.length === 0) {
    ui.eventsList.visible = false
    ui.eventsEmptyText.visible = true
    // Show contextual message based on job state
    const job = snapshot.selectedJob
    if (eventFilter) {
      ui.eventsEmptyText.content = "No events match filter."
    } else if (job?.status === "succeeded" || job?.status === "failed" || job?.status === "completed") {
      ui.eventsEmptyText.content = "No events recorded for this job.\n\nEvents may not have been persisted during execution."
    } else if (job?.status === "running" || job?.status === "queued") {
      ui.eventsEmptyText.content = "Waiting for events...\n\nEvents will appear as the job progresses."
    } else {
      ui.eventsEmptyText.content = "No events yet."
    }
    return
  }
  const total = recentAll.length
  const effectiveVisible = Math.max(1, visibleCount)
  selectedEventIndex = clamp(selectedEventIndex, 0, Math.max(0, total - 1))
  eventWindowStart = clamp(eventWindowStart, 0, Math.max(0, total - effectiveVisible))
  if (selectedEventIndex < eventWindowStart) {
    eventWindowStart = selectedEventIndex
  } else if (selectedEventIndex >= eventWindowStart + effectiveVisible) {
    eventWindowStart = selectedEventIndex - effectiveVisible + 1
  }
  const recent = recentAll.slice(eventWindowStart, eventWindowStart + effectiveVisible)
  ui.eventsEmptyText.visible = false
  ui.eventsList.visible = true
  ui.eventsList.gap = gap
  for (const card of ui.eventCards) {
    ui.eventsList.remove(card.box.id)
  }
  ui.eventCards = []
  recent.forEach((event, index) => {
    const globalIndex = eventWindowStart + index
    const isSelected = globalIndex === selectedEventIndex
    const detail = event.message ?? formatEventData(event.data)
    const isLong = detail.length > eventCollapseLimit
    const isExpanded = event.expanded || (isSelected && !isLong)
    const cardHeight = isExpanded ? expandedHeight : collapsedHeight
    const box = new BoxRenderable(renderer, {
      id: `event-card-${index}`,
      width: "auto",
      height: cardHeight,
      borderStyle: "single",
      borderColor: isSelected ? "#60a5fa" : "#1f2a44",
      backgroundColor: isSelected ? "#0f172a" : "#0b1220",
      border: true,
    })
    const text = new TextRenderable(renderer, {
      id: `event-card-text-${index}`,
      content: formatEventCardText(event, { isExpanded, isLong }),
      fg: "#e2e8f0",
    })
    box.add(text)
    ui.eventsList.add(box)
    ui.eventCards.push({ box, text })
  })
}

function getEventLayoutMetrics(): {
  collapsedHeight: number
  expandedHeight: number
  gap: number
  visibleCount: number
} {
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const compact = rows < 32
  const collapsedHeight = compact ? 3 : 4
  const expandedHeight = compact ? 5 : 7
  const gap = compact ? 0 : 1
  const available = Math.max(1, rows - 24)
  const maxVisible = Math.max(1, Math.floor((available + gap) / (collapsedHeight + gap)))
  const target = Math.max(1, eventVisibleCount)
  const visibleCount = Math.max(1, Math.min(target, maxVisible))
  return { collapsedHeight, expandedHeight, gap, visibleCount }
}

function formatEventData(data: unknown): string {
  if (data == null) return ""
  if (typeof data === "string") return data
  if (typeof data === "number" || typeof data === "boolean") return String(data)
  try {
    const text = JSON.stringify(data)
    return text.length > 120 ? `${text.slice(0, 117)}...` : text
  } catch {
    return String(data)
  }
}

function getFilteredEvents(): JobEvent[] {
  const filter = eventFilter.trim().toLowerCase()
  const list = filter.length
    ? snapshot.events.filter((event) => eventMatchesFilter(event, filter))
    : snapshot.events
  return [...list].sort((a, b) => eventSortKey(b) - eventSortKey(a))
}

function eventMatchesFilter(event: JobEvent, filter: string): boolean {
  const haystack = [
    event.type,
    event.message,
    event.timestamp,
    event.data ? safeEventDataText(event.data) : "",
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase()
  return haystack.includes(filter)
}

function eventSortKey(event: JobEvent): number {
  if (Number.isFinite(event.seq)) {
    return Number(event.seq)
  }
  const ts = event.timestamp
  if (typeof ts === "string") {
    const normalized = ts.trim().replace(" ", "T").replace(/(\.\d{3})\d+/, "$1")
    const parsed = Date.parse(normalized)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return 0
}

function safeEventDataText(data: unknown): string {
  if (data == null) return ""
  if (typeof data === "string") return data
  if (typeof data === "number" || typeof data === "boolean") return String(data)
  try {
    return JSON.stringify(data)
  } catch {
    return ""
  }
}

function getFilteredJobs(): JobSummary[] {
  if (!jobStatusFilter.size) return snapshot.jobs
  return snapshot.jobs.filter((job) =>
    jobStatusFilter.has(String(job.status || "unknown").toLowerCase()),
  )
}

function buildJobStatusOptions(jobs: JobSummary[]): Array<{ status: string; count: number }> {
  const counts = new Map<string, number>()
  for (const job of jobs) {
    const status = String(job.status || "unknown").toLowerCase()
    counts.set(status, (counts.get(status) || 0) + 1)
  }
  const order = ["running", "queued", "succeeded", "failed", "canceled", "cancelled", "unknown"]
  const statuses = Array.from(counts.keys()).sort((a, b) => {
    const ai = order.indexOf(a)
    const bi = order.indexOf(b)
    if (ai === -1 && bi === -1) return a.localeCompare(b)
    if (ai === -1) return 1
    if (bi === -1) return -1
    return ai - bi
  })
  return statuses.map((status) => ({
    status,
    count: counts.get(status) || 0,
  }))
}

function collectJobStatuses(jobs: JobSummary[]): string[] {
  const statuses = new Set<string>()
  for (const job of jobs) {
    statuses.add(String(job.status || "unknown").toLowerCase())
  }
  return Array.from(statuses)
}

function formatEventCardText(
  event: JobEvent,
  opts?: { isExpanded?: boolean; isLong?: boolean },
): string {
  const seq = String(event.seq).padStart(5, " ")
  const header = `${seq} ${event.type}`
  const detail = event.message ?? formatEventData(event.data)
  if (!detail) return header
  if (opts?.isExpanded) {
    const clipped = detail.length > 900 ? `${detail.slice(0, 897)}...` : detail
    return `${header}\n${clipped}`
  }
  const trimmed =
    detail.length > 120
      ? `${detail.slice(0, 117)}...${opts?.isLong ? " (enter to view)" : ""}`
      : detail
  return `${header}\n${trimmed}`
}

function formatValue(value: unknown): string {
  if (value == null) return "-"
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toFixed(4) : String(value)
  }
  if (typeof value === "string") return value
  if (typeof value === "boolean") return value ? "true" : "false"
  try {
    const text = JSON.stringify(value)
    return text.length > 120 ? `${text.slice(0, 117)}...` : text
  } catch {
    return String(value)
  }
}

function extractBestSnapshotId(payload: any): string | null {
  if (!payload) return null
  return (
    payload.best_snapshot_id ||
    payload.prompt_best_snapshot_id ||
    payload.best_snapshot?.id ||
    null
  )
}

function extractBestPrompt(snapshotPayload: Record<string, any>): Record<string, any> | null {
  if (!snapshotPayload) return null
  return (
    (isRecord(snapshotPayload.best_prompt) && snapshotPayload.best_prompt) ||
    (isRecord(snapshotPayload.best_prompt_template) && snapshotPayload.best_prompt_template) ||
    (isRecord(snapshotPayload.best_prompt_pattern) && snapshotPayload.best_prompt_pattern) ||
    null
  )
}

function extractBestPromptText(snapshotPayload: Record<string, any>): string | null {
  if (!snapshotPayload) return null
  const text =
    snapshotPayload.best_prompt_text ||
    snapshotPayload.best_prompt_preview ||
    snapshotPayload.best_prompt_rendered
  return typeof text === "string" ? text : null
}

function extractPromptSections(bestPrompt: Record<string, any>): Array<Record<string, any>> {
  if (!bestPrompt) return []
  if (Array.isArray(bestPrompt.sections)) return bestPrompt.sections
  if (Array.isArray(bestPrompt.prompt_sections)) return bestPrompt.prompt_sections
  return []
}

function truncate(value: string, max: number): string {
  if (value.length <= max) return value
  return `${value.slice(0, Math.max(0, max - 3))}...`
}

function isRecord(value: any): value is Record<string, any> {
  return value != null && typeof value === "object" && !Array.isArray(value)
}

function formatStatus(): string {
  const ts = snapshot.lastRefresh ? new Date(snapshot.lastRefresh).toLocaleTimeString() : "-"
  const baseLabel = getActiveBaseRoot().replace(/^https?:\/\//, "")
  const health = `health=${healthStatus}`
  if (snapshot.lastError) {
    return `Last refresh: ${ts} | ${health} | ${baseLabel} | Error: ${snapshot.lastError}`
  }
  return `Last refresh: ${ts} | ${health} | ${baseLabel} | ${snapshot.status}`
}

function footerText(): string {
  const filterLabel = eventFilter ? `filter=${eventFilter}` : "filter=off"
  const jobFilterLabel = jobStatusFilter.size
    ? `status=${Array.from(jobStatusFilter).join(",")}`
    : "status=all"
  return `Keys: e events | b jobs | tab toggle | j/k nav | enter view | r refresh | l login | L logout | t settings | f ${filterLabel} | shift+j ${jobFilterLabel} | c cancel | a artifacts | s snapshot | q quit`
}

function toggleModal(visible: boolean): void {
  ui.modalVisible = visible
  ui.modalBox.visible = visible
  ui.modalLabel.visible = visible
  ui.modalInput.visible = visible
  if (visible) {
    ui.modalInput.value = ""
    ui.modalInput.focus()
  } else {
    ui.jobsSelect.focus()
  }
}

function toggleFilterModal(visible: boolean): void {
  ui.filterModalVisible = visible
  ui.filterBox.visible = visible
  ui.filterLabel.visible = visible
  ui.filterInput.visible = visible
  if (visible) {
    ui.filterInput.value = eventFilter
    ui.filterInput.focus()
  } else if (activePane === "jobs") {
    ui.jobsSelect.focus()
  }
}

function toggleJobFilterModal(visible: boolean): void {
  ui.jobFilterModalVisible = visible
  ui.jobFilterBox.visible = visible
  ui.jobFilterLabel.visible = visible
  ui.jobFilterHelp.visible = visible
  ui.jobFilterListText.visible = visible
  if (visible) {
    ui.jobFilterLabel.content = "Job filter (status)"
    refreshJobFilterOptions()
    ui.jobsSelect.blur()
    jobFilterCursor = 0
    jobFilterWindowStart = 0
    renderJobFilterList()
  } else if (activePane === "jobs") {
    ui.jobsSelect.focus()
  }
}

function refreshJobFilterOptions(): void {
  jobFilterOptions = buildJobStatusOptions(snapshot.jobs)
  const maxIndex = Math.max(0, jobFilterOptions.length - 1)
  jobFilterCursor = clamp(jobFilterCursor, 0, maxIndex)
  jobFilterWindowStart = clamp(jobFilterWindowStart, 0, Math.max(0, maxIndex))
  renderJobFilterList()
}

function toggleSelectedJobFilter(): void {
  const option = jobFilterOptions[jobFilterCursor]
  if (!option) return
  if (jobStatusFilter.has(option.status)) {
    jobStatusFilter.delete(option.status)
  } else {
    jobStatusFilter.add(option.status)
  }
  renderJobFilterList()
  applyJobFilterSelection()
}

function moveJobFilterSelection(delta: number): void {
  const max = Math.max(0, jobFilterOptions.length - 1)
  jobFilterCursor = clamp(jobFilterCursor + delta, 0, max)
  if (jobFilterCursor < jobFilterWindowStart) {
    jobFilterWindowStart = jobFilterCursor
  } else if (jobFilterCursor >= jobFilterWindowStart + jobFilterVisibleCount) {
    jobFilterWindowStart = jobFilterCursor - jobFilterVisibleCount + 1
  }
  renderJobFilterList()
}

function renderJobFilterList(): void {
  const max = Math.max(0, jobFilterOptions.length - 1)
  jobFilterCursor = clamp(jobFilterCursor, 0, max)
  const start = clamp(jobFilterWindowStart, 0, Math.max(0, max))
  const end = Math.min(jobFilterOptions.length, start + jobFilterVisibleCount)
  const lines: string[] = []
  for (let idx = start; idx < end; idx++) {
    const option = jobFilterOptions[idx]
    const active = jobStatusFilter.has(option.status)
    const cursor = idx === jobFilterCursor ? ">" : " "
    lines.push(`${cursor} [${active ? "x" : " "}] ${option.status} (${option.count})`)
  }
  if (!lines.length) {
    lines.push("  (no statuses available)")
  }
  ui.jobFilterListText.content = lines.join("\n")
  renderer.requestRender()
}

function applyJobFilterSelection(): void {
  const filteredJobs = getFilteredJobs()
  if (!filteredJobs.length) {
    snapshot.selectedJob = null
    snapshot.events = []
    snapshot.metrics = {}
    snapshot.bestSnapshotId = null
    snapshot.bestSnapshot = null
    snapshot.allCandidates = []
    selectedEventIndex = 0
    eventWindowStart = 0
    snapshot.status = jobStatusFilter.size
      ? "No jobs with selected status"
      : "No prompt-learning jobs found"
    renderSnapshot()
    return
  }
  if (!snapshot.selectedJob || !filteredJobs.some((job) => job.job_id === snapshot.selectedJob?.job_id)) {
    void selectJob(filteredJobs[0].job_id)
    return
  }
  renderSnapshot()
}

function buildSettingsOptions(): BackendConfig[] {
  return [backendConfigs.prod, backendConfigs.dev, backendConfigs.local]
}

function toggleSettingsModal(visible: boolean): void {
  ui.settingsModalVisible = visible
  ui.settingsBox.visible = visible
  ui.settingsTitle.visible = visible
  ui.settingsHelp.visible = visible
  ui.settingsListText.visible = visible
  ui.settingsInfoText.visible = visible
  if (visible) {
    settingsOptions = buildSettingsOptions()
    settingsCursor = Math.max(
      0,
      settingsOptions.findIndex((opt) => opt.id === currentBackend),
    )
    ui.jobsSelect.blur()
    renderSettingsList()
  } else {
    if (ui.keyModalVisible) {
      toggleKeyModal(false)
    }
    if (ui.envKeyModalVisible) {
      toggleEnvKeyModal(false)
    }
    if (activePane === "jobs") {
    ui.jobsSelect.focus()
    }
  }
  renderer.requestRender()
}

function moveSettingsSelection(delta: number): void {
  const max = Math.max(0, settingsOptions.length - 1)
  settingsCursor = clamp(settingsCursor + delta, 0, max)
  renderSettingsList()
}

function renderSettingsList(): void {
  if (!ui.settingsModalVisible) return
  const lines: string[] = []
  for (let idx = 0; idx < settingsOptions.length; idx++) {
    const option = settingsOptions[idx]
    const cursor = idx === settingsCursor ? ">" : " "
    const active = option.id === currentBackend ? "*" : " "
    lines.push(`${cursor} [${active}] ${option.label}`)
  }
  if (!lines.length) {
    lines.push("  (no backends available)")
  }
  ui.settingsListText.content = lines.join("\n")
  const selected = settingsOptions[settingsCursor]
  if (selected) {
    const key = backendKeys[selected.id]
    const masked = maskKey(key)
    const baseRoot = selected.baseUrl.replace(/\/api$/, "")
    const devNote =
      selected.id === "prod" ? null : "Note: Dev/Local are for Synth devs only"
    const source = backendKeySources[selected.id]
    const sourceLine = source?.sourcePath
      ? `Source: ${truncatePath(source.sourcePath, 44)}`
      : source?.varName
        ? `Source: ${source.varName}`
        : "Source: -"
    const varLine =
      source?.varName && source.varName !== "manual"
        ? `Var: ${source.varName}`
        : null
    ui.settingsInfoText.content = [
      `Base: ${baseRoot}`,
      `Key: ${masked}`,
      devNote,
      sourceLine,
      varLine,
    ]
      .filter(Boolean)
      .join("\n")
  } else {
    ui.settingsInfoText.content = "Base: -\nKey: -"
  }
}

function maskKey(key: string): string {
  if (!key) return "(missing)"
  if (key.length <= 8) return "****"
  return `${key.slice(0, 4)}...${key.slice(-4)}`
}

async function applySettingsSelection(): Promise<void> {
  const option = settingsOptions[settingsCursor]
  if (!option) return
  if (option.id !== currentBackend) {
    await switchBackend(option.id)
  }
  toggleSettingsModal(false)
}

async function switchBackend(nextBackend: BackendId): Promise<void> {
  if (currentBackend === nextBackend) return
  currentBackend = nextBackend
  void persistSettings()
  snapshot.lastError = null
  snapshot.status = `Switching to ${getBackendConfig().label}...`
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
  snapshot.orgId = null
  snapshot.userId = null
  snapshot.balanceDollars = null
  lastSeq = 0
  selectedEventIndex = 0
  eventWindowStart = 0
  autoSelected = false
  renderSnapshot()
  if (!getActiveApiKey()) {
    snapshot.lastError = `Missing API key for ${getBackendConfig().label}`
    snapshot.status = "Auth required"
    renderSnapshot()
    return
  }
  await refreshIdentity()
  await refreshJobs()
  if (snapshot.jobs.length > 0) {
    await selectJob(snapshot.jobs[0].job_id)
  }
  scheduleJobsPoll(0)
  scheduleEventsPoll(0)
}

function openKeyModal(): void {
  const option = settingsOptions[settingsCursor]
  if (!option) return
  keyModalBackend = option.id
  toggleKeyModal(true)
}

function toggleKeyModal(visible: boolean): void {
  ui.keyModalVisible = visible
  ui.keyModalBox.visible = visible
  ui.keyModalLabel.visible = visible
  ui.keyModalInput.visible = visible
  ui.keyModalHelp.visible = visible
  if (visible) {
    ui.keyModalInput.value = ""
    keyPasteActive = false
    keyPasteBuffer = ""
    ui.keyModalInput.focus()
  } else {
    ui.keyModalInput.value = ""
    ui.keyModalInput.blur()
    if (ui.settingsModalVisible) {
      renderSettingsList()
    }
  }
  renderer.requestRender()
}

function applyKeyModal(value: string): void {
  const trimmed = value.trim()
  backendKeys[keyModalBackend] = trimmed
  backendKeySources[keyModalBackend] = trimmed
    ? { sourcePath: "manual", varName: null }
    : { sourcePath: null, varName: null }
  toggleKeyModal(false)
  void persistSettings()
  if (!getActiveApiKey()) {
    snapshot.lastError = `Missing API key for ${getBackendConfig().label}`
    snapshot.status = "Auth required"
  } else if (keyModalBackend === currentBackend) {
    void switchBackend(currentBackend)
  }
  renderSnapshot()
}

async function openEnvKeyModal(): Promise<void> {
  const option = settingsOptions[settingsCursor]
  if (!option) return
  keyModalBackend = option.id
  toggleEnvKeyModal(true)
  await rescanEnvKeys()
}

function toggleEnvKeyModal(visible: boolean): void {
  ui.envKeyModalVisible = visible
  ui.envKeyModalBox.visible = visible
  ui.envKeyModalTitle.visible = visible
  ui.envKeyModalHelp.visible = visible
  ui.envKeyModalListText.visible = visible
  ui.envKeyModalInfoText.visible = visible
  if (visible) {
    envKeyCursor = 0
    envKeyWindowStart = 0
    envKeyOptions = []
    envKeyError = null
    envKeyScanInProgress = false
    ui.envKeyModalTitle.content = `Settings - API Key (${getBackendConfig().label})`
    ui.jobsSelect.blur()
    renderEnvKeyList()
  } else {
    if (ui.settingsModalVisible) {
      renderSettingsList()
    }
  }
  renderer.requestRender()
}

async function rescanEnvKeys(): Promise<void> {
  if (envKeyScanInProgress) return
  envKeyScanInProgress = true
  envKeyError = null
  ui.envKeyModalInfoText.content = `Scan: ${envKeyScanRoot}`
  ui.envKeyModalListText.content = "Scanning for SYNTH_API_KEY..."
  renderer.requestRender()
  try {
    envKeyOptions = await scanEnvKeys(envKeyScanRoot)
    envKeyCursor = 0
    envKeyWindowStart = 0
  } catch (err: any) {
    envKeyError = err?.message || "Failed to scan .env files"
    envKeyOptions = []
  } finally {
    envKeyScanInProgress = false
    renderEnvKeyList()
  }
}

function moveEnvKeySelection(delta: number): void {
  const max = Math.max(0, envKeyOptions.length - 1)
  envKeyCursor = clamp(envKeyCursor + delta, 0, max)
  if (envKeyCursor < envKeyWindowStart) {
    envKeyWindowStart = envKeyCursor
  } else if (envKeyCursor >= envKeyWindowStart + envKeyVisibleCount) {
    envKeyWindowStart = envKeyCursor - envKeyVisibleCount + 1
  }
  renderEnvKeyList()
}

function renderEnvKeyList(): void {
  if (!ui.envKeyModalVisible) return
  const max = Math.max(0, envKeyOptions.length - 1)
  envKeyCursor = clamp(envKeyCursor, 0, max)
  envKeyWindowStart = clamp(envKeyWindowStart, 0, Math.max(0, max))
  const start = envKeyWindowStart
  const end = Math.min(envKeyOptions.length, start + envKeyVisibleCount)
  const lines: string[] = []
  const activeKey = backendKeys[keyModalBackend]
  for (let idx = start; idx < end; idx++) {
    const option = envKeyOptions[idx]
    const cursor = idx === envKeyCursor ? ">" : " "
    const active = option.key === activeKey ? "x" : " "
    lines.push(
      `${cursor} [${active}] ${maskKeyPrefix(option.key)}  ${formatEnvKeySource(option)}`,
    )
  }
  if (!lines.length) {
    if (envKeyScanInProgress) {
      lines.push("  Scanning...")
    } else if (envKeyError) {
      lines.push("  (scan failed)")
    } else {
      lines.push("  (no SYNTH_API_KEY entries found)")
    }
  }
  ui.envKeyModalListText.content = lines.join("\n")
  const infoLines = [`Scan: ${envKeyScanRoot}`]
  infoLines.push(`Save: ${settingsFilePath}`)
  if (envKeyError) {
    infoLines.push(`Error: ${envKeyError}`)
  } else {
    infoLines.push(`Found: ${envKeyOptions.length}`)
  }
  ui.envKeyModalInfoText.content = infoLines.join("\n")
  renderer.requestRender()
}

function applyEnvKeySelection(): void {
  const option = envKeyOptions[envKeyCursor]
  if (!option) return
  backendKeys[keyModalBackend] = option.key
  backendKeySources[keyModalBackend] = {
    sourcePath: option.sources[0] || null,
    varName: option.varNames[0] || null,
  }
  toggleEnvKeyModal(false)
  void persistSettings()
  if (!getActiveApiKey()) {
    snapshot.lastError = `Missing API key for ${getBackendConfig().label}`
    snapshot.status = "Auth required"
  } else if (keyModalBackend === currentBackend) {
    void switchBackend(currentBackend)
  }
  renderSnapshot()
}

async function scanEnvKeys(rootDir: string): Promise<EnvKeyOption[]> {
  const results = new Map<string, EnvKeyOption>()
  await walkEnvDir(rootDir, results)
  return Array.from(results.values()).sort((a, b) => a.key.localeCompare(b.key))
}

async function loadPersistedSettings(): Promise<void> {
  try {
    const content = await fs.readFile(settingsFilePath, "utf8")
    const values = parseEnvFile(content)
    const backend = values.SYNTH_TUI_BACKEND
    if (backend) {
      currentBackend = normalizeBackendId(backend)
    }
    const prodKey = values.SYNTH_TUI_API_KEY_PROD
    const devKey = values.SYNTH_TUI_API_KEY_DEV
    const localKey = values.SYNTH_TUI_API_KEY_LOCAL
    if (typeof prodKey === "string") backendKeys.prod = prodKey
    if (typeof devKey === "string") backendKeys.dev = devKey
    if (typeof localKey === "string") backendKeys.local = localKey
    backendKeySources.prod = {
      sourcePath: values.SYNTH_TUI_API_KEY_PROD_SOURCE || null,
      varName: values.SYNTH_TUI_API_KEY_PROD_VAR || null,
    }
    backendKeySources.dev = {
      sourcePath: values.SYNTH_TUI_API_KEY_DEV_SOURCE || null,
      varName: values.SYNTH_TUI_API_KEY_DEV_VAR || null,
    }
    backendKeySources.local = {
      sourcePath: values.SYNTH_TUI_API_KEY_LOCAL_SOURCE || null,
      varName: values.SYNTH_TUI_API_KEY_LOCAL_VAR || null,
    }
  } catch (err: any) {
    if (err?.code !== "ENOENT") {
      // Ignore missing file, keep other errors silent for now.
    }
  }
}

async function persistSettings(): Promise<void> {
  try {
    await fs.mkdir(path.dirname(settingsFilePath), { recursive: true })
    const lines = [
      "# synth-ai tui settings",
      formatEnvLine("SYNTH_TUI_BACKEND", currentBackend),
      formatEnvLine("SYNTH_TUI_API_KEY_PROD", backendKeys.prod),
      formatEnvLine(
        "SYNTH_TUI_API_KEY_PROD_SOURCE",
        backendKeySources.prod.sourcePath || "",
      ),
      formatEnvLine("SYNTH_TUI_API_KEY_PROD_VAR", backendKeySources.prod.varName || ""),
      formatEnvLine("SYNTH_TUI_API_KEY_DEV", backendKeys.dev),
      formatEnvLine(
        "SYNTH_TUI_API_KEY_DEV_SOURCE",
        backendKeySources.dev.sourcePath || "",
      ),
      formatEnvLine("SYNTH_TUI_API_KEY_DEV_VAR", backendKeySources.dev.varName || ""),
      formatEnvLine("SYNTH_TUI_API_KEY_LOCAL", backendKeys.local),
      formatEnvLine(
        "SYNTH_TUI_API_KEY_LOCAL_SOURCE",
        backendKeySources.local.sourcePath || "",
      ),
      formatEnvLine(
        "SYNTH_TUI_API_KEY_LOCAL_VAR",
        backendKeySources.local.varName || "",
      ),
    ]
    await fs.writeFile(settingsFilePath, `${lines.join("\n")}\n`, "utf8")
  } catch (err: any) {
    snapshot.lastError = `Failed to save settings: ${err?.message || "unknown"}`
    renderer.requestRender()
  }
}

function parseEnvFile(content: string): Record<string, string> {
  const values: Record<string, string> = {}
  const lines = content.split(/\r?\n/)
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith("#")) continue
    const match = trimmed.match(/^(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.+)$/)
    if (!match) continue
    const key = match[1]
    let value = match[2].trim()
    if (
      (value.startsWith("\"") && value.endsWith("\"")) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      const quoted = value
      value = value.slice(1, -1)
      if (quoted.startsWith("\"")) {
        value = value.replace(/\\\\/g, "\\").replace(/\\"/g, "\"")
      }
    } else {
      value = value.split(/\s+#/)[0].trim()
    }
    values[key] = value
  }
  return values
}

function formatEnvLine(key: string, value: string): string {
  return `${key}=${escapeEnvValue(value)}`
}

function escapeEnvValue(value: string): string {
  const safe = value ?? ""
  return `"${safe.replace(/\\/g, "\\\\").replace(/\"/g, '\\"')}"`
}

async function walkEnvDir(
  dir: string,
  results: Map<string, EnvKeyOption>,
): Promise<void> {
  const ignoreDirs = new Set([
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".next",
    ".turbo",
    ".cache",
    "dist",
    "build",
    "out",
  ])
  let entries: Array<import("node:fs").Dirent>
  try {
    entries = await fs.readdir(dir, { withFileTypes: true })
  } catch {
    return
  }
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      if (ignoreDirs.has(entry.name)) continue
      await walkEnvDir(fullPath, results)
      continue
    }
    if (!entry.isFile()) continue
    if (!/^\.env(\.|$)/.test(entry.name)) continue
    try {
      const stat = await fs.stat(fullPath)
      if (stat.size > 256 * 1024) continue
      const content = await fs.readFile(fullPath, "utf8")
      const found = parseEnvKeys(content, fullPath)
      for (const item of found) {
        const existing = results.get(item.key)
        if (existing) {
          if (!existing.sources.includes(item.source)) {
            existing.sources.push(item.source)
          }
          if (!existing.varNames.includes(item.varName)) {
            existing.varNames.push(item.varName)
          }
        } else {
          results.set(item.key, {
            key: item.key,
            sources: [item.source],
            varNames: [item.varName],
          })
        }
      }
    } catch {
      // ignore parse errors
    }
  }
}

function parseEnvKeys(
  content: string,
  sourcePath: string,
): Array<{ key: string; source: string; varName: string }> {
  const results: Array<{ key: string; source: string; varName: string }> = []
  const lines = content.split(/\r?\n/)
  const keyNames = new Set([
    "SYNTH_API_KEY",
    "SYNTH_TUI_API_KEY_PROD",
    "SYNTH_TUI_API_KEY_DEV",
    "SYNTH_TUI_API_KEY_LOCAL",
  ])
  const relPath = path.relative(envKeyScanRoot, sourcePath) || sourcePath
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith("#")) continue
    const match = trimmed.match(/^(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.+)$/)
    if (!match) continue
    const varName = match[1]
    if (!keyNames.has(varName)) continue
    let value = match[2].trim()
    if (!value || value.startsWith("$")) continue
    if (
      (value.startsWith("\"") && value.endsWith("\"")) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1)
    } else {
      value = value.split(/\s+#/)[0].trim()
    }
    if (!value) continue
    results.push({ key: value, source: relPath, varName })
  }
  return results
}

function formatEnvKeySource(option: EnvKeyOption): string {
  if (!option.sources.length) return "-"
  const first = option.sources[0]
  if (option.sources.length === 1) return truncatePath(first, 36)
  return `${truncatePath(first, 28)} +${option.sources.length - 1}`
}

function truncatePath(value: string, max: number): string {
  if (value.length <= max) return value
  return `...${value.slice(Math.max(0, value.length - max + 3))}`
}

function maskKeyPrefix(key: string): string {
  if (!key) return "(missing)"
  return `${key.slice(0, 5)}...`
}

function beginPasteCapture(sequence: string): void {
  keyPasteActive = true
  keyPasteBuffer = ""
  const stripped = sequence.replace("\u001b[200~", "")
  if (stripped) {
    continuePasteCapture(stripped)
  }
}

function continuePasteCapture(sequence: string): void {
  const endIndex = sequence.indexOf("\u001b[201~")
  if (endIndex !== -1) {
    keyPasteBuffer += sequence.slice(0, endIndex)
    finalizePasteCapture()
    const remainder = sequence.slice(endIndex + "\u001b[201~".length)
    if (remainder) {
      ui.keyModalInput.value = (ui.keyModalInput.value || "") + remainder
    }
    renderer.requestRender()
    return
  }
  keyPasteBuffer += sequence
}

function finalizePasteCapture(): void {
  keyPasteActive = false
  const sanitized = keyPasteBuffer.replace(/\s+/g, "")
  if (sanitized) {
    ui.keyModalInput.value = (ui.keyModalInput.value || "") + sanitized
  }
  keyPasteBuffer = ""
}

function pasteKeyFromClipboard(): void {
  try {
    if (process.platform !== "darwin") return
    const result = execCommandSync("pbpaste")
    if (!result) return
    const sanitized = result.replace(/\s+/g, "")
    if (!sanitized) return
    const current = ui.keyModalInput.value || ""
    ui.keyModalInput.value = current + sanitized
  } catch {
    // Ignore clipboard errors
  }
  renderer.requestRender()
}

function execCommandSync(cmd: string): string | null {
  try {
    const proc = require("child_process").spawnSync(cmd, [], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    })
    if (proc.status !== 0) return null
    return proc.stdout ? String(proc.stdout) : null
  } catch {
    return null
  }
}

function toggleEventModal(visible: boolean): void {
  ui.eventModalVisible = visible
  ui.eventModalBox.visible = visible
  ui.eventModalTitle.visible = visible
  ui.eventModalText.visible = visible
  ui.eventModalHint.visible = visible
  if (!visible) {
    ui.eventModalText.content = ""
  }
  renderer.requestRender()
}

function toggleResultsModal(visible: boolean): void {
  ui.resultsModalVisible = visible
  ui.resultsModalBox.visible = visible
  ui.resultsModalTitle.visible = visible
  ui.resultsModalText.visible = visible
  ui.resultsModalHint.visible = visible
  if (visible) {
    // Blur jobs select to prevent background scrolling
    ui.jobsSelect.blur()
  } else {
    ui.resultsModalText.content = ""
    // Refocus jobs select when closing
    if (activePane === "jobs") {
      ui.jobsSelect.focus()
    }
  }
  renderer.requestRender()
}

function openResultsModal(): void {
  const payload = formatResultsExpanded()
  if (!payload) return
  resultsModalOffset = 0
  ui.resultsModalPayload = payload
  toggleResultsModal(true)
  updateResultsModalContent()
}

function moveResultsModal(delta: number): void {
  if (!ui.resultsModalPayload) return
  const { maxLines } = getResultsModalLayout()
  const lines = wrapModalText(ui.resultsModalPayload, getResultsModalLayout().textWidth)
  const maxOffset = Math.max(0, lines.length - maxLines)
  resultsModalOffset = clamp(resultsModalOffset + delta, 0, maxOffset)
  updateResultsModalContent()
}

function updateResultsModalContent(): void {
  if (!ui.resultsModalVisible || !ui.resultsModalPayload) return
  const { width, height, left, top, maxLines, textWidth } = getResultsModalLayout()
  ui.resultsModalBox.width = width
  ui.resultsModalBox.height = height
  ui.resultsModalBox.left = left
  ui.resultsModalBox.top = top
  ui.resultsModalTitle.left = left + 2
  ui.resultsModalTitle.top = top + 1
  ui.resultsModalText.left = left + 2
  ui.resultsModalText.top = top + 2
  ui.resultsModalText.width = width - 4
  ui.resultsModalHint.left = left + 2
  ui.resultsModalHint.top = top + height - 2
  const lines = wrapModalText(ui.resultsModalPayload, textWidth)
  const sliced = lines.slice(resultsModalOffset, resultsModalOffset + maxLines)
  ui.resultsModalText.content = sliced.join("\n")
  const pos = lines.length <= maxLines ? "end" : `${resultsModalOffset + 1}-${Math.min(resultsModalOffset + maxLines, lines.length)}`
  ui.resultsModalHint.content = `Results (${pos} of ${lines.length}) | j/k scroll | y copy | esc/q/enter close`
}

function getResultsModalLayout(): {
  width: number
  height: number
  left: number
  top: number
  maxLines: number
  textWidth: number
} {
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
  const width = Math.max(60, Math.floor(cols * 0.9))
  const height = Math.max(12, Math.floor(rows * 0.8))
  const left = Math.max(0, Math.floor((cols - width) / 2))
  const top = Math.max(1, Math.floor((rows - height) / 2))
  const maxLines = Math.max(1, height - 4)
  const textWidth = Math.max(30, width - 4)
  return { width, height, left, top, maxLines, textWidth }
}

function formatResultsExpanded(): string | null {
  const job = snapshot.selectedJob
  if (!job) return null
  if (!snapshot.bestSnapshot && !snapshot.bestSnapshotId) {
    return "No best snapshot available yet.\n\nPress 'p' to try loading the best snapshot."
  }
  const lines: string[] = []
  lines.push(`Job: ${job.job_id}`)
  lines.push(`Status: ${job.status}`)
  lines.push(`Best Score: ${job.best_score ?? "-"}`)
  lines.push(`Best Snapshot ID: ${snapshot.bestSnapshotId || "-"}`)
  lines.push("")
  if (snapshot.bestSnapshot) {
    // GEPA stores best_prompt and best_prompt_messages directly in the snapshot
    const bestPrompt = snapshot.bestSnapshot.best_prompt
    const bestPromptMessages = snapshot.bestSnapshot.best_prompt_messages

    if (bestPrompt && typeof bestPrompt === "object") {
      const promptId = bestPrompt.id || bestPrompt.template_id
      const promptName = bestPrompt.name
      if (promptName) lines.push(`Prompt Name: ${promptName}`)
      if (promptId) lines.push(`Prompt ID: ${promptId}`)
      lines.push("")

      // Extract sections from best_prompt (each section = a stage)
      const sections = bestPrompt.sections || bestPrompt.prompt_sections || []
      if (Array.isArray(sections) && sections.length > 0) {
        lines.push(`=== PROMPT TEMPLATE (${sections.length} stage${sections.length > 1 ? "s" : ""}) ===`)
        lines.push("")
        for (let i = 0; i < sections.length; i++) {
          const section = sections[i]
          const role = section.role || "stage"
          const name = section.name || section.id || ""
          const content = section.content || ""
          const order = section.order !== undefined ? section.order : i
          lines.push(`┌─ Stage ${order + 1}: ${role}${name ? ` (${name})` : ""} ─┐`)
          lines.push("")
          if (content) {
            lines.push(content)
          } else {
            lines.push("(empty)")
          }
          lines.push("")
          lines.push(`└${"─".repeat(40)}┘`)
          lines.push("")
        }
      }
    }

    // Show rendered messages (best_prompt_messages)
    if (Array.isArray(bestPromptMessages) && bestPromptMessages.length > 0) {
      lines.push(`=== RENDERED MESSAGES (${bestPromptMessages.length} message${bestPromptMessages.length > 1 ? "s" : ""}) ===`)
      lines.push("")
      for (let i = 0; i < bestPromptMessages.length; i++) {
        const msg = bestPromptMessages[i]
        const role = msg.role || "unknown"
        const content = msg.content || ""
        lines.push(`┌─ Message ${i + 1}: [${role}] ─┐`)
        lines.push("")
        lines.push(content)
        lines.push("")
        lines.push(`└${"─".repeat(40)}┘`)
        lines.push("")
      }
    }

    // Fallback: check for legacy extractors if nothing found
    if (!bestPrompt && !bestPromptMessages) {
      const legacyPrompt = extractBestPrompt(snapshot.bestSnapshot)
      const legacyText = extractBestPromptText(snapshot.bestSnapshot)

      if (legacyPrompt) {
        const sections = extractPromptSections(legacyPrompt)
        if (sections.length > 0) {
          lines.push(`=== PROMPT SECTIONS (${sections.length} stage${sections.length > 1 ? "s" : ""}) ===`)
          lines.push("")
          for (let i = 0; i < sections.length; i++) {
            const section = sections[i]
            const role = section.role || "stage"
            const name = section.name || section.id || ""
            const content = section.content || ""
            lines.push(`┌─ Stage ${i + 1}: ${role}${name ? ` (${name})` : ""} ─┐`)
            lines.push("")
            if (content) {
              lines.push(content)
            }
            lines.push("")
            lines.push(`└${"─".repeat(40)}┘`)
            lines.push("")
          }
        }
      }

      if (legacyText) {
        lines.push("=== RENDERED PROMPT ===")
        lines.push("")
        lines.push(legacyText)
      }

      // Last resort: show raw data
      if (!legacyPrompt && !legacyText) {
        lines.push("=== RAW SNAPSHOT DATA ===")
        lines.push("")
        try {
          lines.push(JSON.stringify(snapshot.bestSnapshot, null, 2))
        } catch {
          lines.push(String(snapshot.bestSnapshot))
        }
      }
    }
  } else {
    lines.push("Best snapshot data not loaded. Press 'p' to load.")
  }
  return lines.join("\n")
}

function getPromptForClipboard(): string | null {
  if (!snapshot.bestSnapshot) return null

  // Try best_prompt_messages first (rendered format)
  const messages = snapshot.bestSnapshot.best_prompt_messages
  if (Array.isArray(messages) && messages.length > 0) {
    return messages.map((msg: any) => {
      const role = msg.role || "unknown"
      const content = msg.content || ""
      return `[${role}]\n${content}`
    }).join("\n\n")
  }

  // Fall back to best_prompt sections
  const bestPrompt = snapshot.bestSnapshot.best_prompt
  if (bestPrompt && typeof bestPrompt === "object") {
    const sections = bestPrompt.sections || bestPrompt.prompt_sections || []
    if (Array.isArray(sections) && sections.length > 0) {
      return sections.map((section: any) => {
        const role = section.role || "stage"
        const name = section.name || ""
        const content = section.content || ""
        return `--- ${role}${name ? `: ${name}` : ""} ---\n${content}`
      }).join("\n\n")
    }
  }

  return null
}

async function copyToClipboard(text: string): Promise<void> {
  // Use pbcopy on macOS
  const proc = Bun.spawn(["pbcopy"], {
    stdin: "pipe",
  })
  proc.stdin.write(text)
  proc.stdin.end()
  await proc.exited
}

async function copyPromptToClipboard(): Promise<void> {
  const promptText = getPromptForClipboard()
  if (!promptText) {
    snapshot.status = "No prompt to copy"
    renderSnapshot()
    return
  }

  try {
    await copyToClipboard(promptText)
    snapshot.status = "Prompt copied to clipboard!"
  } catch (err: any) {
    snapshot.lastError = err?.message || "Failed to copy to clipboard"
    snapshot.status = "Copy failed"
  }
  renderSnapshot()
}

// Config modal state
let configModalOffset = 0

function toggleConfigModal(visible: boolean): void {
  ui.configModalVisible = visible
  ui.configModalBox.visible = visible
  ui.configModalTitle.visible = visible
  ui.configModalText.visible = visible
  ui.configModalHint.visible = visible
  if (visible) {
    ui.jobsSelect.blur()
  } else {
    ui.configModalText.content = ""
    if (activePane === "jobs") {
      ui.jobsSelect.focus()
    }
  }
  renderer.requestRender()
}

function openConfigModal(): void {
  const payload = formatConfigMetadata()
  if (!payload) return
  configModalOffset = 0
  ui.configModalPayload = payload
  toggleConfigModal(true)
  updateConfigModalContent()
}

function moveConfigModal(delta: number): void {
  if (!ui.configModalPayload) return
  const { maxLines } = getConfigModalLayout()
  const lines = wrapModalText(ui.configModalPayload, getConfigModalLayout().textWidth)
  const maxOffset = Math.max(0, lines.length - maxLines)
  configModalOffset = clamp(configModalOffset + delta, 0, maxOffset)
  updateConfigModalContent()
}

function updateConfigModalContent(): void {
  if (!ui.configModalVisible || !ui.configModalPayload) return
  const { width, height, left, top, maxLines, textWidth } = getConfigModalLayout()
  ui.configModalBox.width = width
  ui.configModalBox.height = height
  ui.configModalBox.left = left
  ui.configModalBox.top = top
  ui.configModalTitle.left = left + 2
  ui.configModalTitle.top = top + 1
  ui.configModalText.left = left + 2
  ui.configModalText.top = top + 2
  ui.configModalText.width = width - 4
  ui.configModalHint.left = left + 2
  ui.configModalHint.top = top + height - 2
  const lines = wrapModalText(ui.configModalPayload, textWidth)
  const sliced = lines.slice(configModalOffset, configModalOffset + maxLines)
  ui.configModalText.content = sliced.join("\n")
  const pos = lines.length <= maxLines ? "end" : `${configModalOffset + 1}-${Math.min(configModalOffset + maxLines, lines.length)}`
  ui.configModalHint.content = `Config (${pos} of ${lines.length}) | j/k scroll | esc/q/enter close`
}

function getConfigModalLayout(): {
  width: number
  height: number
  left: number
  top: number
  maxLines: number
  textWidth: number
} {
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
  const width = Math.max(60, Math.floor(cols * 0.9))
  const height = Math.max(12, Math.floor(rows * 0.8))
  const left = Math.max(0, Math.floor((cols - width) / 2))
  const top = Math.max(1, Math.floor((rows - height) / 2))
  const maxLines = Math.max(1, height - 4)
  const textWidth = Math.max(30, width - 4)
  return { width, height, left, top, maxLines, textWidth }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prompt Browser Modal - view baseline and all candidate prompts with scores
// ─────────────────────────────────────────────────────────────────────────────

function togglePromptBrowserModal(visible: boolean): void {
  ui.promptBrowserVisible = visible
  ui.promptBrowserBox.visible = visible
  ui.promptBrowserTitle.visible = visible
  ui.promptBrowserText.visible = visible
  ui.promptBrowserHint.visible = visible
  if (visible) {
    ui.jobsSelect.blur()
  } else {
    ui.promptBrowserText.content = ""
    if (activePane === "jobs") {
      ui.jobsSelect.focus()
    }
  }
  renderer.requestRender()
}

async function openPromptBrowserModal(): Promise<void> {
  const job = snapshot.selectedJob
  if (!job || job.job_source === "learning" || isEvalJob(job)) {
    snapshot.status = "Prompt browser not available for this job type"
    return
  }

  // Fetch candidates if not already loaded
  if (snapshot.allCandidates.length === 0) {
    snapshot.status = "Loading prompt candidates..."
    renderSnapshot()
    await fetchAllCandidates()
  }

  if (snapshot.allCandidates.length === 0) {
    snapshot.status = "No prompt candidates found for this job"
    return
  }

  promptBrowserIndex = 0
  promptBrowserOffset = 0
  togglePromptBrowserModal(true)
  updatePromptBrowserContent()
}

function movePromptBrowserScroll(delta: number): void {
  const { maxLines, textWidth } = getPromptBrowserLayout()
  const content = formatCandidateContent(snapshot.allCandidates[promptBrowserIndex])
  const lines = wrapModalText(content, textWidth)
  const maxOffset = Math.max(0, lines.length - maxLines)
  promptBrowserOffset = clamp(promptBrowserOffset + delta, 0, maxOffset)
  updatePromptBrowserContent()
}

function movePromptBrowserCandidate(delta: number): void {
  const total = snapshot.allCandidates.length
  if (total === 0) return
  promptBrowserIndex = ((promptBrowserIndex + delta) % total + total) % total
  promptBrowserOffset = 0 // Reset scroll when changing candidate
  updatePromptBrowserContent()
}

function updatePromptBrowserContent(): void {
  if (!ui.promptBrowserVisible) return
  const candidates = snapshot.allCandidates
  if (candidates.length === 0) {
    ui.promptBrowserText.content = "No candidates available."
    return
  }

  const { width, height, left, top, maxLines, textWidth } = getPromptBrowserLayout()
  ui.promptBrowserBox.width = width
  ui.promptBrowserBox.height = height
  ui.promptBrowserBox.left = left
  ui.promptBrowserBox.top = top
  ui.promptBrowserTitle.left = left + 2
  ui.promptBrowserTitle.top = top + 1
  ui.promptBrowserText.left = left + 2
  ui.promptBrowserText.top = top + 2
  ui.promptBrowserText.width = width - 4
  ui.promptBrowserHint.left = left + 2
  ui.promptBrowserHint.top = top + height - 2

  const candidate = candidates[promptBrowserIndex]
  const content = formatCandidateContent(candidate)
  const lines = wrapModalText(content, textWidth)
  const sliced = lines.slice(promptBrowserOffset, promptBrowserOffset + maxLines)
  ui.promptBrowserText.content = sliced.join("\n")

  // Build title with navigation info
  const idx = promptBrowserIndex + 1
  const total = candidates.length
  const label = candidate.isBaseline ? " [BASELINE]" : (candidate.tag === "best" ? " [BEST]" : "")
  const scoreStr = candidate.score != null ? ` | Score: ${candidate.score.toFixed(3)}` : ""
  ui.promptBrowserTitle.content = `Prompt Browser (${idx}/${total})${label}${scoreStr}`

  // Build hint
  const scrollPos = lines.length <= maxLines ? "end" : `${promptBrowserOffset + 1}-${Math.min(promptBrowserOffset + maxLines, lines.length)}`
  ui.promptBrowserHint.content = `(${scrollPos} of ${lines.length}) | h/l prev/next | j/k scroll | y copy | esc close`
  renderer.requestRender()
}

function formatCandidateContent(candidate: PromptCandidate | undefined): string {
  if (!candidate) return "No candidate selected."
  const lines: string[] = []

  // Header
  if (candidate.isBaseline) {
    lines.push("═══ BASELINE PROMPT ═══")
    lines.push("")
    lines.push("This is the initial prompt configuration before optimization.")
  } else {
    const scoreStr = candidate.score != null ? candidate.score.toFixed(4) : "N/A"
    lines.push(`═══ CANDIDATE: ${candidate.id.slice(0, 12)}... ═══`)
    lines.push("")
    lines.push(`Score: ${scoreStr}`)
    if (candidate.tag) {
      lines.push(`Tag: ${candidate.tag}`)
    }
    // Show additional metadata from events
    const payload = candidate.payload
    if (payload) {
      if (payload.generation != null) lines.push(`Generation: ${payload.generation}`)
      if (payload.mutation_type) lines.push(`Mutation: ${payload.mutation_type}`)
      if (payload.status) lines.push(`Status: ${payload.status}`)
    }
  }
  lines.push("")

  const payload = candidate.payload
  if (!payload || typeof payload !== "object") {
    lines.push("(No payload data)")
    return lines.join("\n")
  }

  // Format 0: prompt_text directly available (from events)
  const promptText = payload.prompt_text
  if (promptText && typeof promptText === "string" && promptText.length > 0) {
    lines.push("=== PROMPT TEXT ===")
    lines.push("")
    lines.push(promptText)
    lines.push("")

    // Also show stages if available for structured view
    const stages = payload.stages
    if (stages && typeof stages === "object" && Object.keys(stages).length > 0) {
      lines.push("")
      lines.push("=== STAGES (STRUCTURED) ===")
      for (const [stageId, stageData] of Object.entries(stages)) {
        if (!stageData || typeof stageData !== "object") continue
        const sd = stageData as Record<string, any>
        const instruction = sd.instruction || ""
        lines.push(`┌─ [${stageId.toUpperCase()}] ─┐`)
        lines.push(instruction || "(empty)")
        lines.push(`└${"─".repeat(30)}┘`)
        lines.push("")
      }
    }
    return lines.join("\n")
  }

  // Format 1: stages from events (structured prompt)
  const stages = payload.stages
  if (stages && typeof stages === "object" && Object.keys(stages).length > 0) {
    const stageKeys = Object.keys(stages)
    lines.push(`=== STAGES (${stageKeys.length}) ===`)
    lines.push("")
    for (const stageId of stageKeys) {
      const stageData = stages[stageId]
      if (!stageData || typeof stageData !== "object") continue
      const instruction = stageData.instruction || ""
      const rules = stageData.rules || {}
      lines.push(`┌─ [${stageId.toUpperCase()}] ─┐`)
      lines.push(instruction || "(empty)")
      if (Object.keys(rules).length > 0) {
        lines.push("")
        lines.push("Rules:")
        for (const [ruleKey, ruleVal] of Object.entries(rules)) {
          lines.push(`  • ${ruleKey}: ${ruleVal}`)
        }
      }
      lines.push(`└${"─".repeat(30)}┘`)
      lines.push("")
    }
    return lines.join("\n")
  }

  // Format 2: GEPA best_prompt_messages (rendered messages)
  const bestPromptMessages = payload.best_prompt_messages
  if (Array.isArray(bestPromptMessages) && bestPromptMessages.length > 0) {
    lines.push(`=== MESSAGES (${bestPromptMessages.length}) ===`)
    lines.push("")
    for (let i = 0; i < bestPromptMessages.length; i++) {
      const msg = bestPromptMessages[i]
      const role = msg.role || "unknown"
      const content = msg.content || ""
      lines.push(`┌─ [${role}] ─┐`)
      lines.push(content)
      lines.push(`└${"─".repeat(30)}┘`)
      lines.push("")
    }
    return lines.join("\n")
  }

  // Format 3: GEPA best_prompt (structured prompt object)
  const bestPrompt = payload.best_prompt
  if (bestPrompt && typeof bestPrompt === "object") {
    const sections = bestPrompt.sections || bestPrompt.prompt_sections || []
    if (Array.isArray(sections) && sections.length > 0) {
      lines.push(`=== PROMPT SECTIONS (${sections.length}) ===`)
      lines.push("")
      for (let i = 0; i < sections.length; i++) {
        const section = sections[i]
        const role = section.role || "stage"
        const name = section.name || ""
        const content = section.content || ""
        lines.push(`┌─ ${role}${name ? `: ${name}` : ""} ─┐`)
        lines.push(content || "(empty)")
        lines.push(`└${"─".repeat(30)}┘`)
        lines.push("")
      }
      return lines.join("\n")
    }
  }

  // Format 4: Baseline from initial_prompt (serialized prompt from initial snapshot)
  if (candidate.isBaseline) {
    const initialPrompt = payload.initial_prompt
    if (initialPrompt) {
      // initial_prompt can be a string (JSON serialized) or an object
      let promptObj = initialPrompt
      if (typeof initialPrompt === "string") {
        try {
          promptObj = JSON.parse(initialPrompt)
        } catch {
          // If it's not valid JSON, show as-is
          lines.push("=== INITIAL PROMPT ===")
          lines.push("")
          lines.push(initialPrompt)
          return lines.join("\n")
        }
      }

      if (promptObj && typeof promptObj === "object") {
        // Handle PromptPattern format with data.messages: {data: {messages: [{role, order, pattern}, ...]}}
        const dataMessages = promptObj.data?.messages
        if (Array.isArray(dataMessages) && dataMessages.length > 0) {
          // Sort by order if available
          const sortedMessages = [...dataMessages].sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
          lines.push(`=== INITIAL PROMPT MESSAGES (${sortedMessages.length}) ===`)
          lines.push("")
          for (const msg of sortedMessages) {
            const role = msg.role || "unknown"
            const pattern = msg.pattern || msg.content || ""
            lines.push(`┌─ [${role.toUpperCase()}] ─┐`)
            lines.push(pattern || "(empty)")
            lines.push(`└${"─".repeat(30)}┘`)
            lines.push("")
          }
          return lines.join("\n")
        }

        // Handle PromptPattern format: {stages: {system: {instruction, rules}}}
        const promptStages = promptObj.stages || {}
        if (Object.keys(promptStages).length > 0) {
          lines.push(`=== INITIAL PROMPT STAGES (${Object.keys(promptStages).length}) ===`)
          lines.push("")
          for (const [stageId, stageData] of Object.entries(promptStages)) {
            if (!stageData || typeof stageData !== "object") continue
            const sd = stageData as Record<string, any>
            const instruction = sd.instruction || ""
            lines.push(`┌─ [${stageId.toUpperCase()}] ─┐`)
            lines.push(instruction || "(empty)")
            lines.push(`└${"─".repeat(30)}┘`)
            lines.push("")
          }
          return lines.join("\n")
        }

        // Handle PromptTemplate format: {sections: [...]}
        const sections = promptObj.sections || promptObj.prompt_sections || []
        if (Array.isArray(sections) && sections.length > 0) {
          lines.push(`=== INITIAL PROMPT SECTIONS (${sections.length}) ===`)
          lines.push("")
          for (const section of sections) {
            const role = section.role || "stage"
            const name = section.name || section.id || ""
            const content = section.content || section.template || ""
            lines.push(`┌─ ${role}${name ? `: ${name}` : ""} ─┐`)
            lines.push(content || "(empty)")
            lines.push(`└${"─".repeat(30)}┘`)
            lines.push("")
          }
          return lines.join("\n")
        }

        // Handle simple instruction format
        if (promptObj.instruction) {
          lines.push("=== INITIAL PROMPT ===")
          lines.push("")
          lines.push(promptObj.instruction)
          return lines.join("\n")
        }
      }
    }

    // Legacy: prompt_config fallback
    const promptConfig = payload.prompt_config
    if (promptConfig && typeof promptConfig === "object") {
      const sections = promptConfig.sections || promptConfig.prompt_sections || []
      if (Array.isArray(sections) && sections.length > 0) {
        lines.push(`=== INITIAL PROMPT SECTIONS (${sections.length}) ===`)
        lines.push("")
        for (const section of sections) {
          const role = section.role || "stage"
          const name = section.name || section.id || ""
          const content = section.content || section.template || ""
          lines.push(`┌─ ${role}${name ? `: ${name}` : ""} ─┐`)
          lines.push(content || "(empty)")
          lines.push(`└${"─".repeat(30)}┘`)
          lines.push("")
        }
        return lines.join("\n")
      }
      if (promptConfig.name || promptConfig.template) {
        lines.push("=== INITIAL PROMPT ===")
        lines.push("")
        if (promptConfig.name) lines.push(`Name: ${promptConfig.name}`)
        if (promptConfig.template) {
          lines.push("")
          lines.push(promptConfig.template)
        }
        return lines.join("\n")
      }
    }
  }

  // Fallback: show raw JSON (but exclude large nested objects)
  lines.push("=== RAW PAYLOAD ===")
  lines.push("")
  try {
    // Show a simplified view, excluding program_candidate to reduce noise
    const simplified = { ...payload }
    delete simplified.program_candidate
    lines.push(JSON.stringify(simplified, null, 2))
  } catch {
    lines.push(String(payload))
  }
  return lines.join("\n")
}

function getPromptBrowserLayout(): {
  width: number
  height: number
  left: number
  top: number
  maxLines: number
  textWidth: number
} {
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
  const width = Math.max(60, Math.floor(cols * 0.9))
  const height = Math.max(12, Math.floor(rows * 0.8))
  const left = Math.max(0, Math.floor((cols - width) / 2))
  const top = Math.max(1, Math.floor((rows - height) / 2))
  const maxLines = Math.max(1, height - 4)
  const textWidth = Math.max(30, width - 4)
  return { width, height, left, top, maxLines, textWidth }
}

function getPromptBrowserClipboardContent(): string | null {
  const candidates = snapshot.allCandidates
  if (candidates.length === 0) return null
  const candidate = candidates[promptBrowserIndex]
  if (!candidate) return null

  // Try to get messages first
  const messages = candidate.payload?.best_prompt_messages
  if (Array.isArray(messages) && messages.length > 0) {
    return messages.map((msg: any) => {
      const role = msg.role || "unknown"
      const content = msg.content || ""
      return `[${role}]\n${content}`
    }).join("\n\n")
  }

  // Try sections
  const sections = candidate.payload?.best_prompt?.sections || candidate.payload?.prompt_config?.sections
  if (Array.isArray(sections) && sections.length > 0) {
    return sections.map((s: any) => {
      const role = s.role || "stage"
      const name = s.name || ""
      const content = s.content || s.template || ""
      return `--- ${role}${name ? `: ${name}` : ""} ---\n${content}`
    }).join("\n\n")
  }

  // Fallback to JSON
  try {
    return JSON.stringify(candidate.payload, null, 2)
  } catch {
    return null
  }
}

function formatConfigMetadata(): string | null {
  const job = snapshot.selectedJob
  if (!job) return null

  const lines: string[] = []
  lines.push(`Job: ${job.job_id}`)
  lines.push(`Status: ${job.status}`)
  lines.push(`Type: ${job.training_type || "-"}`)
  lines.push(`Source: ${job.job_source || "unknown"}`)
  lines.push("")

  // Check for errors
  if (snapshot.lastError && snapshot.status?.includes("Error")) {
    lines.push("═══ Error Loading Metadata ═══")
    lines.push(snapshot.lastError)
    lines.push("")
    lines.push("The job details could not be loaded.")
    return lines.join("\n")
  }

  const meta = job.metadata
  if (!meta || Object.keys(meta).length === 0) {
    // Check if job details are still loading
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

  // Extract description
  const desc = meta.request_metadata?.description || meta.description
  if (desc) {
    lines.push(`Description: ${desc}`)
    lines.push("")
  }

  // Extract algorithm config (nested under different paths for different job types)
  // Prompt-learning jobs: meta.prompt_initial_snapshot.raw_config.prompt_learning
  // Learning/GEPA jobs: meta.config, meta.job_config, or top-level
  const rawConfig =
    meta.prompt_initial_snapshot?.raw_config?.prompt_learning ||
    meta.config?.prompt_learning ||
    meta.job_config?.prompt_learning ||
    meta.prompt_learning ||
    meta.config ||
    meta.job_config ||
    null
  const optimizerConfig =
    meta.prompt_initial_snapshot?.optimizer_config ||
    meta.optimizer_config ||
    null

  // Policy / Model info
  const policy = rawConfig?.policy || optimizerConfig?.policy_config
  if (policy) {
    lines.push("═══ Model Configuration ═══")
    if (policy.model) lines.push(`  Model: ${policy.model}`)
    if (policy.provider) lines.push(`  Provider: ${policy.provider}`)
    if (policy.temperature != null) lines.push(`  Temperature: ${policy.temperature}`)
    if (policy.max_completion_tokens) lines.push(`  Max Tokens: ${policy.max_completion_tokens}`)
    lines.push("")
  }

  // GEPA-specific config
  const gepa = rawConfig?.gepa
  if (gepa) {
    lines.push("═══ GEPA Configuration ═══")

    // Population
    const pop = gepa.population
    if (pop) {
      lines.push("  Population:")
      if (pop.initial_size != null) lines.push(`    Initial Size: ${pop.initial_size}`)
      if (pop.num_generations != null) lines.push(`    Generations: ${pop.num_generations}`)
      if (pop.children_per_generation != null) lines.push(`    Children/Gen: ${pop.children_per_generation}`)
      if (pop.crossover_rate != null) lines.push(`    Crossover Rate: ${pop.crossover_rate}`)
      if (pop.selection_pressure != null) lines.push(`    Selection Pressure: ${pop.selection_pressure}`)
      if (pop.patience_generations != null) lines.push(`    Patience: ${pop.patience_generations}`)
    }

    // Rollout
    const rollout = gepa.rollout
    if (rollout) {
      lines.push("  Rollout:")
      if (rollout.budget != null) lines.push(`    Budget: ${rollout.budget}`)
      if (rollout.max_concurrent != null) lines.push(`    Max Concurrent: ${rollout.max_concurrent}`)
      if (rollout.minibatch_size != null) lines.push(`    Minibatch Size: ${rollout.minibatch_size}`)
    }

    // Mutation
    const mutation = gepa.mutation
    if (mutation) {
      lines.push("  Mutation:")
      if (mutation.rate != null) lines.push(`    Rate: ${mutation.rate}`)
    }

    // Archive
    const archive = gepa.archive
    if (archive) {
      lines.push("  Archive:")
      if (archive.size != null) lines.push(`    Size: ${archive.size}`)
      if (archive.pareto_set_size != null) lines.push(`    Pareto Set Size: ${archive.pareto_set_size}`)
    }

    // Evaluation
    const evaluation = gepa.evaluation
    if (evaluation) {
      lines.push("  Evaluation:")
      if (evaluation.seeds) {
        const seeds = Array.isArray(evaluation.seeds) ? evaluation.seeds : []
        lines.push(`    Seeds: [${seeds.slice(0, 5).join(", ")}${seeds.length > 5 ? `, ... (${seeds.length} total)` : ""}]`)
      }
      if (evaluation.validation_seeds) {
        const vseeds = Array.isArray(evaluation.validation_seeds) ? evaluation.validation_seeds : []
        lines.push(`    Validation Seeds: [${vseeds.slice(0, 5).join(", ")}${vseeds.length > 5 ? `, ... (${vseeds.length} total)` : ""}]`)
      }
      if (evaluation.validation_top_k != null) lines.push(`    Validation Top-K: ${evaluation.validation_top_k}`)
    }

    // Proposer
    if (gepa.proposer_type) lines.push(`  Proposer Type: ${gepa.proposer_type}`)
    if (gepa.proposer_effort) lines.push(`  Proposer Effort: ${gepa.proposer_effort}`)

    lines.push("")
  }

  // Verifier config
  const verifier = rawConfig?.verifier || optimizerConfig?.verifier
  if (verifier && verifier.enabled) {
    lines.push("═══ Verifier Configuration ═══")
    if (verifier.backend_model) lines.push(`  Model: ${verifier.backend_model}`)
    if (verifier.backend_provider) lines.push(`  Provider: ${verifier.backend_provider}`)
    if (verifier.verifier_graph_id) lines.push(`  Graph ID: ${verifier.verifier_graph_id}`)
    if (verifier.reward_source) lines.push(`  Reward Source: ${verifier.reward_source}`)
    if (verifier.concurrency != null) lines.push(`  Concurrency: ${verifier.concurrency}`)
    if (verifier.timeout != null) lines.push(`  Timeout: ${verifier.timeout}s`)
    lines.push("")
  }

  // Environment config
  const envName = rawConfig?.env_name || optimizerConfig?.env_name || gepa?.env_name
  const envConfig = rawConfig?.env_config || optimizerConfig?.env_config
  if (envName || envConfig) {
    lines.push("═══ Environment ═══")
    if (envName) lines.push(`  Name: ${envName}`)
    if (envConfig?.env_params) {
      const params = envConfig.env_params
      if (params.max_steps != null) lines.push(`  Max Steps: ${params.max_steps}`)
    }
    lines.push("")
  }

  // Task app
  const taskAppUrl = meta.task_app_url || rawConfig?.task_app_url
  const taskAppId = rawConfig?.task_app_id
  if (taskAppUrl || taskAppId) {
    lines.push("═══ Task App ═══")
    if (taskAppId) lines.push(`  ID: ${taskAppId}`)
    if (taskAppUrl) lines.push(`  URL: ${taskAppUrl}`)
    lines.push("")
  }

  // Fallback: show raw metadata structure
  if (lines.length <= 5) {
    lines.push("═══ Raw Metadata ═══")
    try {
      lines.push(JSON.stringify(meta, null, 2))
    } catch {
      lines.push(String(meta))
    }
  }

  return lines.join("\n")
}

function openSelectedEventModal(): void {
  const recent = getFilteredEvents()
  const event = recent[selectedEventIndex]
  if (!event) return
  const header = `${event.type}`
  const detail = event.message ?? formatEventData(event.data)
  const payload = detail ? `${header}\n\n${detail}` : header
  eventModalOffset = 0
  ui.eventModalPayload = payload
  toggleEventModal(true)
  updateEventModalContent()
}

function moveEventModal(delta: number): void {
  if (!ui.eventModalPayload) return
  const { maxLines } = getEventModalLayout()
  const lines = wrapModalText(ui.eventModalPayload, getEventModalLayout().textWidth)
  const maxOffset = Math.max(0, lines.length - maxLines)
  eventModalOffset = clamp(eventModalOffset + delta, 0, maxOffset)
  updateEventModalContent()
}

function updateEventModalContent(): void {
  if (!ui.eventModalVisible || !ui.eventModalPayload) return
  const { width, height, left, top, maxLines, textWidth } = getEventModalLayout()
  ui.eventModalBox.width = width
  ui.eventModalBox.height = height
  ui.eventModalBox.left = left
  ui.eventModalBox.top = top
  ui.eventModalTitle.left = left + 2
  ui.eventModalTitle.top = top + 1
  ui.eventModalText.left = left + 2
  ui.eventModalText.top = top + 2
  ui.eventModalText.width = width - 4
  ui.eventModalHint.left = left + 2
  ui.eventModalHint.top = top + height - 2
  const lines = wrapModalText(ui.eventModalPayload, textWidth)
  const sliced = lines.slice(eventModalOffset, eventModalOffset + maxLines)
  ui.eventModalText.content = sliced.join("\n")
  const pos = lines.length <= maxLines ? "end" : `${eventModalOffset + 1}-${Math.min(eventModalOffset + maxLines, lines.length)}`
  ui.eventModalHint.content = `Event details (${pos} of ${lines.length}) | esc/q/enter close`
}

function getEventModalLayout(): {
  width: number
  height: number
  left: number
  top: number
  maxLines: number
  textWidth: number
} {
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
  const width = Math.max(50, Math.floor(cols * 0.85))
  const height = Math.max(8, Math.floor(rows * 0.6))
  const left = Math.max(0, Math.floor((cols - width) / 2))
  const top = Math.max(2, Math.floor((rows - height) / 2))
  const maxLines = Math.max(1, height - 4)
  const textWidth = Math.max(20, width - 4)
  return { width, height, left, top, maxLines, textWidth }
}

function wrapModalText(text: string, width: number): string[] {
  const lines: string[] = []
  const rawLines = text.split("\n")
  for (const raw of rawLines) {
    if (raw.length === 0) {
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

function setActivePane(pane: "jobs" | "events"): void {
  if (activePane === pane) return
  activePane = pane
  if (pane === "jobs") {
    ui.jobsSelect.focus()
  } else {
    ui.jobsSelect.blur()
  }
  updatePaneIndicators()
  renderer.requestRender()
}

function updatePaneIndicators(): void {
  ui.jobsTabText.fg = activePane === "jobs" ? "#f8fafc" : "#94a3b8"
  ui.eventsTabText.fg = activePane === "events" ? "#f8fafc" : "#94a3b8"
  ui.jobsBox.borderColor = activePane === "jobs" ? "#60a5fa" : "#334155"
  ui.eventsBox.borderColor = activePane === "events" ? "#60a5fa" : "#334155"
}

function moveEventSelection(delta: number): void {
  const filtered = getFilteredEvents()
  if (!filtered.length) return
  const recentCount = Math.min(eventHistoryLimit, filtered.length)
  selectedEventIndex = clamp(selectedEventIndex + delta, 0, Math.max(0, recentCount - 1))
  renderSnapshot()
}

function toggleSelectedEventExpanded(): void {
  const recent = getFilteredEvents()
  const event = recent[selectedEventIndex]
  if (!event) return
  const detail = event.message ?? formatEventData(event.data)
  if (detail.length <= eventCollapseLimit) return
  event.expanded = !event.expanded
  renderSnapshot()
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function scheduleJobsPoll(delayMs: number): void {
  if (jobsTimer) clearTimeout(jobsTimer)
  jobsTimer = setTimeout(pollJobs, delayMs)
}

async function pollJobs(): Promise<void> {
  if (jobsInFlight) {
    scheduleJobsPoll(jobsPollMs)
    return
  }
  jobsInFlight = true
  const ok = await refreshJobs()
  jobsInFlight = false
  if (ok) {
    jobsPollMs = Math.max(1, refreshInterval) * 1000
  } else {
    jobsPollMs = Math.min(jobsPollMs * 2, Math.max(1, maxRefreshInterval) * 1000)
  }
  scheduleJobsPoll(jobsPollMs)
}

function scheduleEventsPoll(delayMs: number): void {
  if (eventsTimer) clearTimeout(eventsTimer)
  eventsTimer = setTimeout(pollEvents, delayMs)
}

async function pollEvents(): Promise<void> {
  if (eventsInFlight) {
    scheduleEventsPoll(eventsPollMs)
    return
  }
  eventsInFlight = true
  const ok = await refreshEvents()
  eventsInFlight = false
  if (ok) {
    eventsPollMs = Math.max(0.5, eventInterval) * 1000
  } else {
    eventsPollMs = Math.min(eventsPollMs * 2, Math.max(0.5, maxEventInterval) * 1000)
  }
  scheduleEventsPoll(eventsPollMs)
}

async function apiGet(path: string): Promise<any> {
  const { baseUrl, apiKey, label } = getBackendConfig()
  if (!apiKey) {
    throw new Error(`Missing API key for ${label}`)
  }
  const res = await fetch(`${baseUrl}${path}`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  })
  if (!res.ok) {
    const body = await res.text().catch(() => "")
    const suffix = body ? ` - ${body.slice(0, 160)}` : ""
    throw new Error(`GET ${path}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  return res.json()
}

async function apiGetV1(path: string): Promise<any> {
  const { baseRoot, apiKey, label } = getBackendConfig()
  if (!apiKey) {
    throw new Error(`Missing API key for ${label}`)
  }
  const res = await fetch(`${baseRoot}/api/v1${path}`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  })
  if (!res.ok) {
    const body = await res.text().catch(() => "")
    const suffix = body ? ` - ${body.slice(0, 160)}` : ""
    throw new Error(`GET /api/v1${path}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  return res.json()
}

async function apiPost(path: string, body: any): Promise<any> {
  const { baseUrl, apiKey, label } = getBackendConfig()
  if (!apiKey) {
    throw new Error(`Missing API key for ${label}`)
  }
  const res = await fetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    const suffix = text ? ` - ${text.slice(0, 160)}` : ""
    throw new Error(`POST ${path}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  return res.json().catch(() => ({}))
}

async function refreshHealth(): Promise<void> {
  try {
    const res = await fetch(`${getActiveBaseRoot()}/health`)
    healthStatus = res.ok ? "ok" : `bad(${res.status})`
  } catch (err: any) {
    healthStatus = `err(${err?.message || "unknown"})`
  }
}

// Helper to extract environment name from job metadata
function extractEnvName(job: JobSummary | null): string | null {
  if (!job?.metadata) return null
  const meta = job.metadata
  return (
    meta.prompt_initial_snapshot?.raw_config?.prompt_learning?.env_name ||
    meta.prompt_initial_snapshot?.optimizer_config?.env_name ||
    meta.config?.env_name ||
    meta.env_name ||
    null
  )
}

function calculateTotalTokensFromEvents(): number {
  let total = 0
  for (const event of snapshot.events) {
    const data = event.data as any
    if (!data) continue
    // Sum up token fields from various event types
    if (typeof data.prompt_tokens === "number") total += data.prompt_tokens
    if (typeof data.completion_tokens === "number") total += data.completion_tokens
    if (typeof data.reasoning_tokens === "number") total += data.reasoning_tokens
    // Also check for total_tokens directly
    if (typeof data.total_tokens === "number") total += data.total_tokens
  }
  return total
}

function ensureApiBase(base: string): string {
  let out = base.replace(/\/+$/, "")
  if (!out.endsWith("/api")) {
    out = `${out}/api`
  }
  return out
}

function normalizeBackendId(value: string): BackendId {
  const lowered = value.toLowerCase()
  if (lowered === "prod" || lowered === "dev" || lowered === "local") {
    return lowered
  }
  return "prod"
}

function getBackendConfig(id: BackendId = currentBackend): {
  id: BackendId
  label: string
  baseUrl: string
  baseRoot: string
  apiKey: string
} {
  const config = backendConfigs[id]
  return {
    id,
    label: config.label,
    baseUrl: config.baseUrl,
    baseRoot: config.baseUrl.replace(/\/api$/, ""),
    apiKey: backendKeys[id],
  }
}

function getActiveApiKey(): string {
  return getBackendConfig().apiKey
}

function getActiveBaseUrl(): string {
  return getBackendConfig().baseUrl
}

function getActiveBaseRoot(): string {
  return getBackendConfig().baseRoot
}

function buildLayout(renderer: any) {
  const root = new BoxRenderable(renderer, {
    id: "root",
    width: "auto",
    height: "auto",
    flexGrow: 1,
    flexShrink: 1,
    flexDirection: "column",
    backgroundColor: "#0b1120",
    border: false,
  })
  renderer.root.add(root)

  const headerBox = new BoxRenderable(renderer, {
    id: "header-box",
    width: "auto",
    height: 3,
    backgroundColor: "#1e293b",
    borderStyle: "single",
    borderColor: "#334155",
    flexGrow: 0,
    flexShrink: 0,
    flexDirection: "row",
    border: true,
  })
  const headerText = new TextRenderable(renderer, {
    id: "header-text",
    content: "Synth AI Prompt Learning Monitor",
    fg: "#e2e8f0",
  })
  const headerSpacer = new BoxRenderable(renderer, {
    id: "header-spacer",
    width: "auto",
    height: "auto",
    flexGrow: 1,
    flexShrink: 1,
    border: false,
  })
  const headerMetaText = new TextRenderable(renderer, {
    id: "header-meta-text",
    content: "",
    fg: "#94a3b8",
  })
  headerBox.add(headerText)
  headerBox.add(headerSpacer)
  headerBox.add(headerMetaText)
  root.add(headerBox)

  const tabsBox = new BoxRenderable(renderer, {
    id: "tabs-box",
    width: "auto",
    height: 2,
    backgroundColor: "#111827",
    borderStyle: "single",
    borderColor: "#1f2937",
    flexDirection: "row",
    gap: 2,
    border: true,
  })
  const jobsTabText = new TextRenderable(renderer, {
    id: "tabs-jobs",
    content: "[Jobs] (b)",
    fg: "#f8fafc",
  })
  const eventsTabText = new TextRenderable(renderer, {
    id: "tabs-events",
    content: "[Events] (e)",
    fg: "#94a3b8",
  })
  tabsBox.add(jobsTabText)
  tabsBox.add(eventsTabText)
  root.add(tabsBox)

  const main = new BoxRenderable(renderer, {
    id: "main",
    width: "auto",
    height: "auto",
    flexDirection: "row",
    flexGrow: 1,
    flexShrink: 1,
    border: false,
  })
  root.add(main)

  const jobsBox = new BoxRenderable(renderer, {
    id: "jobs-box",
    width: 36,
    height: "auto",
    minWidth: 36,
    flexGrow: 0,
    flexShrink: 0,
    borderStyle: "single",
    borderColor: "#334155",
    title: "Jobs",
    titleAlignment: "left",
    border: true,
  })
  const jobsSelect = new SelectRenderable(renderer, {
    id: "jobs-select",
    width: "auto",
    height: "auto",
    options: [],
    backgroundColor: "#0f172a",
    focusedBackgroundColor: "#1e293b",
    textColor: "#e2e8f0",
    focusedTextColor: "#f8fafc",
    selectedBackgroundColor: "#2563eb",
    selectedTextColor: "#ffffff",
    descriptionColor: "#94a3b8",
    selectedDescriptionColor: "#e2e8f0",
    showScrollIndicator: true,
    wrapSelection: true,
    showDescription: true,
    flexGrow: 1,
    flexShrink: 1,
  })
  jobsBox.add(jobsSelect)
  main.add(jobsBox)

  const detailColumn = new BoxRenderable(renderer, {
    id: "detail-column",
    width: "auto",
    height: "auto",
    flexDirection: "column",
    flexGrow: 2,
    flexShrink: 1,
    border: false,
  })
  main.add(detailColumn)

  const detailBox = new BoxRenderable(renderer, {
    id: "detail-box",
    width: "auto",
    height: 12,
    borderStyle: "single",
    borderColor: "#334155",
    title: "Details",
    titleAlignment: "left",
    border: true,
  })
  const detailText = new TextRenderable(renderer, {
    id: "detail-text",
    content: "No job selected.",
    fg: "#e2e8f0",
  })
  detailBox.add(detailText)
  detailColumn.add(detailBox)

  const resultsBox = new BoxRenderable(renderer, {
    id: "results-box",
    width: "auto",
    height: 6,
    borderStyle: "single",
    borderColor: "#334155",
    title: "Results",
    titleAlignment: "left",
    backgroundColor: "#0b1220",
    border: true,
  })
  const resultsText = new TextRenderable(renderer, {
    id: "results-text",
    content: "Results: -",
    fg: "#e2e8f0",
  })
  resultsBox.add(resultsText)
  detailColumn.add(resultsBox)

  const metricsBox = new BoxRenderable(renderer, {
    id: "metrics-box",
    width: "auto",
    height: 5,
    borderStyle: "single",
    borderColor: "#334155",
    title: "Metrics",
    titleAlignment: "left",
    border: true,
  })
  const metricsText = new TextRenderable(renderer, {
    id: "metrics-text",
    content: "Metrics: -",
    fg: "#cbd5f5",
  })
  metricsBox.add(metricsText)
  detailColumn.add(metricsBox)

  const eventsBox = new BoxRenderable(renderer, {
    id: "events-box",
    width: "auto",
    height: "auto",
    flexGrow: 1,
    flexShrink: 1,
    borderStyle: "single",
    borderColor: "#334155",
    title: "Events",
    titleAlignment: "left",
    border: true,
  })
  const eventsList = new BoxRenderable(renderer, {
    id: "events-list",
    width: "auto",
    height: "auto",
    flexDirection: "column",
    flexGrow: 1,
    flexShrink: 1,
    gap: 1,
    border: false,
  })
  const eventsEmptyText = new TextRenderable(renderer, {
    id: "events-empty-text",
    content: "No events yet.",
    fg: "#e2e8f0",
  })
  eventsBox.add(eventsList)
  eventsBox.add(eventsEmptyText)
  detailColumn.add(eventsBox)

  const statusBox = new BoxRenderable(renderer, {
    id: "status-box",
    width: "auto",
    height: 3,
    backgroundColor: "#0f172a",
    borderStyle: "single",
    borderColor: "#334155",
    flexGrow: 0,
    flexShrink: 0,
    border: true,
  })
  const statusText = new TextRenderable(renderer, {
    id: "status-text",
    content: "Ready.",
    fg: "#e2e8f0",
  })
  statusBox.add(statusText)
  root.add(statusBox)

  const footerBox = new BoxRenderable(renderer, {
    id: "footer-box",
    width: "auto",
    height: 2,
    backgroundColor: "#111827",
    flexGrow: 0,
    flexShrink: 0,
  })
  const footerTextNode = new TextRenderable(renderer, {
    id: "footer-text",
    content: footerText(),
    fg: "#94a3b8",
  })
  footerBox.add(footerTextNode)
  root.add(footerBox)

  const modalBox = new BoxRenderable(renderer, {
    id: "modal-box",
    width: 50,
    height: 5,
    position: "absolute",
    left: 4,
    top: 4,
    backgroundColor: "#0f172a",
    borderStyle: "single",
    borderColor: "#94a3b8",
    border: true,
    zIndex: 5,
  })
  const modalLabel = new TextRenderable(renderer, {
    id: "modal-label",
    content: "Snapshot ID:",
    fg: "#e2e8f0",
    position: "absolute",
    left: 6,
    top: 5,
    zIndex: 6,
  })
  const modalInput = new InputRenderable(renderer, {
    id: "modal-input",
    width: 44,
    height: 1,
    position: "absolute",
    left: 6,
    top: 6,
    placeholder: "Enter snapshot id",
    backgroundColor: "#111827",
    focusedBackgroundColor: "#1f2937",
    textColor: "#e2e8f0",
    focusedTextColor: "#ffffff",
  })
  modalBox.visible = false
  modalLabel.visible = false
  modalInput.visible = false
  renderer.root.add(modalBox)
  renderer.root.add(modalLabel)
  renderer.root.add(modalInput)

  const filterBox = new BoxRenderable(renderer, {
    id: "filter-box",
    width: 52,
    height: 5,
    position: "absolute",
    left: 6,
    top: 6,
    backgroundColor: "#0f172a",
    borderStyle: "single",
    borderColor: "#60a5fa",
    border: true,
    zIndex: 5,
  })
  const filterLabel = new TextRenderable(renderer, {
    id: "filter-label",
    content: "Event filter:",
    fg: "#e2e8f0",
    position: "absolute",
    left: 8,
    top: 7,
    zIndex: 6,
  })
  const filterInput = new InputRenderable(renderer, {
    id: "filter-input",
    width: 46,
    height: 1,
    position: "absolute",
    left: 8,
    top: 8,
    placeholder: "Type to filter events",
    backgroundColor: "#111827",
    focusedBackgroundColor: "#1f2937",
    textColor: "#e2e8f0",
    focusedTextColor: "#ffffff",
  })
  filterBox.visible = false
  filterLabel.visible = false
  filterInput.visible = false
  renderer.root.add(filterBox)
  renderer.root.add(filterLabel)
  renderer.root.add(filterInput)

  const jobFilterBox = new BoxRenderable(renderer, {
    id: "job-filter-box",
    width: 52,
    height: 11,
    position: "absolute",
    left: 6,
    top: 6,
    backgroundColor: "#0f172a",
    borderStyle: "single",
    borderColor: "#60a5fa",
    border: true,
    zIndex: 5,
  })
  const jobFilterLabel = new TextRenderable(renderer, {
    id: "job-filter-label",
    content: "Job filter (status: all)",
    fg: "#e2e8f0",
    position: "absolute",
    left: 8,
    top: 7,
    zIndex: 6,
  })
  const jobFilterHelp = new TextRenderable(renderer, {
    id: "job-filter-help",
    content: "Enter/space toggle | a select all | x clear | q close",
    fg: "#94a3b8",
    position: "absolute",
    left: 8,
    top: 8,
    zIndex: 6,
  })
  const jobFilterListText = new TextRenderable(renderer, {
    id: "job-filter-list",
    content: "",
    fg: "#e2e8f0",
    position: "absolute",
    left: 8,
    top: 9,
    zIndex: 6,
  })
  jobFilterBox.visible = false
  jobFilterLabel.visible = false
  jobFilterHelp.visible = false
  jobFilterListText.visible = false
  renderer.root.add(jobFilterBox)
  renderer.root.add(jobFilterLabel)
  renderer.root.add(jobFilterHelp)
  renderer.root.add(jobFilterListText)

  const eventModalBox = new BoxRenderable(renderer, {
    id: "event-modal-box",
    width: 80,
    height: 16,
    position: "absolute",
    left: 6,
    top: 6,
    backgroundColor: "#0b1220",
    borderStyle: "single",
    borderColor: "#60a5fa",
    border: true,
    zIndex: 6,
  })
  const eventModalTitle = new TextRenderable(renderer, {
    id: "event-modal-title",
    content: "Event details",
    fg: "#e2e8f0",
    position: "absolute",
    left: 8,
    top: 7,
    zIndex: 7,
  })
  const eventModalText = new TextRenderable(renderer, {
    id: "event-modal-text",
    content: "",
    fg: "#e2e8f0",
    position: "absolute",
    left: 8,
    top: 8,
    zIndex: 7,
  })
  const eventModalHint = new TextRenderable(renderer, {
    id: "event-modal-hint",
    content: "Event details",
    fg: "#94a3b8",
    position: "absolute",
    left: 8,
    top: 9,
    zIndex: 7,
  })
  eventModalBox.visible = false
  eventModalTitle.visible = false
  eventModalText.visible = false
  eventModalHint.visible = false
  renderer.root.add(eventModalBox)
  renderer.root.add(eventModalTitle)
  renderer.root.add(eventModalText)
  renderer.root.add(eventModalHint)

  const resultsModalBox = new BoxRenderable(renderer, {
    id: "results-modal-box",
    width: 100,
    height: 24,
    position: "absolute",
    left: 6,
    top: 4,
    backgroundColor: "#0b1220",
    borderStyle: "single",
    borderColor: "#22c55e",
    border: true,
    zIndex: 8,
  })
  const resultsModalTitle = new TextRenderable(renderer, {
    id: "results-modal-title",
    content: "Results - Best Prompt",
    fg: "#22c55e",
    position: "absolute",
    left: 8,
    top: 5,
    zIndex: 9,
  })
  const resultsModalText = new TextRenderable(renderer, {
    id: "results-modal-text",
    content: "",
    fg: "#e2e8f0",
    position: "absolute",
    left: 8,
    top: 6,
    zIndex: 9,
  })
  const resultsModalHint = new TextRenderable(renderer, {
    id: "results-modal-hint",
    content: "Results | j/k scroll | esc/q/enter close",
    fg: "#94a3b8",
    position: "absolute",
    left: 8,
    top: 26,
    zIndex: 9,
  })
  resultsModalBox.visible = false
  resultsModalTitle.visible = false
  resultsModalText.visible = false
  resultsModalHint.visible = false
  renderer.root.add(resultsModalBox)
  renderer.root.add(resultsModalTitle)
  renderer.root.add(resultsModalText)
  renderer.root.add(resultsModalHint)

  const configModalBox = new BoxRenderable(renderer, {
    id: "config-modal-box",
    width: 100,
    height: 24,
    position: "absolute",
    left: 6,
    top: 4,
    backgroundColor: "#0b1220",
    borderStyle: "single",
    borderColor: "#f59e0b",
    border: true,
    zIndex: 8,
  })
  const configModalTitle = new TextRenderable(renderer, {
    id: "config-modal-title",
    content: "Job Configuration",
    fg: "#f59e0b",
    position: "absolute",
    left: 8,
    top: 5,
    zIndex: 9,
  })
  const configModalText = new TextRenderable(renderer, {
    id: "config-modal-text",
    content: "",
    fg: "#e2e8f0",
    position: "absolute",
    left: 8,
    top: 6,
    zIndex: 9,
  })
  const configModalHint = new TextRenderable(renderer, {
    id: "config-modal-hint",
    content: "Config | j/k scroll | esc/q/enter close",
    fg: "#94a3b8",
    position: "absolute",
    left: 8,
    top: 26,
    zIndex: 9,
  })
  configModalBox.visible = false
  configModalTitle.visible = false
  configModalText.visible = false
  configModalHint.visible = false
  renderer.root.add(configModalBox)
  renderer.root.add(configModalTitle)
  renderer.root.add(configModalText)
  renderer.root.add(configModalHint)

  // Prompt Browser Modal - for viewing baseline and all candidate prompts
  const promptBrowserBox = new BoxRenderable(renderer, {
    id: "prompt-browser-box",
    width: 100,
    height: 24,
    position: "absolute",
    left: 6,
    top: 4,
    backgroundColor: "#0b1220",
    borderStyle: "single",
    borderColor: "#a855f7",
    border: true,
    zIndex: 10,
  })
  const promptBrowserTitle = new TextRenderable(renderer, {
    id: "prompt-browser-title",
    content: "Prompt Browser",
    fg: "#a855f7",
    position: "absolute",
    left: 8,
    top: 5,
    zIndex: 11,
  })
  const promptBrowserText = new TextRenderable(renderer, {
    id: "prompt-browser-text",
    content: "",
    fg: "#e2e8f0",
    position: "absolute",
    left: 8,
    top: 6,
    zIndex: 11,
  })
  const promptBrowserHint = new TextRenderable(renderer, {
    id: "prompt-browser-hint",
    content: "Prompts | h/l prev/next | j/k scroll | y copy | esc close",
    fg: "#94a3b8",
    position: "absolute",
    left: 8,
    top: 26,
    zIndex: 11,
  })
  promptBrowserBox.visible = false
  promptBrowserTitle.visible = false
  promptBrowserText.visible = false
  promptBrowserHint.visible = false
  renderer.root.add(promptBrowserBox)
  renderer.root.add(promptBrowserTitle)
  renderer.root.add(promptBrowserText)
  renderer.root.add(promptBrowserHint)

  const settingsBox = new BoxRenderable(renderer, {
    id: "settings-modal-box",
    width: 64,
    height: 14,
    position: "absolute",
    left: 6,
    top: 6,
    backgroundColor: "#0b1220",
    borderStyle: "single",
    borderColor: "#38bdf8",
    border: true,
    zIndex: 8,
  })
  const settingsTitle = new TextRenderable(renderer, {
    id: "settings-modal-title",
    content: "Settings - Backend",
    fg: "#38bdf8",
    position: "absolute",
    left: 8,
    top: 7,
    zIndex: 9,
  })
  const settingsHelp = new TextRenderable(renderer, {
    id: "settings-modal-help",
    content: "Enter apply | j/k navigate | a pick key | m manual | q close",
    fg: "#94a3b8",
    position: "absolute",
    left: 8,
    top: 8,
    zIndex: 9,
  })
  const settingsListText = new TextRenderable(renderer, {
    id: "settings-modal-list",
    content: "",
    fg: "#e2e8f0",
    position: "absolute",
    left: 8,
    top: 9,
    zIndex: 9,
  })
  const settingsInfoText = new TextRenderable(renderer, {
    id: "settings-modal-info",
    content: "",
    fg: "#94a3b8",
    position: "absolute",
    left: 8,
    top: 12,
    zIndex: 9,
  })
  settingsBox.visible = false
  settingsTitle.visible = false
  settingsHelp.visible = false
  settingsListText.visible = false
  settingsInfoText.visible = false
  renderer.root.add(settingsBox)
  renderer.root.add(settingsTitle)
  renderer.root.add(settingsHelp)
  renderer.root.add(settingsListText)
  renderer.root.add(settingsInfoText)

  const keyModalBox = new BoxRenderable(renderer, {
    id: "key-modal-box",
    width: 70,
    height: 7,
    position: "absolute",
    left: 8,
    top: 8,
    backgroundColor: "#0b1220",
    borderStyle: "single",
    borderColor: "#7dd3fc",
    border: true,
    zIndex: 10,
  })
  const keyModalLabel = new TextRenderable(renderer, {
    id: "key-modal-label",
    content: "Set API key (saved for this session only)",
    fg: "#7dd3fc",
    position: "absolute",
    left: 10,
    top: 9,
    zIndex: 11,
  })
  const keyModalInput = new InputRenderable(renderer, {
    id: "key-modal-input",
    width: 62,
    height: 1,
    position: "absolute",
    left: 10,
    top: 10,
    backgroundColor: "#0f172a",
    borderStyle: "single",
    borderColor: "#1d4ed8",
    border: true,
    fg: "#e2e8f0",
    zIndex: 11,
  })
  const keyModalHelp = new TextRenderable(renderer, {
    id: "key-modal-help",
    content: "Paste any way | enter save | q close | empty clears",
    fg: "#94a3b8",
    position: "absolute",
    left: 10,
    top: 12,
    zIndex: 11,
  })
  keyModalBox.visible = false
  keyModalLabel.visible = false
  keyModalInput.visible = false
  keyModalHelp.visible = false
  renderer.root.add(keyModalBox)
  renderer.root.add(keyModalLabel)
  renderer.root.add(keyModalInput)
  renderer.root.add(keyModalHelp)

  const envKeyModalBox = new BoxRenderable(renderer, {
    id: "env-key-modal-box",
    width: 78,
    height: 14,
    position: "absolute",
    left: 8,
    top: 6,
    backgroundColor: "#0b1220",
    borderStyle: "single",
    borderColor: "#7dd3fc",
    border: true,
    zIndex: 11,
  })
  const envKeyModalTitle = new TextRenderable(renderer, {
    id: "env-key-modal-title",
    content: "Settings - API Key",
    fg: "#7dd3fc",
    position: "absolute",
    left: 10,
    top: 7,
    zIndex: 12,
  })
  const envKeyModalHelp = new TextRenderable(renderer, {
    id: "env-key-modal-help",
    content: "Enter apply | j/k navigate | r rescan | m manual | q close",
    fg: "#94a3b8",
    position: "absolute",
    left: 10,
    top: 8,
    zIndex: 12,
  })
  const envKeyModalListText = new TextRenderable(renderer, {
    id: "env-key-modal-list",
    content: "",
    fg: "#e2e8f0",
    position: "absolute",
    left: 10,
    top: 9,
    zIndex: 12,
  })
  const envKeyModalInfoText = new TextRenderable(renderer, {
    id: "env-key-modal-info",
    content: "",
    fg: "#94a3b8",
    position: "absolute",
    left: 10,
    top: 13,
    zIndex: 12,
  })
  envKeyModalBox.visible = false
  envKeyModalTitle.visible = false
  envKeyModalHelp.visible = false
  envKeyModalListText.visible = false
  envKeyModalInfoText.visible = false
  renderer.root.add(envKeyModalBox)
  renderer.root.add(envKeyModalTitle)
  renderer.root.add(envKeyModalHelp)
  renderer.root.add(envKeyModalListText)
  renderer.root.add(envKeyModalInfoText)

  // Login modal
  const loginModalBox = new BoxRenderable(renderer, {
    id: "login-modal-box",
    width: 60,
    height: 10,
    position: "absolute",
    left: 10,
    top: 6,
    backgroundColor: "#0b1220",
    borderStyle: "single",
    borderColor: "#22c55e",
    border: true,
    zIndex: 15,
  })
  const loginModalTitle = new TextRenderable(renderer, {
    id: "login-modal-title",
    content: "Sign In",
    fg: "#22c55e",
    position: "absolute",
    left: 12,
    top: 7,
    zIndex: 16,
  })
  const loginModalText = new TextRenderable(renderer, {
    id: "login-modal-text",
    content: "Press Enter to open browser and sign in...",
    fg: "#e2e8f0",
    position: "absolute",
    left: 12,
    top: 9,
    zIndex: 16,
  })
  const loginModalHelp = new TextRenderable(renderer, {
    id: "login-modal-help",
    content: "Enter start | q cancel",
    fg: "#94a3b8",
    position: "absolute",
    left: 12,
    top: 13,
    zIndex: 16,
  })
  loginModalBox.visible = false
  loginModalTitle.visible = false
  loginModalText.visible = false
  loginModalHelp.visible = false
  renderer.root.add(loginModalBox)
  renderer.root.add(loginModalTitle)
  renderer.root.add(loginModalText)
  renderer.root.add(loginModalHelp)

  return {
    jobsBox,
    eventsBox,
    jobsSelect,
    detailText,
    resultsText,
    metricsText,
    eventsList,
    eventsEmptyText,
    jobsTabText,
    eventsTabText,
    headerMetaText,
    statusText,
    footerText: footerTextNode,
    modalBox,
    modalLabel,
    modalInput,
    modalVisible: false,
    filterBox,
    filterLabel,
    filterInput,
    filterModalVisible: false,
    jobFilterBox,
    jobFilterLabel,
    jobFilterHelp,
    jobFilterListText,
    jobFilterModalVisible: false,
    eventModalBox,
    eventModalTitle,
    eventModalText,
    eventModalHint,
    eventModalVisible: false,
    eventModalPayload: "",
    resultsModalBox,
    resultsModalTitle,
    resultsModalText,
    resultsModalHint,
    resultsModalVisible: false,
    resultsModalPayload: "",
    configModalBox,
    configModalTitle,
    configModalText,
    configModalHint,
    configModalVisible: false,
    configModalPayload: "",
    promptBrowserBox,
    promptBrowserTitle,
    promptBrowserText,
    promptBrowserHint,
    promptBrowserVisible: false,
    settingsBox,
    settingsTitle,
    settingsHelp,
    settingsListText,
    settingsInfoText,
    settingsModalVisible: false,
    keyModalBox,
    keyModalLabel,
    keyModalInput,
    keyModalHelp,
    keyModalVisible: false,
    envKeyModalBox,
    envKeyModalTitle,
    envKeyModalHelp,
    envKeyModalListText,
    envKeyModalInfoText,
    envKeyModalVisible: false,
    loginModalBox,
    loginModalTitle,
    loginModalText,
    loginModalHelp,
    loginModalVisible: false,
    eventCards: [] as Array<{ box: BoxRenderable; text: TextRenderable }>,
  }
}
