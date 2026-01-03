import {
  createCliRenderer,
  BoxRenderable,
  TextRenderable,
  SelectRenderable,
  InputRenderable,
  SelectRenderableEvents,
  InputRenderableEvents,
} from "@opentui/core"

type JobSummary = {
  job_id: string
  status: string
  training_type?: string | null
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
  best_score?: number | null
  best_snapshot_id?: string | null
  total_tokens?: number | null
  total_cost_usd?: number | null
  error?: string | null
}

type JobEvent = {
  seq: number
  type: string
  message?: string | null
  data?: unknown
  timestamp?: string | null
  expanded?: boolean
}

type Snapshot = {
  jobs: JobSummary[]
  selectedJob: JobSummary | null
  events: JobEvent[]
  metrics: Record<string, unknown>
  bestSnapshotId: string | null
  bestSnapshot: Record<string, any> | null
  artifacts: Array<Record<string, unknown>>
  orgId: string | null
  userId: string | null
  balanceDollars: number | null
  status: string
  lastError: string | null
  lastRefresh: number | null
}

const baseUrl = ensureApiBase(
  process.env.SYNTH_TUI_API_BASE || "https://api.usesynth.ai/api",
)
const baseRoot = baseUrl.replace(/\/api$/, "")
const apiKey = process.env.SYNTH_API_KEY || ""
const initialJobId = process.env.SYNTH_TUI_JOB_ID || ""
const refreshInterval = parseFloat(process.env.SYNTH_TUI_REFRESH_INTERVAL || "5")
const eventInterval = parseFloat(process.env.SYNTH_TUI_EVENT_INTERVAL || "2")
const maxRefreshInterval = parseFloat(process.env.SYNTH_TUI_REFRESH_MAX || "60")
const maxEventInterval = parseFloat(process.env.SYNTH_TUI_EVENT_MAX || "15")
const eventHistoryLimit = parseInt(process.env.SYNTH_TUI_EVENT_CARDS || "200", 10)
const eventCollapseLimit = parseInt(process.env.SYNTH_TUI_EVENT_COLLAPSE || "160", 10)
const eventVisibleCount = parseInt(process.env.SYNTH_TUI_EVENT_VISIBLE || "6", 10)
const jobLimit = parseInt(process.env.SYNTH_TUI_LIMIT || "50", 10)

const snapshot: Snapshot = {
  jobs: [],
  selectedJob: null,
  events: [],
  metrics: {},
  bestSnapshotId: null,
  bestSnapshot: null,
  artifacts: [],
  orgId: null,
  userId: null,
  balanceDollars: null,
  status: "Loading jobs...",
  lastError: null,
  lastRefresh: null,
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
let jobSelectToken = 0
let eventsToken = 0
let eventFilter = ""
let jobStatusFilter = new Set<string>()
let jobFilterOptions: Array<{ status: string; count: number }> = []
let jobFilterCursor = 0
let jobFilterWindowStart = 0
const jobFilterVisibleCount = 6

const renderer = await createCliRenderer({
  useConsole: false,
  useAlternateScreen: true,
  openConsoleOnError: false,
  backgroundColor: "#0b1120",
})
const ui = buildLayout(renderer)
renderer.start()

renderer.keyInput.on("keypress", (key: any) => {
  if (key.ctrl && key.name === "c") {
    renderer.stop()
    renderer.destroy()
    process.exit(0)
  }
  if (key.name === "q" || key.name === "escape") {
    if (ui.jobFilterModalVisible) {
      toggleJobFilterModal(false)
      return
    }
    if (ui.eventModalVisible) {
      toggleEventModal(false)
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
  if (key.shift && key.name === "j") toggleJobFilterModal(true)
  if (key.name === "c") cancelSelected()
  if (key.name === "a") fetchArtifacts()
  if (key.name === "s") {
    if (snapshot.selectedJob) toggleModal(true)
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

if (!apiKey) {
  snapshot.lastError = "Missing SYNTH_API_KEY"
  snapshot.status = "Auth required"
  renderSnapshot()
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
    const balance = await apiGetV1("/balance/current")
    const dollars = balance?.balance_dollars
    snapshot.balanceDollars =
      typeof dollars === "number" && Number.isFinite(dollars) ? dollars : null
  } catch (err: any) {
    snapshot.balanceDollars = snapshot.balanceDollars || null
  }
  renderSnapshot()
}

async function refreshJobs(): Promise<boolean> {
  try {
    snapshot.status = "Refreshing jobs..."
    const payload = await apiGet(`/prompt-learning/online/jobs?limit=${jobLimit}&offset=0`)
    const jobs = extractJobs(payload)
    snapshot.jobs = jobs
    snapshot.lastRefresh = Date.now()
    snapshot.lastError = null
    if (!snapshot.selectedJob && jobs.length > 0 && !autoSelected) {
      autoSelected = true
      await selectJob(jobs[0].job_id)
      return
    }
    if (snapshot.selectedJob) {
      const match = jobs.find((j) => j.job_id === snapshot.selectedJob?.job_id)
      if (match) snapshot.selectedJob = match
    }
    if (jobs.length === 0) {
      snapshot.status = "No prompt-learning jobs found"
    } else {
      snapshot.status = `Loaded ${jobs.length} prompt-learning job(s)`
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
    } as JobSummary)
  snapshot.status = `Loading job ${jobId}...`
  renderSnapshot()
  try {
    const job = await apiGet(
      `/prompt-learning/online/jobs/${jobId}?include_events=false&include_snapshot=false`,
    )
    if (token !== jobSelectToken || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.selectedJob = coerceJob(job)
    snapshot.bestSnapshotId = extractBestSnapshotId(job)
    snapshot.status = `Selected job ${jobId}`
  } catch (err: any) {
    if (token !== jobSelectToken || snapshot.selectedJob?.job_id !== jobId) {
      return
    }
    snapshot.lastError = err?.message || `Failed to load job ${jobId}`
  }
  await fetchBestSnapshot(token)
  renderSnapshot()
}

async function fetchMetrics(): Promise<void> {
  const job = snapshot.selectedJob
  if (!job) return
  const jobId = job.job_id
  try {
    snapshot.status = "Loading metrics..."
    const payload = await apiGet(`/prompt-learning/online/jobs/${job.job_id}/metrics`)
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

async function fetchBestSnapshot(token?: number): Promise<void> {
  const job = snapshot.selectedJob
  if (!job) return
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

async function refreshEvents(): Promise<boolean> {
  const job = snapshot.selectedJob
  if (!job) return true
  const jobId = job.job_id
  const token = eventsToken
  try {
    const payload = await apiGet(
      `/prompt-learning/online/jobs/${job.job_id}/events?since_seq=${lastSeq}&limit=200`,
    )
    if (token !== eventsToken || snapshot.selectedJob?.job_id !== jobId) {
      return true
    }
    const { events, nextSeq } = extractEvents(payload)
    if (events.length > 0) {
      snapshot.events.push(...events)
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
        const desc = `${job.status} | ${job.training_type || "prompt"} | best=${score}`
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
  renderer.requestRender()
}

function formatDetails(): string {
  const job = snapshot.selectedJob
  if (!job) return "No job selected."
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
  const lines = [
    `Job: ${job.job_id}`,
    `Status: ${job.status}`,
    `Type: ${job.training_type || "-"}`,
    `Created: ${formatTimestamp(job.created_at)}`,
    `Started: ${formatTimestamp(job.started_at)}`,
    `Finished: ${formatTimestamp(job.finished_at)}`,
    `Last event: ${lastEventTs}`,
    `Best score: ${job.best_score ?? "-"}`,
    `Tokens: ${job.total_tokens ?? "-"}`,
    `Cost: ${job.total_cost_usd ?? "-"}`,
  ]
  if (job.error) lines.push(`Error: ${job.error}`)
  if (snapshot.artifacts.length) lines.push(`Artifacts: ${snapshot.artifacts.length}`)
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

function formatHeaderMeta(): string {
  const org = snapshot.orgId || "-"
  const user = snapshot.userId || "-"
  const balance =
    snapshot.balanceDollars == null ? "-" : `$${snapshot.balanceDollars.toFixed(2)}`
  return `org: ${org}  user: ${user}  balance: ${balance}`
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
    const parsed = Date.parse(trimmed)
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
    ui.eventsEmptyText.content = eventFilter ? "No events match filter." : "No events yet."
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
  if (!filter) return snapshot.events
  return snapshot.events.filter((event) => {
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
  })
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
  const baseLabel = baseRoot.replace(/^https?:\/\//, "")
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
  return `Keys: e events | b jobs | tab toggle | j/k or arrows navigate | enter view event | r refresh | m metrics | p best | f ${filterLabel} | shift+j ${jobFilterLabel} | c cancel | a artifacts | s snapshot | q quit`
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

function openSelectedEventModal(): void {
  const recent = getFilteredEvents().slice(-eventHistoryLimit)
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
  const recent = getFilteredEvents().slice(-eventHistoryLimit)
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
    const res = await fetch(`${baseRoot}/health`)
    healthStatus = res.ok ? "ok" : `bad(${res.status})`
  } catch (err: any) {
    healthStatus = `err(${err?.message || "unknown"})`
  }
}

function extractJobs(payload: any): JobSummary[] {
  const list = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.jobs)
      ? payload.jobs
      : Array.isArray(payload?.data)
        ? payload.data
        : []
  return list.map(coerceJob)
}

function extractEvents(payload: any): { events: JobEvent[]; nextSeq: number | null } {
  const list = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.events)
      ? payload.events
      : []
  const events = list.map((e: any, idx: number) => ({
    seq: Number(e.seq ?? e.sequence ?? e.id ?? idx),
    type: String(e.type || e.event_type || "event"),
    message: e.message || null,
    data: e.data ?? e.payload ?? null,
    timestamp: e.timestamp || e.created_at || null,
  }))
  const nextSeq = typeof payload?.next_seq === "number" ? payload.next_seq : null
  return { events, nextSeq }
}

function coerceJob(payload: any): JobSummary {
  return {
    job_id: String(payload?.job_id || payload?.id || ""),
    status: String(payload?.status || "unknown"),
    training_type: payload?.training_type || null,
    created_at: payload?.created_at || null,
    started_at: payload?.started_at || null,
    finished_at: payload?.finished_at || null,
    best_score: num(payload?.best_score),
    best_snapshot_id:
      payload?.best_snapshot_id || payload?.prompt_best_snapshot_id || payload?.best_snapshot?.id || null,
    total_tokens: int(payload?.total_tokens),
    total_cost_usd: num(payload?.total_cost_usd || payload?.total_cost),
    error: payload?.error || null,
  }
}

function num(value: any): number | null {
  if (value == null) return null
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function int(value: any): number | null {
  if (value == null) return null
  const n = parseInt(String(value), 10)
  return Number.isFinite(n) ? n : null
}

function ensureApiBase(base: string): string {
  let out = base.replace(/\/+$/, "")
  if (!out.endsWith("/api")) {
    out = `${out}/api`
  }
  return out
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
    height: 9,
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
    eventCards: [] as Array<{ box: BoxRenderable; text: TextRenderable }>,
  }
}
