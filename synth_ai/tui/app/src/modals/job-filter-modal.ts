/**
 * Job status filter modal controller.
 */
import type { AppContext } from "../context"
import { buildJobStatusOptions, getFilteredJobs } from "../selectors/jobs"
import { createModalUI, clamp, type ModalController } from "./base"
import { focusManager } from "../focus"

export function createJobFilterModal(ctx: AppContext): ModalController & {
  open: () => void
  move: (delta: number) => void
  toggleSelected: () => void
  clearAll: () => void
} {
  const { renderer } = ctx
  const { appState, snapshot, config } = ctx.state

  const modal = createModalUI(renderer, {
    id: "job-filter-modal",
    width: 52,
    height: 11,
    borderColor: "#60a5fa",
    titleColor: "#60a5fa",
    zIndex: 5,
  })

  function toggle(visible: boolean): void {
    if (visible) {
      focusManager.push({
        id: "job-filter-modal",
        handleKey,
      })
      modal.setTitle("Job filter (status)")
      modal.center()
      refreshOptions()
      appState.jobFilterCursor = 0
      appState.jobFilterWindowStart = 0
      renderList()
    } else {
      focusManager.pop("job-filter-modal")
    }
    modal.setVisible(visible)
  }

  function refreshOptions(): void {
    appState.jobFilterOptions = buildJobStatusOptions(snapshot.jobs)
    const maxIndex = Math.max(0, appState.jobFilterOptions.length - 1)
    appState.jobFilterCursor = clamp(appState.jobFilterCursor, 0, maxIndex)
    appState.jobFilterWindowStart = clamp(appState.jobFilterWindowStart, 0, Math.max(0, maxIndex))
  }

  function renderList(): void {
    const max = Math.max(0, appState.jobFilterOptions.length - 1)
    appState.jobFilterCursor = clamp(appState.jobFilterCursor, 0, max)
    const start = clamp(appState.jobFilterWindowStart, 0, Math.max(0, max))
    const end = Math.min(appState.jobFilterOptions.length, start + config.jobFilterVisibleCount)

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
    modal.setContent(lines.join("\n"))
    modal.setHint("j/k move | space select | c clear | q close")
  }

  function move(delta: number): void {
    const max = Math.max(0, appState.jobFilterOptions.length - 1)
    appState.jobFilterCursor = clamp(appState.jobFilterCursor + delta, 0, max)
    if (appState.jobFilterCursor < appState.jobFilterWindowStart) {
      appState.jobFilterWindowStart = appState.jobFilterCursor
    } else if (appState.jobFilterCursor >= appState.jobFilterWindowStart + config.jobFilterVisibleCount) {
      appState.jobFilterWindowStart = appState.jobFilterCursor - config.jobFilterVisibleCount + 1
    }
    renderList()
  }

  function toggleSelected(): void {
    const option = appState.jobFilterOptions[appState.jobFilterCursor]
    if (!option) return
    if (appState.jobStatusFilter.has(option.status)) {
      appState.jobStatusFilter.delete(option.status)
    } else {
      appState.jobStatusFilter.add(option.status)
    }
    renderList()
    applySelection()
  }

  function clearAll(): void {
    appState.jobStatusFilter.clear()
    renderList()
    applySelection()
  }

  function applySelection(): void {
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
      ctx.render()
      return
    }
    if (!snapshot.selectedJob || !filteredJobs.some((job) => job.job_id === snapshot.selectedJob?.job_id)) {
      import("../api/jobs").then(({ selectJob }) => {
        void selectJob(ctx, filteredJobs[0].job_id).then(() => ctx.render()).catch(() => {})
      })
      return
    }
    ctx.render()
  }

  function open(): void {
    toggle(true)
  }

  function handleKey(key: any): boolean {
    if (!modal.visible) return false

    if (key.name === "up" || key.name === "k") {
      move(-1)
      return true
    }
    if (key.name === "down" || key.name === "j") {
      move(1)
      return true
    }
    if (key.name === "space" || key.name === "return" || key.name === "enter") {
      toggleSelected()
      return true
    }
    if (key.name === "c") {
      clearAll()
      return true
    }
    if (key.name === "q" || key.name === "escape") {
      toggle(false)
      return true
    }
    return true // consume all keys when modal is open
  }

  const controller = {
    get isVisible() {
      return modal.visible
    },
    toggle,
    open,
    move,
    toggleSelected,
    clearAll,
    handleKey,
  }

  return controller
}
