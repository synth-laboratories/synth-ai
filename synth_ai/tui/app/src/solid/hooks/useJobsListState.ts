import { batch, type Accessor, createEffect, createMemo, createSignal, onCleanup } from "solid-js"

import type { AppData } from "../../types"
import type { AppState } from "../../state/app-state"
import type { ActivePane } from "../../types"
import type { JobSummary } from "../../tui_data"
import { ListPane } from "../../types"
import { buildJobTypeOptions, getFilteredJobsByType, getJobTypeLabel } from "../../selectors/jobs"
import { formatTimestamp } from "../formatters/time"
import { deriveSelectedIndex, moveSelectionById } from "../utils/list"
import { formatListFilterTitle, getListFilterCount } from "../utils/listFilter"
import { type ListWindowState, useListWindow } from "./useListWindow"
import { formatActionKeys } from "../../input/keymap"

export type JobsListRow = {
  id: string
  type: string
  status: string
  date: string
}

export type JobsListState = {
  selectedJobId: Accessor<string | null>
  selectedIndex: Accessor<number>
  filteredJobs: Accessor<JobSummary[]>
  listWindow: ListWindowState<JobsListRow>
  title: Accessor<string>
  totalCount: Accessor<number>
  moveSelection: (delta: number) => boolean
  selectCurrent: () => void
}

type UseJobsListStateOptions = {
  data: AppData
  ui: AppState
  activePane: Accessor<ActivePane>
  height: Accessor<number>
  onSelectJob?: (jobId: string) => void
  onSelectionIntent?: () => void
}

function getRelevantDate(job: JobSummary): string {
  const dateStr = job.finished_at || job.started_at || job.created_at
  return formatTimestamp(dateStr)
}

function getJobTimestamp(job: JobSummary): number {
  const value = job.created_at || job.started_at || job.finished_at
  if (!value) return 0
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function getStatusLabel(status: string): string {
  const normalized = (status || "unknown").toLowerCase()
  switch (normalized) {
    case "running":
      return "Running"
    case "completed":
    case "succeeded":
      return "Completed"
    case "failed":
    case "error":
      return "Error"
    case "queued":
      return "Queued"
    case "canceled":
    case "cancelled":
      return "Canceled"
    default:
      return status || "-"
  }
}

function formatJobRow(job: JobSummary): JobsListRow {
  return {
    id: job.job_id,
    type: getJobTypeLabel(job),
    status: getStatusLabel(job.status),
    date: getRelevantDate(job),
  }
}

const SELECTION_DEBOUNCE_MS = 180

export function useJobsListState(options: UseJobsListStateOptions): JobsListState {
  const [selectedJobId, setSelectedJobId] = createSignal<string | null>(null)
  const [pendingSelectId, setPendingSelectId] = createSignal<string | null>(null)
  const jobs = createMemo(() => options.data.jobs)
  const filteredJobs = createMemo(() => {
    const list = jobs()
    const filters = options.ui.listFilterSelections[ListPane.Jobs]
    return getFilteredJobsByType(list, filters)
  })
  const selectedIndex = createMemo(() =>
    deriveSelectedIndex(filteredJobs(), selectedJobId(), (job) => job.job_id),
  )
  const listItems = createMemo(() => filteredJobs().map(formatJobRow))
  const totalFilterOptions = createMemo(() => buildJobTypeOptions(jobs()).length)
  const listWindow = useListWindow({
    items: listItems,
    selectedIndex,
    height: options.height,
    rowHeight: 2,
    chromeHeight: 2,
  })
  const cacheRemaining = createMemo(() => {
    if (!options.data.jobsCache.length) return 0
    const seen = new Set(options.data.jobs.map((job) => job.job_id))
    const oldest = options.data.jobs[options.data.jobs.length - 1]
    const oldestTs = oldest ? getJobTimestamp(oldest) : 0
    let count = 0
    for (const job of options.data.jobsCache) {
      if (!job.job_id || seen.has(job.job_id)) continue
      if (oldestTs && getJobTimestamp(job) > oldestTs) continue
      count += 1
    }
    return count
  })
  const loadMoreHint = createMemo(() => {
    if (options.ui.jobsListLoadingMore) return "Loading more..."
    if (options.ui.jobsListHasMore) {
      return `More: ${formatActionKeys("jobs.loadMore", { primaryOnly: true })}`
    }
    if (cacheRemaining() > 0) {
      return `More (cached): ${formatActionKeys("jobs.loadMore", { primaryOnly: true })}`
    }
    return ""
  })
  const title = createMemo(() => {
    const count = getListFilterCount(options.ui, ListPane.Jobs)
    const base = formatListFilterTitle("Jobs", count, totalFilterOptions())
    const hint = loadMoreHint()
    return hint ? `${base} | ${hint}` : base
  })
  const totalCount = createMemo(() => filteredJobs().length)

  createEffect(() => {
    const list = filteredJobs()
    const selected = selectedJobId()
    if (!list.length) {
      if (selected !== null || pendingSelectId() !== null) {
        batch(() => {
          setSelectedJobId(null)
          setPendingSelectId(null)
        })
      }
      return
    }
    if (!selected || !list.some((job) => job.job_id === selected)) {
      const nextId = list[0].job_id
      batch(() => {
        setPendingSelectId(nextId)
        setSelectedJobId(nextId)
      })
    }
  })

  createEffect(() => {
    const jobId = selectedJobId()
    if (!jobId || options.activePane() !== ListPane.Jobs) return
    const current = options.data.selectedJob?.job_id ?? null
    if (current === jobId) {
      setPendingSelectId(null)
      return
    }
    setPendingSelectId(jobId)
    const timer = setTimeout(() => {
      if (pendingSelectId() === jobId) {
        options.onSelectJob?.(jobId)
      }
    }, SELECTION_DEBOUNCE_MS)
    onCleanup(() => {
      clearTimeout(timer)
    })
  })

  const moveSelection = (delta: number): boolean => {
    const nextId = moveSelectionById(
      filteredJobs(),
      selectedJobId(),
      delta,
      (job) => job.job_id,
    )
    if (!nextId || nextId === selectedJobId()) return false
    options.onSelectionIntent?.()
    batch(() => {
      setPendingSelectId(nextId)
      setSelectedJobId(nextId)
    })
    return true
  }

  const selectCurrent = () => {
    const list = filteredJobs()
    const job = list[selectedIndex()]
    if (job?.job_id) {
      options.onSelectionIntent?.()
      batch(() => {
        setPendingSelectId(job.job_id)
        setSelectedJobId(job.job_id)
      })
    }
  }

  return {
    selectedJobId,
    selectedIndex,
    filteredJobs,
    listWindow,
    title,
    totalCount,
    moveSelection,
    selectCurrent,
  }
}
