import { type Accessor, createEffect, createMemo, createSignal, onCleanup } from "solid-js"

import type { AppData } from "../../types"
import type { AppState } from "../../state/app-state"
import type { PrimaryView } from "../../types"
import type { JobSummary } from "../../tui_data"
import { ListPane } from "../../types"
import { getFilteredJobsByType, getJobTypeLabel } from "../../selectors/jobs"
import { formatTimestamp } from "../formatters/time"
import { deriveSelectedIndex, moveSelectionById, uniqueById } from "../utils/list"
import { formatListTitle, getListFilterCount } from "../../utils/listTitle"
import { log } from "../../utils/log"
import { watchArrayMutations } from "../../utils/array-mutation"
import { getJobsList } from "../../state/jobs-index"
import { type ListWindowState, useListWindow } from "./useListWindow"

export type JobsListRow = {
  id: string
  type: string
  status: string
  date: string
}

export type JobsListState = {
  selectedJobId: Accessor<string | null>
  selectedIndex: Accessor<number>
  listWindow: ListWindowState<JobsListRow>
  title: Accessor<string>
  totalCount: Accessor<number>
  loadMoreHint: Accessor<string>
  moveSelection: (delta: number) => boolean
  selectCurrent: () => void
}

type UseJobsListStateOptions = {
  data: AppData
  ui: AppState
  primaryView: Accessor<PrimaryView>
  height: Accessor<number>
  onSelectJob?: (jobId: string) => void
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

type DuplicateIdInfo = {
  id: string
  count: number
  indices: number[]
}

function findDuplicateIds(ids: string[]): DuplicateIdInfo[] {
  const counts = new Map<string, DuplicateIdInfo>()
  ids.forEach((id, idx) => {
    if (!id) return
    const entry = counts.get(id)
    if (!entry) {
      counts.set(id, { id, count: 1, indices: [idx] })
    } else {
      entry.count += 1
      entry.indices.push(idx)
    }
  })
  return Array.from(counts.values()).filter((entry) => entry.count > 1)
}

function buildDuplicateKey(duplicates: DuplicateIdInfo[]): string {
  return duplicates.map((entry) => `${entry.id}:${entry.count}:${entry.indices.join(",")}`).join("|")
}

function countOrderChanges(
  prevIds: string[],
  nextIds: string[],
): { lengthChanged: boolean; changedCount: number } {
  const lengthChanged = prevIds.length !== nextIds.length
  const compareCount = lengthChanged ? Math.min(prevIds.length, nextIds.length) : prevIds.length
  let changedCount = 0
  for (let i = 0; i < compareCount; i += 1) {
    if (prevIds[i] !== nextIds[i]) {
      changedCount += 1
    }
  }
  return { lengthChanged, changedCount }
}

const SELECTION_DEBOUNCE_MS = 250

export function useJobsListState(options: UseJobsListStateOptions): JobsListState {
  const [selectedJobId, setSelectedJobId] = createSignal<string | null>(null)
  const jobs = createMemo(() => getJobsList(options.data))
  const filteredJobs = createMemo(() => {
    const list = jobs()
    const filters = options.ui.listFilterSelections[ListPane.Jobs]
    const mode = options.ui.listFilterMode[ListPane.Jobs]
    return getFilteredJobsByType(list, filters, mode)
  })
  const listItems = createMemo(() =>
    uniqueById(filteredJobs().map(formatJobRow), (row) => row.id),
  )
  const listIds = createMemo(() => listItems().map((row) => row.id))
  const jobIds = createMemo(() => options.data.jobsOrder)
  const filteredIds = createMemo(() => filteredJobs().map((job) => job.job_id))
  const selectedIndex = createMemo(() =>
    deriveSelectedIndex(listItems(), selectedJobId(), (row) => row.id),
  )
  const listWindow = useListWindow({
    items: listItems,
    selectedIndex,
    height: options.height,
    rowHeight: 2,
    chromeHeight: 2,
  })
  const cacheRemaining = createMemo(() => {
    if (!options.data.jobsCache.length) return 0
    const currentJobs = jobs()
    const seen = new Set(currentJobs.map((job) => job.job_id))
    const oldest = currentJobs[currentJobs.length - 1]
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
    if (options.ui.jobsListLoadingMore) return "Loading..."
    if (options.ui.jobsListHasMore) return "More (L)"
    if (cacheRemaining() > 0) return "More (L)"
    return ""
  })
  const title = createMemo(() => {
    const count = getListFilterCount(options.ui, ListPane.Jobs)
    const mode = options.ui.listFilterMode[ListPane.Jobs]
    const total = listItems().length
    const idx = selectedIndex()
    return formatListTitle("Jobs", mode, count, idx, total)
  })
  const totalCount = createMemo(() => listItems().length)

  createEffect(() => {
    watchArrayMutations(options.data.jobsOrder, {
      label: "jobs-order",
      getId: (id) => id,
    })
  })

  createEffect((prev: { key: string } | null) => {
    const mode = options.ui.listFilterMode[ListPane.Jobs]
    const selections = Array.from(options.ui.listFilterSelections[ListPane.Jobs] ?? []).sort()
    const key = `${mode}:${selections.join(",")}`
    if (!prev || prev.key !== key) {
      log("state", "jobs list filters", {
        mode,
        selections,
        selectionsCount: selections.length,
        totalJobs: jobs().length,
        filteredCount: filteredJobs().length,
      })
    }
    return { key }
  }, null)

  createEffect((prevSnapshot: { ids: string[]; dupKey: string; ref: string[] } | null) => {
    const current = options.data.jobsOrder
    const ids = jobIds().slice()
    const duplicates = findDuplicateIds(ids)
    const dupKey = buildDuplicateKey(duplicates)
    if (!prevSnapshot) {
      if (duplicates.length > 0) {
        log("state", "jobs raw duplicates", { duplicates, ids })
      }
      return { ids, dupKey, ref: current }
    }
    const prevIds = prevSnapshot.ids
    const sameRef = prevSnapshot.ref === current
    const { lengthChanged, changedCount } = countOrderChanges(prevIds, ids)
    const orderChanged = lengthChanged || changedCount > 0
    const duplicatesChanged = dupKey !== prevSnapshot.dupKey
    if (orderChanged || duplicatesChanged) {
      const prevDuplicates = findDuplicateIds(prevIds)
      const headCount = Math.min(12, ids.length)
      const prevHeadCount = Math.min(12, prevIds.length)
      const tailCount = Math.min(12, ids.length)
      const prevTailCount = Math.min(12, prevIds.length)
      log("state", "jobs raw order change", {
        prevLength: prevIds.length,
        nextLength: ids.length,
        changedCount: lengthChanged ? null : changedCount,
        sameRef,
        prevDuplicates,
        nextDuplicates: duplicates,
        prevHead: prevIds.slice(0, prevHeadCount),
        nextHead: ids.slice(0, headCount),
        prevTail: prevTailCount ? prevIds.slice(-prevTailCount) : [],
        nextTail: tailCount ? ids.slice(-tailCount) : [],
        prevIds,
        nextIds: ids,
      })
      if (sameRef) {
        log("state", "jobs raw mutation", {
          reason: orderChanged ? "order-change" : "duplicates-change",
          prevLength: prevIds.length,
          nextLength: ids.length,
          changedCount: lengthChanged ? null : changedCount,
          prevDuplicates,
          nextDuplicates: duplicates,
          prevIds,
          nextIds: ids,
        })
      }
    }
    return { ids, dupKey, ref: current }
  }, null)

  createEffect((prevSnapshot: { ids: string[]; dupKey: string; ref: JobSummary[] } | null) => {
    const current = filteredJobs()
    const ids = filteredIds()
    const duplicates = findDuplicateIds(ids)
    const dupKey = buildDuplicateKey(duplicates)
    if (!prevSnapshot) {
      if (duplicates.length > 0) {
        log("state", "jobs filtered duplicates", { duplicates, ids })
      }
      return { ids, dupKey, ref: current }
    }
    const prevIds = prevSnapshot.ids
    const sameRef = prevSnapshot.ref === current
    const { lengthChanged, changedCount } = countOrderChanges(prevIds, ids)
    const orderChanged = lengthChanged || changedCount > 0
    const duplicatesChanged = dupKey !== prevSnapshot.dupKey
    if (orderChanged || duplicatesChanged) {
      const prevDuplicates = findDuplicateIds(prevIds)
      const headCount = Math.min(12, ids.length)
      const prevHeadCount = Math.min(12, prevIds.length)
      const tailCount = Math.min(12, ids.length)
      const prevTailCount = Math.min(12, prevIds.length)
      log("state", "jobs filtered order change", {
        prevLength: prevIds.length,
        nextLength: ids.length,
        changedCount: lengthChanged ? null : changedCount,
        sameRef,
        prevDuplicates,
        nextDuplicates: duplicates,
        prevHead: prevIds.slice(0, prevHeadCount),
        nextHead: ids.slice(0, headCount),
        prevTail: prevTailCount ? prevIds.slice(-prevTailCount) : [],
        nextTail: tailCount ? ids.slice(-tailCount) : [],
        prevIds,
        nextIds: ids,
      })
      if (sameRef) {
        log("state", "jobs filtered mutation", {
          reason: orderChanged ? "order-change" : "duplicates-change",
          prevLength: prevIds.length,
          nextLength: ids.length,
          changedCount: lengthChanged ? null : changedCount,
          prevDuplicates,
          nextDuplicates: duplicates,
          prevIds,
          nextIds: ids,
        })
      }
    }
    return { ids, dupKey, ref: current }
  }, null)

  createEffect(() => {
    const list = listItems()
    const selected = selectedJobId()
    if (!list.length) {
      if (selected !== null) {
        setSelectedJobId(null)
        log("state", "jobs list selection reset", { reason: "empty" })
      }
      return
    }
    if (!selected || !list.some((row) => row.id === selected)) {
      const nextId = list[0].id
      setSelectedJobId(nextId)
      log("state", "jobs list selection reset", { reason: "missing", nextId })
    }
  })

  createEffect((prevId: string | null) => {
    const nextId = selectedJobId()
    if (nextId !== prevId) {
      const start = listWindow.windowStart()
      const end = listWindow.windowEnd()
      const visibleCount = listWindow.visibleCount()
      const index = selectedIndex()
      const visible = index >= start && index < end
      log("state", "jobs list selection", {
        from: prevId,
        to: nextId,
        index,
        total: listItems().length,
        windowStart: start,
        windowEnd: end,
        visibleCount,
        selectedVisible: visible,
      })
      if (!visible) {
        log("state", "jobs list selection hidden", {
          jobId: nextId,
          index,
          total: listItems().length,
          windowStart: start,
          windowEnd: end,
          visibleCount,
        })
      }
    }
    return nextId
  }, null as string | null)

  createEffect((prevSnapshot: { selectedId: string | null; index: number; idAtIndex: string | null } | null) => {
    const list = listItems()
    const selected = selectedJobId()
    const index = selectedIndex()
    const idAtIndex = list[index]?.id ?? null
    const mismatch = Boolean(selected && idAtIndex && selected !== idAtIndex)
    const changed =
      !prevSnapshot ||
      prevSnapshot.selectedId !== selected ||
      prevSnapshot.index !== index ||
      prevSnapshot.idAtIndex !== idAtIndex
    if (mismatch && changed) {
      log("state", "jobs list selection mismatch", {
        selectedId: selected,
        index,
        idAtIndex,
        total: list.length,
      })
    }
    return { selectedId: selected, index, idAtIndex }
  }, null)

  createEffect((prevDupKey: string | null) => {
    const visibleIds = listWindow.visibleItems().map((entry) => entry.item.id)
    const duplicates = findDuplicateIds(visibleIds)
    const dupKey = buildDuplicateKey(duplicates)
    if (dupKey && dupKey !== prevDupKey) {
      log("state", "jobs list window duplicates", {
        duplicates,
        visibleIds,
        windowStart: listWindow.windowStart(),
        windowEnd: listWindow.windowEnd(),
        selectedIndex: selectedIndex(),
        selectedJobId: selectedJobId(),
      })
    }
    return dupKey || ""
  }, null)

  createEffect((prevCount: number) => {
    const nextCount = listItems().length
    if (nextCount !== prevCount) {
      log("state", "jobs list size", {
        from: prevCount,
        to: nextCount,
        selectedIndex: selectedIndex(),
        windowStart: listWindow.windowStart(),
        windowEnd: listWindow.windowEnd(),
        visibleCount: listWindow.visibleCount(),
        selectedJobId: selectedJobId(),
      })
    }
    return nextCount
  }, -1)

  createEffect((prevSnapshot: { ids: string[]; selectedId: string | null } | null) => {
    const ids = listIds()
    const selectedId = selectedJobId()
    if (!prevSnapshot) {
      return { ids, selectedId }
    }
    const prevIds = prevSnapshot.ids
    const { lengthChanged, changedCount } = countOrderChanges(prevIds, ids)
    const orderChanged = lengthChanged || changedCount > 0
    const prevDuplicates = findDuplicateIds(prevIds)
    const nextDuplicates = findDuplicateIds(ids)
    const duplicatesChanged = buildDuplicateKey(prevDuplicates) !== buildDuplicateKey(nextDuplicates)
    if (orderChanged || duplicatesChanged) {
      const prevSelectedIndex = selectedId ? prevIds.indexOf(selectedId) : -1
      const nextSelectedIndex = selectedId ? ids.indexOf(selectedId) : -1
      const headCount = Math.min(12, ids.length)
      const prevHeadCount = Math.min(12, prevIds.length)
      const tailCount = Math.min(12, ids.length)
      const prevTailCount = Math.min(12, prevIds.length)
      log("state", "jobs list order change", {
        prevLength: prevIds.length,
        nextLength: ids.length,
        changedCount: lengthChanged ? null : changedCount,
        selectedId,
        prevSelectedIndex,
        nextSelectedIndex,
        prevDuplicates,
        nextDuplicates,
        prevHead: prevIds.slice(0, prevHeadCount),
        nextHead: ids.slice(0, headCount),
        prevTail: prevTailCount ? prevIds.slice(-prevTailCount) : [],
        nextTail: tailCount ? ids.slice(-tailCount) : [],
        prevIds,
        nextIds: ids,
      })
    }
    return { ids, selectedId }
  }, null)

  createEffect((prevStart: number) => {
    const start = listWindow.windowStart()
    if (start !== prevStart) {
      log("state", "jobs list window", {
        windowStart: start,
        windowEnd: listWindow.windowEnd(),
        visibleCount: listWindow.visibleCount(),
        selectedIndex: selectedIndex(),
        total: listItems().length,
        selectedJobId: selectedJobId(),
      })
    }
    return start
  }, -1)

  createEffect(() => {
    const jobId = selectedJobId()
    if (!jobId || options.primaryView() !== "jobs") return
    const current = options.data.selectedJob?.job_id ?? null
    if (current === jobId) return
    const timer = setTimeout(() => {
      log("state", "jobs list commit", {
        jobId,
        debounceMs: SELECTION_DEBOUNCE_MS,
        index: selectedIndex(),
        total: listItems().length,
      })
      options.onSelectJob?.(jobId)
    }, SELECTION_DEBOUNCE_MS)
    onCleanup(() => {
      clearTimeout(timer)
    })
  })

  const moveSelection = (delta: number): boolean => {
    const moveStart = Date.now()
    const list = listItems()
    if (!list.length) {
      log("action", "jobs list move", { delta, reason: "empty" })
      return false
    }
    log("state", "jobs list move start", {
      delta,
      listSize: list.length,
      selectedJobId: selectedJobId(),
      selectedIndex: selectedIndex(),
    })
    const currentId = selectedJobId()
    const resolveStart = Date.now()
    const nextId = moveSelectionById(list, currentId, delta, (row) => row.id)
    const resolveMs = Date.now() - resolveStart
    log("state", "jobs list move resolve", {
      delta,
      currentId,
      nextId,
      listSize: list.length,
      resolveMs,
    })
    if (!nextId || nextId === currentId) {
      log("action", "jobs list move", {
        delta,
        currentId,
        nextId,
        moved: false,
        listSize: list.length,
      })
      return false
    }
    const nextIndex = list.findIndex((row) => row.id === nextId)
    setSelectedJobId(nextId)
    log("action", "jobs list move", {
      delta,
      currentId,
      nextId,
      moved: true,
      listSize: list.length,
      nextIndex,
    })
    const totalMs = Date.now() - moveStart
    log("state", "jobs list move end", {
      delta,
      currentId,
      nextId,
      listSize: list.length,
      nextIndex,
      totalMs,
    })
    return true
  }

  const selectCurrent = () => {
    const list = listItems()
    const row = list[selectedIndex()]
    if (!row?.id) return
    if (row.id !== selectedJobId()) {
      setSelectedJobId(row.id)
    }
  }

  return {
    selectedJobId,
    selectedIndex,
    listWindow,
    title,
    totalCount,
    loadMoreHint,
    moveSelection,
    selectCurrent,
  }
}
