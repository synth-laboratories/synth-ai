import { type Accessor, createMemo } from "solid-js"

import type { ActivePane, ListFilterMode, PrincipalPane } from "../../types"
import { ListPane } from "../../types"
import type { AppState } from "../../state/app-state"
import { formatTimestamp } from "../formatters/time"
import { useLiveLogs } from "../utils/live-logs"
import type { LogFileInfo } from "../utils/logs"
import { formatListTitle, getListFilterCount } from "../utils/listTitle"
import { type ListWindowState, useListWindow } from "./useListWindow"

export type LogsListRow = {
  id: string
  type: string
  date: string
}

export type LogsListState = {
  liveLogs: ReturnType<typeof useLiveLogs>
  logFiles: Accessor<LogFileInfo[]>
  selectedIndex: Accessor<number>
  selectedFile: Accessor<LogFileInfo | null>
  listWindow: ListWindowState<LogsListRow>
  listTitle: Accessor<string>
  totalCount: Accessor<number>
}

type UseLogsListStateOptions = {
  activePane: Accessor<ActivePane>
  principalPane: Accessor<PrincipalPane>
  height: Accessor<number>
  ui: AppState
}

// Maps log file suffix to display label
// Files are named {timestamp}_{suffix}.log
const LOG_SUFFIXES = {
  all: "All",
  key: "Key",
  action: "Action",
  lifecycle: "Lifecycle",
  http: "HTTP",
  modal: "Modal",
  state: "State",
  error: "Error",
  console: "Console",
} as const

type LogType = (typeof LOG_SUFFIXES)[keyof typeof LOG_SUFFIXES] | "Other"

function getLogTypeSuffix(name: string): string {
  // Files are named {timestamp}_{suffix}.log
  // Extract suffix by removing .log and taking last part after _
  const baseName = name.replace(/\.log$/, "")
  const lastUnderscore = baseName.lastIndexOf("_")
  if (lastUnderscore === -1) return "other"
  return baseName.slice(lastUnderscore + 1).toLowerCase()
}

function formatLogType(name: string): LogType {
  const suffix = getLogTypeSuffix(name)
  const label = LOG_SUFFIXES[suffix as keyof typeof LOG_SUFFIXES]
  return label ?? "Other"
}

export function getLogTypeLabel(file: LogFileInfo): string {
  return formatLogType(file.name)
}

export function buildLogTypeOptions(
  files: LogFileInfo[],
): Array<{ id: string; label: string; count: number }> {
  const counts = new Map<string, { label: string; count: number }>()
  for (const file of files) {
    const key = getLogTypeSuffix(file.name)
    const label = formatLogType(file.name)
    const current = counts.get(key)
    if (current) {
      current.count += 1
    } else {
      counts.set(key, { label, count: 1 })
    }
  }
  return Array.from(counts.entries())
    .sort((a, b) => a[1].label.localeCompare(b[1].label))
    .map(([id, data]) => ({ id, label: data.label, count: data.count }))
}

export function getFilteredLogsByType(
  files: LogFileInfo[],
  typeFilter: ReadonlySet<string>,
  mode: ListFilterMode,
): LogFileInfo[] {
  if (mode === "none") return []
  if (mode === "all") return files
  if (!typeFilter.size) return []
  return files.filter((file) => typeFilter.has(getLogTypeSuffix(file.name)))
}

function formatLogRow(file: LogFileInfo): LogsListRow {
  return {
    id: file.path,
    type: formatLogType(file.name),
    date: formatTimestamp(file.mtimeMs),
  }
}

export function useLogsListState(options: UseLogsListStateOptions): LogsListState {
  const liveLogs = useLiveLogs({
    listActive: () => options.activePane() === ListPane.Logs,
    detailActive: () => options.activePane() === ListPane.Logs && options.principalPane() === "jobs",
  })

  const allLogFiles = createMemo(() =>
    options.activePane() === ListPane.Logs ? liveLogs.files() : [],
  )
  const logFiles = createMemo(() => {
    const files = allLogFiles()
    const filters = options.ui.listFilterSelections[ListPane.Logs]
    const mode = options.ui.listFilterMode[ListPane.Logs]
    return getFilteredLogsByType(files, filters, mode)
  })
  const listItems = createMemo(() => logFiles().map(formatLogRow))
  const listWindow = useListWindow({
    items: listItems,
    selectedIndex: liveLogs.selectedIndex,
    height: options.height,
    rowHeight: 2,
    chromeHeight: 2,
  })
  const selectedFile = createMemo(() => {
    const files = logFiles()
    const index = liveLogs.selectedIndex()
    return files[index] ?? null
  })
  const listTitle = createMemo(() => {
    const count = getListFilterCount(options.ui, ListPane.Logs)
    const mode = options.ui.listFilterMode[ListPane.Logs]
    const total = logFiles().length
    const idx = liveLogs.selectedIndex()
    return formatListTitle("Logs", mode, count, idx, total)
  })
  const totalCount = createMemo(() => logFiles().length)

  return {
    liveLogs,
    logFiles,
    selectedIndex: liveLogs.selectedIndex,
    selectedFile,
    listWindow,
    listTitle,
    totalCount,
  }
}
