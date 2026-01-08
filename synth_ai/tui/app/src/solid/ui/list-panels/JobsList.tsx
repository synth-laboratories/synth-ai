import { For, Show, createMemo } from "solid-js"
import { COLORS } from "../../theme"
import type { JobSummary } from "../../../tui_data"
import { formatTimestamp } from "../../formatters/time"

interface JobsListProps {
  jobs: JobSummary[]
  selectedIndex: number
  focused: boolean
  width: number
  height: number
}

function getRelevantDate(job: JobSummary): string {
  const dateStr = job.finished_at || job.started_at || job.created_at
  return formatTimestamp(dateStr)
}

function getJobTypeLabel(job: JobSummary): string {
  // Return a human-readable job type
  const type = job.training_type || job.job_source || "job"
  
  // Map known types to readable labels
  switch (type.toLowerCase()) {
    case "mipro":
    case "mipro_v2":
      return "MIPRO"
    case "gepa":
    case "gepa_v1":
      return "GEPA"
    case "prompt-learning":
    case "prompt_learning":
      return "Prompt Learning"
    case "eval":
      return "Eval"
    case "learning":
      return "Learning"
    case "graph_evolve":
      return "Graph Evolve"
    default:
      // Capitalize and clean up
      return type
        .replace(/_/g, " ")
        .replace(/-/g, " ")
        .split(" ")
        .map(w => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ")
  }
}

function getStatusLabel(status: string): string {
  const s = (status || "unknown").toLowerCase()
  switch (s) {
    case "running": return "Running"
    case "completed": case "succeeded": return "Completed"
    case "failed": case "error": return "Failed"
    case "queued": return "Queued"
    case "canceled": case "cancelled": return "Canceled"
    default: return status || "-"
  }
}

function formatJobCard(job: JobSummary) {
  const jobType = getJobTypeLabel(job)
  const status = getStatusLabel(job.status)
  const dateStr = getRelevantDate(job)

  return {
    id: job.job_id,
    type: jobType,
    status: status,
    date: dateStr,
  }
}

/**
 * Jobs list panel component.
 */
export function JobsList(props: JobsListProps) {
  const items = createMemo(() => props.jobs.map(formatJobCard))

  const visibleItems = createMemo(() => {
    const list = items()
    const height = props.height - 2 // Account for border
    const selected = props.selectedIndex

    let start = 0
    if (selected >= start + height) {
      start = selected - height + 1
    }
    if (selected < start) {
      start = selected
    }

    return list.slice(start, start + height).map((item, idx) => ({
      ...item,
      globalIndex: start + idx,
    }))
  })

  // Calculate max widths for alignment
  const maxTypeWidth = createMemo(() => {
    const list = items()
    if (!list.length) return 12
    return Math.max(12, ...list.map((item) => item.type.length))
  })

  const maxStatusWidth = createMemo(() => {
    const list = items()
    if (!list.length) return 10
    return Math.max(10, ...list.map((item) => item.status.length))
  })

  return (
    <box
      width={props.width}
      height={props.height}
      borderStyle="single"
      borderColor={props.focused ? COLORS.textAccent : COLORS.border}
      title="Jobs"
      titleAlignment="left"
      paddingLeft={1}
      paddingRight={1}
      flexDirection="column"
    >
      <Show
        when={props.jobs.length > 0}
        fallback={<text fg={COLORS.textDim}>No jobs yet. Press r to refresh.</text>}
      >
        <For each={visibleItems()}>
          {(item) => {
            const isSelected = item.globalIndex === props.selectedIndex
            const fg = isSelected ? COLORS.textSelected : COLORS.text
            const bg = isSelected ? COLORS.bgSelection : undefined
            const statusFg = isSelected ? COLORS.textSelected : 
              item.status === "Running" ? COLORS.warning :
              item.status === "Completed" ? COLORS.success :
              item.status === "Failed" ? COLORS.error :
              COLORS.textDim
            
            const typeStr = item.type.padEnd(maxTypeWidth(), " ")
            const statusStr = item.status.padEnd(maxStatusWidth(), " ")

            return (
              <box flexDirection="row" backgroundColor={bg}>
                <text fg={fg}>{typeStr}</text>
                <text fg={COLORS.textDim}> | </text>
                <text fg={statusFg}>{statusStr}</text>
                <text fg={COLORS.textDim}> | </text>
                <text fg={isSelected ? COLORS.textSelected : COLORS.textDim}>{item.date}</text>
              </box>
            )
          }}
        </For>
        <Show when={props.jobs.length > visibleItems().length}>
          <text fg={COLORS.textDim}>... ({props.jobs.length - visibleItems().length} more)</text>
        </Show>
      </Show>
    </box>
  )
}
