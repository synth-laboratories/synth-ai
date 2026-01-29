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
  const type = job.job_type || job.job_source || "job"
  
  // Map known types to readable labels
  switch (type.toLowerCase()) {
    case "gepa":
    case "gepa_v1":
      return "GEPA"
    case "prompt-learning":
    case "prompt_learning":
      return "Prompt Optimization"
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
    case "failed": case "error": return "Error"
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
 * 
 * Gold reference format (two lines per job):
 *   ▸ Eval
 *     Error | Jan 8 at 10:32 AM
 */
export function JobsList(props: JobsListProps) {
  const items = createMemo(() => props.jobs.map(formatJobCard))

  // Each job takes 2 lines, so we can show (height - 2) / 2 jobs
  const visibleCount = createMemo(() => Math.floor((props.height - 2) / 2))

  const visibleItems = createMemo(() => {
    const list = items()
    const maxVisible = visibleCount()
    const selected = props.selectedIndex

    let start = 0
    if (selected >= start + maxVisible) {
      start = selected - maxVisible + 1
    }
    if (selected < start) {
      start = selected
    }

    return list.slice(start, start + maxVisible).map((item, idx) => ({
      ...item,
      globalIndex: start + idx,
    }))
  })

  return (
    <box
      width={props.width}
      height={props.height}
      borderStyle="single"
      borderColor={props.focused ? COLORS.textAccent : COLORS.border}
      title="Jobs"
      titleAlignment="left"
      flexDirection="column"
    >
      <Show
        when={props.jobs.length > 0}
        fallback={<text fg={COLORS.textDim}> No jobs yet. Press r to refresh.</text>}
      >
        <For each={visibleItems()}>
          {(item) => {
            const isSelected = item.globalIndex === props.selectedIndex
            const bg = isSelected ? COLORS.bgSelection : undefined
            const typeFg = isSelected ? COLORS.textBright : COLORS.text
            const statusFg = isSelected ? COLORS.textBright : COLORS.textDim
            const indicator = isSelected ? "▸ " : "  "

            return (
              <box flexDirection="column">
                {/* Line 1: indicator + job type */}
                <box flexDirection="row" backgroundColor={bg} width="100%">
                  <text fg={typeFg}>{indicator}{item.type}</text>
                </box>
                {/* Line 2: status | date (indented) */}
                <box flexDirection="row" backgroundColor={bg} width="100%">
                  <text fg={statusFg}>  {item.status} | {item.date}</text>
                </box>
              </box>
            )
          }}
        </For>
      </Show>
    </box>
  )
}
