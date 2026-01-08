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

function formatJobCard(job: JobSummary) {
  const jobType = job.training_type || job.job_source || "job"
  const status = job.status || "-"
  const dateStr = getRelevantDate(job)
  const shortId = job.job_id.slice(-8)

  return {
    id: job.job_id,
    name: jobType,
    description: `${status} | ${shortId} | ${dateStr}`,
  }
}

/**
 * Jobs list panel component.
 */
export function JobsList(props: JobsListProps) {
  const items = createMemo(() => props.jobs.map(formatJobCard))

  const maxNameWidth = createMemo(() => {
    const list = items()
    if (!list.length) return 0
    return Math.max(...list.map((item) => item.name.length))
  })

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
            const name = item.name.padEnd(maxNameWidth(), " ")
            const description = item.description || ""

            return (
              <text fg={fg} bg={bg}>
                {`${name} ${description}`}
              </text>
            )
          }}
        </For>
        <Show when={props.jobs.length > visibleItems().length}>
          <text fg={COLORS.textDim}>...</text>
        </Show>
      </Show>
    </box>
  )
}
