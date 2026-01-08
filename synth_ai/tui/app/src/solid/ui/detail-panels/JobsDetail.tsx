import { For, Show, createMemo } from "solid-js"
import { COLORS } from "../../theme"
import type { Snapshot } from "../../../types"
import type { JobEvent } from "../../../tui_data"
import { formatDetails } from "../../formatters/job-details"
import { formatResults } from "../../formatters/results"
import { formatMetrics } from "../../formatters/metrics"
import { formatEventData } from "../../../formatters"

interface JobsDetailProps {
  snapshot: Snapshot
  events: JobEvent[]
  eventWindow: {
    slice: JobEvent[]
    windowStart: number
    selected: number
  }
  lastError: string | null
  detailWidth: number
}

function formatEventSummary(event: JobEvent, maxWidth: number): string {
  const seq = String(event.seq).padStart(5, " ")
  const detail = event.message ?? formatEventData(event.data)
  const text = detail ? `${seq} ${event.type} ${detail}` : `${seq} ${event.type}`
  if (text.length <= maxWidth) return text
  return `${text.slice(0, Math.max(0, maxWidth - 3))}...`
}

/**
 * Jobs detail panels (right side).
 */
export function JobsDetail(props: JobsDetailProps) {
  const detailsText = createMemo(() => formatDetails(props.snapshot))
  const resultsText = createMemo(() => formatResults(props.snapshot))
  const metricsText = createMemo(() => formatMetrics(props.snapshot.metrics))

  return (
    <box flexDirection="column" flexGrow={1} border={false} gap={1}>
      {/* Details Box */}
      <box
        border
        borderStyle="single"
        borderColor={COLORS.border}
        title="Details"
        titleAlignment="left"
        paddingLeft={1}
        paddingTop={1}
        height={6}
      >
        <text fg={COLORS.text}>{detailsText()}</text>
      </box>

      {/* Results Box */}
      <box
        border
        borderStyle="single"
        borderColor={COLORS.border}
        title="Results"
        titleAlignment="left"
        paddingLeft={1}
        paddingTop={1}
        height={4}
      >
        <text fg={COLORS.text}>{resultsText()}</text>
      </box>

      {/* Metrics Box */}
      <box
        border
        borderStyle="single"
        borderColor={COLORS.border}
        title="Metrics"
        titleAlignment="left"
        paddingLeft={1}
        paddingTop={1}
        height={4}
      >
        <text fg={COLORS.text}>{metricsText()}</text>
      </box>

      {/* Events Box */}
      <box
        flexGrow={1}
        border
        borderStyle="single"
        borderColor={COLORS.border}
        title="Events"
        titleAlignment="left"
        paddingLeft={1}
        paddingTop={1}
        flexDirection="column"
      >
        <Show
          when={props.events.length > 0}
          fallback={<text fg={COLORS.textDim}>No events yet.</text>}
        >
          <For each={props.eventWindow.slice}>
            {(event, idx) => {
              const globalIndex = props.eventWindow.windowStart + idx()
              const isSelected = globalIndex === props.eventWindow.selected
              const maxWidth = Math.max(10, props.detailWidth - 6)
              const summary = formatEventSummary(event, maxWidth)
              return (
                <text fg={isSelected ? COLORS.textAccent : COLORS.text}>
                  {`${isSelected ? ">" : " "} ${summary}`}
                </text>
              )
            }}
          </For>
        </Show>
      </box>

      <Show when={props.lastError}>
        <text fg={COLORS.error}>{`Error: ${props.lastError}`}</text>
      </Show>
    </box>
  )
}
