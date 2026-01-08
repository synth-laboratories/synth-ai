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
  eventsFocused?: boolean
}

/**
 * Format event type for display (truncate long types)
 */
function formatEventType(type: string, maxWidth: number): string {
  if (type.length <= maxWidth) return type
  return type.slice(0, maxWidth - 3) + "..."
}

/**
 * Get event message/description
 */
function getEventMessage(event: JobEvent): string {
  if (event.message) return event.message
  const data = formatEventData(event.data)
  return data || ""
}

/**
 * Jobs detail panels (right side).
 */
export function JobsDetail(props: JobsDetailProps) {
  const detailsText = createMemo(() => formatDetails(props.snapshot))
  const resultsText = createMemo(() => formatResults(props.snapshot))
  const metricsText = createMemo(() => formatMetrics(props.snapshot.metrics))

  return (
    <box flexDirection="column" flexGrow={1} border={false} gap={0}>
      {/* Details Box */}
      <box
        border
        borderStyle="single"
        borderColor={COLORS.border}
        title="Details"
        titleAlignment="left"
        paddingLeft={1}
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
        height={4}
      >
        <text fg={COLORS.text}>{metricsText()}</text>
      </box>

      {/* Events Box - gold reference style with two-line cards */}
      <box
        flexGrow={1}
        border
        borderStyle="single"
        borderColor={props.eventsFocused ? COLORS.textAccent : COLORS.border}
        title="Events"
        titleAlignment="left"
        flexDirection="column"
      >
        <Show
          when={props.events.length > 0}
          fallback={<text fg={COLORS.textDim}>  No events yet.</text>}
        >
          <For each={props.eventWindow.slice}>
            {(event, idx) => {
              const globalIndex = props.eventWindow.windowStart + idx()
              const isSelected = globalIndex === props.eventWindow.selected
              const bg = isSelected ? COLORS.bgSelection : undefined
              const fgPrimary = isSelected ? COLORS.textBright : COLORS.text
              const fgSecondary = isSelected ? COLORS.textBright : COLORS.textDim
              const maxTypeWidth = Math.max(10, props.detailWidth - 10)
              const eventType = formatEventType(event.type, maxTypeWidth)
              const eventMessage = getEventMessage(event)

              return (
                <box flexDirection="column">
                  {/* Line 1: sequence number + event type */}
                  <box flexDirection="row" backgroundColor={bg} width="100%">
                    <text fg={fgSecondary}>  </text>
                    <text fg={fgSecondary}>{String(event.seq).padStart(3, " ")} </text>
                    <text fg={fgPrimary}>{eventType}</text>
                  </box>
                  {/* Line 2: event message (indented) */}
                  <box flexDirection="row" backgroundColor={bg} width="100%">
                    <text fg={fgSecondary}>  </text>
                    <text fg={fgSecondary}>{eventMessage}</text>
                  </box>
                </box>
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
