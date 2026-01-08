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
 * Truncate text to max width
 */
function truncate(text: string, maxWidth: number): string {
  if (text.length <= maxWidth) return text
  return text.slice(0, maxWidth - 3) + "..."
}

/**
 * Format a single event line for display
 * Gold reference format: "  seq type.name message"
 */
function formatEventLine(event: JobEvent, maxWidth: number): string {
  const seq = String(event.seq).padStart(3, " ")
  const type = event.type || ""
  const message = event.message || formatEventData(event.data) || ""
  
  // Build the line: seq + space + type + space + message
  const prefix = `${seq} ${type}`
  
  if (message) {
    const fullLine = `${prefix} ${message}`
    return truncate(fullLine, maxWidth)
  }
  
  return truncate(prefix, maxWidth)
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

      {/* Events Box - clean text list */}
      <box
        flexGrow={1}
        border
        borderStyle="single"
        borderColor={props.eventsFocused ? COLORS.textAccent : COLORS.border}
        title="Events"
        titleAlignment="left"
        flexDirection="column"
        paddingLeft={1}
      >
        <Show
          when={props.events.length > 0}
          fallback={<text fg={COLORS.textDim}>No events yet.</text>}
        >
          {(() => {
            const maxWidth = Math.max(20, props.detailWidth - 6)
            const selected = props.eventWindow.selected
            const windowStart = props.eventWindow.windowStart
            
            return (
              <For each={props.eventWindow.slice}>
                {(event, idx) => {
                  const globalIdx = windowStart + idx()
                  const isSel = globalIdx === selected
                  const line = formatEventLine(event, maxWidth)
                  
                  return (
                    <text fg={isSel ? COLORS.textAccent : COLORS.text}>
                      {`${isSel ? ">" : " "} ${line}`}
                    </text>
                  )
                }}
              </For>
            )
          })()}
        </Show>
      </box>

      <Show when={props.lastError}>
        <text fg={COLORS.error}>{`Error: ${props.lastError}`}</text>
      </Show>
    </box>
  )
}
