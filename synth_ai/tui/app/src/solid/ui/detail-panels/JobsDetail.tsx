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
 * Wrap text to maxWidth (simple whitespace wrapping).
 * Kept local so the Events cards don't depend on SolidShell internals.
 */
function wrapText(text: string, maxWidth: number): string[] {
  const raw = (text ?? "").toString()
  if (!raw) return []
  const words = raw.replace(/\s+/g, " ").trim().split(" ")
  const lines: string[] = []
  let current = ""
  for (const w of words) {
    if (!current) {
      current = w
      continue
    }
    if ((current + " " + w).length <= maxWidth) {
      current = current + " " + w
      continue
    }
    // If a single word is longer than maxWidth, hard-split it.
    if (current.length === 0 && w.length > maxWidth) {
      lines.push(w.slice(0, maxWidth))
      current = w.slice(maxWidth)
      continue
    }
    lines.push(current)
    current = w
  }
  if (current) lines.push(current)
  // Hard truncate overly long lines (e.g. long unbroken strings)
  return lines.map((l) => (l.length > maxWidth ? truncate(l, maxWidth) : l))
}

function formatEventHeader(event: JobEvent): string {
  const seq = String(event.seq).padStart(3, " ")
  const type = event.type || ""
  return `${seq} ${type}`.trimEnd()
}

function formatEventBody(event: JobEvent): string {
  return event.message || formatEventData(event.data) || ""
}

function formatEventTimestamp(event: JobEvent): string {
  const ts = (event as any).timestamp
  if (typeof ts !== "string" || ts.length === 0) return ""
  // Keep it short but human-readable
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return ""
  return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
}

function buildEventCardLines(event: JobEvent, width: number, isSelected: boolean): string[] {
  // NOTE: The box-drawing version was causing terminal rendering glitches for some fonts/terminals.
  // Use a plain-text "card" made of separators and indentation.
  const prefix = isSelected ? "> " : "  "
  const ts = formatEventTimestamp(event)
  const header = ts ? `${formatEventHeader(event)}  ${ts}` : formatEventHeader(event)

  const body = formatEventBody(event)
  const bodyLines = wrapText(body, Math.max(10, width - 4)).slice(0, 4)
  const content =
    bodyLines.length > 0
      ? bodyLines.map((l) => `${prefix}  ${l}`)
      : [`${prefix}  (no content)`]

  const needsEllipsis = wrapText(body, Math.max(10, width - 4)).length > bodyLines.length
  const ellipsis = needsEllipsis ? [`${prefix}  …`] : []

  const sep = `${prefix}${"-".repeat(Math.max(0, width - prefix.length))}`
  return [sep, `${prefix}${header}`, ...content, ...ellipsis]
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

      {/* Events Box - compact per-event cards (pure text) */}
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
            const selected = props.eventWindow.selected
            const windowStart = props.eventWindow.windowStart
            const width = Math.max(40, props.detailWidth - 6)

            return (
              <box flexDirection="column">
                <For each={props.eventWindow.slice}>
                  {(event, idx) => {
                    const globalIdx = windowStart + idx()
                    const isSel = globalIdx === selected
                    const lines = buildEventCardLines(event, width, isSel)
                    return (
                      <box flexDirection="column" paddingBottom={1}>
                        <For each={lines}>
                          {(line) => (
                            <text fg={isSel ? COLORS.textBright : COLORS.textDim}>
                              {line}
                            </text>
                          )}
                        </For>
                      </box>
                    )
                  }}
                </For>
              </box>
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
