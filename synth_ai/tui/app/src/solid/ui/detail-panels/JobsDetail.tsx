import { For, Show, createMemo } from "solid-js"
import { COLORS } from "../../theme"
import type { AppData } from "../../../types"
import type { JobEvent } from "../../../tui_data"
import { isEvalJob } from "../../../tui_data"
import { formatDetails } from "../../formatters/job-details"
import { formatResults } from "../../formatters/results"
import { GraphEvolveResultsPanel } from "./GraphEvolveResultsPanel"
import { formatMetrics, formatMetricsCharts } from "../../formatters/metrics"
import { ListCard, getIndicator } from "../../components/ListCard"

interface JobsDetailProps {
  data: AppData
  events: JobEvent[]
  eventWindow: {
    slice: JobEvent[]
    windowStart: number
    selected: number
  }
  lastError: string | null
  detailWidth: number
  detailHeight: number
  resultsFocused?: boolean
  eventsFocused?: boolean
  metricsFocused?: boolean
  metricsView: "latest" | "charts"
  verifierEvolveGenerationIndex: number
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function padLine(text: string, width: number): string {
  if (text.length >= width) return text
  return text.padEnd(width, " ")
}

function formatEventHeader(event: JobEvent): string {
  const seq = String(event.seq).padStart(3, " ")
  const typeRaw = event.type || ""
  const type = typeRaw.replace(/^prompt\.learning\./, "")
  return `${seq} ${type}`.trimEnd()
}

function formatEventTimestamp(event: JobEvent): string {
  const ts = (event as any).timestamp
  if (typeof ts !== "string" || ts.length === 0) return ""
  // Keep it short but human-readable
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return ""
  return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })
}


/**
 * Jobs detail panels (right side).
 */
export function JobsDetail(props: JobsDetailProps) {
  const detailsText = createMemo(() => formatDetails(props.data))
  const resultsText = createMemo(() => formatResults(props.data))
  const isGraphEvolveVerifier = createMemo(() => {
    const job = props.data.selectedJob
    const meta = job?.metadata as Record<string, any> | null
    return job?.training_type === "graph_evolve" && meta?.graph_type === "verifier"
  })
  const metricPointsCount = createMemo(() => {
    const m: any = props.data.metrics || {}
    const pts = Array.isArray(m?.points) ? m.points : []
    return pts.length
  })
  const detailsHeight = createMemo(() => {
    const lines = detailsText().split("\n").length
    return clamp(lines + 2, 6, 9)
  })
  const resultsHeight = createMemo(() => {
    if (isGraphEvolveVerifier()) return 9
    const lines = resultsText().split("\n").length
    return clamp(lines + 2, 6, 10)
  })
  const resultsTitle = createMemo(() => {
    const job = props.data.selectedJob
    return job && isEvalJob(job) ? "Results" : "Summary"
  })
  const resultsInnerHeight = createMemo(() => Math.max(4, resultsHeight() - 2))
  const metricsPanelHeight = createMemo(() => {
    // Reserve fixed space for Details/Results and ensure Events always has room.
    const detailsH = detailsHeight()
    const resultsH = resultsHeight()
    const minEventsH = 12
    const maxH = Math.max(4, props.detailHeight - (detailsH + resultsH + minEventsH))

    if (props.metricsView === "charts") {
      const desired = metricPointsCount() > 0 ? 22 : 18
      return clamp(desired, 12, maxH)
    }
    // Latest mode: expand a bit when we actually have metrics.
    const desired = metricPointsCount() > 0 ? 8 : 4
    return clamp(desired, 4, maxH)
  })
  const metricsText = createMemo(() => {
    if (props.metricsView === "charts") {
      const innerWidth = Math.max(30, props.detailWidth - 6)
      const panelHeight = metricsPanelHeight()
      // In charts mode we use the full panel height for larger charts.
      return formatMetricsCharts(props.data.metrics, {
        width: innerWidth,
        height: panelHeight,
      })
    }
    return formatMetrics(props.data.metrics)
  })

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
        height={detailsHeight()}
      >
        <text fg={COLORS.text}>{detailsText()}</text>
      </box>

      {/* Results Box */}
      <box
        border
        borderStyle="single"
        borderColor={props.resultsFocused && isGraphEvolveVerifier() ? COLORS.textAccent : COLORS.border}
        title={resultsTitle()}
        titleAlignment="left"
        paddingLeft={1}
        height={resultsHeight()}
      >
        <Show
          when={isGraphEvolveVerifier()}
          fallback={<text fg={COLORS.text}>{resultsText()}</text>}
        >
          <GraphEvolveResultsPanel
            data={props.data}
            width={Math.max(30, props.detailWidth - 6)}
            height={resultsInnerHeight()}
            focused={!!props.resultsFocused && isGraphEvolveVerifier()}
            selectedGenerationIndex={props.verifierEvolveGenerationIndex}
          />
        </Show>
      </box>

      {/* Metrics Box */}
      <box
        border
        borderStyle="single"
        borderColor={props.metricsFocused ? COLORS.textAccent : COLORS.border}
        title="Metrics"
        titleAlignment="left"
        paddingLeft={1}
        height={metricsPanelHeight()}
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
            const lineWidth = Math.max(20, props.detailWidth - 6)
            return (
              <box flexDirection="column">
                <For each={props.eventWindow.slice}>
                  {(event, idx) => {
                    const globalIdx = windowStart + idx()
                    const isSel = globalIdx === selected
                    const title = formatEventHeader(event)
                    const timestamp = formatEventTimestamp(event) || "-"
                    return (
                      <ListCard isSelected={isSel}>
                        {(ctx) => (
                          <box flexDirection="column">
                            <box flexDirection="row" backgroundColor={ctx.bg} width="100%">
                              <text fg={ctx.fg}>
                                {padLine(`${getIndicator(ctx.isSelected)}${title}`, lineWidth)}
                              </text>
                            </box>
                            <box flexDirection="row" backgroundColor={ctx.bg} width="100%">
                              <text fg={ctx.fgDim}>
                                {padLine(`  ${timestamp}`, lineWidth)}
                              </text>
                            </box>
                          </box>
                        )}
                      </ListCard>
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
