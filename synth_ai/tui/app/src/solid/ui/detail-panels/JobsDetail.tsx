import { Show, createMemo } from "solid-js"
import { Dynamic } from "solid-js/web"

import type { AppData } from "../../../types"
import type { JobEvent } from "../../../tui_data"
import { isEvalJob } from "../../../tui_data"
import type { ListWindowItem } from "../../utils/list"
import type { PanelRegistryKey } from "./types"

// Import registries
import { getDetailsPanel } from "./registries/details-registry"
import { getResultsPanel } from "./registries/results-registry"
import { getMetricsPanel, type MetricsView } from "./registries/metrics-registry"

// Import panel modules to register them
import "./details"
import "./results"
import "./metrics"

// Import events panel directly (not registry-based)
import { EventsListPanel } from "./events"
import { COLORS, PANEL, TEXT, getPanelBorderColor } from "../../theme"

interface JobsDetailProps {
  data: AppData
  eventItems: ListWindowItem<JobEvent>[]
  totalEvents: number
  selectedIndex: number
  lastError: string | null
  detailWidth: number
  detailHeight: number
  resultsFocused?: boolean
  eventsFocused?: boolean
  metricsFocused?: boolean
  metricsView: MetricsView
  verifierEvolveGenerationIndex: number
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

/**
 * Jobs detail panels (right side).
 * Uses registry pattern for pluggable panel components.
 */
export function JobsDetail(props: JobsDetailProps) {
  // Derive registry key from selected job
  const registryKey = createMemo((): PanelRegistryKey => {
    const job = props.data.selectedJob
    const meta = job?.metadata as Record<string, any> | null
    return {
      trainingType: job?.training_type ?? null,
      graphType: meta?.graph_type ?? null,
      jobSource: job?.job_source ?? null,
    }
  })

  // Check for special job types
  const isGraphEvolveVerifier = createMemo(() => {
    const key = registryKey()
    return key.trainingType === "graph_evolve" && key.graphType === "verifier"
  })

  // Get panel components from registries
  const DetailsPanel = createMemo(() => getDetailsPanel(registryKey()))
  const ResultsPanel = createMemo(() => getResultsPanel(registryKey()))
  const MetricsPanel = createMemo(() => getMetricsPanel(props.metricsView))

  // Calculate panel heights
  const metricPointsCount = createMemo(() => {
    const m: any = props.data.metrics || {}
    const pts = Array.isArray(m?.points) ? m.points : []
    return pts.length
  })

  const detailsHeight = createMemo(() => {
    // Details panel is typically 6-9 lines
    return clamp(7, 6, 9)
  })

  const resultsHeight = createMemo(() => {
    if (isGraphEvolveVerifier()) return 9
    return clamp(8, 6, 10)
  })

  const metricsPanelHeight = createMemo(() => {
    const detailsH = detailsHeight()
    const resultsH = resultsHeight()
    const minEventsH = 12
    const maxH = Math.max(4, props.detailHeight - (detailsH + resultsH + minEventsH))

    if (props.metricsView === "charts") {
      const desired = metricPointsCount() > 0 ? 22 : 18
      return clamp(desired, 12, maxH)
    }
    const desired = metricPointsCount() > 0 ? 8 : 4
    return clamp(desired, 4, maxH)
  })

  // Fixed height for events panel
  const eventsPanelHeight = createMemo(() => 14)

  // Results panel title varies by job type
  const resultsTitle = createMemo(() => {
    if (isGraphEvolveVerifier()) return "Reward"
    const job = props.data.selectedJob
    return job && isEvalJob(job) ? "Results" : "Summary"
  })

  // Determine if results panel needs special rendering (graph-evolve uses internal box)
  const resultsNeedsWrapper = createMemo(() => !isGraphEvolveVerifier())

  return (
    <box flexDirection="column" flexGrow={1} border={false} gap={0} overflow="hidden">
      {/* Details Panel */}
      <Dynamic
        component={DetailsPanel()}
        data={props.data}
        width={props.detailWidth}
        height={detailsHeight()}
      />

      {/* Results Panel */}
      <Show
        when={resultsNeedsWrapper()}
        fallback={
          <box
            border={PANEL.border}
            borderStyle={PANEL.borderStyle}
            borderColor={getPanelBorderColor(!!props.resultsFocused)}
            title={resultsTitle()}
            titleAlignment={PANEL.titleAlignment}
            paddingLeft={PANEL.paddingLeft}
            height={resultsHeight()}
          >
            <Dynamic
              component={ResultsPanel()}
              data={props.data}
              width={Math.max(30, props.detailWidth - 6)}
              height={Math.max(4, resultsHeight() - 2)}
              focused={!!props.resultsFocused}
              extra={{ selectedGenerationIndex: props.verifierEvolveGenerationIndex }}
            />
          </box>
        }
      >
        <Dynamic
          component={ResultsPanel()}
          data={props.data}
          width={props.detailWidth}
          height={resultsHeight()}
          focused={!!props.resultsFocused}
        />
      </Show>

      {/* Metrics Panel */}
      <Dynamic
        component={MetricsPanel()}
        metrics={props.data.metrics}
        width={props.detailWidth}
        height={metricsPanelHeight()}
        focused={!!props.metricsFocused}
      />

      {/* Events Panel */}
      <EventsListPanel
        eventItems={props.eventItems}
        totalEvents={props.totalEvents}
        selectedIndex={props.selectedIndex}
        focused={!!props.eventsFocused}
        width={props.detailWidth}
        height={eventsPanelHeight()}
        emptyFallback={<text fg={TEXT.fg}>No events yet.</text>}
      />

      <Show when={props.lastError}>
        <text fg={COLORS.error}>{`Error: ${props.lastError}`}</text>
      </Show>
    </box>
  )
}
