import { Show, createMemo } from "solid-js"
import { Dynamic } from "solid-js/web"

import type { AppData } from "../../../types"
import type { JobEvent } from "../../../tui_data"
import { isEvalJob } from "../../../tui_data"
import type { ListWindowItem } from "../../utils/list"
import type { PanelRegistryKey } from "./types"
import type { JobsDetailLayout } from "../../hooks/useJobsDetailLayout"

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
import { PromptDiffPanel } from "./PromptDiffPanel"

interface JobsDetailProps {
  data: AppData
  eventItems: ListWindowItem<JobEvent>[]
  totalEvents: number
  selectedIndex: number
  lastError: string | null
  detailWidth: number
  detailHeight: number
  scrollOffset: number
  layout: JobsDetailLayout
  detailsFocused?: boolean
  resultsFocused?: boolean
  promptDiffFocused?: boolean
  eventsFocused?: boolean
  metricsFocused?: boolean
  metricsView: MetricsView
  verifierEvolveGenerationIndex: number
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

  const sectionHeight = (id: keyof JobsDetailLayout["byId"], fallback: number) =>
    props.layout.byId[id]?.height ?? fallback
  const hasPromptDiff = createMemo(() => Boolean(props.layout.byId.promptDiff))

  // Results panel title varies by job type
  const resultsTitle = createMemo(() => {
    if (isGraphEvolveVerifier()) return "Reward"
    const job = props.data.selectedJob
    return job && isEvalJob(job) ? "Results" : "Summary"
  })

  // Determine if results panel needs special rendering (graph-evolve uses internal box)
  const resultsNeedsWrapper = createMemo(() => !isGraphEvolveVerifier())

  return (
    <box
      flexDirection="column"
      flexGrow={1}
      border={false}
      gap={0}
      overflow="hidden"
      height={props.detailHeight}
      position="relative"
    >
      <box
        flexDirection="column"
        gap={0}
        width={props.detailWidth}
        height={props.layout.contentHeight}
        position="absolute"
        left={0}
        top={-props.scrollOffset}
      >
        {/* Details Panel */}
        <Dynamic
          component={DetailsPanel()}
          data={props.data}
          width={props.detailWidth}
          height={sectionHeight("details", 7)}
          focused={!!props.detailsFocused}
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
              height={sectionHeight("results", 8)}
            >
              <Dynamic
                component={ResultsPanel()}
                data={props.data}
                width={Math.max(30, props.detailWidth - 6)}
                height={Math.max(4, sectionHeight("results", 8) - 2)}
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
            height={sectionHeight("results", 8)}
            focused={!!props.resultsFocused}
          />
        </Show>

        {/* Prompt Diff Panel */}
        <Show when={hasPromptDiff()}>
          <PromptDiffPanel
            data={props.data}
            width={props.detailWidth}
            height={sectionHeight("promptDiff", 8)}
            focused={!!props.promptDiffFocused}
          />
        </Show>

        {/* Metrics Panel */}
        <Dynamic
          component={MetricsPanel()}
          metrics={props.data.metrics}
          width={props.detailWidth}
          height={sectionHeight("metrics", 8)}
          focused={!!props.metricsFocused}
        />

        {/* Events Panel */}
        <EventsListPanel
          eventItems={props.eventItems}
          totalEvents={props.totalEvents}
          selectedIndex={props.selectedIndex}
          focused={!!props.eventsFocused}
          width={props.detailWidth}
          height={sectionHeight("events", 14)}
          emptyFallback={<text fg={TEXT.fg}>No events yet.</text>}
        />

        <Show when={props.lastError}>
          <text fg={COLORS.error}>{`Error: ${props.lastError}`}</text>
        </Show>
      </box>
    </box>
  )
}
