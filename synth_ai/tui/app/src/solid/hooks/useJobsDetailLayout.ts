import { createMemo, type Accessor } from "solid-js"

import type { AppData } from "../../types"
import { getPanelContentWidth } from "../../utils/panel"
import type { DetailsPanelProps } from "../ui/detail-panels/details/types"
import type { MetricsPanelProps } from "../ui/detail-panels/metrics/types"
import type { ResultsPanelProps } from "../ui/detail-panels/results/types"
import {
  getDetailsPanel,
} from "../ui/detail-panels/registries/details-registry"
import {
  getMetricsPanel,
  type MetricsView,
} from "../ui/detail-panels/registries/metrics-registry"
import {
  getResultsPanel,
} from "../ui/detail-panels/registries/results-registry"
import type { PanelRegistryKey, TextPanelComponent } from "../ui/detail-panels/types"
import { PromptDiffPanel, shouldShowPromptDiffPanel } from "../ui/detail-panels/PromptDiffPanel"
import "../ui/detail-panels/details"
import "../ui/detail-panels/results"
import "../ui/detail-panels/metrics"

export type JobsDetailSectionId =
  | "details"
  | "results"
  | "promptDiff"
  | "metrics"
  | "events"
  | "error"

export type JobsDetailSectionLayout = {
  id: JobsDetailSectionId
  height: number
  top: number
}

export type JobsDetailLayout = {
  sections: JobsDetailSectionLayout[]
  byId: Partial<Record<JobsDetailSectionId, JobsDetailSectionLayout>>
  contentHeight: number
  resultsInteractive: boolean
}

type UseJobsDetailLayoutOptions = {
  data: AppData
  detailWidth: Accessor<number>
  metricsView: Accessor<MetricsView>
  lastError: Accessor<string | null>
}

const DEFAULT_DETAILS_HEIGHT = 7
const DEFAULT_RESULTS_HEIGHT = 8
const DEFAULT_PROMPT_DIFF_HEIGHT = 8
const GRAPH_EVOLVE_RESULTS_HEIGHT = 9
const EVENTS_PANEL_HEIGHT = 14
const METRICS_LATEST_WITH_POINTS = 8
const METRICS_LATEST_EMPTY = 4
const METRICS_CHARTS_WITH_POINTS = 22
const METRICS_CHARTS_EMPTY = 18

function clampHeight(value: number, min = 3): number {
  return Math.max(min, Math.floor(value))
}

export function useJobsDetailLayout(options: UseJobsDetailLayoutOptions): Accessor<JobsDetailLayout> {
  return createMemo(() => {
    const width = Math.max(1, Math.floor(options.detailWidth()))
    const contentWidth = getPanelContentWidth(width)
    const job = options.data.selectedJob
    const meta = (job?.metadata as Record<string, any> | null) ?? null
    const registryKey: PanelRegistryKey = {
      trainingType: job?.training_type ?? null,
      graphType: meta?.graph_type ?? null,
      jobSource: job?.job_source ?? null,
    }
    const isGraphEvolveVerifier =
      registryKey.trainingType === "graph_evolve" && registryKey.graphType === "verifier"

    const metricPoints = Array.isArray((options.data.metrics as any)?.points)
      ? (options.data.metrics as any).points
      : []
    const metricPointsCount = metricPoints.length

    const DetailsPanel = getDetailsPanel(registryKey) as TextPanelComponent<DetailsPanelProps>
    const ResultsPanel = getResultsPanel(registryKey) as TextPanelComponent<ResultsPanelProps>
    const MetricsPanel = getMetricsPanel(options.metricsView()) as TextPanelComponent<MetricsPanelProps>
    const PromptDiff = PromptDiffPanel as TextPanelComponent<{ data: AppData; width: number; height: number }>

    const detailsLines = DetailsPanel.getLines
      ? DetailsPanel.getLines({ data: options.data, width, height: 0 }, contentWidth)
      : null
    const resultsLines = ResultsPanel.getLines
      ? ResultsPanel.getLines({ data: options.data, width, height: 0, focused: false }, contentWidth)
      : null
    const promptDiffEnabled = shouldShowPromptDiffPanel(options.data)
    const promptDiffLines = promptDiffEnabled && PromptDiff.getLines
      ? PromptDiff.getLines({ data: options.data, width, height: 0 }, contentWidth)
      : null
    const metricsLines = MetricsPanel.getLines
      ? MetricsPanel.getLines({ metrics: options.data.metrics, width, height: 0, focused: false }, contentWidth)
      : null

    const detailsHeight = detailsLines
      ? clampHeight(detailsLines.length + 2)
      : DEFAULT_DETAILS_HEIGHT

    const resultsHeight = resultsLines
      ? clampHeight(resultsLines.length + 2)
      : (isGraphEvolveVerifier ? GRAPH_EVOLVE_RESULTS_HEIGHT : DEFAULT_RESULTS_HEIGHT)

    const promptDiffHeight = promptDiffLines
      ? clampHeight(promptDiffLines.length + 2)
      : DEFAULT_PROMPT_DIFF_HEIGHT

    const metricsHeight = metricsLines
      ? clampHeight(metricsLines.length + 2)
      : clampHeight(
          options.metricsView() === "charts"
            ? (metricPointsCount > 0 ? METRICS_CHARTS_WITH_POINTS : METRICS_CHARTS_EMPTY)
            : (metricPointsCount > 0 ? METRICS_LATEST_WITH_POINTS : METRICS_LATEST_EMPTY),
        )

    const sections: JobsDetailSectionLayout[] = []
    const byId: Partial<Record<JobsDetailSectionId, JobsDetailSectionLayout>> = {}
    let cursor = 0

    const pushSection = (id: JobsDetailSectionId, height: number) => {
      const section = { id, height, top: cursor }
      sections.push(section)
      byId[id] = section
      cursor += height
    }

    pushSection("details", detailsHeight)
    pushSection("results", resultsHeight)
    if (promptDiffEnabled) {
      pushSection("promptDiff", promptDiffHeight)
    }
    pushSection("metrics", metricsHeight)
    pushSection("events", EVENTS_PANEL_HEIGHT)

    if (options.lastError()) {
      pushSection("error", 1)
    }

    return {
      sections,
      byId,
      contentHeight: cursor,
      resultsInteractive: isGraphEvolveVerifier,
    }
  })
}
