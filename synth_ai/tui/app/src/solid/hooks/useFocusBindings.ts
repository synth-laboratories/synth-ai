import type { AppContext } from "../../context"
import type { AppState } from "../../state/app-state"
import type { JobsDetailState } from "./useJobsDetailState"
import type { JobsListState } from "./useJobsListState"
import type { LogsDetailState } from "./useLogsDetailState"
import type { LogsListState } from "./useLogsListState"
import type { JobEvent } from "../../tui_data"
import { focusManager } from "../../focus"
import { ListPane } from "../../types"
import { moveEventSelection } from "../utils/events"
import { getSelectedVerifierEvolveGeneration, moveVerifierEvolveGenerationSelection } from "../utils/verifier-evolve"

type UseFocusBindingsOptions = {
  ctx: AppContext
  ui: AppState
  jobsList: JobsListState
  logsList: LogsListState
  jobsDetail: JobsDetailState
  logsDetail: LogsDetailState
  openEventModal: (event: JobEvent) => void
  openLogModal: (filePath: string) => Promise<void>
  openMetricsAndFetch: () => void
  openCandidatesForGeneration: (generation: number) => void
}

export function useFocusBindings(options: UseFocusBindingsOptions): void {
  const isVerifierEvolveJob = () => {
    const job = options.ctx.state.data.selectedJob
    const meta = job?.metadata
    const graphType = meta && typeof meta === "object" ? (meta as any).graph_type : null
    return job?.training_type === "graph_evolve" && graphType === "verifier"
  }

  focusManager.register({
    id: "list",
    order: 0,
    enabled: () => true,
    handleAction: (action) => {
      if (action === "nav.down" || action === "nav.up") {
        const delta = action === "nav.down" ? 1 : -1
        if (options.ui.activePane === ListPane.Jobs) {
          options.jobsList.moveSelection(delta)
        } else if (options.ui.activePane === ListPane.Logs) {
          options.logsList.liveLogs.moveSelection(delta)
        }
        return true
      }
      if (action === "pane.select") {
        if (options.ui.activePane === ListPane.Jobs) {
          options.jobsList.selectCurrent()
        } else if (options.ui.activePane === ListPane.Logs) {
          const file = options.logsList.selectedFile()
          if (file) {
            void options.openLogModal(file.path)
          }
        }
        return true
      }
      return false
    },
  })

  focusManager.register({
    id: "results",
    order: 1,
    enabled: () =>
      options.ui.principalPane === "jobs" &&
      options.ui.activePane === ListPane.Jobs &&
      isVerifierEvolveJob(),
    handleAction: (action) => {
      if (action === "nav.down" || action === "nav.up") {
        const delta = action === "nav.down" ? 1 : -1
        moveVerifierEvolveGenerationSelection(options.ctx, delta)
        return true
      }
      if (action === "pane.select") {
        const generation = getSelectedVerifierEvolveGeneration(
          options.ctx.state.data,
          options.ui.verifierEvolveGenerationIndex,
        )
        if (generation != null) {
          options.openCandidatesForGeneration(generation)
        }
        return true
      }
      return false
    },
  })

  focusManager.register({
    id: "metrics",
    order: 2,
    enabled: () => options.ui.principalPane === "jobs" && options.ui.activePane === ListPane.Jobs,
    handleAction: (action) => {
      if (action === "pane.select") {
        options.openMetricsAndFetch()
        return true
      }
      return false
    },
  })

  focusManager.register({
    id: "events",
    order: 3,
    enabled: () => options.ui.principalPane === "jobs" && options.ui.activePane === ListPane.Jobs,
    handleAction: (action) => {
      if (action === "nav.down" || action === "nav.up") {
        const delta = action === "nav.down" ? 1 : -1
        moveEventSelection(options.ctx, delta)
        return true
      }
      if (action === "pane.select") {
        const event = options.jobsDetail.events()[options.jobsDetail.eventWindow().selected]
        if (event) {
          options.openEventModal(event)
        }
        return true
      }
      return false
    },
  })

  focusManager.register({
    id: "agent",
    order: 4,
    enabled: () => options.ui.principalPane === "opencode",
  })

  focusManager.register({
    id: "logs-detail",
    order: 5,
    enabled: () => options.ui.principalPane === "jobs" && options.ui.activePane === ListPane.Logs,
    onBlur: () => options.logsDetail.onBlur(),
    onFocus: () => options.logsDetail.onFocus(),
    handleAction: (action) => {
      if (action === "nav.down" || action === "nav.up") {
        const delta = action === "nav.down" ? 1 : -1
        options.logsDetail.scrollBy(delta)
        return true
      }
      if (action === "pane.select") {
        options.logsDetail.jumpToTail()
        return true
      }
      return false
    },
  })
}
