import type { Accessor } from "solid-js"
import type { AppContext } from "../../context"
import type { AppState } from "../../state/app-state"
import type { JobsDetailState } from "./useJobsDetailState"
import type { JobsDetailSectionId } from "./useJobsDetailLayout"
import type { JobsListState } from "./useJobsListState"
import type { LogsDetailState } from "./useLogsDetailState"
import type { LogsListState } from "./useLogsListState"
import type { SessionsListState } from "./useSessionsListState"
import type { JobEvent } from "../../tui_data"
import { focusManager } from "../../focus"
import { log } from "../../utils/log"
import type { PrimaryView } from "../../types"
import { moveEventSelection } from "../../utils/events"
import {
  getBestVerifierEvolveGenerationIndex,
  getSelectedVerifierEvolveGeneration,
  moveVerifierEvolveGenerationSelection,
} from "../../utils/verifier-evolve"
import { shouldShowPromptDiffPanel } from "../ui/detail-panels/PromptDiffPanel"

type UseFocusBindingsOptions = {
  ctx: AppContext
  ui: AppState
  primaryView: Accessor<PrimaryView>
  jobsList: JobsListState
  logsList: LogsListState
  sessionsList: SessionsListState
  jobsDetail: JobsDetailState
  logsDetail: LogsDetailState
  scrollJobsDetailBy: (delta: number) => boolean
  ensureJobsDetailSectionVisible: (id: JobsDetailSectionId) => void
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
        const view = options.primaryView()
        const beforeIndex =
          view === "jobs"
            ? options.jobsList.selectedIndex()
            : view === "logs"
              ? options.logsList.selectedIndex()
              : options.sessionsList.selectedIndex()
        const listSize =
          view === "jobs"
            ? options.jobsList.totalCount()
            : view === "logs"
              ? options.logsList.totalCount()
              : options.sessionsList.totalCount()
        let moved = false
        if (view === "jobs") {
          moved = options.jobsList.moveSelection(delta)
        } else if (view === "logs") {
          moved = options.logsList.liveLogs.moveSelection(delta)
        } else {
          moved = options.sessionsList.moveSelection(delta)
        }
        const afterIndex =
          view === "jobs"
            ? options.jobsList.selectedIndex()
            : view === "logs"
              ? options.logsList.selectedIndex()
              : options.sessionsList.selectedIndex()
        log("action", "list.nav", {
          view,
          delta,
          moved,
          beforeIndex,
          afterIndex,
          listSize,
        })
        return true
      }
      if (action === "pane.select") {
        if (options.primaryView() === "jobs") {
          options.jobsList.selectCurrent()
        } else if (options.primaryView() === "logs") {
          const file = options.logsList.selectedFile()
          if (file) {
            void options.openLogModal(file.path)
          }
        } else {
          if (options.sessionsList.selectCurrent()) {
            focusManager.setFocus("agent")
          }
        }
        return true
      }
      return false
    },
  })

  focusManager.register({
    id: "agent",
    order: 1,
    enabled: () => options.primaryView() === "agent",
  })

  focusManager.register({
    id: "conversation",
    order: 2,
    enabled: () => options.primaryView() === "agent",
  })

  focusManager.register({
    id: "principal",
    order: 3,
    enabled: () => options.primaryView() !== "agent",
    onBlur: () => {
      if (options.primaryView() === "logs") {
        options.logsDetail.onBlur()
      }
    },
    onFocus: () => {
      if (options.primaryView() === "logs") {
        options.logsDetail.onFocus()
      }
    },
    handleAction: (action) => {
      if (action === "nav.down" || action === "nav.up") {
        const delta = action === "nav.down" ? 1 : -1
        if (options.primaryView() === "jobs") {
          options.scrollJobsDetailBy(delta)
        } else if (options.primaryView() === "logs") {
          options.logsDetail.scrollBy(delta)
        }
        return true
      }
      if (action === "pane.select" && options.primaryView() === "logs") {
        options.logsDetail.jumpToTail()
        return true
      }
      return false
    },
  })

  focusManager.register({
    id: "details",
    order: 4,
    enabled: () => options.primaryView() === "jobs",
    onFocus: () => options.ensureJobsDetailSectionVisible("details"),
    handleAction: (action) => {
      if (action === "nav.down" || action === "nav.up") {
        const delta = action === "nav.down" ? 1 : -1
        options.scrollJobsDetailBy(delta)
        return true
      }
      return false
    },
  })

  focusManager.register({
    id: "results",
    order: 5,
    enabled: () => options.primaryView() === "jobs",
    onFocus: () => {
      options.ensureJobsDetailSectionVisible("results")
      if (!isVerifierEvolveJob()) return
      const bestIndex = getBestVerifierEvolveGenerationIndex(options.ctx.state.data)
      if (bestIndex == null) return
      if (options.ui.verifierEvolveGenerationIndex === 0) {
        options.ctx.setUi("verifierEvolveGenerationIndex", bestIndex)
      }
    },
    handleAction: (action) => {
      if (action === "nav.down" || action === "nav.up") {
        const delta = action === "nav.down" ? 1 : -1
        if (isVerifierEvolveJob()) {
          moveVerifierEvolveGenerationSelection(options.ctx, delta)
        } else {
          options.scrollJobsDetailBy(delta)
        }
        return true
      }
      if (action === "pane.select") {
        if (!isVerifierEvolveJob()) return false
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
    id: "promptDiff",
    order: 6,
    enabled: () =>
      options.primaryView() === "jobs" && shouldShowPromptDiffPanel(options.ctx.state.data),
    onFocus: () => options.ensureJobsDetailSectionVisible("promptDiff"),
    handleAction: (action) => {
      if (action === "nav.down" || action === "nav.up") {
        const delta = action === "nav.down" ? 1 : -1
        options.scrollJobsDetailBy(delta)
        return true
      }
      return false
    },
  })

  focusManager.register({
    id: "metrics",
    order: 7,
    enabled: () => options.primaryView() === "jobs",
    onFocus: () => options.ensureJobsDetailSectionVisible("metrics"),
    handleAction: (action) => {
      if (action === "nav.down" || action === "nav.up") {
        const delta = action === "nav.down" ? 1 : -1
        options.scrollJobsDetailBy(delta)
        return true
      }
      if (action === "pane.select") {
        options.openMetricsAndFetch()
        return true
      }
      return false
    },
  })

  focusManager.register({
    id: "events",
    order: 8,
    enabled: () => options.primaryView() === "jobs",
    onFocus: () => options.ensureJobsDetailSectionVisible("events"),
    handleAction: (action) => {
      if (action === "nav.down" || action === "nav.up") {
        const delta = action === "nav.down" ? 1 : -1
        moveEventSelection(options.ctx, delta)
        return true
      }
      if (action === "pane.select") {
        const event = options.jobsDetail.events()[options.jobsDetail.selectedIndex()]
        if (event) {
          options.openEventModal(event)
        }
        return true
      }
      return false
    },
  })

}
