import { useKeyboard } from "@opentui/solid"
import path from "node:path"
import type { Accessor, Component, Setter } from "solid-js"
import type { SetStoreFunction } from "solid-js/store"

import { getTextInput, matchAction, type KeyEvent } from "../../input/keymap"
import { log, logKey } from "../../utils/log"
import { focusManager, setListPane } from "../../focus"
import { ListPane } from "../../types"
import type { AppState } from "../../state/app-state"
import { shutdown } from "../../lifecycle"
import { readLogFile } from "../utils/logs"
import { copyToClipboard } from "../../utils/clipboard"
import type { AppData } from "../../types"
import type { ActiveModal, ModalState } from "../modals/types"

type KeyEventWithPrevent = KeyEvent & { preventDefault?: () => void }

type UseAppKeybindingsOptions = {
  onExit?: () => void
  data: AppData
  ui: AppState
  setUi: SetStoreFunction<AppState>
  setData: SetStoreFunction<AppData>
  modal: Accessor<ModalState | null>
  setModal: Setter<ModalState | null>
  activeModal: Accessor<ActiveModal | null>
  showCreateJobModal: Accessor<boolean>
  setShowCreateJobModal: Setter<boolean>
  createJobModalComponent: Accessor<Component<any> | null>
  chatPaneComponent: Accessor<Component<any> | null>
  closeActiveModal: () => void
  applyFilterModal: () => void
  applySnapshotModal: () => Promise<void>
  moveSettingsCursor: (delta: number) => void
  selectSettingsBackend: () => Promise<void>
  openUsageBilling: () => void
  moveTaskAppsSelection: (delta: number) => void
  copySelectedTunnelUrl: () => Promise<void> | void
  moveSessionsSelection: (delta: number) => void
  copySelectedSessionUrl: () => Promise<void> | void
  connectLocalSession: () => Promise<void> | void
  disconnectSelectedSession: () => Promise<void> | void
  refreshSessionsModal: () => Promise<void> | void
  selectSession: () => void
  moveListFilter: (delta: number) => void
  toggleListFilterSelection: () => void
  selectAllListFilterSelection: () => void
  startLoginAuth: () => Promise<void>
  openFilterModal: () => void
  openConfigModal: () => void
  openProfileModal: () => void
  openResultsModal: () => void
  openListFilterModal: () => void
  openSnapshotModal: () => void
  openUrlsModal: () => void
  openSettingsModal: () => void
  openUsageModal: () => void
  openTaskAppsModal: () => void
  openSessionsModal: () => void
  openMetricsAndFetch: () => void
  openTracesModal: () => void
  ensureCreateJobModal: () => void
  logout: () => Promise<void>
  runAction: (key: string, task: () => Promise<void>) => void
  refreshData: () => Promise<boolean>
  ensureOpenCodeServer: () => Promise<void>
  cancelSelectedJob: () => Promise<void>
  fetchArtifacts: () => Promise<void>
  refreshMetrics: () => Promise<void>
  loadMoreJobs: () => Promise<void>
}

export function useAppKeybindings(options: UseAppKeybindingsOptions): void {
  function handleGlobalBack(evt: KeyEventWithPrevent): void {
    evt.preventDefault?.()

    if (options.showCreateJobModal()) {
      const handler = (options.createJobModalComponent() as any)?.handleKeyPress
      const handled = handler ? handler(evt) : false
      if (handled) return
      options.setShowCreateJobModal(false)
      return
    }

    if (options.activeModal()) {
      options.closeActiveModal()
      return
    }

    if (options.modal()) {
      options.setModal(null)
      return
    }

    if (options.ui.principalPane === "opencode") {
      const chatBack = (options.chatPaneComponent() as any)?.handleBack
      if (typeof chatBack === "function" && chatBack()) {
        return
      }
      options.setUi("principalPane", "jobs")
      return
    }
  }

  useKeyboard((evt: KeyEventWithPrevent) => {
    const currentContext = options.activeModal()
      ? `modal:${options.activeModal()}`
      : options.modal()
        ? `detail:${options.modal()?.type}`
        : options.showCreateJobModal()
          ? "createJob"
          : options.ui.principalPane === "opencode"
            ? "opencode"
            : options.ui.activePane // "jobs" or "logs"
    logKey(evt, currentContext)

    const globalAction = matchAction(evt, "app.global")
    if (globalAction) {
      log("action",globalAction, "app.global")
    }
    switch (globalAction) {
      case "app.forceQuit":
        evt.preventDefault?.()
        options.onExit?.()
        void shutdown(0)
        return
      case "app.back":
        handleGlobalBack(evt)
        return
      case "app.quit":
        evt.preventDefault?.()
        options.onExit?.()
        void shutdown(0)
        return
      default:
        break
    }

    const overlayModal = options.activeModal()
    if (overlayModal) {
      const context = (() => {
        switch (overlayModal) {
          case "filter":
            return "modal.filter"
          case "snapshot":
            return "modal.snapshot"
          case "settings":
            return "modal.settings"
          case "usage":
            return "modal.usage"
          case "metrics":
            return "modal.metrics"
          case "task-apps":
            return "modal.taskApps"
          case "sessions":
            return "modal.sessions"
          case "list-filter":
            return "modal.listFilter"
          case "config":
            return "modal.config"
          case "profile":
            return "modal.profile"
          case "urls":
            return "modal.urls"
          case "login":
            return "modal.login"
          default:
            return null
        }
      })()

      if (context) {
        const action = matchAction(evt, context)
        if (action) {
          log("action",action, context)
          evt.preventDefault?.()
          switch (overlayModal) {
            case "filter":
              switch (action) {
                case "modal.confirm":
                  options.applyFilterModal()
                  return
                default:
                  return
              }
            case "snapshot":
              switch (action) {
                case "modal.confirm":
                  options.runAction("snapshot", () => options.applySnapshotModal())
                  return
                default:
                  return
              }
            case "settings":
              switch (action) {
                case "nav.up":
                  options.moveSettingsCursor(-1)
                  return
                case "nav.down":
                  options.moveSettingsCursor(1)
                  return
                case "modal.confirm":
                  void options.selectSettingsBackend()
                  return
                default:
                  return
              }
            case "usage":
              switch (action) {
                case "usage.openBilling":
                  options.openUsageBilling()
                  return
                case "nav.up":
                  options.setUi("usageModalOffset", Math.max(0, (options.ui.usageModalOffset || 0) - 1))
                  return
                case "nav.down":
                  options.setUi("usageModalOffset", (options.ui.usageModalOffset || 0) + 1)
                  return
                case "modal.confirm":
                  options.closeActiveModal()
                  return
                default:
                  return
              }
            case "metrics":
              switch (action) {
                case "nav.up":
                  options.setUi("metricsModalOffset", Math.max(0, (options.ui.metricsModalOffset || 0) - 1))
                  return
                case "nav.down":
                  options.setUi("metricsModalOffset", (options.ui.metricsModalOffset || 0) + 1)
                  return
                case "metrics.refresh":
                  options.runAction("metrics", () => options.refreshMetrics())
                  return
                case "modal.confirm":
                  options.closeActiveModal()
                  return
                default:
                  return
              }
            case "task-apps":
              switch (action) {
                case "nav.up":
                  options.moveTaskAppsSelection(-1)
                  return
                case "nav.down":
                  options.moveTaskAppsSelection(1)
                  return
                case "modal.copy":
                  void options.copySelectedTunnelUrl()
                  return
                case "modal.confirm":
                  options.closeActiveModal()
                  return
                default:
                  return
              }
            case "sessions":
              switch (action) {
                case "nav.up":
                  options.moveSessionsSelection(-1)
                  return
                case "nav.down":
                  options.moveSessionsSelection(1)
                  return
                case "sessions.copy":
                  void options.copySelectedSessionUrl()
                  return
                case "sessions.connect":
                  options.runAction("sessions-connect", async () => {
                    await options.connectLocalSession()
                  })
                  return
                case "sessions.disconnect":
                  options.runAction("sessions-disconnect", async () => {
                    await options.disconnectSelectedSession()
                  })
                  return
                case "sessions.refresh":
                  options.runAction("sessions-refresh", async () => {
                    await options.refreshSessionsModal()
                  })
                  return
                case "modal.confirm":
                  options.selectSession()
                  return
                default:
                  return
              }
            case "list-filter":
              switch (action) {
                case "nav.up":
                  options.moveListFilter(-1)
                  return
                case "nav.down":
                  options.moveListFilter(1)
                  return
                case "listFilter.toggle":
                  options.toggleListFilterSelection()
                  return
                case "listFilter.all":
                  options.selectAllListFilterSelection()
                  return
                default:
                  return
              }
            case "config":
              switch (action) {
                case "nav.up":
                  options.setUi("configModalOffset", Math.max(0, options.ui.configModalOffset - 1))
                  return
                case "nav.down":
                  options.setUi("configModalOffset", options.ui.configModalOffset + 1)
                  return
                case "modal.confirm":
                  options.closeActiveModal()
                  return
                default:
                  return
              }
            case "profile":
            case "urls":
              switch (action) {
                case "modal.confirm":
                  options.closeActiveModal()
                  return
                default:
                  return
              }
            case "login":
              switch (action) {
                case "login.confirm":
                  void options.startLoginAuth()
                  return
                default:
                  return
              }
            default:
              return
          }
        }
      }
    }

    if (!overlayModal) {
      const detailModal = options.modal()
      if (detailModal) {
        const action = matchAction(evt, "modal.detail")
        if (action) {
          evt.preventDefault?.()
          switch (action) {
            case "detail.toggleFullscreen":
              options.setModal({ ...detailModal, fullscreen: !detailModal.fullscreen })
              return
            case "modal.confirm":
              options.setModal(null)
              return
            case "nav.down":
              if (detailModal.type === "log") {
                options.setModal({ ...detailModal, offset: detailModal.offset + 1, tail: false })
              } else {
                options.setModal({ ...detailModal, offset: detailModal.offset + 1 })
              }
              return
            case "nav.up":
              if (detailModal.type === "log") {
                options.setModal({ ...detailModal, offset: detailModal.offset - 1, tail: false })
              } else {
                options.setModal({ ...detailModal, offset: detailModal.offset - 1 })
              }
              return
            case "detail.tail":
              if (detailModal.type === "log") {
                options.setModal({ ...detailModal, tail: true })
              }
              return
            case "modal.copy":
              if (detailModal.type === "log") {
                void readLogFile(detailModal.path)
                  .then((raw) => copyToClipboard(raw))
                  .then(() => {
                    options.setData("status", `Copied: ${path.basename(detailModal.path)}`)
                  })
              }
              return
            default:
              return
          }
        }
      }
    }

    if (options.showCreateJobModal()) {
      const handler = (options.createJobModalComponent() as any)?.handleKeyPress
      const handled = handler ? handler(evt) : true
      if (handled) return
    }

    if (options.ui.principalPane === "opencode") {
      const action = matchAction(evt, "app.opencode")
      if (action) {
        evt.preventDefault?.()
        switch (action) {
          case "pane.openSessions":
            options.openSessionsModal()
            return
          default:
            return
        }
      }
      if (options.ui.focusTarget === "agent") {
        return
      }
    }

    const action = matchAction(evt, "app.global")
    if (!action) return

    if ((options.activeModal() || options.modal()) && !action.startsWith("modal.open.")) {
      return
    }

    const agentFocused = options.ui.principalPane === "opencode" && options.ui.focusTarget === "agent"
    const isNavAction = action === "nav.down" || action === "nav.up" || action === "pane.select"
    if (agentFocused && (isNavAction || getTextInput(evt))) {
      return
    }

    evt.preventDefault?.()

    switch (action) {
      case "focus.next":
      case "focus.prev": {
        focusManager.route(action)
        return
      }
      default:
        break
    }

    if (focusManager.route(action)) {
      return
    }

    switch (action) {
      case "app.refresh":
        options.runAction("refresh", async () => { await options.refreshData() })
        return
      case "jobs.loadMore":
        if (options.ui.activePane !== ListPane.Jobs || options.ui.focusTarget !== "list") {
          return
        }
        options.runAction("jobs-load-more", () => options.loadMoreJobs())
        return
      case "pane.jobs":
        setListPane(ListPane.Jobs)
        return
      case "pane.logs":
        setListPane(ListPane.Logs)
        return
      case "pane.togglePrincipal": {
        const nextPane = options.ui.principalPane === "jobs" ? "opencode" : "jobs"
        options.setUi("principalPane", nextPane)
        if (nextPane === "opencode") {
          options.runAction("opencode-ensure", () => options.ensureOpenCodeServer())
          focusManager.setFocus("agent")
        } else {
          focusManager.ensureValid()
        }
        return
      }
      case "app.logout":
        void options.logout()
        return
      case "modal.open.filter":
        options.openFilterModal()
        return
      case "modal.open.config":
        options.openConfigModal()
        return
      case "modal.open.profile":
        if (!process.env.SYNTH_API_KEY) return
        options.openProfileModal()
        return
      case "modal.open.results":
        options.openResultsModal()
        return
      case "modal.open.listFilter":
        options.openListFilterModal()
        return
      case "modal.open.snapshot":
        options.openSnapshotModal()
        return
      case "modal.open.urls":
        options.openUrlsModal()
        return
      case "modal.open.settings":
        options.openSettingsModal()
        return
      case "modal.open.usage":
        options.openUsageModal()
        return
      case "modal.open.taskApps":
        options.openTaskAppsModal()
        return
      case "modal.open.createJob":
        options.setShowCreateJobModal(true)
        void options.ensureCreateJobModal()
        return
      case "modal.open.metrics":
        options.openMetricsAndFetch()
        return
      case "modal.open.traces":
        if (options.data.selectedJob) {
          options.openTracesModal()
        }
        return
      case "job.cancel":
        options.runAction("cancel-job", () => options.cancelSelectedJob())
        return
      case "job.artifacts":
        options.runAction("artifacts", () => options.fetchArtifacts())
        return
      default:
        return
    }
  })
}
