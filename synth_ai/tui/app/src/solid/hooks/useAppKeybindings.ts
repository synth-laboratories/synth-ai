import { useKeyboard } from "@opentui/solid"
import path from "node:path"
import type { Accessor, Component, Setter } from "solid-js"

import { getTextInput, matchAction, type KeyEvent } from "../../input/keymap"
import { focusManager, setListPane } from "../../focus"
import { appState } from "../../state/app-state"
import { shutdown } from "../../lifecycle"
import { readLogFile } from "../utils/logs"
import { copyToClipboard } from "../../utils/clipboard"
import type { Snapshot } from "../../types"
import type { ActiveModal, ModalState } from "../modals/types"

type KeyEventWithPrevent = KeyEvent & { preventDefault?: () => void }

type UseAppKeybindingsOptions = {
  onExit?: () => void
  render: () => void
  snapshot: Snapshot
  modal: Accessor<ModalState | null>
  setModal: Setter<ModalState | null>
  activeModal: Accessor<ActiveModal | null>
  setActiveModal: Setter<ActiveModal | null>
  showCreateJobModal: Accessor<boolean>
  setShowCreateJobModal: Setter<boolean>
  createJobModalComponent: Accessor<Component<any> | null>
  chatPaneComponent: Accessor<Component<any> | null>
  closeActiveModal: () => void
  applyFilterModal: () => void
  applySnapshotModal: () => Promise<void>
  applyKeyModal: () => void
  pasteKeyModal: () => Promise<void>
  moveSettingsCursor: (delta: number) => void
  selectSettingsBackend: () => Promise<void>
  openKeyModal: () => void
  openUsageBilling: () => void
  moveTaskAppsSelection: (delta: number) => void
  copySelectedTunnelUrl: () => Promise<void> | void
  moveSessionsSelection: (delta: number) => void
  copySelectedSessionUrl: () => Promise<void> | void
  connectLocalSession: () => Promise<void> | void
  disconnectSelectedSession: () => Promise<void> | void
  refreshSessionsModal: () => Promise<void> | void
  selectSession: () => void
  moveJobFilter: (delta: number) => void
  toggleJobFilterSelection: () => void
  clearJobFilterSelection: () => void
  startLoginAuth: () => Promise<void>
  openFilterModal: () => void
  openConfigModal: () => void
  openProfileModal: () => void
  openResultsModal: () => void
  openJobFilterModal: () => void
  openSnapshotModal: () => void
  openUrlsModal: () => void
  openSettingsModal: () => void
  openUsageModal: () => void
  openTaskAppsModal: () => void
  openSessionsModal: () => void
  openMetricsAndFetch: () => void
  ensureCreateJobModal: () => void
  ensureTraceViewerModal: () => void
  logout: () => Promise<void>
  runAction: (key: string, task: () => Promise<void>) => void
  refreshData: () => Promise<void>
  ensureOpenCodeServer: () => Promise<void>
  cancelSelectedJob: () => Promise<void>
  fetchArtifacts: () => Promise<void>
  refreshMetrics: () => Promise<void>
}

export function useAppKeybindings(options: UseAppKeybindingsOptions): void {
  function handleGlobalBack(evt: KeyEventWithPrevent): void {
    evt.preventDefault?.()

    if (options.showCreateJobModal()) {
      const handler = (options.createJobModalComponent() as any)?.handleKeyPress
      const handled = handler ? handler(evt) : false
      if (handled) {
        options.render()
        return
      }
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

    if (appState.principalPane === "opencode") {
      const chatBack = (options.chatPaneComponent() as any)?.handleBack
      if (typeof chatBack === "function" && chatBack()) {
        return
      }
      appState.principalPane = "jobs"
      options.render()
      return
    }
  }

  useKeyboard((evt: KeyEventWithPrevent) => {
    const globalAction = matchAction(evt, "app.global")
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

    const detailModal = options.modal()
    if (detailModal) {
      const action = matchAction(evt, "modal.detail")
      if (!action) return
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
                options.snapshot.status = `Copied: ${path.basename(detailModal.path)}`
                options.render()
              })
          }
          return
        default:
          return
      }
    }

    const overlayModal = options.activeModal()
    if (overlayModal) {
      const context = (() => {
        switch (overlayModal) {
          case "filter":
            return "modal.filter"
          case "snapshot":
            return "modal.snapshot"
          case "key":
            return "modal.key"
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
          case "job-filter":
            return "modal.jobFilter"
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

      if (!context) return
      const action = matchAction(evt, context)
      if (!action) return
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
        case "key":
          switch (action) {
            case "modal.confirm":
              options.applyKeyModal()
              return
            case "modal.paste":
              void options.pasteKeyModal()
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
            case "settings.openKey":
              options.closeActiveModal()
              options.openKeyModal()
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
              appState.usageModalOffset = Math.max(0, (appState.usageModalOffset || 0) - 1)
              options.render()
              return
            case "nav.down":
              appState.usageModalOffset = (appState.usageModalOffset || 0) + 1
              options.render()
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
              appState.metricsModalOffset = Math.max(0, (appState.metricsModalOffset || 0) - 1)
              options.render()
              return
            case "nav.down":
              appState.metricsModalOffset = (appState.metricsModalOffset || 0) + 1
              options.render()
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
        case "job-filter":
          switch (action) {
            case "nav.up":
              options.moveJobFilter(-1)
              return
            case "nav.down":
              options.moveJobFilter(1)
              return
            case "jobFilter.toggle":
              options.toggleJobFilterSelection()
              return
            case "jobFilter.clear":
              options.clearJobFilterSelection()
              return
            default:
              return
          }
        case "config":
          switch (action) {
            case "nav.up":
              appState.configModalOffset = Math.max(0, appState.configModalOffset - 1)
              options.render()
              return
            case "nav.down":
              appState.configModalOffset = appState.configModalOffset + 1
              options.render()
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

    if (options.showCreateJobModal()) {
      const handler = (options.createJobModalComponent() as any)?.handleKeyPress
      const handled = handler ? handler(evt) : true
      if (handled) {
        options.render()
        return
      }
    }

    if (appState.principalPane === "opencode") {
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
      if (appState.focusTarget === "agent") {
        return
      }
    }

    const action = matchAction(evt, "app.global")
    if (!action) return

    const agentFocused = appState.principalPane === "opencode" && appState.focusTarget === "agent"
    const isNavAction = action === "nav.down" || action === "nav.up" || action === "pane.select"
    if (agentFocused && (isNavAction || getTextInput(evt))) {
      return
    }

    evt.preventDefault?.()

    switch (action) {
      case "focus.next":
      case "focus.prev": {
        const moved = focusManager.route(action)
        if (moved) {
          options.render()
        }
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
        options.runAction("refresh", () => options.refreshData())
        return
      case "pane.jobs":
        if (setListPane("jobs")) {
          options.render()
        }
        return
      case "pane.logs":
        if (setListPane("logs")) {
          options.render()
        }
        return
      case "pane.togglePrincipal": {
        const nextPane = appState.principalPane === "jobs" ? "opencode" : "jobs"
        appState.principalPane = nextPane
        if (nextPane === "opencode") {
          options.runAction("opencode-ensure", () => options.ensureOpenCodeServer())
          focusManager.setFocus("agent")
        } else {
          focusManager.ensureValid()
        }
        options.render()
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
      case "modal.open.jobFilter":
        options.openJobFilterModal()
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
        if (options.snapshot.selectedJob) {
          options.setActiveModal("traces")
          void options.ensureTraceViewerModal()
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
