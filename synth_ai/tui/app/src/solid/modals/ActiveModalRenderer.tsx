import { type Accessor, type Component } from "solid-js"

import type { AuthStatus } from "../../auth"
import type { AppData, SessionHealthResult, SessionRecord } from "../../types"
import type { AppState } from "../../state/app-state"
import type { ActiveModal, UsageData } from "./types"
import { LoadingModal } from "./ModalShared"
import { FilterModal } from "./FilterModal"
import { SnapshotModal } from "./SnapshotModal"
import { SettingsModal } from "./SettingsModal"
import { UsageModal } from "./UsageModal"
import { MetricsModal } from "./MetricsModal"
import { TaskAppsModal } from "./TaskAppsModal"
import { SessionsModal } from "./SessionsModal"
import { ConfigModal } from "./ConfigModal"
import { ProfileModal } from "./ProfileModal"
import { ListFilterModal } from "./ListFilterModal"
import { LoginModal } from "./LoginModal"

type ActiveModalRendererProps = {
  kind: ActiveModal
  dimensions: Accessor<{ width: number; height: number }>
  ui: AppState
  setModalInputValue: (value: string) => void
  setModalInputRef: (ref: any) => void
  settingsCursor: Accessor<number>
  usageData: Accessor<UsageData | null>
  sessionsCache: Accessor<SessionRecord[]>
  sessionsHealthCache: Accessor<Map<string, SessionHealthResult>>
  sessionsSelectedIndex: Accessor<number>
  sessionsScrollOffset: Accessor<number>
  loginStatus: Accessor<AuthStatus>
  candidatesModalComponent: Accessor<Component<any> | null>
  graphEvolveGenerationsModalComponent: Accessor<Component<any> | null>
  traceViewerModalComponent: Accessor<Component<any> | null>
  closeActiveModal: () => void
  onStatusUpdate: (message: string) => void
  openCandidatesForGeneration: (generation: number) => void
  data: AppData
}

export function ActiveModalRenderer(props: ActiveModalRendererProps) {
  switch (props.kind) {
    case "config":
      return (
        <ConfigModal
          dimensions={props.dimensions}
          data={props.data}
          offset={props.ui.configModalOffset}
        />
      )
    case "filter":
      return (
        <FilterModal
          dimensions={props.dimensions}
          setModalInputValue={props.setModalInputValue}
          setModalInputRef={props.setModalInputRef}
        />
      )
    case "generations": {
      const Loaded = props.graphEvolveGenerationsModalComponent()
      if (!Loaded) {
        return (
          <LoadingModal title="Generations" dimensions={props.dimensions} message="Loading generations..." />
        )
      }
      return (
        <Loaded
          visible={true}
          data={props.data}
          width={props.dimensions().width}
          height={props.dimensions().height}
          onClose={props.closeActiveModal}
          onStatus={(message: string) => {
            props.onStatusUpdate(message)
          }}
          onOpenCandidates={props.openCandidatesForGeneration}
        />
      )
    }
    case "list-filter":
      return <ListFilterModal dimensions={props.dimensions} ui={props.ui} />
    case "login":
      return <LoginModal dimensions={props.dimensions} loginStatus={props.loginStatus} />
    case "metrics":
      return (
        <MetricsModal
          dimensions={props.dimensions}
          data={props.data}
          offset={props.ui.metricsModalOffset || 0}
        />
      )
    case "profile":
      return <ProfileModal dimensions={props.dimensions} data={props.data} />
    case "results": {
      const Loaded = props.candidatesModalComponent()
      if (!Loaded) {
        return (
          <LoadingModal title="Candidates" dimensions={props.dimensions} message="Loading candidates..." />
        )
      }
      return (
        <Loaded
          visible={true}
          data={props.data}
          generationFilter={props.ui.candidatesGenerationFilter}
          width={props.dimensions().width}
          height={props.dimensions().height}
          onGenerationChange={props.openCandidatesForGeneration}
          onClose={props.closeActiveModal}
          onStatus={(message: string) => {
            props.onStatusUpdate(message)
          }}
        />
      )
    }
    case "sessions":
      return (
        <SessionsModal
          dimensions={props.dimensions}
          sessions={props.sessionsCache}
          sessionsHealth={props.sessionsHealthCache}
          selectedIndex={props.sessionsSelectedIndex}
          scrollOffset={props.sessionsScrollOffset}
          openCodeUrl={props.ui.openCodeUrl}
        />
      )
    case "settings":
      return (
        <SettingsModal
          dimensions={props.dimensions}
          ui={props.ui}
          settingsCursor={props.settingsCursor}
        />
      )
    case "snapshot":
      return (
        <SnapshotModal
          dimensions={props.dimensions}
          setModalInputValue={props.setModalInputValue}
          setModalInputRef={props.setModalInputRef}
        />
      )
    case "task-apps":
      return (
        <TaskAppsModal
          dimensions={props.dimensions}
          data={props.data}
          selectedIndex={props.ui.taskAppsModalSelectedIndex || 0}
          offset={props.ui.taskAppsModalOffset || 0}
        />
      )
    case "traces": {
      const Loaded = props.traceViewerModalComponent()
      if (!Loaded) {
        return (
          <LoadingModal title="Traces" dimensions={props.dimensions} message="Loading traces..." />
        )
      }
      return (
        <Loaded
          visible={true}
          data={props.data}
          width={props.dimensions().width}
          height={props.dimensions().height}
          onClose={props.closeActiveModal}
          onStatus={(message: string) => {
            props.onStatusUpdate(message)
          }}
        />
      )
    }
    case "usage":
      return (
        <UsageModal
          dimensions={props.dimensions}
          usageData={props.usageData}
          offset={props.ui.usageModalOffset || 0}
        />
      )
    default:
      return null
  }
}
