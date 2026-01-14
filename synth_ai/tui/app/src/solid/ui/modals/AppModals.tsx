import { Show, type Accessor, type Component } from "solid-js"
import { Dynamic } from "solid-js/web"

import type { AppData, SessionHealthResult, SessionRecord } from "../../../types"
import type { AuthStatus } from "../../../auth"
import type { AppState } from "../../../state/app-state"
import type { ActiveModal, ModalState, UsageData } from "../../modals/types"
import type { JobCreatedInfo } from "../../modals/CreateJobModal"
import type { DetailModalLayout, DetailModalView } from "../../hooks/useDetailModal"
import { ActiveModalRenderer } from "../../modals/ActiveModalRenderer"
import { ModalFrame } from "../../components/ModalFrame"

type AppModalsProps = {
  detail: {
    modal: Accessor<ModalState | null>
    modalLayout: Accessor<DetailModalLayout>
    modalView: Accessor<DetailModalView | null>
    modalHint: Accessor<string>
  }
  overlay: {
    activeModal: Accessor<ActiveModal | null>
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
  createJob: {
    createJobModalComponent: Accessor<Component<any> | null>
    showCreateJobModal: Accessor<boolean>
    onClose: () => void
    onJobCreated: (info: JobCreatedInfo) => void
    onStatusUpdate: (status: string) => void
    onError: (error: string) => void
    localApiFiles: Accessor<string[]>
    width: Accessor<number>
    height: Accessor<number>
    dimensions: Accessor<{ width: number; height: number }>
  }
}

export function AppModals(props: AppModalsProps) {
  return (
    <>
      <Show when={props.detail.modal()}>
        <box
          position="absolute"
          left={props.detail.modalLayout().left}
          top={props.detail.modalLayout().top}
          width={props.detail.modalLayout().width}
          height={props.detail.modalLayout().height}
          backgroundColor="#0b1220"
          border
          borderStyle="single"
          borderColor="#60a5fa"
          zIndex={20}
          flexDirection="column"
          paddingLeft={2}
          paddingRight={2}
          paddingTop={1}
          paddingBottom={1}
        >
          <text fg="#60a5fa">
            {props.detail.modal()!.title}
          </text>
          <box flexGrow={1}>
            <text fg="#e2e8f0">{props.detail.modalView()?.visible.join("\n") ?? ""}</text>
          </box>
          <text fg="#94a3b8">{props.detail.modalHint()}</text>
        </box>
      </Show>

      <Show when={props.overlay.activeModal()}>
        {(kind) => (
          <ActiveModalRenderer
            kind={kind()}
            dimensions={props.overlay.dimensions}
            ui={props.overlay.ui}
            setModalInputValue={props.overlay.setModalInputValue}
            setModalInputRef={props.overlay.setModalInputRef}
            settingsCursor={props.overlay.settingsCursor}
            usageData={props.overlay.usageData}
            sessionsCache={props.overlay.sessionsCache}
            sessionsHealthCache={props.overlay.sessionsHealthCache}
            sessionsSelectedIndex={props.overlay.sessionsSelectedIndex}
            sessionsScrollOffset={props.overlay.sessionsScrollOffset}
            loginStatus={props.overlay.loginStatus}
            candidatesModalComponent={props.overlay.candidatesModalComponent}
            graphEvolveGenerationsModalComponent={props.overlay.graphEvolveGenerationsModalComponent}
            traceViewerModalComponent={props.overlay.traceViewerModalComponent}
            closeActiveModal={props.overlay.closeActiveModal}
            onStatusUpdate={props.overlay.onStatusUpdate}
            openCandidatesForGeneration={props.overlay.openCandidatesForGeneration}
            data={props.overlay.data}
          />
        )}
      </Show>

      <Show
        when={props.createJob.createJobModalComponent()}
        fallback={
          <Show when={props.createJob.showCreateJobModal()}>
            <ModalFrame
              title="Create Job"
              width={50}
              height={8}
              borderColor="#60a5fa"
              titleColor="#60a5fa"
              hint="Loading..."
              dimensions={props.createJob.dimensions}
            >
              <text fg="#e2e8f0">Loading create job flow...</text>
            </ModalFrame>
          </Show>
        }
      >
        <Dynamic
          component={props.createJob.createJobModalComponent() as Component<any>}
          visible={props.createJob.showCreateJobModal()}
          onClose={props.createJob.onClose}
          onJobCreated={props.createJob.onJobCreated}
          onStatusUpdate={props.createJob.onStatusUpdate}
          onError={props.createJob.onError}
          localApiFiles={props.createJob.localApiFiles()}
          width={props.createJob.width()}
          height={props.createJob.height()}
        />
      </Show>
    </>
  )
}
