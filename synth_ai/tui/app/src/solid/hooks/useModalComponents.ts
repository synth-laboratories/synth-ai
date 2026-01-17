import { createMemo, createSignal, type Component, type Setter } from "solid-js"

import { useLocalApiScanner } from "../utils/localapi-watch"
import { toDisplayPath } from "../../utils/files"

export type ModalComponentsState = {
  candidatesModalComponent: () => Component<any> | null
  graphEvolveGenerationsModalComponent: () => Component<any> | null
  traceViewerModalComponent: () => Component<any> | null
  createJobModalComponent: () => Component<any> | null
  showCreateJobModal: () => boolean
  setShowCreateJobModal: Setter<boolean>
  localApiFiles: () => string[]
  ensureCandidatesModal: () => Promise<void>
  ensureGraphEvolveGenerationsModal: () => Promise<void>
  ensureTraceViewerModal: () => Promise<void>
  ensureCreateJobModal: () => Promise<void>
}

export function useModalComponents(): ModalComponentsState {
  const [candidatesModalComponent, setCandidatesModalComponent] = createSignal<Component<any> | null>(null)
  const [graphEvolveGenerationsModalComponent, setGraphEvolveGenerationsModalComponent] = createSignal<Component<any> | null>(null)
  const [traceViewerModalComponent, setTraceViewerModalComponent] = createSignal<Component<any> | null>(null)
  const [createJobModalComponent, setCreateJobModalComponent] = createSignal<Component<any> | null>(null)
  const [showCreateJobModal, setShowCreateJobModal] = createSignal(false)
  let candidatesModalLoading = false
  let graphEvolveGenerationsModalLoading = false
  let traceViewerModalLoading = false
  let createJobModalLoading = false
  const localApiScanner = useLocalApiScanner({
    enabled: showCreateJobModal,
    directories: () => [process.cwd()],
  })
  const localApiFiles = createMemo(() =>
    localApiScanner.files().map((api) => toDisplayPath(api.filepath)),
  )

  const ensureCandidatesModal = async (): Promise<void> => {
    if (candidatesModalComponent() || candidatesModalLoading) return
    candidatesModalLoading = true
    const mod = await import("../modals/CandidatesModal")
    setCandidatesModalComponent(() => mod.CandidatesModal)
    candidatesModalLoading = false
  }

  const ensureGraphEvolveGenerationsModal = async (): Promise<void> => {
    if (graphEvolveGenerationsModalComponent() || graphEvolveGenerationsModalLoading) return
    graphEvolveGenerationsModalLoading = true
    const mod = await import("../modals/GraphEvolveGenerationsModal")
    setGraphEvolveGenerationsModalComponent(() => mod.GraphEvolveGenerationsModal)
    graphEvolveGenerationsModalLoading = false
  }

  const ensureTraceViewerModal = async (): Promise<void> => {
    if (traceViewerModalComponent() || traceViewerModalLoading) return
    traceViewerModalLoading = true
    const mod = await import("../modals/TraceViewerModal")
    setTraceViewerModalComponent(() => mod.TraceViewerModal)
    traceViewerModalLoading = false
  }

  const ensureCreateJobModal = async (): Promise<void> => {
    if (createJobModalComponent() || createJobModalLoading) return
    createJobModalLoading = true
    const mod = await import("../modals/CreateJobModal")
    setCreateJobModalComponent(() => mod.CreateJobModal)
    createJobModalLoading = false
  }

  return {
    candidatesModalComponent,
    graphEvolveGenerationsModalComponent,
    traceViewerModalComponent,
    createJobModalComponent,
    showCreateJobModal,
    setShowCreateJobModal,
    localApiFiles,
    ensureCandidatesModal,
    ensureGraphEvolveGenerationsModal,
    ensureTraceViewerModal,
    ensureCreateJobModal,
  }
}
