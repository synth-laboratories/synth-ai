import { createMemo, type Accessor, type Component } from "solid-js"

import { formatActionKeys } from "../../input/keymap"
import { formatMetricsCharts } from "../../formatters/metrics"
import {
  formatConfigMetadata,
  formatPlanName,
  formatSessionDetails,
  formatTunnelDetails,
  formatUsageDetails,
} from "../../formatters/modals"
import type { AuthStatus } from "../../auth"
import type { AppData, SessionHealthResult, SessionRecord } from "../../types"
import { modeUrls } from "../../state/app-state"
import type { AppState } from "../../state/app-state"
import { ModalFrame } from "../components/ModalFrame"
import { resolveSelectionWindow } from "../utils/list"
import { buildScrollableModal } from "../utils/modal"
import type { ActiveModal, UsageData } from "./types"

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
  const ui = props.ui
  const listFilterHint = "up/down move | space select | a all/none | esc close"
  const listFilterView = createMemo(() => {
    const totalOptions = ui.listFilterOptions.length
    if (!totalOptions) {
      return ["  (no filters available)"]
    }
    const total = totalOptions + 1
    const window = resolveSelectionWindow(
      total,
      ui.listFilterCursor,
      ui.listFilterWindowStart,
      ui.listFilterVisibleCount,
    )
    const selections = ui.listFilterSelections[ui.listFilterPane]
    const mode = ui.listFilterMode[ui.listFilterPane]
    const lines: string[] = []
    const totalItems = ui.listFilterOptions.reduce((sum, option) => sum + option.count, 0)
    const allSelected = mode === "all"
    for (let idx = window.windowStart; idx < window.windowEnd; idx++) {
      const cursor = idx === window.selectedIndex ? ">" : " "
      if (idx === 0) {
        lines.push(`${cursor} [${allSelected ? "x" : " "}] All (${totalItems})`)
        continue
      }
      const option = ui.listFilterOptions[idx - 1]
      if (!option) continue
      const active = mode === "all" ? true : mode === "subset" && selections?.has(option.id)
      lines.push(`${cursor} [${active ? "x" : " "}] ${option.label} (${option.count})`)
    }
    return lines
  })
  const listFilterFrameWidth = createMemo(() => {
    const lines = listFilterView()
    const maxLine = Math.max(
      ...lines.map((line) => line.length),
      listFilterHint.length,
      "List filter".length,
    )
    return Math.min(
      Math.max(48, maxLine + 6),
      Math.max(48, props.dimensions().width - 4),
    )
  })
  const listFilterFrameHeight = createMemo(() => {
    const lines = listFilterView()
    return Math.min(
      Math.max(8, props.dimensions().height - 4),
      Math.max(8, lines.length + 6),
    )
  })

  if (props.kind === "filter") {
    return (
      <ModalFrame
        title="Event Filter"
        width={52}
        height={7}
        borderColor="#60a5fa"
        titleColor="#60a5fa"
        hint={`${formatActionKeys("modal.confirm")} apply | ${formatActionKeys("app.back")} close`}
        dimensions={props.dimensions}
      >
        <box flexDirection="column" gap={1}>
          <text fg="#e2e8f0">Event filter:</text>
          <input
            placeholder="Type to filter events"
            onInput={(value) => props.setModalInputValue(value)}
            ref={(ref) => {
              props.setModalInputRef(ref)
            }}
          />
        </box>
      </ModalFrame>
    )
  }

  if (props.kind === "snapshot") {
    return (
      <ModalFrame
        title="Snapshot ID"
        width={50}
        height={7}
        borderColor="#60a5fa"
        titleColor="#60a5fa"
        hint={`${formatActionKeys("modal.confirm")} apply | ${formatActionKeys("app.back")} close`}
        dimensions={props.dimensions}
      >
        <box flexDirection="column" gap={1}>
          <text fg="#e2e8f0">Snapshot ID:</text>
          <input
            placeholder="Enter snapshot id"
            onInput={(value) => props.setModalInputValue(value)}
            ref={(ref) => {
              props.setModalInputRef(ref)
            }}
          />
        </box>
      </ModalFrame>
    )
  }

  if (props.kind === "settings") {
    const modeLabels: Record<string, string> = { prod: "Prod", dev: "Dev", local: "Local" }
    const settingsContent = () => {
      const cursorIdx = props.settingsCursor()
      const lines: string[] = []
      for (let idx = 0; idx < ui.settingsOptions.length; idx++) {
        const mode = ui.settingsOptions[idx]
        const active = ui.currentMode === mode
        const cursor = idx === cursorIdx ? ">" : " "
        lines.push(`${cursor} [${active ? "x" : " "}] ${modeLabels[mode] || mode} (${mode})`)
      }
      const selectedMode = ui.settingsOptions[cursorIdx]
      if (selectedMode) {
        const urls = modeUrls[selectedMode]
        const key = ui.settingsKeys[selectedMode] || ""
        const keyPreview = key.trim() ? `...${key.slice(-8)}` : "(no key)"
        lines.push("")
        lines.push(`Backend: ${urls?.backendUrl || "(unset)"}`)
        lines.push(`Frontend: ${urls?.frontendUrl || "(unset)"}`)
        lines.push(`Key: ${keyPreview}`)
      }
      return lines.join("\n")
    }

    return (
      <ModalFrame
        title="Settings - Mode"
        width={64}
        height={14}
        borderColor="#38bdf8"
        titleColor="#38bdf8"
        hint={`${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} navigate | ${formatActionKeys("modal.confirm")} select | ${formatActionKeys("app.back")} close`}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{settingsContent()}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "usage") {
    const raw = formatUsageDetails(props.usageData())
    const view = buildScrollableModal(raw, 72, 28, ui.usageModalOffset || 0)
    const range = view.lines.length > view.bodyHeight
      ? `[${view.offset + 1}-${view.offset + view.visible.length}/${view.lines.length}] `
      : ""
    const scrollHint = `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} scroll`
    return (
      <ModalFrame
        title={`Usage & Plan - ${formatPlanName(props.usageData()?.plan_type || "free")} ${range}`.trim()}
        width={72}
        height={28}
        borderColor="#10b981"
        titleColor="#10b981"
        hint={`${scrollHint} | ${formatActionKeys("usage.openBilling")} billing | ${formatActionKeys("app.back")} close`}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{view.visible.join("\n")}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "metrics") {
    const m: any = props.data.metrics || {}
    const pts = Array.isArray(m?.points) ? m.points : []
    const raw = formatMetricsCharts(props.data.metrics, {
      width: props.dimensions().width - 6,
      height: props.dimensions().height - 8,
    })
    const view = buildScrollableModal(
      raw,
      props.dimensions().width - 4,
      props.dimensions().height - 6,
      ui.metricsModalOffset || 0,
    )
    const scrollHint = `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} scroll`
    const refreshHint = `${formatActionKeys("metrics.refresh")} refresh`
    const closeHint = `${formatActionKeys("app.back")} close`
    const hint = view.lines.length > view.bodyHeight
      ? `[${view.offset + 1}-${view.offset + view.visible.length}/${view.lines.length}] ${scrollHint} | ${refreshHint} | ${closeHint}`
      : `${refreshHint} | ${closeHint}`
    return (
      <ModalFrame
        title={`Metrics (${pts.length} points)`}
        width={props.dimensions().width - 4}
        height={props.dimensions().height - 6}
        borderColor="#8b5cf6"
        titleColor="#8b5cf6"
        hint={hint}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{view.visible.join("\n")}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "task-apps") {
    const raw = formatTunnelDetails(
      props.data.tunnels,
      props.data.tunnelHealthResults,
      ui.taskAppsModalSelectedIndex || 0,
    )
    const view = buildScrollableModal(raw, 90, 20, ui.taskAppsModalOffset || 0)
    const selectHint = `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} select`
    const copyHint = `${formatActionKeys("modal.copy", { primaryOnly: true })} copy hostname`
    const closeHint = `${formatActionKeys("app.back")} close`
    const hint = view.lines.length > view.bodyHeight
      ? `[${view.offset + 1}-${view.offset + view.visible.length}/${view.lines.length}] ${selectHint} | ${copyHint} | ${closeHint}`
      : `${selectHint} | ${copyHint} | ${closeHint}`
    return (
      <ModalFrame
        title={`Task Apps (${props.data.tunnels.length} tunnel${props.data.tunnels.length !== 1 ? "s" : ""})`}
        width={90}
        height={20}
        borderColor="#06b6d4"
        titleColor="#06b6d4"
        hint={hint}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{view.visible.join("\n")}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "sessions") {
    const sessions = props.sessionsCache()
    const raw = formatSessionDetails(
      sessions,
      props.sessionsHealthCache(),
      props.sessionsSelectedIndex(),
      ui.openCodeUrl,
    )
    const view = buildScrollableModal(raw, 70, 20, props.sessionsScrollOffset())
    const selectHint = `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} select`
    const connectHint = `${formatActionKeys("sessions.connect")} connect local`
    const disconnectHint = `${formatActionKeys("sessions.disconnect")} disconnect`
    const copyHint = `${formatActionKeys("sessions.copy", { primaryOnly: true })} copy URL`
    const confirmHint = `${formatActionKeys("modal.confirm")} select`
    const closeHint = `${formatActionKeys("app.back")} close`
    const hint = view.lines.length > view.bodyHeight
      ? `[${view.offset + 1}-${view.offset + view.visible.length}/${view.lines.length}] ${selectHint} | ${connectHint} | ${disconnectHint} | ${copyHint} | ${confirmHint} | ${closeHint}`
      : `${selectHint} | ${connectHint} | ${disconnectHint} | ${copyHint} | ${confirmHint} | ${closeHint}`
    return (
      <ModalFrame
        title={`OpenCode Sessions (${sessions.filter((s) => s.state === "connected" || s.state === "connecting" || s.state === "reconnecting").length} active)`}
        width={70}
        height={20}
        borderColor="#60a5fa"
        titleColor="#60a5fa"
        hint={hint}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{view.visible.join("\n")}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "config") {
    const raw = formatConfigMetadata(props.data)
    const view = buildScrollableModal(raw, 100, 24, ui.configModalOffset)
    const scrollHint = `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} scroll`
    const closeHint = `${formatActionKeys("app.back")} close`
    const hint = view.lines.length > view.bodyHeight
      ? `[${view.offset + 1}-${view.offset + view.visible.length}/${view.lines.length}] ${scrollHint} | ${closeHint}`
      : closeHint
    return (
      <ModalFrame
        title="Job Configuration"
        width={100}
        height={24}
        borderColor="#f59e0b"
        titleColor="#f59e0b"
        hint={hint}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{view.visible.join("\n")}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "results") {
    const Loaded = props.candidatesModalComponent()
    if (!Loaded) {
      return (
        <ModalFrame
          title="Candidates"
          width={60}
          height={8}
          borderColor="#60a5fa"
          titleColor="#60a5fa"
          hint="Loading..."
          dimensions={props.dimensions}
        >
          <text fg="#e2e8f0">Loading candidates...</text>
        </ModalFrame>
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

  if (props.kind === "generations") {
    const Loaded = props.graphEvolveGenerationsModalComponent()
    if (!Loaded) {
      return (
        <ModalFrame
          title="Generations"
          width={60}
          height={8}
          borderColor="#60a5fa"
          titleColor="#60a5fa"
          hint="Loading..."
          dimensions={props.dimensions}
        >
          <text fg="#e2e8f0">Loading generations...</text>
        </ModalFrame>
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

  if (props.kind === "traces") {
    const Loaded = props.traceViewerModalComponent()
    if (!Loaded) {
      return (
        <ModalFrame
          title="Traces"
          width={60}
          height={8}
          borderColor="#60a5fa"
          titleColor="#60a5fa"
          hint="Loading..."
          dimensions={props.dimensions}
        >
          <text fg="#e2e8f0">Loading traces...</text>
        </ModalFrame>
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

  if (props.kind === "profile") {
    const org = props.data.orgName || "-"
    const user = props.data.userEmail || "-"
    const apiKey = process.env.SYNTH_API_KEY || "-"
    return (
      <ModalFrame
        title="Profile"
        width={72}
        height={15}
        borderColor="#818cf8"
        titleColor="#818cf8"
        hint={`${formatActionKeys("app.back")} close`}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{`Organization:\n${org}\n\nEmail:\n${user}\n\nAPI Key:\n${apiKey}`}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "urls") {
    return (
      <ModalFrame
        title="URLs"
        width={60}
        height={10}
        borderColor="#f59e0b"
        titleColor="#f59e0b"
        hint={`${formatActionKeys("app.back")} close`}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{`Backend:\n${process.env.SYNTH_BACKEND_URL || "-"}\n\nFrontend:\n${process.env.SYNTH_FRONTEND_URL || "-"}`}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "list-filter") {
    return (
      <ModalFrame
        title="List filter"
        width={listFilterFrameWidth()}
        height={listFilterFrameHeight()}
        borderColor="#60a5fa"
        titleColor="#60a5fa"
        hint={listFilterHint}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{listFilterView().join("\n")}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "login") {
    const loginCopy = createMemo(() => {
      const status = props.loginStatus()
      const confirmKey = formatActionKeys("login.confirm")
      const closeKey = formatActionKeys("app.back")
      let content = "Press Enter to open browser and sign in..."
      let hint = `${confirmKey} start | ${closeKey} cancel`
      switch (status.state) {
        case "initializing":
          content = "Initializing..."
          hint = "Please wait..."
          break
        case "waiting":
          content = `Browser opened. Complete sign-in there.\n\nURL: ${status.verificationUri}`
          hint = `Waiting for browser auth... | ${closeKey} cancel`
          break
        case "polling":
          content = "Browser opened. Complete sign-in there.\n\nChecking for completion..."
          hint = `Waiting for browser auth... | ${closeKey} cancel`
          break
        case "success":
          content = "Authentication successful!"
          hint = "Loading..."
          break
        case "error":
          content = `Error: ${status.message}`
          hint = `${confirmKey} retry | ${closeKey} close`
          break
        default:
          break
      }
      return { content, hint }
    })
    return (
      <ModalFrame
        title="Sign In / Sign Up"
        width={60}
        height={10}
        borderColor="#22c55e"
        titleColor="#22c55e"
        hint={loginCopy().hint}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{loginCopy().content}</text>
      </ModalFrame>
    )
  }

  return null
}
