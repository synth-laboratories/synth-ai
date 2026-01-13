import { type Accessor, type Component } from "solid-js"

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
import type { SessionHealthResult, SessionRecord, Snapshot } from "../../types"
import { modeKeys, modeUrls, appState } from "../../state/app-state"
import { config } from "../../state/polling"
import { ModalFrame } from "../components/ModalFrame"
import { buildScrollableModal } from "../utils/modal"
import type { ActiveModal, UsageData } from "./types"

type ActiveModalRendererProps = {
  kind: ActiveModal
  dataVersion: Accessor<number>
  dimensions: Accessor<{ width: number; height: number }>
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
  traceViewerModalComponent: Accessor<Component<any> | null>
  closeActiveModal: () => void
  onStatusUpdate: (message: string) => void
  snapshot: Snapshot
}

export function ActiveModalRenderer(props: ActiveModalRendererProps) {
  // Modal content is mostly derived from non-reactive state objects (appState/snapshot).
  // Make modal rendering depend on the reactive version signal so calls to
  // `data.ctx.render()` repaint the modal (e.g. settings cursor).
  props.dataVersion()

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

  if (props.kind === "key") {
    return (
      <ModalFrame
        title="API Key"
        width={70}
        height={7}
        borderColor="#7dd3fc"
        titleColor="#7dd3fc"
        hint={`Paste or type key | ${formatActionKeys("modal.confirm")} apply | ${formatActionKeys("app.back")} close`}
        dimensions={props.dimensions}
      >
        <box flexDirection="column" gap={1}>
          <text fg="#e2e8f0">API Key:</text>
          <input
            placeholder=""
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
      for (let idx = 0; idx < appState.settingsOptions.length; idx++) {
        const mode = appState.settingsOptions[idx]
        const active = appState.currentMode === mode
        const cursor = idx === cursorIdx ? ">" : " "
        lines.push(`${cursor} [${active ? "x" : " "}] ${modeLabels[mode] || mode} (${mode})`)
      }
      const selectedMode = appState.settingsOptions[cursorIdx]
      if (selectedMode) {
        const urls = modeUrls[selectedMode]
        const key = modeKeys[selectedMode]
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
        hint={`${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} navigate | ${formatActionKeys("modal.confirm")} select | ${formatActionKeys("settings.openKey")} key | ${formatActionKeys("app.back")} close`}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{settingsContent()}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "usage") {
    const raw = formatUsageDetails(props.usageData())
    const view = buildScrollableModal(raw, 72, 28, appState.usageModalOffset || 0)
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
    const m: any = props.snapshot.metrics || {}
    const pts = Array.isArray(m?.points) ? m.points : []
    const job = props.snapshot.selectedJob
    const isGepa = job?.training_type === "gepa" || job?.training_type === "graph_gepa"

    const raw = formatMetricsCharts(props.snapshot.metrics, {
      width: props.dimensions().width - 6,
      height: props.dimensions().height - 8,
      isGepa,
    })
    const view = buildScrollableModal(
      raw,
      props.dimensions().width - 4,
      props.dimensions().height - 6,
      appState.metricsModalOffset || 0,
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
      props.snapshot.tunnels,
      props.snapshot.tunnelHealthResults,
      appState.taskAppsModalSelectedIndex || 0,
    )
    const view = buildScrollableModal(raw, 90, 20, appState.taskAppsModalOffset || 0)
    const selectHint = `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} select`
    const copyHint = `${formatActionKeys("modal.copy", { primaryOnly: true })} copy hostname`
    const closeHint = `${formatActionKeys("app.back")} close`
    const hint = view.lines.length > view.bodyHeight
      ? `[${view.offset + 1}-${view.offset + view.visible.length}/${view.lines.length}] ${selectHint} | ${copyHint} | ${closeHint}`
      : `${selectHint} | ${copyHint} | ${closeHint}`
    return (
      <ModalFrame
        title={`Task Apps (${props.snapshot.tunnels.length} tunnel${props.snapshot.tunnels.length !== 1 ? "s" : ""})`}
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
      appState.openCodeUrl,
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
    const raw = formatConfigMetadata(props.snapshot)
    const view = buildScrollableModal(raw, 100, 24, appState.configModalOffset)
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
        snapshot={props.snapshot}
        width={props.dimensions().width}
        height={props.dimensions().height}
        onClose={props.closeActiveModal}
        onStatus={(message: string) => {
          props.onStatusUpdate(message)
        }}
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
        snapshot={props.snapshot}
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
    const org = props.snapshot.orgId || "-"
    const user = props.snapshot.userId || "-"
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
        <text fg="#e2e8f0">{`Organization:\n${org}\n\nUser:\n${user}\n\nAPI Key:\n${apiKey}`}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "urls") {
    const backend = process.env.SYNTH_BACKEND_URL || "-"
    const frontend = process.env.SYNTH_FRONTEND_URL || "-"
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
        <text fg="#e2e8f0">{`Backend:\n${backend}\n\nFrontend:\n${frontend}`}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "job-filter") {
    const max = Math.max(0, appState.jobFilterOptions.length - 1)
    const start = clamp(appState.jobFilterWindowStart, 0, Math.max(0, max))
    const end = Math.min(appState.jobFilterOptions.length, start + config.jobFilterVisibleCount)
    const lines: string[] = []
    for (let idx = start; idx < end; idx++) {
      const option = appState.jobFilterOptions[idx]
      const active = appState.jobStatusFilter.has(option.status)
      const cursor = idx === appState.jobFilterCursor ? ">" : " "
      lines.push(`${cursor} [${active ? "x" : " "}] ${option.status} (${option.count})`)
    }
    if (!lines.length) {
      lines.push("  (no statuses available)")
    }
    return (
      <ModalFrame
        title="Job filter (status)"
        width={52}
        height={11}
        borderColor="#60a5fa"
        titleColor="#60a5fa"
        hint={`${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} move | ${formatActionKeys("jobFilter.toggle", { primaryOnly: true })} select | ${formatActionKeys("jobFilter.clear", { primaryOnly: true })} clear | ${formatActionKeys("app.back")} close`}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{lines.join("\n")}</text>
      </ModalFrame>
    )
  }

  if (props.kind === "login") {
    const status = props.loginStatus()
    let content = ""
    const confirmKey = formatActionKeys("login.confirm")
    const closeKey = formatActionKeys("app.back")
    let hint = `${confirmKey} start | ${closeKey} cancel`
    if (status.state === "idle") {
      content = "Press Enter to open browser and sign in..."
    } else if (status.state === "initializing") {
      content = "Initializing..."
      hint = "Please wait..."
    } else if (status.state === "waiting") {
      content = `Browser opened. Complete sign-in there.\n\nURL: ${status.verificationUri}`
      hint = `Waiting for browser auth... | ${closeKey} cancel`
    } else if (status.state === "polling") {
      content = "Browser opened. Complete sign-in there.\n\nChecking for completion..."
      hint = `Waiting for browser auth... | ${closeKey} cancel`
    } else if (status.state === "success") {
      content = "Authentication successful!"
      hint = "Loading..."
    } else if (status.state === "error") {
      content = `Error: ${status.message}`
      hint = `${confirmKey} retry | ${closeKey} close`
    }
    return (
      <ModalFrame
        title="Sign In / Sign Up"
        width={60}
        height={10}
        borderColor="#22c55e"
        titleColor="#22c55e"
        hint={hint}
        dimensions={props.dimensions}
      >
        <text fg="#e2e8f0">{content}</text>
      </ModalFrame>
    )
  }

  return null
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}
