/**
 * TaskAppsList - List of task apps (tunnels + local apps)
 *
 * Displays discovered task apps in a compact 2-line format.
 */
import { For, Show, createMemo } from "solid-js"
import { COLORS } from "../../theme"
import type { TaskApp } from "../../../types"

interface TaskAppsListProps {
  apps: TaskApp[]
  selectedIndex: number
  focused: boolean
  width: number
  height: number
}

function getHealthIcon(status: "healthy" | "unhealthy" | "unknown"): string {
  switch (status) {
    case "healthy":
      return "✓"
    case "unhealthy":
      return "✗"
    case "unknown":
      return "?"
  }
}

function getHealthColor(status: "healthy" | "unhealthy" | "unknown"): string {
  switch (status) {
    case "healthy":
      return "#10b981" // green
    case "unhealthy":
      return "#ef4444" // red
    case "unknown":
      return "#6b7280" // gray
  }
}

function formatAppCard(app: TaskApp) {
  const healthIcon = getHealthIcon(app.health_status)
  const locationLabel = app.type === "tunnel"
    ? "tunnel"
    : app.port
      ? `local:${app.port}`
      : "local"

  return {
    id: app.id,
    name: app.name,
    healthIcon,
    healthStatus: app.health_status,
    locationLabel,
  }
}

/**
 * Task apps list panel component.
 *
 * Format (two lines per app):
 *   ▸ banking77
 *     ✓ Healthy | local:8000
 */
export function TaskAppsList(props: TaskAppsListProps) {
  const items = createMemo(() => props.apps.map(formatAppCard))

  // Each app takes 2 lines, so we can show (height - 2) / 2 apps
  const visibleCount = createMemo(() => Math.floor((props.height - 2) / 2))

  const visibleItems = createMemo(() => {
    const list = items()
    const maxVisible = visibleCount()
    const selected = props.selectedIndex

    let start = 0
    if (selected >= start + maxVisible) {
      start = selected - maxVisible + 1
    }
    if (selected < start) {
      start = selected
    }

    return list.slice(start, start + maxVisible).map((item, idx) => ({
      ...item,
      globalIndex: start + idx,
    }))
  })

  return (
    <box
      width={props.width}
      height={props.height}
      borderStyle="single"
      borderColor={props.focused ? COLORS.textAccent : COLORS.border}
      title="Task Apps"
      titleAlignment="left"
      flexDirection="column"
    >
      <Show
        when={props.apps.length > 0}
        fallback={<text fg={COLORS.textDim}> No task apps found</text>}
      >
        <For each={visibleItems()}>
          {(item) => {
            const isSelected = item.globalIndex === props.selectedIndex
            const bg = isSelected ? COLORS.bgSelection : undefined
            const nameFg = isSelected ? COLORS.textBright : COLORS.text
            const detailFg = isSelected ? COLORS.textBright : COLORS.textDim
            const indicator = isSelected ? "▸ " : "  "

            return (
              <box flexDirection="column">
                {/* Line 1: indicator + app name */}
                <box flexDirection="row" backgroundColor={bg} width="100%">
                  <text fg={nameFg}>{indicator}{item.name}</text>
                </box>
                {/* Line 2: health icon + status | location */}
                <box flexDirection="row" backgroundColor={bg} width="100%">
                  <text>
                    <span style={{ fg: getHealthColor(item.healthStatus) }}>  {item.healthIcon}</span>
                    <span style={{ fg: detailFg }}> {item.healthStatus} | {item.locationLabel}</span>
                  </text>
                </box>
              </box>
            )
          }}
        </For>
      </Show>
    </box>
  )
}
