import { For, Show, createMemo } from "solid-js"
import { COLORS } from "../../theme"

interface LogFileInfo {
  name: string
  path: string
}

interface LogsListProps {
  logs: LogFileInfo[]
  selectedIndex: number
  focused: boolean
  width: number
  height: number
}

function formatLogType(name: string): string {
  if (name.includes("_deploy_")) return "deploy"
  if (name.includes("_serve_")) return "serve"
  return "log"
}

function formatLogDate(name: string): string {
  const match = name.match(/(\d{4}_\d{2}_\d{2})_([0-9]{2}[:\-][0-9]{2}[:\-][0-9]{2})/)
  if (match) {
    const date = match[1].replace(/_/g, "-")
    const time = match[2].replace(/-/g, ":")
    return `${date} ${time}`
  }
  return name
}

/**
 * Logs list panel component.
 */
export function LogsList(props: LogsListProps) {
  const items = createMemo(() =>
    props.logs.map((file) => ({
      id: file.path,
      name: formatLogType(file.name),
      description: formatLogDate(file.name),
    }))
  )

  const maxNameWidth = createMemo(() => {
    const list = items()
    if (!list.length) return 0
    return Math.max(...list.map((item) => item.name.length))
  })

  const visibleItems = createMemo(() => {
    const list = items()
    const height = props.height - 2 // Account for border
    const selected = props.selectedIndex

    let start = 0
    if (selected >= start + height) {
      start = selected - height + 1
    }
    if (selected < start) {
      start = selected
    }

    return list.slice(start, start + height).map((item, idx) => ({
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
      title="Logs"
      titleAlignment="left"
      paddingLeft={1}
      paddingRight={1}
      flexDirection="column"
    >
      <Show
        when={props.logs.length > 0}
        fallback={<text fg={COLORS.textDim}>No log files found.</text>}
      >
        <For each={visibleItems()}>
          {(item) => {
            const isSelected = item.globalIndex === props.selectedIndex
            const fg = isSelected ? COLORS.textSelected : COLORS.text
            const bg = isSelected ? COLORS.bgSelection : undefined
            const name = item.name.padEnd(maxNameWidth(), " ")
            const description = item.description || ""

            return (
              <text fg={fg} bg={bg}>
                {`${name} ${description}`}
              </text>
            )
          }}
        </For>
        <Show when={props.logs.length > visibleItems().length}>
          <text fg={COLORS.textDim}>...</text>
        </Show>
      </Show>
    </box>
  )
}
