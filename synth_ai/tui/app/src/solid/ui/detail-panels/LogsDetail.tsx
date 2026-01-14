import { Show } from "solid-js"
import { COLORS } from "../../theme"

interface LogsDetailProps {
  title: string
  filePath: string | null
  lines: string[]
  visibleLines: string[]
  focused: boolean
  tail: boolean
  offset: number
  maxOffset: number
}

/**
 * Logs detail panel (right side).
 */
export function LogsDetail(props: LogsDetailProps) {
  const scrollInfo = () => {
    if (props.lines.length === 0) return ""
    const current = props.offset + 1
    const end = Math.min(props.offset + props.visibleLines.length, props.lines.length)
    const total = props.lines.length
    const tailIndicator = props.tail ? " [TAIL]" : ""
    return ` [${current}-${end}/${total}]${tailIndicator}`
  }

  return (
    <box
      flexGrow={1}
      border
      borderStyle="single"
      borderColor={props.focused ? COLORS.textAccent : COLORS.border}
      title={`${props.title}${scrollInfo()}`}
      titleAlignment="left"
      paddingLeft={1}
      paddingRight={1}
      flexDirection="column"
    >
      <Show when={props.filePath}>
        <box paddingBottom={1}>
          <text fg={COLORS.textDim}>{props.filePath}</text>
        </box>
      </Show>
      <Show
        when={props.lines.length > 0}
        fallback={<text fg={COLORS.textDim}>No log content.</text>}
      >
        <text fg={COLORS.text}>
          {props.visibleLines.join("\n")}
        </text>
      </Show>
    </box>
  )
}
