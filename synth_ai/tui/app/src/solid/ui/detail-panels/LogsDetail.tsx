import { Show } from "solid-js"
import { TEXT, PANEL, getPanelBorderColor } from "../../theme"

interface LogsDetailProps {
  title: string
  filePath: string | null
  lines: string[]
  visibleLines: string[]
  focused: boolean
  tail: boolean
  offset: number
  maxOffset: number
  framed?: boolean
}

/**
 * Logs detail panel (right side).
 */
export function LogsDetail(props: LogsDetailProps) {
  const framed = () => props.framed !== false

  const scrollInfo = () => {
    if (props.lines.length === 0) return ""
    const current = props.offset + 1
    const end = Math.min(props.offset + props.visibleLines.length, props.lines.length)
    const total = props.lines.length
    const tailIndicator = props.tail ? " [TAIL]" : ""
    return ` [${current}-${end}/${total}]${tailIndicator}`
  }

  const borderProps = () => {
    if (!framed()) {
      return { border: false }
    }
    return {
      border: PANEL.border,
      borderStyle: PANEL.borderStyle,
      borderColor: getPanelBorderColor(props.focused),
      title: `${props.title}${scrollInfo()}`,
      titleAlignment: PANEL.titleAlignment,
    }
  }

  return (
    <box
      {...borderProps()}
      flexGrow={1}
      paddingLeft={PANEL.paddingLeft}
      paddingRight={1}
      flexDirection="column"
    >
      <Show when={props.filePath}>
        <box paddingBottom={1}>
          <text fg={TEXT.fgDim}>{props.filePath}</text>
        </box>
      </Show>
      <Show
        when={props.lines.length > 0}
        fallback={<text fg={TEXT.fg}>No log content.</text>}
      >
        <text fg={TEXT.fg}>
          {props.visibleLines.join("\n")}
        </text>
      </Show>
    </box>
  )
}
