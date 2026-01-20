import { Show } from "solid-js"
import { COLORS } from "../../theme"

interface LogsDetailProps {
  title: string
  lines: string[]
  visibleLines: string[]
}

/**
 * Logs detail panel (right side).
 */
export function LogsDetail(props: LogsDetailProps) {
  return (
    <box
      flexGrow={1}
      border
      borderStyle="single"
      borderColor={COLORS.border}
      title={props.title}
      titleAlignment="left"
      paddingLeft={1}
      paddingRight={1}
      paddingTop={1}
      flexDirection="column"
    >
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
