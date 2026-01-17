import { type Accessor, createMemo } from "solid-js"

import { defaultLayoutSpec } from "../../layout"
import { clampLine } from "../../../utils/text"

type StatusBarProps = {
  statusText: Accessor<string>
  width: Accessor<number>
}

export function StatusBar(props: StatusBarProps) {
  const contentWidth = createMemo(() => Math.max(1, props.width() - 3))
  const statusLine = createMemo(() => clampLine(props.statusText(), contentWidth()))

  return (
    <box
      height={defaultLayoutSpec.statusHeight}
      backgroundColor="#0f172a"
      border
      borderStyle="single"
      borderColor="#334155"
      paddingLeft={1}
      alignItems="center"
    >
      <text fg="#e2e8f0">{statusLine()}</text>
    </box>
  )
}
