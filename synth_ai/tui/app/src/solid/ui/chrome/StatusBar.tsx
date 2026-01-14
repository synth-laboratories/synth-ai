import { type Accessor } from "solid-js"

import { defaultLayoutSpec } from "../../layout"

type StatusBarProps = {
  statusText: Accessor<string>
}

export function StatusBar(props: StatusBarProps) {
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
      <text fg="#e2e8f0">{props.statusText()}</text>
    </box>
  )
}
