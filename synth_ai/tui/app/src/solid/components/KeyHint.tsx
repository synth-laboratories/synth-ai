import { COLORS } from "../theme"

interface KeyHintProps {
  description: string
  keyLabel: string
  active?: boolean
}

/**
 * KeyHint component that renders exactly "Description (key)" with colors
 * matching the gold/reference TUI style.
 */
export function KeyHint(props: KeyHintProps) {
  return (
    <text fg={props.active ? COLORS.textBright : COLORS.textDim}>
      {`${props.description} (${props.keyLabel})`}
    </text>
  )
}


