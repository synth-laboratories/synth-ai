import { COLORS } from "../theme"
import { getActionHint, type KeyAction } from "../../input/keymap"

interface KeyHintProps {
  action: KeyAction
  active?: boolean
}

/**
 * KeyHint component that renders the hint for a key action.
 * Uses centralized hints with embedded key format, e.g. "(r)efresh", "(f)ilter"
 */
export function KeyHint(props: KeyHintProps) {
  return (
    <text fg={props.active ? COLORS.textBright : COLORS.textDim}>
      {getActionHint(props.action)}
    </text>
  )
}


