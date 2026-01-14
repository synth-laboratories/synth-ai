import { COLORS } from "../../theme"
import { defaultLayoutSpec } from "../../layout"

export function AppHeader() {
  return (
    <box
      height={defaultLayoutSpec.headerHeight}
      backgroundColor={COLORS.bgHeader}
      border
      borderStyle="single"
      borderColor={COLORS.border}
      alignItems="center"
    >
      <text fg={COLORS.text}>Synth AI</text>
    </box>
  )
}
