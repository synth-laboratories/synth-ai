import type { JSX } from "solid-js"
import { COLORS } from "../theme"
import { getSelectionStyle } from "../utils/list"

/**
 * Style context passed to ListCard children render function.
 */
export interface ListCardStyleContext {
  /** Foreground color for primary text */
  fg: string
  /** Background color (undefined when not selected) */
  bg: string | undefined
  /** Whether this card is currently selected */
  isSelected: boolean
  /** Foreground color for secondary/dim text */
  fgDim: string
}

export interface ListCardProps {
  /** Whether this card is currently selected */
  isSelected: boolean
  /** Whether the parent panel/list has focus */
  panelFocused?: boolean
  /** Render function receiving style context */
  children: (ctx: ListCardStyleContext) => JSX.Element
}

/**
 * Shared list card component with consistent selection styling.
 * Uses render-prop pattern for flexible content.
 */
export function ListCard(props: ListCardProps) {
  const styleContext = (): ListCardStyleContext => {
    // Only show selection highlight when panel is focused (default to true for backwards compat)
    const showSelection = props.isSelected && (props.panelFocused ?? true)
    const sel = getSelectionStyle(showSelection)
    return {
      fg: sel.fg,
      bg: sel.bg,
      isSelected: props.isSelected,
      fgDim: showSelection ? COLORS.textBright : COLORS.textDim,
    }
  }

  return <>{props.children(styleContext())}</>
}

/**
 * Get selection indicator string.
 */
export function getIndicator(isSelected: boolean): string {
  return isSelected ? "▸ " : "  "
}
