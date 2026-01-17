import { createMemo, type JSX } from "solid-js"
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
  isSelected: boolean | (() => boolean)
  /** Whether the parent panel/list has focus */
  panelFocused?: boolean | (() => boolean)
  /** Render function receiving style context */
  children: (ctx: ListCardStyleContext) => JSX.Element
}

/**
 * Shared list card component with consistent selection styling.
 * Uses render-prop pattern for flexible content.
 */
export function ListCard(props: ListCardProps) {
  const ctx = createMemo<ListCardStyleContext>(() => {
    const isSelected = typeof props.isSelected === "function" ? props.isSelected() : props.isSelected
    const panelFocused =
      props.panelFocused === undefined
        ? true
        : typeof props.panelFocused === "function"
          ? props.panelFocused()
          : props.panelFocused
    // Only show selection highlight when panel is focused (default to true for backwards compat)
    const showSelection = isSelected && panelFocused
    const sel = getSelectionStyle(showSelection)
    return {
      fg: sel.fg,
      bg: sel.bg,
      isSelected,
      fgDim: showSelection ? COLORS.textBright : COLORS.textDim,
    }
  })

  return <>{props.children(ctx())}</>
}

/**
 * Get selection indicator string.
 */
export function getIndicator(isSelected: boolean): string {
  return isSelected ? "▸ " : "  "
}
