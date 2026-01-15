import type { JSX } from "solid-js"
import { COLORS } from "../theme"

export interface DetailContainerProps {
  title: string
  scrollInfo?: string
  focused: boolean
  children: JSX.Element
  paddingLeft?: number
  paddingRight?: number
  flexGrow?: number
  height?: number
  overflow?: "hidden" | "visible"
}

/**
 * Shared container for detail panels with consistent styling.
 * Handles border, focus color, title with optional scroll info.
 */
export function DetailContainer(props: DetailContainerProps) {
  const fullTitle = () =>
    props.scrollInfo ? `${props.title}${props.scrollInfo}` : props.title

  return (
    <box
      flexGrow={props.flexGrow ?? 1}
      height={props.height}
      border
      borderStyle="single"
      borderColor={props.focused ? COLORS.textAccent : COLORS.border}
      title={fullTitle()}
      titleAlignment="left"
      paddingLeft={props.paddingLeft ?? 1}
      paddingRight={props.paddingRight}
      flexDirection="column"
      overflow={props.overflow}
    >
      {props.children}
    </box>
  )
}
