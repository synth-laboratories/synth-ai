import { COLORS } from "../theme"
import {
  clampIndex,
  moveSelectionIndex,
  resolveSelectionWindow,
  wrapIndex,
  computeListWindow,
  computeListWindowFromStart,
  deriveSelectedIndex,
  resolveSelectionIndexById,
  moveSelectionById,
  uniqueById,
} from "../../utils/list"

export type { ListWindowItem, SelectionWindow, SelectionWindowMode } from "../../utils/list"

export type SelectionStyle = {
  fg: string
  bg: string | undefined
}

export function getSelectionStyle(isSelected: boolean): SelectionStyle {
  return {
    fg: isSelected ? COLORS.textSelected : COLORS.text,
    bg: isSelected ? COLORS.bgSelection : undefined,
  }
}

export {
  clampIndex,
  moveSelectionIndex,
  resolveSelectionWindow,
  wrapIndex,
  computeListWindow,
  computeListWindowFromStart,
  deriveSelectedIndex,
  resolveSelectionIndexById,
  moveSelectionById,
  uniqueById,
}
