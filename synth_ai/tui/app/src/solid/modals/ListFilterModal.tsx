import { type Accessor, createMemo } from "solid-js"

import type { AppState } from "../../state/app-state"
import { resolveSelectionWindow } from "../utils/list"
import { TextContentModal } from "./ModalShared"

type ListFilterModalProps = {
  dimensions: Accessor<{ width: number; height: number }>
  ui: AppState
}

export function ListFilterModal(props: ListFilterModalProps) {
  const hint = "up/down move | space select | a all/none | esc close"
  const view = createMemo(() => {
    const totalOptions = props.ui.listFilterOptions.length
    if (!totalOptions) {
      return ["  (no filters available)"]
    }
    const total = totalOptions + 1
    const window = resolveSelectionWindow(
      total,
      props.ui.listFilterCursor,
      props.ui.listFilterWindowStart,
      props.ui.listFilterVisibleCount,
    )
    const selections = props.ui.listFilterSelections[props.ui.listFilterPane]
    const mode = props.ui.listFilterMode[props.ui.listFilterPane]
    const lines: string[] = []
    const totalItems = props.ui.listFilterOptions.reduce((sum, option) => sum + option.count, 0)
    const allSelected = mode === "all"
    for (let idx = window.windowStart; idx < window.windowEnd; idx++) {
      const cursor = idx === window.selectedIndex ? ">" : " "
      if (idx === 0) {
        lines.push(`${cursor} [${allSelected ? "x" : " "}] All (${totalItems})`)
        continue
      }
      const option = props.ui.listFilterOptions[idx - 1]
      if (!option) continue
      const active = mode === "all" ? true : mode === "subset" && selections?.has(option.id)
      lines.push(`${cursor} [${active ? "x" : " "}] ${option.label} (${option.count})`)
    }
    return lines
  })
  const frameWidth = createMemo(() => {
    const lines = view()
    const maxLine = Math.max(
      ...lines.map((line) => line.length),
      hint.length,
      "List filter".length,
    )
    return Math.min(Math.max(48, maxLine + 6), Math.max(48, props.dimensions().width - 4))
  })
  const frameHeight = createMemo(() => {
    const lines = view()
    return Math.min(Math.max(8, props.dimensions().height - 4), Math.max(8, lines.length + 6))
  })

  return (
    <TextContentModal
      title="List filter"
      width={frameWidth()}
      height={frameHeight()}
      borderColor="#60a5fa"
      titleColor="#60a5fa"
      hint={hint}
      dimensions={props.dimensions}
      text={view().join("\n")}
    />
  )
}
