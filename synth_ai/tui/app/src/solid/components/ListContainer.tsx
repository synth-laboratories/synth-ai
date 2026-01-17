import { For, Show, createEffect, createMemo, type JSX } from "solid-js"

import { COLORS } from "../theme"
import { ListCard, type ListCardStyleContext } from "./ListCard"
import type { ListWindowItem } from "../utils/list"
import { log } from "../../utils/log"

type ListSelectionSnapshot = {
  title: string
  selectedIndex: number
  itemsLength: number
  firstIndex: number | null
  lastIndex: number | null
  selectedCount: number
  focused: boolean
}

export type ListContainerProps<T> = {
  items: Array<ListWindowItem<T>>
  selectedIndex: number
  focused: boolean
  title: string
  totalCount: number
  emptyFallback: JSX.Element
  renderItem: (item: T, ctx: ListCardStyleContext, globalIndex: number) => JSX.Element
  footer?: JSX.Element
  width?: number
  height?: number
  flexGrow?: number
  paddingLeft?: number
  paddingRight?: number
  paddingTop?: number
  paddingBottom?: number
  borderColor?: string
  border?: boolean
  titleAlignment?: "left" | "center" | "right"
}

export function ListContainer<T>(props: ListContainerProps<T>) {
  const borderColor = () =>
    props.borderColor ?? (props.focused ? COLORS.textAccent : COLORS.border)
  const titleAlignment = () => props.titleAlignment ?? "left"
  const borderProps = () => (props.border === undefined ? {} : { border: props.border })
  const title = createMemo(() => props.title || "")

  createEffect((prev: ListSelectionSnapshot | null) => {
    const items = props.items
    const selectedIndex = props.selectedIndex
    const itemsLength = items.length
    const firstIndex = itemsLength > 0 ? items[0].globalIndex : null
    const lastIndex = itemsLength > 0 ? items[itemsLength - 1].globalIndex : null
    let selectedCount = 0
    for (const entry of items) {
      if (entry.globalIndex === selectedIndex) selectedCount += 1
    }
    const snapshot: ListSelectionSnapshot = {
      title: title(),
      selectedIndex,
      itemsLength,
      firstIndex,
      lastIndex,
      selectedCount,
      focused: props.focused,
    }
    const changed =
      !prev ||
      prev.title !== snapshot.title ||
      prev.selectedIndex !== snapshot.selectedIndex ||
      prev.itemsLength !== snapshot.itemsLength ||
      prev.firstIndex !== snapshot.firstIndex ||
      prev.lastIndex !== snapshot.lastIndex ||
      prev.selectedCount !== snapshot.selectedCount ||
      prev.focused !== snapshot.focused
    if (changed) {
      log("state", "list render selection", snapshot)
      const selectedInRange =
        firstIndex != null &&
        lastIndex != null &&
        selectedIndex >= firstIndex &&
        selectedIndex <= lastIndex
      if (itemsLength > 0 && selectedInRange && selectedCount !== 1) {
        log("state", "list render selection mismatch", snapshot)
      }
    }
    return snapshot
  }, null)

  return (
    <box
      {...borderProps()}
      width={props.width}
      height={props.height}
      flexGrow={props.flexGrow}
      borderStyle="single"
      borderColor={borderColor()}
      title={title()}
      titleAlignment={titleAlignment()}
      flexDirection="column"
      paddingLeft={props.paddingLeft}
      paddingRight={props.paddingRight}
      paddingTop={props.paddingTop}
      paddingBottom={props.paddingBottom}
    >
      <Show when={props.items.length > 0} fallback={props.emptyFallback}>
        <box flexDirection="column">
          <For each={props.items}>
            {(entry) => {
              const isSelected = () => entry.globalIndex === props.selectedIndex
              return (
                <ListCard isSelected={isSelected} panelFocused={() => props.focused}>
                  {(ctx) => props.renderItem(entry.item, ctx, entry.globalIndex)}
                </ListCard>
              )
            }}
          </For>
          <Show when={props.footer}>{props.footer}</Show>
        </box>
      </Show>
    </box>
  )
}
