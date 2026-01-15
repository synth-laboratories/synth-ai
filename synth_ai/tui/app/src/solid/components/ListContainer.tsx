import { For, Show, type JSX } from "solid-js"

import { COLORS } from "../theme"
import { ListCard, type ListCardStyleContext } from "./ListCard"
import type { ListWindowItem } from "../utils/list"

export type ListContainerProps<T> = {
  items: Array<ListWindowItem<T>>
  selectedIndex: number
  focused: boolean
  title: string
  totalCount: number
  emptyFallback: JSX.Element
  renderItem: (item: T, ctx: ListCardStyleContext, globalIndex: number) => JSX.Element
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

  return (
    <box
      {...borderProps()}
      width={props.width}
      height={props.height}
      flexGrow={props.flexGrow}
      borderStyle="single"
      borderColor={borderColor()}
      title={props.title}
      titleAlignment={titleAlignment()}
      flexDirection="column"
      paddingLeft={props.paddingLeft}
      paddingRight={props.paddingRight}
      paddingTop={props.paddingTop}
      paddingBottom={props.paddingBottom}
    >
      <Show when={props.totalCount > 0} fallback={props.emptyFallback}>
        <For each={props.items}>
          {(entry) => {
            const isSelected = entry.globalIndex === props.selectedIndex
            return (
              <ListCard isSelected={isSelected}>
                {(ctx) => props.renderItem(entry.item, ctx, entry.globalIndex)}
              </ListCard>
            )
          }}
        </For>
      </Show>
    </box>
  )
}
