import { For, Show, createMemo } from "solid-js"

import { clampLine } from "../../utils/text"
import { COLORS } from "../theme"

export type SuggestionItem = {
  id: string
  label: string
  description?: string
}

export type SuggestionPopupProps = {
  title: string
  query?: string
  items: SuggestionItem[]
  selectedIndex: number
  width: number
  height: number
  left: number
  top: number
  emptyText?: string
  labelColor?: string
}

export function SuggestionPopup(props: SuggestionPopupProps) {
  const contentWidth = createMemo(() => Math.max(1, props.width - 2))
  const labelMax = createMemo(() => Math.max(1, Math.floor(contentWidth() * 0.6)))
  const emptyText = () => props.emptyText ?? "No matches"
  const baseLabelColor = () => props.labelColor ?? COLORS.textAccent

  return (
    <box
      position="absolute"
      left={props.left}
      top={props.top}
      width={props.width}
      height={props.height}
      backgroundColor={COLORS.bgTabs}
      border
      borderStyle="single"
      borderColor={COLORS.borderAccent}
      flexDirection="column"
    >
      <box paddingLeft={1} paddingRight={1} backgroundColor={COLORS.bgHeader}>
        <text fg={COLORS.text}>
          <span style={{ bold: true }}>{props.title}</span>
          <Show when={props.query}>
            <span style={{ fg: COLORS.textDim }}> {props.query}</span>
          </Show>
        </text>
      </box>
      <box flexDirection="column" paddingLeft={1} paddingRight={1} flexGrow={1}>
        <Show
          when={props.items.length > 0}
          fallback={<text fg={COLORS.textDim}>{emptyText()}</text>}
        >
          <For each={props.items}>
            {(item, index) => {
              const isSelected = () => index() === props.selectedIndex
              const labelColor = isSelected() ? COLORS.textSelected : baseLabelColor()
              const descriptionColor = isSelected() ? COLORS.textBright : COLORS.textDim
              const maxWidth = contentWidth()
              const label = item.label
              if (label.length >= maxWidth) {
                return (
                  <box backgroundColor={isSelected() ? COLORS.bgSelection : undefined}>
                    <text fg={labelColor}>{clampLine(label, maxWidth)}</text>
                  </box>
                )
              }
              const remaining = Math.max(0, maxWidth - label.length - 2)
              const description = remaining > 0 ? clampLine(item.description ?? "", remaining) : ""
              return (
                <box backgroundColor={isSelected() ? COLORS.bgSelection : undefined}>
                  <text>
                    <span style={{ fg: labelColor }}>{clampLine(label, labelMax())}</span>
                    <Show when={description}>
                      <span style={{ fg: descriptionColor }}>{"  " + description}</span>
                    </Show>
                  </text>
                </box>
              )
            }}
          </For>
        </Show>
      </box>
    </box>
  )
}
