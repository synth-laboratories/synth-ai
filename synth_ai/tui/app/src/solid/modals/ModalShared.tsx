import { type Accessor, createMemo, type JSX } from "solid-js"

import { ModalFrame } from "../components/ModalFrame"
import { buildScrollableModal, type ScrollableModalView } from "../utils/modal"

type HintVisibility = "auto" | "always" | "never"

type ScrollableHintOptions = {
  baseHints?: string[]
  scrollHint?: string
  showScrollHint?: HintVisibility
  showRange?: HintVisibility
}

function buildScrollableHint(view: ScrollableModalView, options: ScrollableHintOptions): string {
  const baseHints = (options.baseHints ?? []).filter((hint) => hint.trim().length > 0)
  const showRange = options.showRange ?? "auto"
  const showScrollHint = options.showScrollHint ?? "auto"
  const isScrollable = view.lines.length > view.bodyHeight
  const parts: string[] = []

  if (showRange === "always" || (showRange === "auto" && isScrollable)) {
    parts.push(`[${view.offset + 1}-${view.offset + view.visible.length}/${view.lines.length}]`)
  }

  if (options.scrollHint && (showScrollHint === "always" || (showScrollHint === "auto" && isScrollable))) {
    parts.push(options.scrollHint)
  }

  parts.push(...baseHints)
  return parts.join(" | ")
}

type BaseModalProps = {
  title: string
  width: number
  height: number
  borderColor: string
  titleColor: string
  hint: string
  dimensions: Accessor<{ width: number; height: number }>
  children: JSX.Element
}

export function BaseModalFrame(props: BaseModalProps) {
  return (
    <ModalFrame
      title={props.title}
      width={props.width}
      height={props.height}
      borderColor={props.borderColor}
      titleColor={props.titleColor}
      hint={props.hint}
      dimensions={props.dimensions}
    >
      {props.children}
    </ModalFrame>
  )
}

type TextContentModalProps = Omit<BaseModalProps, "children"> & {
  text: string
  textColor?: string
}

export function TextContentModal(props: TextContentModalProps) {
  return (
    <BaseModalFrame {...props} hint={props.hint}>
      <text fg={props.textColor ?? "#e2e8f0"}>{props.text}</text>
    </BaseModalFrame>
  )
}

type TextInputModalProps = Omit<BaseModalProps, "children"> & {
  label: string
  placeholder: string
  onInput: (value: string) => void
  setInputRef: (ref: any) => void
  textColor?: string
}

export function TextInputModal(props: TextInputModalProps) {
  return (
    <BaseModalFrame {...props} hint={props.hint}>
      <box flexDirection="column" gap={1}>
        <text fg={props.textColor ?? "#e2e8f0"}>{props.label}</text>
        <input
          placeholder={props.placeholder}
          onInput={(value) => props.onInput(value)}
          ref={(ref) => {
            props.setInputRef(ref)
          }}
        />
      </box>
    </BaseModalFrame>
  )
}

type ScrollableTextModalProps = Omit<BaseModalProps, "children" | "hint"> & {
  raw: string
  offset: number
  hint: ScrollableHintOptions
  showRangeInTitle?: HintVisibility
  textColor?: string
}

export function ScrollableTextModal(props: ScrollableTextModalProps) {
  const view = createMemo(() => buildScrollableModal(props.raw, props.width, props.height, props.offset))
  const title = createMemo(() => {
    const showRangeInTitle = props.showRangeInTitle ?? "never"
    const isScrollable = view().lines.length > view().bodyHeight
    if (showRangeInTitle === "never" || (showRangeInTitle === "auto" && !isScrollable)) {
      return props.title
    }
    const range = `[${view().offset + 1}-${view().offset + view().visible.length}/${view().lines.length}]`
    return `${props.title} ${range}`.trim()
  })
  const hint = createMemo(() => buildScrollableHint(view(), props.hint))

  return (
    <BaseModalFrame
      title={title()}
      width={props.width}
      height={props.height}
      borderColor={props.borderColor}
      titleColor={props.titleColor}
      hint={hint()}
      dimensions={props.dimensions}
    >
      <text fg={props.textColor ?? "#e2e8f0"}>{view().visible.join("\n")}</text>
    </BaseModalFrame>
  )
}

type LoadingModalProps = {
  title: string
  dimensions: Accessor<{ width: number; height: number }>
  borderColor?: string
  titleColor?: string
  hint?: string
  message?: string
  width?: number
  height?: number
  textColor?: string
}

export function LoadingModal(props: LoadingModalProps) {
  const borderColor = props.borderColor ?? "#60a5fa"
  const titleColor = props.titleColor ?? "#60a5fa"
  const hint = props.hint ?? "Loading..."
  const message = props.message ?? "Loading..."
  return (
    <BaseModalFrame
      title={props.title}
      width={props.width ?? 60}
      height={props.height ?? 8}
      borderColor={borderColor}
      titleColor={titleColor}
      hint={hint}
      dimensions={props.dimensions}
    >
      <text fg={props.textColor ?? "#e2e8f0"}>{message}</text>
    </BaseModalFrame>
  )
}
