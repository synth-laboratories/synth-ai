import { Show, createMemo, createSignal } from "solid-js"

import { getActionHint, type KeyAction } from "../../input/keymap"
import { clampLine } from "../../utils/text"
import { COLORS } from "../theme"

export type ChatInputProps = {
  focused: boolean
  width: number
  currentModelDisplay: { modelName: string; providerName: string } | null
  onSubmit: (text: string) => boolean
  onInputChange?: (text: string) => void
  register?: (controller: ChatInputController) => void
}

export type ChatInputController = {
  getInputState: () => { inputText: string; inputTextLength: number }
  handleTextInput: (text: string) => boolean
  handleAction: (action: KeyAction) => boolean
  setInputText: (text: string) => void
  clearInput: () => void
}

export const CHAT_INPUT_HEIGHT = 4

export function ChatInput(props: ChatInputProps) {
  const [inputText, setInputText] = createSignal("")
  const isFocused = () => props.focused === true
  const contentWidth = createMemo(() => Math.max(1, Math.floor(props.width - 4)))

  const handleTextInput = (text: string): boolean => {
    if (!isFocused()) return false
    updateInput(inputText() + text)
    return true
  }

  const handleAction = (action: KeyAction): boolean => {
    if (!isFocused()) return false
    if (action === "chat.backspace") {
      updateInput(inputText().slice(0, -1))
      return true
    }
    if (action === "chat.send") {
      const trimmed = inputText().trim()
      if (!trimmed) return true
      const shouldClear = props.onSubmit(trimmed)
      if (shouldClear) {
        updateInput("")
      }
      return true
    }
    return false
  }

  const updateInput = (next: string) => {
    setInputText(next)
    props.onInputChange?.(next)
  }

  const clearInput = () => updateInput("")

  const inputLine = createMemo(() => {
    const width = contentWidth()
    const prefix = "> "
    const text = inputText()
    if (width <= prefix.length) return prefix.slice(0, width)
    const maxText = width - prefix.length
    if (!text) return prefix.trimEnd()
    if (text.length <= maxText) return prefix + text
    return prefix + text.slice(text.length - maxText)
  })

  const modelLine = createMemo(() => {
    if (!props.currentModelDisplay) return "No model selected"
    return `${props.currentModelDisplay.modelName} (${props.currentModelDisplay.providerName})`
  })

  const actionLine = createMemo(() =>
    `${getActionHint("chat.newSession")} | ` +
    `/model | ` +
    `${getActionHint("chat.abort")} | /abort`,
  )

  const leftMax = createMemo(() => Math.max(1, Math.floor(contentWidth() * 0.6)))
  const rightMax = createMemo(() => Math.max(0, contentWidth() - leftMax()))
  const modelLineClamped = createMemo(() => clampLine(modelLine(), leftMax()))
  const actionLineClamped = createMemo(() => {
    const max = rightMax()
    if (max <= 0) return ""
    return clampLine(actionLine(), max)
  })

  const controller: ChatInputController = {
    getInputState: () => ({
      inputText: inputText(),
      inputTextLength: inputText().length,
    }),
    handleTextInput,
    handleAction,
    setInputText: updateInput,
    clearInput,
  }

  props.register?.(controller)

  return (
    <box
      flexDirection="column"
      backgroundColor="#0f172a"
      border
      borderStyle="single"
      borderColor={isFocused() ? COLORS.borderAccent : COLORS.border}
      title="Prompt Input"
      titleAlignment="left"
      width={props.width}
    >
      <box paddingLeft={1} paddingRight={1}>
        <text fg={COLORS.text}>{inputLine()}</text>
      </box>
      <box paddingLeft={1} paddingRight={1} flexDirection="row" justifyContent="space-between">
        <text fg={COLORS.textDim}>{modelLineClamped()}</text>
        <Show when={actionLineClamped()}>
          <text fg="#475569">{actionLineClamped()}</text>
        </Show>
      </box>
    </box>
  )
}
