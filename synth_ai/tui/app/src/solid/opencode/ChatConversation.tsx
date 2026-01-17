import { For, Show, createEffect, createMemo, createSignal } from "solid-js"

import type { AssistantMessage, Part } from "./client"
import { MessageBubble } from "./MessageBubble"
import type { MessageWrapper, VisibleMessage } from "./chat-types"
import { COLORS } from "../theme"
import type { KeyAction } from "../../input/keymap"
import { CHAT_INPUT_HEIGHT } from "./ChatInput"

export type ChatConversationController = {
  handleScrollAction: (action: KeyAction | null) => boolean
  resetScroll: () => void
}

export type ChatConversationProps = {
  width: number
  height: number
  headerHeight: number
  sessionId: string
  messages: MessageWrapper[]
  partsStore: Record<string, Part[]>
  error: string | null
  scrollFocused: boolean
  isLoading: boolean
  register?: (controller: ChatConversationController) => void
}

const MESSAGE_FRAME_INSET = 2

function countWrappedLines(text: string, maxWidth: number): number {
  if (!text) return 0
  let lines = 0
  for (const para of String(text).split("\n")) {
    if (!para) {
      lines += 1
      continue
    }
    if (para.length <= maxWidth) {
      lines += 1
      continue
    }
    const words = para.split(" ")
    let current = ""
    for (const word of words) {
      if (!current) {
        if (word.length <= maxWidth) {
          current = word
        } else {
          lines += Math.ceil(word.length / maxWidth)
          current = ""
        }
        continue
      }
      if (current.length + 1 + word.length <= maxWidth) {
        current += " " + word
      } else {
        lines += 1
        if (word.length <= maxWidth) {
          current = word
        } else {
          lines += Math.ceil(word.length / maxWidth)
          current = ""
        }
      }
    }
    if (current) lines += 1
  }
  return lines
}

export function ChatConversation(props: ChatConversationProps) {
  const messageAreaWidth = createMemo(() => Math.max(0, props.width - MESSAGE_FRAME_INSET))
  const messageAreaHeight = createMemo(() =>
    Math.max(0, props.height - props.headerHeight - CHAT_INPUT_HEIGHT - MESSAGE_FRAME_INSET),
  )
  const bubbleWidth = createMemo(() => {
    const hardMax = Math.min(72, Math.max(32, messageAreaWidth() - 4))
    const target = Math.floor(messageAreaWidth() * 0.62)
    return Math.max(32, Math.min(hardMax, target))
  })
  const [messageScrollOffset, setMessageScrollOffset] = createSignal(0)
  const maxScrollOffset = createMemo(() => Math.max(0, props.messages.length - 1))

  createEffect((prevCount: number) => {
    const count = props.messages.length
    let nextOffset = messageScrollOffset()
    if (prevCount >= 0 && count > prevCount && nextOffset > 0) {
      nextOffset += count - prevCount
    }
    const maxOffset = maxScrollOffset()
    if (nextOffset > maxOffset) {
      nextOffset = maxOffset
    }
    if (nextOffset !== messageScrollOffset()) {
      setMessageScrollOffset(nextOffset)
    }
    return count
  }, 0)

  const scrollBy = (delta: number) => {
    const maxOffset = maxScrollOffset()
    setMessageScrollOffset((current) => {
      const next = Math.max(0, Math.min(current + delta, maxOffset))
      return next
    })
  }

  const scrollToLatest = () => {
    setMessageScrollOffset(0)
  }

  const scrollToOldest = () => {
    setMessageScrollOffset(maxScrollOffset())
  }

  const handleScrollAction = (action: KeyAction | null): boolean => {
    if (!props.scrollFocused) return false
    if (!action) return false
    const pageStep = Math.max(1, Math.floor(messageAreaHeight() / 4))
    if (action === "nav.up") {
      scrollBy(1)
    } else if (action === "nav.down") {
      scrollBy(-1)
    } else if (action === "nav.pageUp") {
      scrollBy(pageStep)
    } else if (action === "nav.pageDown") {
      scrollBy(-pageStep)
    } else if (action === "nav.home") {
      scrollToOldest()
    } else if (action === "nav.end") {
      scrollToLatest()
    } else {
      return false
    }
    return true
  }

  const controller: ChatConversationController = {
    handleScrollAction,
    resetScroll: scrollToLatest,
  }

  props.register?.(controller)

  const visibleMessages = createMemo<VisibleMessage[]>(() => {
    const maxBubbleContentWidth = Math.max(10, bubbleWidth() - 4)
    const maxHeight = messageAreaHeight()
    const msgs = props.messages
    if (!msgs.length) return []

    const result: VisibleMessage[] = []
    let used = 0

    const loadingReserve = props.isLoading ? 1 : 0
    const budget = Math.max(0, maxHeight - loadingReserve)

    const startIndex = Math.min(msgs.length - 1, msgs.length - 1 - messageScrollOffset())
    if (startIndex < 0) return []

    for (let i = startIndex; i >= 0; i--) {
      const wrapper = msgs[i]
      const parts = props.partsStore[wrapper.info.id] || wrapper.parts || []
      let contentLines = 0
      for (const p of parts as any[]) {
        const text = (p?.text ?? p?.content ?? "").toString()
        if (!text) continue
        contentLines += countWrappedLines(text.trim(), maxBubbleContentWidth)
      }
      const bubbleHeight = Math.max(3, contentLines + 2)
      const blockHeight = 1 + bubbleHeight + 1

      if (used + blockHeight <= budget) {
        result.push({ wrapper })
        used += blockHeight
        continue
      }

      if (result.length === 0 && budget > 0) {
        const remainingForBubble = Math.max(0, budget - 2)
        const remainingContent = Math.max(0, remainingForBubble - 2)
        result.push({ wrapper, maxLines: Math.max(1, remainingContent) })
      }
      break
    }

    return result.reverse()
  })

  return (
    <box
      flexDirection="column"
      flexGrow={1}
      border
      borderStyle="single"
      borderColor={props.scrollFocused ? COLORS.borderAccent : COLORS.borderDim}
    >
      <box
        flexDirection="column"
        flexGrow={1}
        overflow="hidden"
        paddingLeft={1}
        paddingRight={1}
        backgroundColor={props.scrollFocused ? COLORS.bgTabs : undefined}
      >
        <Show when={props.error}>
          <box backgroundColor="#7f1d1d" paddingLeft={1} paddingRight={1}>
            <text fg="#fca5a5">Error: {props.error}</text>
          </box>
        </Show>

        <Show when={!props.sessionId && props.messages.length === 0}>
          <box paddingLeft={1} paddingRight={1} paddingTop={1}>
            <text fg={COLORS.textDim}>
              No OpenCode session selected. Choose one from the Sessions list.
            </text>
          </box>
        </Show>

        <For each={visibleMessages()}>
          {(item) => {
            const wrapper = item.wrapper
            const parts = () => props.partsStore[wrapper.info.id] || wrapper.parts || []
            return (
              <box
                flexDirection="row"
                justifyContent={wrapper.info.role === "user" ? "flex-end" : "flex-start"}
                marginBottom={1}
              >
                <box flexDirection="column" width={bubbleWidth()}>
                  <box flexDirection="row" justifyContent="space-between" paddingLeft={1} paddingRight={1} marginBottom={0}>
                    <text fg={wrapper.info.role === "user" ? COLORS.textAccent : COLORS.success}>
                      <span style={{ bold: true }}>{wrapper.info.role === "user" ? "You" : "Assistant"}</span>
                    </text>
                    <Show when={wrapper.info.role === "assistant"}>
                      <text fg={COLORS.textDim}>
                        {(() => {
                          const assistant = wrapper.info as AssistantMessage
                          const duration = assistant.time?.completed
                            ? `${((assistant.time.completed - assistant.time.created) / 1000).toFixed(1)}s`
                            : null
                          return `${assistant.mode} · ${assistant.modelID}${duration ? ` · ${duration}` : ""}`
                        })()}
                      </text>
                    </Show>
                  </box>
                  <MessageBubble msg={wrapper.info} parts={parts} maxWidth={bubbleWidth()} maxLines={item.maxLines} />
                </box>
              </box>
            )
          }}
        </For>

        <Show when={props.isLoading}>
          <text fg={COLORS.textDim}>Thinking...</text>
        </Show>
      </box>
    </box>
  )
}
