import type { Message, Part } from "./client"

// The SDK returns messages in this wrapper format.
export type MessageWrapper = {
  info: Message
  parts: Part[]
}

export type VisibleMessage = {
  wrapper: MessageWrapper
  maxLines?: number
}
