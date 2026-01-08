/**
 * OpenCode SDK Client Wrapper
 *
 * Thin wrapper around @opencode-ai/sdk for use in synth-ai TUI
 */
import { createOpencodeClient, type OpencodeClient } from "@opencode-ai/sdk/v2"

export type { OpencodeClient }

let clientInstance: OpencodeClient | null = null

export function getClient(url: string): OpencodeClient {
  if (!clientInstance) {
    clientInstance = createOpencodeClient({ baseUrl: url })
  }
  return clientInstance
}

export function resetClient(): void {
  clientInstance = null
}

// Re-export types we need
export type {
  Message,
  UserMessage,
  AssistantMessage,
  Part,
  TextPart,
  ToolPart,
  FilePart,
  Event,
  Session,
} from "@opencode-ai/sdk/v2"
