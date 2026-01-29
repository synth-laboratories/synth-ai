import { For, Show, createMemo, type Accessor } from "solid-js"
import { COLORS } from "../theme"
import type { Message, Part } from "./client"

export interface MessageBubbleProps {
  msg: Message
  /** Parts accessor for fine-grained reactivity */
  parts: Accessor<Part[]>
  /** Max bubble width in terminal columns */
  maxWidth: number
  /**
   * Optional hard cap on the number of rendered lines.
   * This prevents terminal scroll / layout corruption when messages are extremely long.
   */
  maxLines?: number
}

function wrapText(text: string, maxWidth: number): string[] {
  const lines: string[] = []
  const paragraphs = text.split("\n")

  for (const para of paragraphs) {
    if (para.length <= maxWidth) {
      lines.push(para)
      continue
    }

    // Word wrap with fallback for long tokens
    const words = para.split(" ")
    let currentLine = ""

    for (const word of words) {
      if (currentLine.length === 0) {
        if (word.length <= maxWidth) {
          currentLine = word
        } else {
          // Hard wrap long single token
          for (let i = 0; i < word.length; i += maxWidth) {
            lines.push(word.slice(i, i + maxWidth))
          }
          currentLine = ""
        }
      } else if (currentLine.length + 1 + word.length <= maxWidth) {
        currentLine += " " + word
      } else {
        lines.push(currentLine)
        if (word.length <= maxWidth) {
          currentLine = word
        } else {
          for (let i = 0; i < word.length; i += maxWidth) {
            lines.push(word.slice(i, i + maxWidth))
          }
          currentLine = ""
        }
      }
    }

    if (currentLine.length > 0) lines.push(currentLine)
  }

  return lines
}

/** Render a single part to {text, color} or null if not renderable */
function renderPart(p: any, contentWidth: number): { text: string; color: string } | null {
  // Text content (main response)
  if (p.type === "text" && p.text) {
    const trimmed = String(p.text).trim()
    if (!trimmed) return null
    return { text: wrapText(trimmed, contentWidth).join("\n"), color: COLORS.text }
  }

  // Reasoning/thinking content
  if (p.type === "reasoning" && p.text) {
    const trimmed = String(p.text).trim()
    if (!trimmed) return null
    return { text: wrapText(trimmed, contentWidth).join("\n"), color: "#a78bfa" }
  }

  // Tool calls
  if (p.type === "tool") {
    const toolName = p.tool || p.state?.title || "tool"
    const status = p.state?.status || "pending"
    return { text: `[${toolName}] ${status}`, color: COLORS.warning }
  }

  // Skill results (from OpenCode skills)
  if (p.type === "skill") {
    const skillName = p.skill || p.name || "skill"
    const status = p.state?.status || "completed"
    return { text: `[${skillName}] ${status}`, color: "#22d3ee" }
  }

  // Skip metadata parts
  if (p.type === "step-start" || p.type === "step-finish") return null

  // Fallback: if part has any text-like content, try to show it
  if (p.content && typeof p.content === "string") {
    const trimmed = p.content.trim()
    if (!trimmed) return null
    return { text: wrapText(trimmed, contentWidth).join("\n"), color: COLORS.text }
  }

  return null
}

export function MessageBubble(props: MessageBubbleProps) {
  const isUser = createMemo(() => props.msg.role === "user")
  const isAssistant = createMemo(() => props.msg.role === "assistant")

  const bubble = createMemo(() => {
    if (isUser()) {
      return {
        borderColor: COLORS.textAccent,
        bodyBg: "#071426",
      }
    }
    if (isAssistant()) {
      return {
        borderColor: COLORS.success,
        bodyBg: "#061a13",
      }
    }
    return {
      borderColor: COLORS.border,
      bodyBg: COLORS.bgTabs,
    }
  })

  // Conservative estimate for inner content width (border + padding)
  const contentWidth = createMemo(() => Math.max(10, props.maxWidth - 4))

  // Pre-filter parts to only those with renderable content
  // Call props.parts() accessor inside memo for fine-grained reactivity
  const renderableParts = createMemo(() => {
    return props.parts()
      .map((part) => renderPart(part as any, contentWidth()))
      .filter((r): r is { text: string; color: string } => r !== null)
  })

  const renderableLines = createMemo(() => {
    const maxLines = props.maxLines
    const lines: Array<{ text: string; color: string }> = []
    for (const part of renderableParts()) {
      const partLines = String(part.text).split("\n")
      for (const line of partLines) {
        lines.push({ text: line, color: part.color })
      }
    }
    if (typeof maxLines === "number" && maxLines > 0 && lines.length > maxLines) {
      return lines.slice(lines.length - maxLines)
    }
    return lines
  })

  return (
    <box
      flexDirection="column"
      width={props.maxWidth}
      borderStyle="single"
      borderColor={bubble().borderColor}
      backgroundColor={bubble().bodyBg}
    >
      {/* Body */}
      <box flexDirection="column" paddingLeft={1} paddingRight={1} paddingTop={0} paddingBottom={0}>
        <Show when={renderableLines().length > 0} fallback={<text fg={COLORS.textDim}>(loading...)</text>}>
          <For each={renderableLines()}>
            {(line) => <text fg={line.color}>{line.text}</text>}
          </For>
        </Show>
      </box>
    </box>
  )
}


