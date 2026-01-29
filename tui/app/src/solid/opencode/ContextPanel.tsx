/**
 * ContextPanel - Right panel showing context for OpenCode
 *
 * Displays static Synth docs and user-editable synth.md instructions.
 */
import { createMemo, Show } from "solid-js"
import { COLORS } from "../theme"

// Import static docs as raw text
import synthDocsContent from "../../data/synth-docs.md?raw"

export interface ContextPanelProps {
  width: number
  height: number
  synthMdPath: string | null
  synthMdContent: string
  onSynthMdChange?: (content: string) => void
  scrollOffset?: number
  /** Whether Synth docs are injected into prompts */
  docsEnabled?: boolean
  /** Whether synth.md is injected into prompts */
  synthMdEnabled?: boolean
}

/**
 * ContextPanel displays:
 * 1. Static Synth documentation (read-only, scrollable)
 * 2. User's synth.md content (editable)
 */
export function ContextPanel(props: ContextPanelProps) {
  const localScrollOffset = createMemo(() => props.scrollOffset || 0)

  // Split heights: 60% for static docs, 40% for synth.md
  const staticDocsHeight = createMemo(() => Math.floor((props.height - 4) * 0.6))
  const synthMdHeight = createMemo(() => props.height - staticDocsHeight() - 4)

  // Truncate static docs to fit in the panel
  const visibleStaticDocs = createMemo(() => {
    const lines = synthDocsContent.split("\n")
    const offset = localScrollOffset()
    const maxLines = staticDocsHeight() - 2
    return lines.slice(offset, offset + maxLines).join("\n")
  })

  // Format file path for display
  const displayPath = createMemo(() => {
    if (!props.synthMdPath) return "synth.md not found"
    // Shorten path if too long
    const maxLen = props.width - 6
    if (props.synthMdPath.length > maxLen) {
      return "..." + props.synthMdPath.slice(-maxLen + 3)
    }
    return props.synthMdPath
  })

  return (
    <box
      flexDirection="column"
      width={props.width}
      height={props.height}
      borderStyle="single"
      borderColor={COLORS.border}
    >
      {/* Header */}
      <box backgroundColor={COLORS.bgHeader} paddingLeft={1} paddingRight={1}>
        <text fg={COLORS.text}>
          <span style={{ bold: true }}>Context</span>
        </text>
      </box>

      {/* Static Docs Section */}
      <box
        flexDirection="column"
        height={staticDocsHeight()}
        borderStyle="single"
        borderColor={props.docsEnabled !== false ? COLORS.borderAccent : COLORS.borderDim}
        marginTop={0}
      >
        <box paddingLeft={1} backgroundColor={COLORS.bgTabs}>
          <text fg={props.docsEnabled !== false ? COLORS.text : COLORS.textDim}>
            Synth Docs {props.docsEnabled !== false ? "[1:ON]" : "[1:OFF]"}
          </text>
        </box>
        <box flexDirection="column" overflow="hidden" paddingLeft={1} paddingRight={1}>
          <text fg={COLORS.textDim}>{visibleStaticDocs()}</text>
        </box>
      </box>

      {/* synth.md Section */}
      <box
        flexDirection="column"
        height={synthMdHeight()}
        borderStyle="single"
        borderColor={props.synthMdEnabled !== false ? COLORS.borderAccent : COLORS.borderDim}
      >
        <box paddingLeft={1} backgroundColor={COLORS.bgTabs}>
          <text fg={props.synthMdEnabled !== false ? COLORS.text : COLORS.textDim}>
            {displayPath()} {props.synthMdEnabled !== false ? "[2:ON]" : "[2:OFF]"}
          </text>
        </box>
        <box flexDirection="column" overflow="hidden" paddingLeft={1} paddingRight={1}>
          <Show
            when={props.synthMdContent}
            fallback={<text fg={COLORS.textDim}>(no custom instructions)</text>}
          >
            <text fg={COLORS.text}>{props.synthMdContent}</text>
          </Show>
        </box>
      </box>
    </box>
  )
}

/**
 * Get the combined context string for injection into prompts.
 */
export function getCombinedContext(synthMdContent: string): string {
  return `<context>
<synth-documentation>
${synthDocsContent}
</synth-documentation>

<user-instructions>
${synthMdContent || "(No custom instructions provided)"}
</user-instructions>
</context>`
}

/**
 * Export the raw static docs for use elsewhere
 */
export { synthDocsContent }
