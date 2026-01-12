import { For, Show, createMemo } from "solid-js"
import { COLORS } from "../theme"

export type RightPanelContextItem = {
  id: string
  name: string
  value: string
}

export type RightPanelEntry = {
  id: string
  kind: "tool" | "skill"
  name: string
  status: string
  detail: string
  /** Unix epoch millis when the tool/skill started (best-effort) */
  startedAtMs?: number
}

export type TodoItem = {
  id: string
  content: string
  status: "pending" | "in_progress" | "completed" | "cancelled"
  priority?: "high" | "medium" | "low"
}

export type RightPanelMode = "tools" | "todos"

export interface RightPanelBrowserProps {
  width: number
  height: number
  title: string
  focused: boolean
  mode: RightPanelMode
  onModeChange?: (mode: RightPanelMode) => void
  contextItems: RightPanelContextItem[]
  entries: RightPanelEntry[]
  todos: TodoItem[]
  selectedIndex: number
  listScroll: number
  detailScroll: number
}

function wrapText(text: string, maxWidth: number): string[] {
  const lines: string[] = []
  const paragraphs = String(text || "").split("\n")
  for (const para of paragraphs) {
    if (!para) {
      lines.push("")
      continue
    }
    if (para.length <= maxWidth) {
      lines.push(para)
      continue
    }
    const words = para.split(" ")
    let current = ""
    for (const word of words) {
      if (!current) {
        if (word.length <= maxWidth) {
          current = word
        } else {
          for (let i = 0; i < word.length; i += maxWidth) lines.push(word.slice(i, i + maxWidth))
          current = ""
        }
        continue
      }
      if (current.length + 1 + word.length <= maxWidth) {
        current += " " + word
      } else {
        lines.push(current)
        if (word.length <= maxWidth) {
          current = word
        } else {
          for (let i = 0; i < word.length; i += maxWidth) lines.push(word.slice(i, i + maxWidth))
          current = ""
        }
      }
    }
    if (current) lines.push(current)
  }
  return lines
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text
  if (max <= 1) return text.slice(0, max)
  return text.slice(0, max - 1) + "…"
}

function formatTime(ms?: number): string {
  if (!ms || !Number.isFinite(ms)) return ""
  const d = new Date(ms)
  // HH:MM:SS (local time). Keep it short but unambiguous.
  const hh = String(d.getHours()).padStart(2, "0")
  const mm = String(d.getMinutes()).padStart(2, "0")
  const ss = String(d.getSeconds()).padStart(2, "0")
  return `${hh}:${mm}:${ss}`
}

function kindColor(kind: RightPanelEntry["kind"]): string {
  return kind === "tool" ? COLORS.warning : "#22d3ee"
}

function todoStatusIcon(status: TodoItem["status"]): string {
  switch (status) {
    case "completed": return "✓"
    case "in_progress": return "▸"
    case "cancelled": return "✗"
    default: return "○"
  }
}

function todoStatusColor(status: TodoItem["status"]): string {
  switch (status) {
    case "completed": return "#22c55e"  // green
    case "in_progress": return "#fbbf24" // yellow/amber
    case "cancelled": return "#64748b"   // gray
    default: return COLORS.textDim       // pending
  }
}

export function RightPanelBrowser(props: RightPanelBrowserProps) {
  const innerWidth = createMemo(() => Math.max(10, props.width - 2))
  const contentWidth = createMemo(() => Math.max(10, props.width - 4))

  const headerHeight = 1
  const ctxHeader = props.contextItems.length ? 1 : 0
  const ctxHeight = props.contextItems.length ? props.contextItems.length : 0
  const ctxBlockHeight = ctxHeader + ctxHeight

  // Layout: header + (context block) + list header + list + detail header + detail
  const remaining = createMemo(() => Math.max(0, props.height - 2 - headerHeight)) // minus border+header
  const listHeaderHeight = 1
  const detailHeaderHeight = 1

  const listHeight = createMemo(() => {
    const avail = remaining() - ctxBlockHeight - listHeaderHeight - detailHeaderHeight - 1 // 1 divider row
    const target = Math.floor(avail * 0.38)
    return Math.max(5, Math.min(avail - 4, target))
  })

  const detailHeight = createMemo(() => {
    const avail = remaining() - ctxBlockHeight - listHeaderHeight - detailHeaderHeight - 1
    return Math.max(3, avail - listHeight())
  })

  const selected = createMemo(() => {
    const idx = Math.max(0, Math.min(props.entries.length - 1, props.selectedIndex))
    return props.entries[idx]
  })

  const selectedDetailLines = createMemo(() => {
    const detail = selected()?.detail || ""
    if (!detail) return ["(no output)"]
    return wrapText(detail, contentWidth())
  })

  const visibleDetailLines = createMemo(() => {
    const lines = selectedDetailLines()
    const max = detailHeight() - 1 // leave one row for breathing room
    const offset = Math.max(0, Math.min(props.detailScroll, Math.max(0, lines.length - max)))
    return lines.slice(offset, offset + max)
  })

  const visibleEntries = createMemo(() => {
    const max = listHeight() - 1
    const offset = Math.max(0, props.listScroll)
    return props.entries.slice(offset, offset + max)
  })

  const selectedId = createMemo(() => selected()?.id)

  return (
    <box
      flexDirection="column"
      width={props.width}
      height={props.height}
      borderStyle="single"
      borderColor={props.focused ? COLORS.borderAccent : COLORS.border}
    >
      <box backgroundColor={COLORS.bgHeader} paddingLeft={1} paddingRight={1} flexDirection="column">
        <box flexDirection="row" justifyContent="space-between">
          <text fg={COLORS.text}>
            <span style={{ bold: true }}>{props.title}</span>
            <span style={{ fg: COLORS.textDim }}>
              {props.focused ? "  [focused]" : "  Ctrl+P"}
            </span>
          </text>
          <text fg={COLORS.textDim}>Tab to switch</text>
        </box>
        <box flexDirection="row" gap={2}>
          <text fg={props.mode === "tools" ? COLORS.text : COLORS.textDim}>
            <span style={{ bold: props.mode === "tools" }}>
              {props.mode === "tools" ? "▸ " : "  "}Tools ({props.entries.length})
            </span>
          </text>
          <text fg={props.mode === "todos" ? "#22c55e" : COLORS.textDim}>
            <span style={{ bold: props.mode === "todos" }}>
              {props.mode === "todos" ? "▸ " : "  "}Todos ({props.todos.length})
            </span>
          </text>
        </box>
      </box>

      <box flexDirection="column" overflow="hidden" paddingLeft={1} paddingRight={1}>
        {/* Context - always shown */}
        <Show when={props.contextItems.length > 0}>
          <text fg={COLORS.text}>Context</text>
          <For each={props.contextItems}>
            {(it) => (
              <text fg={COLORS.textDim}>
                {truncate(`${it.name}: ${it.value}`, contentWidth())}
              </text>
            )}
          </For>
        </Show>

        {/* TODOS VIEW */}
        <Show when={props.mode === "todos"}>
          <box flexDirection="column" height={remaining() - ctxBlockHeight - 2} overflow="hidden">
            <Show
              when={props.todos.length > 0}
              fallback={<text fg={COLORS.textDim}>(no todos yet)</text>}
            >
              <For each={props.todos}>
                {(todo, idx) => (
                  <box
                    width={innerWidth()}
                    backgroundColor={idx() === props.selectedIndex && props.focused ? "#1e293b" : undefined}
                    paddingLeft={1}
                    paddingRight={1}
                  >
                    <text fg={todoStatusColor(todo.status)}>
                      {todoStatusIcon(todo.status)} {truncate(todo.content, contentWidth() - 3)}
                    </text>
                  </box>
                )}
              </For>
            </Show>
          </box>
        </Show>

        {/* TOOLS VIEW */}
        <Show when={props.mode === "tools"}>
          <text fg={COLORS.text}>Tool calls</text>
          <box flexDirection="column" height={listHeight()} overflow="hidden">
            <Show
              when={props.entries.length > 0}
              fallback={<text fg={COLORS.textDim}>(no tool calls yet)</text>}
            >
              <For each={visibleEntries()}>
                {(e) => {
                  const ts = formatTime(e.startedAtMs)
                  const pillText = ts ? `${ts}  [${e.name}] ${e.status}` : `[${e.name}] ${e.status}`
                  const pill = truncate(pillText, contentWidth())
                  const isSel = e.id === selectedId()
                  return (
                    <box
                      width={innerWidth()}
                      backgroundColor={isSel ? (props.focused ? "#1e293b" : "#0b1220") : undefined}
                      paddingLeft={isSel ? 1 : 0}
                      paddingRight={1}
                    >
                      <text fg={isSel ? COLORS.text : kindColor(e.kind)}>
                        {isSel ? "› " : "  "}
                        {pill}
                      </text>
                    </box>
                )
              }}
            </For>
          </Show>
        </box>

          <text fg={COLORS.text}>Details</text>
          <box flexDirection="column" height={detailHeight()} overflow="hidden">
            <For each={visibleDetailLines()}>{(l) => <text fg={COLORS.textDim}>{truncate(l, contentWidth())}</text>}</For>
          </box>
        </Show>
      </box>
    </box>
  )
}

