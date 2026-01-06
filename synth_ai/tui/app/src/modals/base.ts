/**
 * Modal base types and utilities.
 */
import { BoxRenderable, TextRenderable, type CliRenderer } from "@opentui/core"

export type ModalController = {
  readonly isVisible: boolean
  toggle: (visible: boolean) => void
  handleKey?: (key: any) => boolean // returns true if handled
}

/**
 * Configuration for creating a modal UI primitive.
 */
export type ModalConfig = {
  id: string
  width?: number
  height?: number
  borderColor?: string
  titleColor?: string
  zIndex?: number
}

/**
 * Modal UI primitive - provides consistent styling across all modals.
 */
export type ModalUI = {
  box: BoxRenderable
  title: TextRenderable
  content: TextRenderable
  hint: TextRenderable
  visible: boolean
  setVisible: (visible: boolean) => void
  setTitle: (title: string) => void
  setContent: (content: string) => void
  setHint: (hint: string) => void
  center: () => void
}

/**
 * Default modal styling constants.
 */
const MODAL_DEFAULTS = {
  width: 50,
  height: 10,
  borderColor: "#60a5fa",
  titleColor: "#60a5fa",
  backgroundColor: "#0b1220",
  textColor: "#e2e8f0",
  hintColor: "#94a3b8",
  zIndex: 10,
  padding: 2,
}

/**
 * Create a modal UI primitive with consistent styling.
 * This is analogous to a React component - creates all necessary elements
 * and provides a unified interface for controlling them.
 */
export function createModalUI(renderer: CliRenderer, config: ModalConfig): ModalUI {
  const {
    id,
    width = MODAL_DEFAULTS.width,
    height = MODAL_DEFAULTS.height,
    borderColor = MODAL_DEFAULTS.borderColor,
    titleColor = MODAL_DEFAULTS.titleColor,
    zIndex = MODAL_DEFAULTS.zIndex,
  } = config

  // Calculate initial centered position
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
  const left = Math.max(0, Math.floor((cols - width) / 2))
  const top = Math.max(1, Math.floor((rows - height) / 2))

  // Create the modal box
  const box = new BoxRenderable(renderer, {
    id: `${id}-box`,
    width,
    height,
    position: "absolute",
    left,
    top,
    backgroundColor: MODAL_DEFAULTS.backgroundColor,
    borderStyle: "single",
    borderColor,
    border: true,
    zIndex,
  })

  // Create title text
  const title = new TextRenderable(renderer, {
    id: `${id}-title`,
    content: "",
    fg: titleColor,
    position: "absolute",
    left: left + MODAL_DEFAULTS.padding,
    top: top + 1,
    zIndex: zIndex + 1,
  })

  // Create content text
  const content = new TextRenderable(renderer, {
    id: `${id}-content`,
    content: "",
    fg: MODAL_DEFAULTS.textColor,
    position: "absolute",
    left: left + MODAL_DEFAULTS.padding,
    top: top + 3,
    zIndex: zIndex + 1,
  })

  // Create hint text (at bottom of modal)
  const hint = new TextRenderable(renderer, {
    id: `${id}-hint`,
    content: "",
    fg: MODAL_DEFAULTS.hintColor,
    position: "absolute",
    left: left + MODAL_DEFAULTS.padding,
    top: top + height - 2,
    zIndex: zIndex + 1,
  })

  // Set initial visibility to false
  box.visible = false
  title.visible = false
  content.visible = false
  hint.visible = false

  // Add to renderer root
  renderer.root.add(box)
  renderer.root.add(title)
  renderer.root.add(content)
  renderer.root.add(hint)

  let visible = false

  function setVisible(v: boolean): void {
    visible = v
    box.visible = v
    title.visible = v
    content.visible = v
    hint.visible = v
    renderer.requestRender()
  }

  function center(): void {
    const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
    const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
    const newLeft = Math.max(0, Math.floor((cols - width) / 2))
    const newTop = Math.max(1, Math.floor((rows - height) / 2))

    box.left = newLeft
    box.top = newTop
    title.left = newLeft + MODAL_DEFAULTS.padding
    title.top = newTop + 1
    content.left = newLeft + MODAL_DEFAULTS.padding
    content.top = newTop + 3
    hint.left = newLeft + MODAL_DEFAULTS.padding
    hint.top = newTop + height - 2
  }

  return {
    box,
    title,
    content,
    hint,
    get visible() {
      return visible
    },
    setVisible,
    setTitle: (t: string) => {
      title.content = t
    },
    setContent: (c: string) => {
      content.content = c
    },
    setHint: (h: string) => {
      hint.content = h
    },
    center,
  }
}

export function centerModal(
  width: number,
  height: number,
): { left: number; top: number; width: number; height: number } {
  const rows = typeof process.stdout?.rows === "number" ? process.stdout.rows : 40
  const cols = typeof process.stdout?.columns === "number" ? process.stdout.columns : 120
  const left = Math.max(0, Math.floor((cols - width) / 2))
  const top = Math.max(1, Math.floor((rows - height) / 2))
  return { left, top, width, height }
}

export function wrapModalText(text: string, width: number): string[] {
  const lines: string[] = []
  for (const raw of text.split("\n")) {
    if (raw.length <= width) {
      lines.push(raw)
      continue
    }
    if (raw.trim() === "") {
      lines.push("")
      continue
    }
    let start = 0
    while (start < raw.length) {
      lines.push(raw.slice(start, start + width))
      start += width
    }
  }
  return lines
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

