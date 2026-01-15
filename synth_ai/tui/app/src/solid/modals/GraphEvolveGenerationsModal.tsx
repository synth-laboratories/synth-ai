import { Show, createEffect, createMemo, createSignal } from "solid-js"
import { useKeyboard } from "@opentui/solid"
import { formatActionKeys, matchAction } from "../../input/keymap"
import { COLORS } from "../theme"
import type { AppData } from "../../types"
import { moveSelectionIndex, resolveSelectionWindow } from "../utils/list"
import {
  extractGraphEvolveCandidates,
  formatRacePreview,
  groupCandidatesByGeneration,
} from "../../formatters/graph-evolve"

type GraphEvolveGenerationsModalProps = {
  visible: boolean
  data: AppData
  width: number
  height: number
  onClose: () => void
  onStatus: (message: string) => void
  onOpenCandidates: (generation: number) => void
}

function clampLine(text: string, width: number): string {
  if (text.length <= width) return text
  if (width <= 3) return text.slice(0, width)
  return `${text.slice(0, width - 3)}...`
}

export function GraphEvolveGenerationsModal(props: GraphEvolveGenerationsModalProps) {
  const [selectedIndex, setSelectedIndex] = createSignal(0)
  const [windowStart, setWindowStart] = createSignal(0)

  const generations = createMemo(() => {
    const candidates = extractGraphEvolveCandidates(props.data)
    return groupCandidatesByGeneration(candidates)
  })

  const modalWidth = createMemo(() => Math.min(96, Math.max(48, props.width - 4)))
  const modalHeight = createMemo(() => Math.min(28, Math.max(12, props.height - 4)))

  createEffect(() => {
    const total = generations().length
    if (total === 0) {
      setSelectedIndex(0)
      setWindowStart(0)
      return
    }
    if (selectedIndex() >= total) {
      setSelectedIndex(total - 1)
    }
    const visibleCount = Math.max(3, modalHeight() - 8)
    const window = resolveSelectionWindow(total, selectedIndex(), windowStart(), visibleCount, "center")
    if (window.windowStart !== windowStart()) {
      setWindowStart(window.windowStart)
    }
  })

  const layout = createMemo(() => {
    const width = modalWidth()
    const height = modalHeight()
    const contentHeight = Math.max(4, height - 6)
    const listWidth = Math.max(20, Math.floor(width * 0.32))
    const detailWidth = Math.max(10, width - listWidth - 4)
    const visibleCount = Math.max(3, contentHeight - 2)
    const total = generations().length
    const window = resolveSelectionWindow(total, selectedIndex(), windowStart(), visibleCount, "center")

    const listLines: string[] = []
    listLines.push(clampLine("=== Generations ===", listWidth))
    if (total === 0) {
      listLines.push(clampLine("  (no data yet)", listWidth))
    } else {
      if (window.windowStart > 0) {
        listLines.push(clampLine("  ...", listWidth))
      }
      for (let i = window.windowStart; i < window.windowEnd; i += 1) {
        const group = generations()[i]
        if (!group) continue
        const cursor = i === selectedIndex() ? ">" : " "
        const line = `${cursor} Gen ${group.generation} (${group.candidates.length})`
        listLines.push(clampLine(line, listWidth))
      }
      if (window.windowEnd < total) {
        listLines.push(clampLine("  ...", listWidth))
      }
    }

    const selected = generations()[selectedIndex()] ?? null
    const labelWidth = 4
    const trackWidth = Math.max(8, detailWidth - labelWidth - 1)
    const preview = selected
      ? formatRacePreview({
          candidates: selected.candidates,
          maxCandidates: 3,
          trackWidth,
          labelWidth,
          scorePrecision: 2,
        })
      : { lines: ["No candidates yet."], bestReward: null }

    const detailLines: string[] = []
    detailLines.push(clampLine("Reward 0.00-1.00", detailWidth))
    if (selected) {
      detailLines.push(clampLine(`Gen ${selected.generation} (top 3)`, detailWidth))
    }
    for (const line of preview.lines) {
      detailLines.push(clampLine(line, detailWidth))
    }
    detailLines.push("")
    detailLines.push(clampLine("Enter to view all candidates", detailWidth))

    return {
      listWidth,
      detailWidth,
      contentHeight,
      listLines,
      detailLines,
      total,
      selected,
      visibleCount,
    }
  })

  const handleKey = (evt: any) => {
    if (!props.visible) return
    const action = matchAction(evt, "modal.generations")
    if (!action) return
    evt.preventDefault?.()

    const total = generations().length
    if (total === 0) return

    const visibleCount = layout().visibleCount
    const move = (delta: number) => {
      const next = moveSelectionIndex(selectedIndex(), delta, total)
      const window = resolveSelectionWindow(total, next, windowStart(), visibleCount, "center")
      setSelectedIndex(next)
      setWindowStart(window.windowStart)
    }

    switch (action) {
      case "generation.next":
      case "nav.down":
        move(1)
        return
      case "generation.prev":
      case "nav.up":
        move(-1)
        return
      case "nav.pageDown":
        move(Math.max(1, visibleCount - 1))
        return
      case "nav.pageUp":
        move(-Math.max(1, visibleCount - 1))
        return
      case "nav.home":
        setSelectedIndex(0)
        setWindowStart(0)
        return
      case "nav.end":
        setSelectedIndex(total - 1)
        setWindowStart(Math.max(0, total - visibleCount))
        return
      case "modal.confirm": {
        const selected = layout().selected
        if (selected) {
          props.onOpenCandidates(selected.generation)
        }
        return
      }
      default:
        return
    }
  }

  useKeyboard(handleKey)

  const hint = createMemo(() => {
    const range =
      layout().total > layout().visibleCount
        ? `[${windowStart() + 1}-${Math.min(windowStart() + layout().visibleCount, layout().total)}/${layout().total}] `
        : ""
    const tabHint = `${formatActionKeys("generation.prev", { primaryOnly: true })}/${formatActionKeys("generation.next", { primaryOnly: true })} gen`
    const scrollHint = `${formatActionKeys("nav.up", { primaryOnly: true })}/${formatActionKeys("nav.down", { primaryOnly: true })} scroll`
    const openHint = `${formatActionKeys("modal.confirm")} view`
    const closeHint = `${formatActionKeys("app.back")} close`
    return `${range}${tabHint} | ${scrollHint} | ${openHint} | ${closeHint}`
  })

  return (
    <Show when={props.visible}>
      <box
        position="absolute"
        left={Math.max(0, Math.floor((props.width - modalWidth()) / 2))}
        top={Math.max(1, Math.floor((props.height - modalHeight()) / 2))}
        width={modalWidth()}
        height={modalHeight()}
        backgroundColor="#0b1220"
        border
        borderStyle="single"
        borderColor={COLORS.borderAccent}
        zIndex={30}
        flexDirection="column"
        paddingLeft={2}
        paddingRight={2}
        paddingTop={1}
        paddingBottom={1}
      >
        <text fg={COLORS.borderAccent}>
          {clampLine("GraphEvolve - Generations", Math.max(10, modalWidth() - 6))}
        </text>
        <box flexDirection="row" gap={2} height={layout().contentHeight}>
          <box
            width={layout().listWidth}
            height={layout().contentHeight}
            overflow="hidden"
            flexDirection="column"
          >
            <text fg={COLORS.text}>{layout().listLines.join("\n")}</text>
          </box>
          <box
            width={layout().detailWidth}
            height={layout().contentHeight}
            overflow="hidden"
            flexDirection="column"
          >
            <text fg={COLORS.text}>{layout().detailLines.join("\n")}</text>
          </box>
        </box>
        <text fg={COLORS.textDim}>{hint()}</text>
      </box>
    </Show>
  )
}
