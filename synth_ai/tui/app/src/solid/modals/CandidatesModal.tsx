import { Show, createEffect, createMemo, createSignal } from "solid-js"
import { useKeyboard } from "@opentui/solid"

import type { Snapshot } from "../../types"
import { copyToClipboard } from "../../utils/clipboard"

type CandidateView = {
  id: string
  reward: number | null
  meanReward: number | null
  isBaseline: boolean
  isPareto: boolean
  tag: string | null
  createdAt: string | null
  payload: Record<string, any>
}

type CandidatesModalProps = {
  visible: boolean
  snapshot: Snapshot
  width: number
  height: number
  onClose: () => void
  onStatus: (message: string) => void
}

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === "object" && !Array.isArray(value)
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string") {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function mean(values: Array<unknown>): number | null {
  const numeric = values
    .map((value) => toNumber(value))
    .filter((value): value is number => value != null)
  if (numeric.length === 0) return null
  const sum = numeric.reduce((acc, val) => acc + val, 0)
  return sum / numeric.length
}

function extractReward(payload: Record<string, any>): number | null {
  const score = isRecord(payload.score) ? payload.score : null
  return (
    toNumber(payload.reward) ??
    toNumber(payload.accuracy) ??
    toNumber(payload.full_score) ??
    toNumber(payload.minibatch_score) ??
    toNumber(payload.reward_mean) ??
    toNumber(payload.mean_reward) ??
    toNumber(score?.reward) ??
    toNumber(score?.accuracy) ??
    toNumber(score?.reward_mean) ??
    toNumber(score?.mean_reward) ??
    null
  )
}

function extractMeanReward(payload: Record<string, any>): number | null {
  const score = isRecord(payload.score) ? payload.score : null
  const instanceScores =
    (Array.isArray(payload.instance_scores) && payload.instance_scores) ||
    (Array.isArray(payload.instance_rewards) && payload.instance_rewards) ||
    (Array.isArray(score?.instance_scores) && score?.instance_scores) ||
    (Array.isArray(score?.instance_rewards) && score?.instance_rewards) ||
    null
  const computed = instanceScores ? mean(instanceScores) : null
  return computed ?? toNumber(payload.reward_mean) ?? toNumber(payload.mean_reward) ?? null
}

function formatReward(value: number | null): string {
  if (value == null) return "-"
  return Number.isFinite(value) ? value.toFixed(3).replace(/\.?0+$/, "") : "-"
}

function buildCandidateViews(snapshot: Snapshot): CandidateView[] {
  const byId = new Map<string, CandidateView>()
  for (const candidate of snapshot.allCandidates) {
    const payload = isRecord(candidate.payload) ? candidate.payload : {}
    const reward = candidate.reward ?? extractReward(payload)
    const meanReward = extractMeanReward(payload)
    const isPareto = payload.is_pareto === true || payload.pareto === true
    byId.set(candidate.id, {
      id: candidate.id,
      reward,
      meanReward,
      isBaseline: candidate.isBaseline,
      isPareto,
      tag: candidate.tag ?? null,
      createdAt: candidate.createdAt ?? null,
      payload,
    })
  }
  return Array.from(byId.values())
}

function wrapText(text: string, width: number): string[] {
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

function clampLine(text: string, width: number): string {
  if (text.length <= width) return text
  if (width <= 3) return text.slice(0, width)
  return `${text.slice(0, width - 3)}...`
}

function buildCandidateDetail(candidate: CandidateView | null): string {
  if (!candidate) {
    return "No candidates available."
  }
  const payload = candidate.payload
  const lines: string[] = []

  lines.push("=== SELECTED CANDIDATE ===")
  lines.push(`ID: ${candidate.id}`)
  lines.push(`Type: ${candidate.isBaseline ? "baseline" : "optimized"}`)
  lines.push(`Reward: ${formatReward(candidate.reward)}`)
  if (candidate.meanReward != null) {
    lines.push(`Mean Reward: ${formatReward(candidate.meanReward)}`)
  }
  lines.push(`Pareto: ${candidate.isPareto ? "yes" : "no"}`)
  lines.push(`Tag: ${candidate.tag ?? "-"}`)
  lines.push(`Created: ${candidate.createdAt ?? "-"}`)
  lines.push("")

  const prompt =
    payload.prompt ||
    payload.system_prompt ||
    payload.user_prompt ||
    payload.instructions ||
    null

  if (typeof prompt === "string" && prompt.trim().length > 0) {
    lines.push("=== PROMPT ===")
    lines.push(prompt)
    lines.push("")
  }

  const messages = Array.isArray(payload.messages) ? payload.messages : null
  if (messages && messages.length > 0) {
    lines.push("=== MESSAGES ===")
    for (const message of messages) {
      if (!isRecord(message)) continue
      const role = message.role ?? "unknown"
      const content = message.content ?? ""
      lines.push(`[${role}] ${content}`)
    }
    lines.push("")
  }

  try {
    const json = JSON.stringify(payload, null, 2)
    if (json.length > 0) {
      lines.push("=== RAW PAYLOAD ===")
      lines.push(json.length > 4000 ? `${json.slice(0, 4000)}\n... (truncated)` : json)
    }
  } catch {
    // ignore
  }

  return lines.join("\n")
}

export function CandidatesModal(props: CandidatesModalProps) {
  const [selectedIndex, setSelectedIndex] = createSignal(0)
  const [detailOffset, setDetailOffset] = createSignal(0)

  const modalWidth = createMemo(() => Math.max(60, Math.min(props.width - 2, 120)))
  const modalHeight = createMemo(() => Math.max(18, Math.min(props.height - 2, 34)))

  const candidates = createMemo(() => buildCandidateViews(props.snapshot))

  const clampIndex = (index: number) => {
    const total = candidates().length
    if (total === 0) return 0
    return Math.max(0, Math.min(index, total - 1))
  }

  createEffect(() => {
    if (!props.visible) return
    setSelectedIndex(0)
    setDetailOffset(0)
  })

  createEffect(() => {
    const total = candidates().length
    if (total === 0) {
      setSelectedIndex(0)
      setDetailOffset(0)
      return
    }
    const clamped = clampIndex(selectedIndex())
    if (clamped !== selectedIndex()) {
      setSelectedIndex(clamped)
    }
  })

  const layout = createMemo(() => {
    const total = candidates().length
    const selected = total > 0 ? candidates()[clampIndex(selectedIndex())] : null
    const contentWidth = Math.max(10, modalWidth() - 6)
    const contentHeight = Math.max(6, modalHeight() - 6)
    const listWidth = Math.max(24, Math.min(40, Math.floor(contentWidth * 0.35)))
    const detailWidth = Math.max(10, contentWidth - listWidth - 2)
    const maxListHeight = Math.max(1, contentHeight - 2)
    const listWindowStart = Math.max(0, Math.min(clampIndex(selectedIndex()) - Math.floor(maxListHeight / 2), Math.max(0, total - maxListHeight)))
    const listWindowEnd = Math.min(total, listWindowStart + maxListHeight)
    const list: string[] = []

    list.push("")
    list.push(clampLine("=== CANDIDATES ===", listWidth))
    if (total === 0) {
      list.push(clampLine("  (no candidates yet)", listWidth))
    } else {
      if (listWindowStart > 0) {
        list.push(clampLine("  ...", listWidth))
      }
      for (let i = listWindowStart; i < listWindowEnd; i += 1) {
        const candidate = candidates()[i]
        const cursor = i === clampIndex(selectedIndex()) ? ">" : " "
        const meanReward = candidate.meanReward != null ? `mean=${formatReward(candidate.meanReward)}` : null
        const reward = candidate.reward != null ? `reward=${formatReward(candidate.reward)}` : null
        const tags = [
          candidate.isPareto ? "pareto" : null,
          candidate.isBaseline ? "baseline" : null,
        ].filter(Boolean)
        const meta = [meanReward, reward, ...tags].filter(Boolean).join(" | ")
        const line = `${cursor} #${i + 1} ${candidate.id}${meta ? ` | ${meta}` : ""}`
        list.push(clampLine(line, listWidth))
      }
      if (listWindowEnd < total) {
        list.push(clampLine("  ...", listWidth))
      }
    }

    const detailHeight = Math.max(3, contentHeight - 1)
    const detailText = buildCandidateDetail(selected)
    const detailLines = wrapText(detailText, detailWidth).map((line) => clampLine(line, detailWidth))
    const maxOffset = Math.max(0, detailLines.length - detailHeight)
    const clampedOffset = Math.max(0, Math.min(detailOffset(), maxOffset))
    const visibleDetail = detailLines.slice(clampedOffset, clampedOffset + detailHeight)

    const paretoCandidates = candidates().filter((cand) => cand.isPareto)
    const paretoMean = mean(paretoCandidates.map((cand) => cand.meanReward ?? cand.reward).filter((val): val is number => val != null))

    return {
      total,
      paretoCount: paretoCandidates.length,
      paretoMean,
      listLines: list,
      listWidth,
      detailWidth,
      listBodyHeight: Math.max(1, detailHeight - 1),
      detailLines,
      visibleDetail,
      detailHeight,
      maxOffset,
      offset: clampedOffset,
      selected,
    }
  })

  createEffect(() => {
    const maxOffset = layout().maxOffset
    if (detailOffset() > maxOffset) {
      setDetailOffset(maxOffset)
    }
  })

  const handleKey = (evt: any) => {
    if (!props.visible) return
    const name = typeof evt?.name === "string" ? evt.name : ""
    const key = name.toLowerCase()

    const total = candidates().length
    const detailHeight = layout().detailHeight
    const maxOffset = layout().maxOffset
    const clampOffset = (value: number) => Math.max(0, Math.min(value, maxOffset))

    const prev = () => {
      setSelectedIndex((current) => clampIndex(current - 1))
      setDetailOffset(0)
    }
    const next = () => {
      setSelectedIndex((current) => clampIndex(current + 1))
      setDetailOffset(0)
    }
    const scrollUp = (amount: number) => setDetailOffset((current) => clampOffset(current - amount))
    const scrollDown = (amount: number) => setDetailOffset((current) => clampOffset(current + amount))

    if (key === "q" || name === "escape") {
      evt.preventDefault?.()
      props.onClose()
      return
    }
    if (key === "y") {
      evt.preventDefault?.()
      const text = buildCandidateDetail(layout().selected)
      void copyToClipboard(text).then(() => props.onStatus("Candidate copied to clipboard"))
      return
    }
    if (name === "left" || name === "a") {
      evt.preventDefault?.()
      if (total > 0) prev()
      return
    }
    if (name === "right" || name === "d") {
      evt.preventDefault?.()
      if (total > 0) next()
      return
    }
    if (name === "up" || name === "w") {
      evt.preventDefault?.()
      scrollUp(1)
      return
    }
    if (name === "down" || name === "s") {
      evt.preventDefault?.()
      scrollDown(1)
      return
    }
    if (name === "pageup") {
      evt.preventDefault?.()
      scrollUp(Math.max(1, detailHeight - 1))
      return
    }
    if (name === "pagedown") {
      evt.preventDefault?.()
      scrollDown(Math.max(1, detailHeight - 1))
      return
    }
    if (name === "home") {
      evt.preventDefault?.()
      setDetailOffset(0)
      return
    }
    if (name === "end") {
      evt.preventDefault?.()
      setDetailOffset(maxOffset)
      return
    }
  }

  useKeyboard(handleKey)

  const hint = createMemo(() => {
    const offset = layout().offset
    const total = layout().detailLines.length
    const detailHeight = layout().detailHeight
    const range = total > detailHeight
      ? `[${offset + 1}-${Math.min(offset + detailHeight, total)}/${total}] `
      : ""
    return `${range}←/→ candidate | ↑/↓ scroll | y copy | q close`
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
        borderColor="#22c55e"
        zIndex={30}
        flexDirection="column"
        paddingLeft={2}
        paddingRight={2}
        paddingTop={1}
        paddingBottom={1}
      >
        <text fg="#22c55e">
          {clampLine(
            `Results - Candidates${layout().total ? ` (${clampIndex(selectedIndex()) + 1}/${layout().total})` : ""}`,
            Math.max(10, modalWidth() - 6),
          )}
        </text>
        <box flexDirection="row" gap={2} height={layout().detailHeight + 1}>
          <box
            width={layout().detailWidth}
            height={layout().detailHeight + 1}
            overflow="hidden"
            flexDirection="column"
          >
            <text fg="#e2e8f0">{layout().visibleDetail.join("\n")}</text>
          </box>
          <box
            width={layout().listWidth}
            height={layout().detailHeight + 1}
            overflow="hidden"
            flexDirection="column"
            gap={1}
          >
            <text fg="#94a3b8">
              {clampLine(
                `Candidates: ${layout().total} | Pareto: ${layout().paretoCount}${layout().paretoMean != null ? ` (mean=${formatReward(layout().paretoMean)})` : ""}`,
                Math.max(10, layout().listWidth),
              )}
            </text>
            <text fg="#e2e8f0">{layout().listLines.slice(0, layout().listBodyHeight).join("\n")}</text>
          </box>
        </box>
        <text fg="#94a3b8">{hint()}</text>
      </box>
    </Show>
  )
}
