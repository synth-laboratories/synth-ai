type DiffKind = "equal" | "add" | "remove"

export type PromptDiffBlock = {
  title: string
  lines: string[]
}

type TextReplacement = {
  oldText: string
  newText: string
  role?: string
  order?: number
}

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === "object" && !Array.isArray(value)
}

function splitLines(text: string): string[] {
  return text.split(/\r?\n/)
}

function diffLines(oldText: string, newText: string): Array<{ kind: DiffKind; text: string }> {
  const oldLines = splitLines(oldText)
  const newLines = splitLines(newText)
  const rows = oldLines.length + 1
  const cols = newLines.length + 1
  const dp: number[][] = Array.from({ length: rows }, () => Array(cols).fill(0))

  for (let i = rows - 2; i >= 0; i -= 1) {
    for (let j = cols - 2; j >= 0; j -= 1) {
      if (oldLines[i] === newLines[j]) {
        dp[i][j] = dp[i + 1][j + 1] + 1
      } else {
        dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1])
      }
    }
  }

  const result: Array<{ kind: DiffKind; text: string }> = []
  let i = 0
  let j = 0
  while (i < oldLines.length && j < newLines.length) {
    if (oldLines[i] === newLines[j]) {
      result.push({ kind: "equal", text: oldLines[i] })
      i += 1
      j += 1
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      result.push({ kind: "remove", text: oldLines[i] })
      i += 1
    } else {
      result.push({ kind: "add", text: newLines[j] })
      j += 1
    }
  }

  while (i < oldLines.length) {
    result.push({ kind: "remove", text: oldLines[i] })
    i += 1
  }
  while (j < newLines.length) {
    result.push({ kind: "add", text: newLines[j] })
    j += 1
  }

  return result
}

function renderDiffLines(oldText: string, newText: string): string[] {
  return diffLines(oldText, newText).map((line) => {
    if (line.kind === "add") return `+ ${line.text}`
    if (line.kind === "remove") return `- ${line.text}`
    return `  ${line.text}`
  })
}

function coerceTextReplacement(value: unknown): TextReplacement | null {
  if (!isRecord(value)) return null
  const oldText = typeof value.old_text === "string" ? value.old_text : ""
  const newText = typeof value.new_text === "string" ? value.new_text : ""
  if (!oldText && !newText) return null
  const role = typeof value.apply_to_role === "string" ? value.apply_to_role : undefined
  const order = typeof value.apply_to_order === "number" ? value.apply_to_order : undefined
  return { oldText, newText, role, order }
}

function extractTextReplacements(payload: Record<string, any>): TextReplacement[] {
  const sources: Array<Record<string, any> | null> = [
    isRecord(payload.transformation) ? payload.transformation : null,
    isRecord(payload.object?.transformation) ? payload.object.transformation : null,
    isRecord(payload.object) ? payload.object : null,
    isRecord(payload.object?.data) ? payload.object.data : null,
    payload,
  ]

  for (const source of sources) {
    if (!source) continue
    const replacements = source.text_replacements
    if (!Array.isArray(replacements)) continue
    const normalized = replacements.map(coerceTextReplacement).filter(Boolean) as TextReplacement[]
    if (normalized.length > 0) return normalized
  }

  return []
}

function extractStageInstruction(stage: Record<string, any>): string | null {
  if (typeof stage.instruction_text === "string" && stage.instruction_text.trim()) {
    return stage.instruction_text
  }
  if (Array.isArray(stage.instruction_lines) && stage.instruction_lines.length > 0) {
    return stage.instruction_lines.map((line: unknown) => String(line)).join("\n")
  }
  if (typeof stage.instruction === "string" && stage.instruction.trim()) {
    return stage.instruction
  }
  if (typeof stage.content === "string" && stage.content.trim()) {
    return stage.content
  }
  if (typeof stage.prompt_text === "string" && stage.prompt_text.trim()) {
    return stage.prompt_text
  }
  return null
}

function extractBaselineMessages(stage: Record<string, any>): Array<Record<string, any>> | null {
  const messages = stage.baseline_messages ?? stage.baselineMessages
  if (!Array.isArray(messages) || messages.length === 0) return null
  return messages.filter((msg) => isRecord(msg)) as Array<Record<string, any>>
}

function baselineInstructionFromMessages(messages: Array<Record<string, any>>): string | null {
  const system = messages.find((msg) => {
    const role = String(msg.role || "").toLowerCase()
    return role === "system" || role === "developer"
  })
  const candidate = system ?? messages[0]
  const content =
    typeof candidate?.content === "string"
      ? candidate.content
      : typeof candidate?.text === "string"
        ? candidate.text
        : typeof candidate?.pattern === "string"
          ? candidate.pattern
          : ""
  if (content.trim()) return content

  const lines: string[] = []
  for (const msg of messages) {
    const role = String(msg.role || "message")
    const text = typeof msg.content === "string" ? msg.content : typeof msg.text === "string" ? msg.text : ""
    if (text) lines.push(`[${role}] ${text}`)
  }
  return lines.length ? lines.join("\n") : null
}

function normalizeStages(stages: unknown): Record<string, any> | null {
  if (isRecord(stages)) return stages
  if (!Array.isArray(stages)) return null

  const result: Record<string, any> = {}
  stages.forEach((stage, idx) => {
    if (!isRecord(stage)) return
    const id =
      stage.id ??
      stage.stage_id ??
      stage.name ??
      stage.module_id ??
      `stage_${idx + 1}`
    result[String(id)] = stage
  })
  return Object.keys(result).length > 0 ? result : null
}

function extractStages(payload: Record<string, any>): Record<string, any> | null {
  const candidates: unknown[] = [
    payload.stages,
    payload.object?.stages,
    payload.object?.data?.stages,
  ]
  for (const candidate of candidates) {
    const normalized = normalizeStages(candidate)
    if (normalized) return normalized
  }
  return null
}

export function buildPromptDiffBlocks(payload: Record<string, any>): PromptDiffBlock[] {
  const replacements = extractTextReplacements(payload)
  if (replacements.length > 0) {
    return replacements.map((replacement, idx) => {
      const role = replacement.role ?? "message"
      const orderLabel = replacement.order != null ? `#${replacement.order}` : null
      const title = [role, orderLabel].filter(Boolean).join(" ")
      const label = title ? `replace ${title}` : `replace ${idx + 1}`
      return {
        title: label,
        lines: renderDiffLines(replacement.oldText, replacement.newText),
      }
    })
  }

  const stages = extractStages(payload)
  if (!stages) return []

  const blocks: PromptDiffBlock[] = []
  for (const stageId of Object.keys(stages).sort()) {
    const stage = stages[stageId]
    if (!isRecord(stage)) continue
    const baselineMessages = extractBaselineMessages(stage)
    if (!baselineMessages) continue
    const baselineText = baselineInstructionFromMessages(baselineMessages)
    const instruction = extractStageInstruction(stage)
    if (!baselineText || !instruction) continue
    if (baselineText.trim() === instruction.trim()) continue
    blocks.push({
      title: `stage ${stageId}`,
      lines: renderDiffLines(baselineText, instruction),
    })
  }

  return blocks
}

export function formatPromptDiffBlocks(blocks: PromptDiffBlock[]): string[] {
  const lines: string[] = []
  blocks.forEach((block, idx) => {
    const title = block.title ? `-- ${block.title} --` : "-- diff --"
    lines.push(title)
    lines.push(...block.lines)
    if (idx < blocks.length - 1) lines.push("")
  })
  return lines
}
