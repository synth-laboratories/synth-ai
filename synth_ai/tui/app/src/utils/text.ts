export function wrapTextLines(text: string, width: number): string[] {
  const safeWidth = Math.max(1, Math.floor(width))
  const lines: string[] = []
  for (const raw of text.split("\n")) {
    if (raw.length <= safeWidth) {
      lines.push(raw)
      continue
    }
    if (raw.trim() === "") {
      lines.push("")
      continue
    }
    let start = 0
    while (start < raw.length) {
      lines.push(raw.slice(start, start + safeWidth))
      start += safeWidth
    }
  }
  return lines
}

export function clampLines(lines: string[], maxLines: number): string[] {
  const limit = Math.max(1, Math.floor(maxLines))
  if (lines.length <= limit) return lines
  return lines.slice(0, limit)
}

export function clampLine(line: string, width: number): string {
  const limit = Math.max(1, Math.floor(width))
  if (line.length <= limit) return line
  if (limit <= 3) return line.slice(0, limit)
  return `${line.slice(0, limit - 3)}...`
}
