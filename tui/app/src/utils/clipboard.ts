/**
 * Clipboard utilities.
 */

export async function copyToClipboard(text: string): Promise<void> {
  // Use pbcopy on macOS
  const proc = Bun.spawn(["pbcopy"], {
    stdin: "pipe",
  })
  proc.stdin.write(text)
  proc.stdin.end()
  await proc.exited
}

export function execCommandSync(cmd: string): string | null {
  try {
    const result = Bun.spawnSync(["sh", "-c", cmd])
    if (result.exitCode === 0 && result.stdout) {
      const decoder = new TextDecoder()
      return decoder.decode(result.stdout).trim()
    }
    return null
  } catch {
    return null
  }
}
