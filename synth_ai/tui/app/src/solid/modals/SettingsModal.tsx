import { type Accessor, createMemo } from "solid-js"

import { buildCombinedHint, getActionHint } from "../../input/keymap"
import { modeUrls } from "../../state/app-state"
import type { AppState } from "../../state/app-state"
import { TextContentModal } from "./ModalShared"

type SettingsModalProps = {
  dimensions: Accessor<{ width: number; height: number }>
  ui: AppState
  settingsCursor: Accessor<number>
}

export function SettingsModal(props: SettingsModalProps) {
  const modeLabels: Record<string, string> = {
    prod: "Prod",
    dev: "Dev",
    local: "Local",
    reset: "Reset",
  }

  const content = createMemo(() => {
    const cursorIdx = props.settingsCursor()
    const lines: string[] = []
    for (let idx = 0; idx < props.ui.settingsOptions.length; idx++) {
      const mode = props.ui.settingsOptions[idx]
      const active = mode === "reset" ? props.ui.settingsMode === null : props.ui.settingsMode === mode
      const cursor = idx === cursorIdx ? ">" : " "
      lines.push(`${cursor} [${active ? "x" : " "}] ${modeLabels[mode] || mode} (${mode})`)
    }
    const selectedMode = props.ui.settingsOptions[cursorIdx]
    if (selectedMode) {
      lines.push("")
      if (selectedMode === "reset") {
        lines.push("Clears the saved mode selection.")
      } else {
        const urls = modeUrls[selectedMode]
        const key = props.ui.settingsKeys[selectedMode] || ""
        const keyPreview = key.trim() ? `...${key.slice(-8)}` : "(no key)"
        lines.push(`Backend: ${urls?.backendUrl || "(unset)"}`)
        lines.push(`Frontend: ${urls?.frontendUrl || "(unset)"}`)
        lines.push(`Key: ${keyPreview}`)
      }
      const envBackend = process.env.SYNTH_BACKEND_URL || "(unset)"
      const envFrontend = process.env.SYNTH_FRONTEND_URL || "(unset)"
      lines.push("")
      lines.push("Currently loaded URLs:")
      lines.push(`Backend: ${envBackend}`)
      lines.push(`Frontend: ${envFrontend}`)
    }
    return lines.join("\n")
  })

  return (
    <TextContentModal
      title="Dev"
      width={72}
      height={20}
      borderColor="#38bdf8"
      titleColor="#38bdf8"
      hint={`${buildCombinedHint("nav.down", "nav.up", "navigate")} | ${getActionHint("modal.confirm")} | ${getActionHint("app.back")}`}
      dimensions={props.dimensions}
      text={content()}
    />
  )
}
