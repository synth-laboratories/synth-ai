import { type Accessor } from "solid-js"

import { getActionHint } from "../../input/keymap"
import type { AppData } from "../../types"
import { TextContentModal } from "./ModalShared"

type ProfileModalProps = {
  dimensions: Accessor<{ width: number; height: number }>
  data: AppData
}

export function ProfileModal(props: ProfileModalProps) {
  const org = props.data.orgName || "-"
  const user = props.data.userEmail || "-"
  const apiKey = process.env.SYNTH_API_KEY || "-"
  const content = `Organization:\n${org}\n\nEmail:\n${user}\n\nAPI Key:\n${apiKey}`

  return (
    <TextContentModal
      title="Profile"
      width={72}
      height={15}
      borderColor="#818cf8"
      titleColor="#818cf8"
      hint={getActionHint("app.back")}
      dimensions={props.dimensions}
      text={content}
    />
  )
}
