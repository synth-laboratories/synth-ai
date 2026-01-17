import { type Accessor } from "solid-js"

import { getActionHint, buildActionHint } from "../../input/keymap"
import { TextInputModal } from "./ModalShared"

type SnapshotModalProps = {
  dimensions: Accessor<{ width: number; height: number }>
  setModalInputValue: (value: string) => void
  setModalInputRef: (ref: any) => void
}

export function SnapshotModal(props: SnapshotModalProps) {
  return (
    <TextInputModal
      title="Snapshot ID"
      width={50}
      height={7}
      borderColor="#60a5fa"
      titleColor="#60a5fa"
      hint={`${buildActionHint("modal.confirm", "apply")} | ${getActionHint("app.back")}`}
      dimensions={props.dimensions}
      label="Snapshot ID:"
      placeholder="Enter snapshot id"
      onInput={props.setModalInputValue}
      setInputRef={props.setModalInputRef}
    />
  )
}
