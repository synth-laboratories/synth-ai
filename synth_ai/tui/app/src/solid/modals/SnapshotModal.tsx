import { type Accessor } from "solid-js"

import { formatActionKeys } from "../../input/keymap"
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
      hint={`${formatActionKeys("modal.confirm")} apply | ${formatActionKeys("app.back")} close`}
      dimensions={props.dimensions}
      label="Snapshot ID:"
      placeholder="Enter snapshot id"
      onInput={props.setModalInputValue}
      setInputRef={props.setModalInputRef}
    />
  )
}
