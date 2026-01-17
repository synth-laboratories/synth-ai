import { type Accessor } from "solid-js"

import { getActionHint, buildActionHint } from "../../input/keymap"
import { TextInputModal } from "./ModalShared"

type FilterModalProps = {
  dimensions: Accessor<{ width: number; height: number }>
  setModalInputValue: (value: string) => void
  setModalInputRef: (ref: any) => void
}

export function FilterModal(props: FilterModalProps) {
  return (
    <TextInputModal
      title="Event Filter"
      width={52}
      height={7}
      borderColor="#60a5fa"
      titleColor="#60a5fa"
      hint={`${buildActionHint("modal.confirm", "apply")} | ${getActionHint("app.back")}`}
      dimensions={props.dimensions}
      label="Event filter:"
      placeholder="Type to filter events"
      onInput={props.setModalInputValue}
      setInputRef={props.setModalInputRef}
    />
  )
}
