import { Show, type Accessor } from "solid-js"

type Dimensions = { width: number; height: number }

type ModalFrameProps = {
  title: string
  width: number
  height: number
  borderColor: string
  titleColor?: string
  hint?: string
  dimensions: Accessor<Dimensions>
  children: any
}

export function ModalFrame(props: ModalFrameProps) {
  const frameWidth = Math.min(props.width, Math.max(20, props.dimensions().width - 4))
  const frameHeight = Math.min(props.height, Math.max(6, props.dimensions().height - 4))
  const left = Math.max(0, Math.floor((props.dimensions().width - frameWidth) / 2))
  const top = Math.max(1, Math.floor((props.dimensions().height - frameHeight) / 2))
  // Calculate content height: frame - borders(2) - padding(2) - title(1) - hint(1) = height - 6
  const contentHeight = Math.max(1, frameHeight - 6)

  return (
    <box
      position="absolute"
      left={left}
      top={top}
      width={frameWidth}
      height={frameHeight}
      backgroundColor="#0b1220"
      border
      borderStyle="single"
      borderColor={props.borderColor}
      zIndex={30}
      flexDirection="column"
      paddingLeft={2}
      paddingRight={2}
      paddingTop={1}
      paddingBottom={1}
    >
      <text fg={props.titleColor ?? props.borderColor}>
        <b>{props.title}</b>
      </text>
      <box height={contentHeight} overflow="hidden">
        {props.children}
      </box>
      <Show when={props.hint}>
        <text fg="#94a3b8">{props.hint}</text>
      </Show>
    </box>
  )
}
