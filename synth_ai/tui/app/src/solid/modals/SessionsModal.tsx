import { type Accessor, createMemo } from "solid-js"

import { formatActionKeys } from "../../input/keymap"
import { formatSessionDetails } from "../../formatters/modals"
import type { SessionHealthResult, SessionRecord } from "../../types"
import { ScrollableTextModal } from "./ModalShared"

type SessionsModalProps = {
  dimensions: Accessor<{ width: number; height: number }>
  sessions: Accessor<SessionRecord[]>
  sessionsHealth: Accessor<Map<string, SessionHealthResult>>
  selectedIndex: Accessor<number>
  scrollOffset: Accessor<number>
  openCodeUrl: string | null
}

export function SessionsModal(props: SessionsModalProps) {
  const raw = createMemo(() =>
    formatSessionDetails(
      props.sessions(),
      props.sessionsHealth(),
      props.selectedIndex(),
      props.openCodeUrl,
    ),
  )
  const title = createMemo(() => {
    const sessions = props.sessions()
    const active = sessions.filter(
      (session) =>
        session.state === "connected" ||
        session.state === "connecting" ||
        session.state === "reconnecting",
    ).length
    return `OpenCode Sessions (${active} active)`
  })

  return (
    <ScrollableTextModal
      title={title()}
      width={70}
      height={20}
      borderColor="#60a5fa"
      titleColor="#60a5fa"
      dimensions={props.dimensions}
      raw={raw()}
      offset={props.scrollOffset()}
      hint={{
        baseHints: [
          `${formatActionKeys("nav.down", { primaryOnly: true })}/${formatActionKeys("nav.up", { primaryOnly: true })} select`,
          `${formatActionKeys("sessions.connect")} connect local`,
          `${formatActionKeys("sessions.disconnect")} disconnect`,
          `${formatActionKeys("sessions.copy", { primaryOnly: true })} copy URL`,
          `${formatActionKeys("modal.confirm")} select`,
          `${formatActionKeys("app.back")} close`,
        ],
      }}
    />
  )
}
