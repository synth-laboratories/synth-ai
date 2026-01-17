import { type Accessor, createMemo } from "solid-js"

import { getActionHint, buildCombinedHint, buildActionHint } from "../../input/keymap"
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
          buildCombinedHint("nav.down", "nav.up", "select"),
          buildActionHint("sessions.connect", "connect local"),
          getActionHint("sessions.disconnect"),
          buildActionHint("sessions.copy", "copy URL"),
          getActionHint("modal.confirm"),
          getActionHint("app.back"),
        ],
      }}
    />
  )
}
