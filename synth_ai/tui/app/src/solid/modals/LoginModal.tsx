import { type Accessor, createMemo } from "solid-js"

import { getActionHint, buildActionHint } from "../../input/keymap"
import type { AuthStatus } from "../../auth"
import { TextContentModal } from "./ModalShared"

type LoginModalProps = {
  dimensions: Accessor<{ width: number; height: number }>
  loginStatus: Accessor<AuthStatus>
}

export function LoginModal(props: LoginModalProps) {
  const copy = createMemo(() => {
    const status = props.loginStatus()
    const startHint = getActionHint("login.confirm")
    const closeHint = getActionHint("app.back")
    const retryHint = buildActionHint("login.confirm", "retry")
    let content = "Press Enter to open browser and sign in..."
    let hint = `${startHint} | ${buildActionHint("app.back", "cancel")}`
    switch (status.state) {
      case "initializing":
        content = "Initializing..."
        hint = "Please wait..."
        break
      case "waiting":
        content = `Browser opened. Complete sign-in there.\n\nURL: ${status.verificationUri}`
        hint = `Waiting for browser auth... | ${buildActionHint("app.back", "cancel")}`
        break
      case "polling":
        content = "Browser opened. Complete sign-in there.\n\nChecking for completion..."
        hint = `Waiting for browser auth... | ${buildActionHint("app.back", "cancel")}`
        break
      case "success":
        content = "Authentication successful!"
        hint = "Loading..."
        break
      case "error":
        content = `Error: ${status.message}`
        hint = `${retryHint} | ${closeHint}`
        break
      default:
        break
    }
    return { content, hint }
  })

  return (
    <TextContentModal
      title="Sign In / Sign Up"
      width={60}
      height={10}
      borderColor="#22c55e"
      titleColor="#22c55e"
      hint={copy().hint}
      dimensions={props.dimensions}
      text={copy().content}
    />
  )
}
