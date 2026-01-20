declare module "@opencode/tui/embedded" {
  import type { Component, JSX } from "solid-js"
  import type { CliRenderer } from "@opentui/core"

  export type SidebarExtensionPosition =
    | "top"
    | "after-context"
    | "after-mcp"
    | "after-lsp"
    | "after-todo"
    | "after-files"
    | "bottom"

  export type SidebarExtension = {
    id: string
    title?: string
    position: SidebarExtensionPosition
    render: (props: { sessionID: string }) => JSX.Element
  }

  export type EmbeddedOpenCodeTUIProps = {
    url: string
    args?: Record<string, unknown>
    sessionID?: string
    extensions?: SidebarExtension[]
    width?: number
    height?: number
    mode?: "dark" | "light"
    renderer?: CliRenderer
    onExit?: () => void
  }

  export const EmbeddedOpenCodeTUI: Component<EmbeddedOpenCodeTUIProps>
}
