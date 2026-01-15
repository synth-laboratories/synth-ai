import type { UsageData } from "../../types"

export type { UsageData }

export type ModalState =
  | {
      type: "event"
      title: string
      raw: string
      offset: number
      fullscreen?: boolean
    }
  | {
      type: "log"
      title: string
      raw: string
      offset: number
      tail: boolean
      path: string
      fullscreen?: boolean
    }

export type ActiveModal =
  | "filter"
  | "snapshot"
  | "settings"
  | "usage"
  | "task-apps"
  | "sessions"
  | "config"
  | "generations"
  | "results"
  | "profile"
  | "login"
  | "metrics"
  | "traces"
  | "list-filter"
