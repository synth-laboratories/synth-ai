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
  | "job-filter"
  | "snapshot"
  | "key"
  | "settings"
  | "usage"
  | "task-apps"
  | "sessions"
  | "config"
  | "results"
  | "profile"
  | "urls"
  | "login"
  | "metrics"
  | "traces"
