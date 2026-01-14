import os from "node:os"
import path from "node:path"

export const synthBaseDir = path.join(os.homedir(), ".synth-ai")
export const tuiDir = path.join(synthBaseDir, "tui")
export const tuiConfigDir = path.join(tuiDir, "config")
export const tuiLogsDir = path.join(tuiDir, "logs")
export const tuiCacheDir = path.join(tuiDir, "cache")

export const tuiSettingsPath = path.join(tuiConfigDir, "settings.json")
export const tuiJobsCachePath = path.join(tuiCacheDir, "jobs.json")
