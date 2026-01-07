/**
 * Global application state.
 */

// Mutable app state
export const appState = {
  activePane: "jobs" as "jobs" | "events",
  healthStatus: "unknown",
  autoSelected: false,

  // Event state
  lastSeq: 0,
  selectedEventIndex: 0,
  eventWindowStart: 0,
  eventFilter: "",

  // Job filter state
  jobStatusFilter: new Set<string>(),
  jobFilterOptions: [] as Array<{ status: string; count: number }>,
  jobFilterCursor: 0,
  jobFilterWindowStart: 0,

  // Key modal state
  keyPasteActive: false,
  keyPasteBuffer: "",

  // Modal scroll offsets
  eventModalOffset: 0,
  resultsModalOffset: 0,
  configModalOffset: 0,
  promptBrowserIndex: 0,
  promptBrowserOffset: 0,

  // Task Apps modal state
  taskAppsModalOffset: 0,
  taskAppsModalSelectedIndex: 0,

  // Create Job modal state
  createJobCursor: 0,

  // Deploy state
  deployedUrl: null as string | null,
  deployProc: null as import("child_process").ChildProcess | null,

  // Request tokens for cancellation
  jobSelectToken: 0,
  eventsToken: 0,
}

export function setActivePane(pane: "jobs" | "events"): void {
  appState.activePane = pane
}
