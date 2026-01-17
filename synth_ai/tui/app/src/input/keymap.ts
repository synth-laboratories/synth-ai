export type KeyEvent = {
  name?: string
  sequence?: string
  ctrl?: boolean
  meta?: boolean
  shift?: boolean
}

type KeyCombo = string

const KEY = {
  q: "q",
  esc: "escape",
  enter: "enter",
  tab: "tab",
  shiftTab: "shift+tab",
  up: "up",
  down: "down",
  left: "left",
  right: "right",
  pageUp: "pageup",
  pageDown: "pagedown",
  home: "home",
  end: "end",
  space: "space",
  backspace: "backspace",
  slash: "/",
  one: "1",
  two: "2",
  three: "3",
  a: "a",
  b: "b",
  c: "c",
  d: "d",
  e: "e",
  f: "f",
  g: "g",
  h: "h",
  i: "i",
  j: "j",
  k: "k",
  l: "l",
  shiftL: "shift+l",
  m: "m",
  n: "n",
  o: "o",
  p: "p",
  r: "r",
  s: "s",
  t: "t",
  u: "u",
  v: "v",
  y: "y",
  ctrlC: "ctrl+c",
  ctrlK: "ctrl+k",
  ctrlN: "ctrl+n",
  ctrlP: "ctrl+p",
  ctrlV: "ctrl+v",
  ctrlX: "ctrl+x",
  metaV: "meta+v",
} as const

const ACTION_KEYS = {
  "app.forceQuit": [KEY.ctrlC],
  "app.back": [KEY.esc],
  "app.quit": [KEY.q],
  "app.refresh": [KEY.r],
  "jobs.loadMore": [KEY.shiftL],
  "app.logout": [KEY.l],
  "focus.next": [KEY.tab],
  "focus.prev": [KEY.shiftTab],
  "pane.jobs": [KEY.one],
  "pane.agent": [KEY.two],
  "pane.logs": [KEY.three],
  "pane.select": [KEY.enter],
  "modal.open.filter": [KEY.slash],
  "modal.open.config": [KEY.i],
  "modal.open.profile": [KEY.p],
  "modal.open.results": [KEY.v],
  "modal.open.listFilter": [KEY.f],
  "modal.open.snapshot": [KEY.g],
  "modal.open.settings": [KEY.s, KEY.o],
  "modal.open.usage": [KEY.u],
  "modal.open.taskApps": [KEY.a],
  "modal.open.createJob": [KEY.n],
  "modal.open.metrics": [KEY.m],
  "modal.open.traces": [KEY.t],
  "job.cancel": [KEY.c],
  "job.artifacts": [KEY.d],
  "nav.up": [KEY.up, KEY.k],
  "nav.down": [KEY.down, KEY.j],
  "nav.pageUp": [KEY.pageUp],
  "nav.pageDown": [KEY.pageDown],
  "nav.home": [KEY.home],
  "nav.end": [KEY.end],
  "modal.confirm": [KEY.enter],
  "modal.copy": [KEY.y],
  "listFilter.toggle": [KEY.space, KEY.enter],
  "listFilter.all": [KEY.a],
  "usage.openBilling": [KEY.b],
  "metrics.refresh": [KEY.r],
  "detail.toggleFullscreen": [KEY.f],
  "detail.tail": [KEY.t],
  "sessions.connect": [KEY.c],
  "sessions.disconnect": [KEY.d],
  "sessions.refresh": [KEY.r],
  "sessions.copy": [KEY.y],
  "login.confirm": [KEY.enter],
  "selector.up": [KEY.up, KEY.ctrlP],
  "selector.down": [KEY.down, KEY.ctrlN],
  "chat.newSession": [KEY.ctrlN],
  "chat.send": [KEY.enter],
  "chat.backspace": [KEY.backspace],
  "chat.abort": [KEY.ctrlX],
  "candidates.prev": [KEY.left, KEY.h],
  "candidates.next": [KEY.right, KEY.l],
  "candidates.scrollUp": [KEY.up, KEY.k],
  "candidates.scrollDown": [KEY.down, KEY.j],
  "generation.prev": [KEY.shiftTab],
  "generation.next": [KEY.tab],
  "trace.prev": [KEY.left, KEY.h],
  "trace.next": [KEY.right, KEY.l],
  "trace.refresh": [KEY.r],
  "trace.toggleImage": [KEY.i],
} as const

export type KeyAction = keyof typeof ACTION_KEYS

const DISPLAY_ALIASES: Record<string, string> = {
  escape: "esc",
  pageup: "pgup",
  pagedown: "pgdn",
  ctrl: "ctrl",
  shift: "shift",
  meta: "cmd",
}

function formatKeyComboInternal(combo: string): string {
  const parts = combo.split("+").map((part) => DISPLAY_ALIASES[part] ?? part)
  return parts.join("+")
}

type ActionMeta = {
  label: string
  hint: string
}

function buildHint(label: string, keys: readonly string[]): string {
  const primaryKey = formatKeyComboInternal(keys[0])
  const lowerKey = primaryKey.toLowerCase()
  const lowerLabel = label.toLowerCase()
  // Embed key if it's a single char matching the first letter
  if (lowerKey.length === 1 && lowerLabel.startsWith(lowerKey)) {
    return `(${primaryKey})${label.slice(1)}`
  }
  // Otherwise prefix with key
  return `(${primaryKey}) ${label}`
}

const ACTION_META: Record<KeyAction, ActionMeta> = {
  "app.forceQuit": { label: "Force Quit", hint: "(ctrl+c) Force Quit" },
  "app.back": { label: "Back", hint: "(esc) Back" },
  "app.quit": { label: "Quit", hint: buildHint("Quit", ACTION_KEYS["app.quit"]) },
  "app.refresh": { label: "Refresh", hint: buildHint("Refresh", ACTION_KEYS["app.refresh"]) },
  "jobs.loadMore": { label: "Load More", hint: buildHint("Load More", ACTION_KEYS["jobs.loadMore"]) },
  "app.logout": { label: "Logout", hint: buildHint("Logout", ACTION_KEYS["app.logout"]) },
  "focus.next": { label: "Focus Next", hint: "(tab) Focus Next" },
  "focus.prev": { label: "Focus Prev", hint: "(shift+tab) Focus Prev" },
  "pane.jobs": { label: "Jobs", hint: buildHint("Jobs", ACTION_KEYS["pane.jobs"]) },
  "pane.agent": { label: "Agent", hint: buildHint("Agent", ACTION_KEYS["pane.agent"]) },
  "pane.logs": { label: "Logs", hint: buildHint("Logs", ACTION_KEYS["pane.logs"]) },
  "pane.select": { label: "Select", hint: "(enter) Select" },
  "modal.open.filter": { label: "Filter", hint: "(/) Filter" },
  "modal.open.config": { label: "Config", hint: buildHint("Config", ACTION_KEYS["modal.open.config"]) },
  "modal.open.profile": { label: "Profile", hint: buildHint("Profile", ACTION_KEYS["modal.open.profile"]) },
  "modal.open.results": { label: "Results", hint: buildHint("Results", ACTION_KEYS["modal.open.results"]) },
  "modal.open.listFilter": { label: "Filter", hint: buildHint("Filter", ACTION_KEYS["modal.open.listFilter"]) },
  "modal.open.snapshot": { label: "Snapshot", hint: buildHint("Snapshot", ACTION_KEYS["modal.open.snapshot"]) },
  "modal.open.settings": { label: "Settings", hint: buildHint("Settings", ACTION_KEYS["modal.open.settings"]) },
  "modal.open.usage": { label: "Usage", hint: buildHint("Usage", ACTION_KEYS["modal.open.usage"]) },
  "modal.open.taskApps": { label: "Apps", hint: buildHint("Apps", ACTION_KEYS["modal.open.taskApps"]) },
  "modal.open.createJob": { label: "New Job", hint: buildHint("New Job", ACTION_KEYS["modal.open.createJob"]) },
  "modal.open.metrics": { label: "Metrics", hint: buildHint("Metrics", ACTION_KEYS["modal.open.metrics"]) },
  "modal.open.traces": { label: "Traces", hint: buildHint("Traces", ACTION_KEYS["modal.open.traces"]) },
  "job.cancel": { label: "Cancel", hint: buildHint("Cancel", ACTION_KEYS["job.cancel"]) },
  "job.artifacts": { label: "Artifacts", hint: buildHint("Artifacts", ACTION_KEYS["job.artifacts"]) },
  "nav.up": { label: "Up", hint: buildHint("Up", ACTION_KEYS["nav.up"]) },
  "nav.down": { label: "Down", hint: buildHint("Down", ACTION_KEYS["nav.down"]) },
  "nav.pageUp": { label: "Page Up", hint: "(pgup) Page Up" },
  "nav.pageDown": { label: "Page Down", hint: "(pgdn) Page Down" },
  "nav.home": { label: "Home", hint: "(home) Home" },
  "nav.end": { label: "End", hint: "(end) End" },
  "modal.confirm": { label: "Confirm", hint: "(enter) Confirm" },
  "modal.copy": { label: "Copy", hint: buildHint("Copy", ACTION_KEYS["modal.copy"]) },
  "listFilter.toggle": { label: "Toggle", hint: "(space) Toggle" },
  "listFilter.all": { label: "All/None", hint: buildHint("All/None", ACTION_KEYS["listFilter.all"]) },
  "usage.openBilling": { label: "Billing", hint: buildHint("Billing", ACTION_KEYS["usage.openBilling"]) },
  "metrics.refresh": { label: "Refresh", hint: buildHint("Refresh", ACTION_KEYS["metrics.refresh"]) },
  "detail.toggleFullscreen": { label: "Fullscreen", hint: buildHint("Fullscreen", ACTION_KEYS["detail.toggleFullscreen"]) },
  "detail.tail": { label: "Tail", hint: buildHint("Tail", ACTION_KEYS["detail.tail"]) },
  "sessions.connect": { label: "Connect", hint: buildHint("Connect", ACTION_KEYS["sessions.connect"]) },
  "sessions.disconnect": { label: "Disconnect", hint: buildHint("Disconnect", ACTION_KEYS["sessions.disconnect"]) },
  "sessions.refresh": { label: "Refresh", hint: buildHint("Refresh", ACTION_KEYS["sessions.refresh"]) },
  "sessions.copy": { label: "Copy URL", hint: buildHint("Copy URL", ACTION_KEYS["sessions.copy"]) },
  "login.confirm": { label: "Start", hint: "(enter) Start" },
  "selector.up": { label: "Up", hint: buildHint("Up", ACTION_KEYS["selector.up"]) },
  "selector.down": { label: "Down", hint: buildHint("Down", ACTION_KEYS["selector.down"]) },
  "chat.newSession": { label: "New Session", hint: "(ctrl+n) New Session" },
  "chat.send": { label: "Send", hint: "(enter) Send" },
  "chat.backspace": { label: "Backspace", hint: "(backspace) Backspace" },
  "chat.abort": { label: "Abort", hint: "(ctrl+x) Abort" },
  "candidates.prev": { label: "Prev", hint: buildHint("Prev", ACTION_KEYS["candidates.prev"]) },
  "candidates.next": { label: "Next", hint: buildHint("Next", ACTION_KEYS["candidates.next"]) },
  "candidates.scrollUp": { label: "Scroll Up", hint: buildHint("Scroll Up", ACTION_KEYS["candidates.scrollUp"]) },
  "candidates.scrollDown": { label: "Scroll Down", hint: buildHint("Scroll Down", ACTION_KEYS["candidates.scrollDown"]) },
  "generation.prev": { label: "Prev Gen", hint: "(shift+tab) Prev Gen" },
  "generation.next": { label: "Next Gen", hint: "(tab) Next Gen" },
  "trace.prev": { label: "Prev Trace", hint: buildHint("Prev Trace", ACTION_KEYS["trace.prev"]) },
  "trace.next": { label: "Next Trace", hint: buildHint("Next Trace", ACTION_KEYS["trace.next"]) },
  "trace.refresh": { label: "Refresh", hint: buildHint("Refresh", ACTION_KEYS["trace.refresh"]) },
  "trace.toggleImage": { label: "Toggle Image", hint: buildHint("Toggle Image", ACTION_KEYS["trace.toggleImage"]) },
}

const CONTEXT_ACTIONS = {
  "app.global": [
    "app.forceQuit",
    "app.back",
    "app.quit",
    "focus.next",
    "focus.prev",
    "app.refresh",
    "jobs.loadMore",
    "pane.jobs",
    "pane.logs",
    "pane.agent",
    "app.logout",
    "modal.open.filter",
    "modal.open.config",
    "modal.open.profile",
    "modal.open.results",
    "modal.open.listFilter",
    "modal.open.snapshot",
    "modal.open.settings",
    "modal.open.usage",
    "modal.open.taskApps",
    "modal.open.createJob",
    "modal.open.metrics",
    "modal.open.traces",
    "job.cancel",
    "job.artifacts",
    "nav.down",
    "nav.up",
    "pane.select",
  ],
  "modal.detail": [
    "detail.toggleFullscreen",
    "detail.tail",
    "modal.copy",
    "nav.down",
    "nav.up",
    "modal.confirm",
  ],
  "modal.filter": ["modal.confirm"],
  "modal.snapshot": ["modal.confirm"],
  "modal.settings": ["nav.up", "nav.down", "modal.confirm"],
  "modal.usage": ["usage.openBilling", "nav.up", "nav.down", "modal.confirm"],
  "modal.metrics": ["nav.up", "nav.down", "metrics.refresh", "modal.confirm"],
  "modal.taskApps": ["nav.up", "nav.down", "modal.copy", "modal.confirm"],
  "modal.sessions": [
    "nav.up",
    "nav.down",
    "sessions.copy",
    "sessions.connect",
    "sessions.disconnect",
    "sessions.refresh",
    "modal.confirm",
  ],
  "modal.listFilter": ["nav.up", "nav.down", "listFilter.toggle", "listFilter.all"],
  "modal.config": ["nav.up", "nav.down", "modal.confirm"],
  "modal.profile": ["modal.confirm"],
  "modal.login": ["login.confirm", "app.back"],
  "modal.createJob": ["nav.up", "nav.down", "modal.confirm", "app.back"],
  "modal.candidates": [
    "modal.copy",
    "candidates.prev",
    "candidates.next",
    "candidates.scrollUp",
    "candidates.scrollDown",
    "nav.pageUp",
    "nav.pageDown",
    "nav.home",
    "nav.end",
  ],
  "modal.generations": [
    "generation.prev",
    "generation.next",
    "nav.up",
    "nav.down",
    "nav.pageUp",
    "nav.pageDown",
    "nav.home",
    "nav.end",
    "modal.confirm",
  ],
  "modal.trace": [
    "trace.prev",
    "trace.next",
    "nav.up",
    "nav.down",
    "nav.pageUp",
    "nav.pageDown",
    "nav.home",
    "nav.end",
    "trace.refresh",
    "trace.toggleImage",
  ],
  "chat.normal": [
    "chat.newSession",
    "chat.send",
    "chat.backspace",
    "chat.abort",
  ],
  "chat.scroll": [
    "nav.up",
    "nav.down",
    "nav.pageUp",
    "nav.pageDown",
    "nav.home",
    "nav.end",
  ],
} as const satisfies Record<string, readonly KeyAction[]>

const RESERVED_KEYS: Record<KeyCombo, KeyAction[]> = {
  [KEY.esc]: ["app.back"],
  [KEY.ctrlC]: ["app.forceQuit"],
}

function assertReservedKeys(): void {
  const usage = new Map<KeyCombo, KeyAction[]>()
  for (const action of Object.keys(ACTION_KEYS) as KeyAction[]) {
    for (const combo of ACTION_KEYS[action]) {
      const list = usage.get(combo) ?? []
      list.push(action)
      usage.set(combo, list)
    }
  }
  for (const [combo, allowed] of Object.entries(RESERVED_KEYS) as [KeyCombo, KeyAction[]][]) {
    const actions = usage.get(combo) ?? []
    const invalid = actions.filter((action) => !allowed.includes(action))
    if (invalid.length) {
      throw new Error(
        `Reserved key "${combo}" is bound to ${invalid.join(", ")}; only ${allowed.join(", ")} allowed.`,
      )
    }
  }
}

if (process.env.NODE_ENV !== "production") {
  assertReservedKeys()
}

export type KeyContext = keyof typeof CONTEXT_ACTIONS

const NAME_ALIASES: Record<string, string> = {
  return: KEY.enter,
  esc: KEY.esc,
  escape: KEY.esc,
  arrowup: KEY.up,
  arrowdown: KEY.down,
  arrowleft: KEY.left,
  arrowright: KEY.right,
  pgup: KEY.pageUp,
  pageup: KEY.pageUp,
  pgdown: KEY.pageDown,
  pagedown: KEY.pageDown,
  slash: KEY.slash,
}

const SEQUENCE_ALIASES: Record<string, string> = {
  "\u001b": KEY.esc,
  "\u001b[A": KEY.up,
  "\u001b[B": KEY.down,
  "\u001b[C": KEY.right,
  "\u001b[D": KEY.left,
  "\u001b[5~": KEY.pageUp,
  "\u001b[6~": KEY.pageDown,
  "\u001b[H": KEY.home,
  "\u001b[F": KEY.end,
  "\u001bOH": KEY.home,
  "\u001bOF": KEY.end,
}

export function getActionKeys(action: KeyAction): KeyCombo[] {
  return [...ACTION_KEYS[action]]
}

export function getContextActions(context: KeyContext): KeyAction[] {
  return [...(CONTEXT_ACTIONS[context] as readonly KeyAction[])]
}

export function matchAction(event: KeyEvent, context: KeyContext): KeyAction | null {
  const combo = normalizeKeyEvent(event)
  if (!combo) return null
  const actions = CONTEXT_ACTIONS[context] as readonly KeyAction[]
  for (const action of actions) {
    const combos = ACTION_KEYS[action] as readonly KeyCombo[]
    if (combos.includes(combo)) {
      return action
    }
  }
  return null
}

export function formatActionKeys(
  action: KeyAction,
  options: { joiner?: string; primaryOnly?: boolean } = {},
): string {
  const joiner = options.joiner ?? "/"
  const keys = ACTION_KEYS[action]
  const list = options.primaryOnly ? keys.slice(0, 1) : keys
  return list.map(formatKeyCombo).join(joiner)
}

export function formatKeyCombo(combo: KeyCombo): string {
  return formatKeyComboInternal(combo)
}

export function getActionLabel(action: KeyAction): string {
  return ACTION_META[action].label
}

export function getActionHint(action: KeyAction): string {
  return ACTION_META[action].hint
}

export function getActionMeta(action: KeyAction): ActionMeta {
  return ACTION_META[action]
}

/**
 * Build a hint with a custom label but using the key(s) from an action.
 * Format: "(key) label" or "(key)abel" if key matches first letter
 */
export function buildActionHint(action: KeyAction, label: string): string {
  const keys = ACTION_KEYS[action]
  return buildHint(label, keys)
}

/**
 * Build a hint combining keys from two actions (e.g., nav.down/nav.up for "move")
 * Format: "(down/up) label"
 */
export function buildCombinedHint(action1: KeyAction, action2: KeyAction, label: string): string {
  const key1 = formatKeyComboInternal(ACTION_KEYS[action1][0])
  const key2 = formatKeyComboInternal(ACTION_KEYS[action2][0])
  return `(${key1}/${key2}) ${label}`
}

export function getTextInput(event: KeyEvent): string | null {
  const seq = event.sequence || ""
  if (!seq || seq.length !== 1) return null
  if (event.ctrl || event.meta) return null
  if (seq === "\u001b") return null
  return seq
}

function normalizeKeyEvent(event: KeyEvent): KeyCombo | null {
  const rawSequence = event.sequence && SEQUENCE_ALIASES[event.sequence]
  const raw = rawSequence ?? event.name ?? (event.sequence && event.sequence.length === 1 ? event.sequence : undefined)
  if (!raw) return null
  let name = raw
  let shift = !!event.shift
  if (raw === "backtab") {
    name = "tab"
    shift = true
  }
  if (raw.length === 1 && raw >= "A" && raw <= "Z") {
    name = raw.toLowerCase()
    shift = true
  }
  const normalizedName = NAME_ALIASES[name.toLowerCase()] ?? name.toLowerCase()
  const parts: string[] = []
  if (event.ctrl) parts.push("ctrl")
  if (event.meta) parts.push("meta")
  if (shift) parts.push("shift")
  parts.push(normalizedName)
  return parts.join("+")
}
