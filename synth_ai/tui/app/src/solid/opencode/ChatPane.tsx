/**
 * ChatPane - Main OpenCode chat component
 *
 * A thin SolidJS client for OpenCode that replaces the embedded TUI.
 */
import { createSignal, createEffect, createMemo, onCleanup, Show } from "solid-js"
import { createStore, reconcile, produce } from "solid-js/store"
import { useKeyboard, useRenderer } from "@opentui/solid"

import { getActionHint, getTextInput, matchAction } from "../../input/keymap"
import { subscribeToOpenCodeEvents } from "../../api/opencode"
import { getClient, type Message, type Part, type Event, type Session, type AssistantMessage } from "./client"
import { ChatConversation, type ChatConversationController } from "./ChatConversation"
import { ChatInput, CHAT_INPUT_HEIGHT, type ChatInputController } from "./ChatInput"
import { SuggestionPopup, type SuggestionItem } from "./SuggestionPopup"
import type { MessageWrapper } from "./chat-types"
import { buildAvailableModels } from "./model-utils"
import type { AvailableModel, Provider, ProviderListResponse, SelectedModel } from "./model-types"
import { moveSelectionIndex } from "../utils/list"
import { COLORS } from "../theme"

export type ChatPaneProps = {
  url: string
  sessionId?: string
  width: number
  height: number
  framed?: boolean
  scrollFocused?: boolean
  /** Working directory for OpenCode session execution */
  workingDir?: string
  onExit?: () => void
  /** Whether this panel has focus */
  focused?: boolean
}

type SessionStatus = { type: "idle" } | { type: "busy" } | { type: "retry"; delay: number }

type SessionState = {
  id: string
  session: Session | null
  messages: MessageWrapper[]
  providers: Provider[]
  isLoading: boolean
  error: string | null
  sessionStatus: SessionStatus
}

type SuggestionMode = "commands" | "models"

type CommandSuggestion = SuggestionItem & {
  kind: "command"
  command: string
}

type ModelSuggestion = SuggestionItem & {
  kind: "model"
  model: SelectedModel
}

type SuggestionEntry = CommandSuggestion | ModelSuggestion

type SlashContext =
  | { mode: "commands"; query: string }
  | { mode: "models"; filter: string }

type SlashCommand = {
  command: string
  description: string
}

const SLASH_COMMANDS: SlashCommand[] = [
  { command: "model", description: "Select a model" },
  { command: "abort", description: "Abort the current response" },
]

export function ChatPane(props: ChatPaneProps) {
  const [state, setState] = createSignal<SessionState>({
    id: props.sessionId || "",
    session: null,
    messages: [],
    providers: [],
    isLoading: false,
    error: null,
    sessionStatus: { type: "idle" },
  })
  // Use SolidJS store for parts - proper fine-grained reactivity like OpenCode's TUI
  const [partsStore, setPartsStore] = createStore<Record<string, Part[]>>({})
  const [, setDebugLog] = createSignal<string[]>([])
  const log = (msg: string) => setDebugLog((logs) => [...logs.slice(-5), msg])
  const scrollFocused = createMemo(() => props.scrollFocused === true)
  const [selectedModel, setSelectedModel] = createSignal<SelectedModel | null>(null)
  const [showAbortedBanner, setShowAbortedBanner] = createSignal(false)
  const [inputText, setInputText] = createSignal("")
  const [suggestionIndex, setSuggestionIndex] = createSignal(0)
  const renderer = useRenderer()
  const framed = createMemo(() => props.framed !== false)
  const frameInset = createMemo(() => (framed() ? 2 : 0))
  const innerWidth = createMemo(() => Math.max(0, props.width - frameInset()))
  const innerHeight = createMemo(() => Math.max(0, props.height - frameInset()))
  let inputController: ChatInputController | null = null
  let conversationController: ChatConversationController | null = null

  // Buffer for parts that arrive before their message (mutable to avoid signal race conditions)
  const pendingParts = new Map<string, Part[]>()

  let client = getClient(props.url)
  let cancelPolling = () => {}

  createEffect(() => {
    client = getClient(props.url)
  })

  // Flatten models from providers, prioritizing synth when available.
  const availableModels = createMemo<AvailableModel[]>(() => buildAvailableModels(state().providers))

  // Current model display info
  const currentModelDisplay = createMemo(() => {
    const model = selectedModel()
    if (!model) return null
    const provider = state().providers.find((p) => p.id === model.providerID)
    const modelInfo = provider?.models[model.modelID]
    return {
      providerName: provider?.name || model.providerID,
      modelName: modelInfo?.name || model.modelID,
    }
  })

  // Compute context stats from messages
  const contextStats = createMemo(() => {
    const msgs = state().messages
    const providers = state().providers
    const lastAssistant = msgs.findLast((m) => m.info.role === "assistant") as { info: AssistantMessage } | undefined
    if (!lastAssistant) return null

    const msg = lastAssistant.info
    const totalTokens = msg.tokens.input + msg.tokens.output + msg.tokens.reasoning +
      msg.tokens.cache.read + msg.tokens.cache.write

    // Find model context limit
    const provider = providers.find((p) => p.id === msg.providerID)
    const model = provider?.models[msg.modelID]
    const contextLimit = model?.limit?.context
    const percentUsed = contextLimit ? Math.round((totalTokens / contextLimit) * 100) : null

    const totalCost = msgs.reduce((sum, m) => sum + (m.info.role === "assistant" ? (m.info as AssistantMessage).cost : 0), 0)
    const costStr = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(totalCost)

    return {
      tokens: totalTokens.toLocaleString(),
      percentUsed,
      cost: costStr,
    }
  })

  const setDefaultModel = (providers: Provider[]) => {
    if (selectedModel()) return

    const synthProvider = providers.find((p) => p.id === "synth")
    if (!synthProvider) {
      setSelectedModel({ providerID: "synth", modelID: "synth-large-instant" })
      return
    }

    const preferredModels = ["synth-large-instant", "synth-large-thinking", "synth-medium", "synth-small"]
    for (const modelId of preferredModels) {
      if (synthProvider.models[modelId]) {
        setSelectedModel({ providerID: "synth", modelID: modelId })
        return
      }
    }

    const firstModelId = Object.keys(synthProvider.models)[0]
    if (firstModelId) {
      setSelectedModel({ providerID: "synth", modelID: firstModelId })
    }
  }

  const getSlashToken = (rawText: string): string | null => {
    const trimmedStart = rawText.trimStart()
    if (!trimmedStart.startsWith("/")) return null
    const withoutSlash = trimmedStart.slice(1)
    const token = withoutSlash.split(/\s+/)[0] ?? ""
    return token.toLowerCase()
  }

  const getSlashContext = (rawText: string): SlashContext | null => {
    const trimmedStart = rawText.trimStart()
    if (!trimmedStart.startsWith("/")) return null
    const afterSlash = trimmedStart.slice(1)
    const token = (afterSlash.split(/\s+/)[0] ?? "").toLowerCase()
    if (token === "model") {
      const filter = afterSlash.slice(token.length).trim()
      return { mode: "models", filter }
    }
    if (afterSlash.includes(" ")) return null
    return { mode: "commands", query: token }
  }

  const slashContext = createMemo(() => (props.focused === true ? getSlashContext(inputText()) : null))
  const currentModelKey = createMemo(() => {
    const model = selectedModel()
    return model ? `${model.providerID}:${model.modelID}` : null
  })

  const commandSuggestions = createMemo<CommandSuggestion[]>(() => {
    const ctx = slashContext()
    if (!ctx || ctx.mode !== "commands") return []
    const query = ctx.query
    return SLASH_COMMANDS
      .filter((cmd) => cmd.command.startsWith(query))
      .map((cmd) => ({
        kind: "command",
        id: cmd.command,
        command: cmd.command,
        label: `/${cmd.command}`,
        description: cmd.description,
      }))
  })

  const modelSuggestions = createMemo<ModelSuggestion[]>(() => {
    const ctx = slashContext()
    if (!ctx || ctx.mode !== "models") return []
    const filter = ctx.filter.trim().toLowerCase()
    const currentKey = currentModelKey()
    return availableModels()
      .filter((model) => {
        if (!filter) return true
        const haystack = `${model.modelName} ${model.modelID} ${model.providerName}`.toLowerCase()
        return haystack.includes(filter)
      })
      .map((model) => {
        const id = `${model.providerID}:${model.modelID}`
        const isCurrent = currentKey === id
        return {
          kind: "model",
          id,
          label: `${isCurrent ? "* " : "  "}${model.modelName}`,
          description: model.providerName,
          model: { providerID: model.providerID, modelID: model.modelID },
        }
      })
  })

  const maxSuggestionItems = createMemo(() => Math.max(1, Math.min(12, innerHeight() - 3)))
  const suggestionItems = createMemo<SuggestionEntry[]>(() => {
    const ctx = slashContext()
    if (!ctx) return []
    const list = ctx.mode === "models" ? modelSuggestions() : commandSuggestions()
    return list.slice(0, maxSuggestionItems())
  })

  const showSuggestions = createMemo(() => slashContext() !== null)
  const suggestionTitle = createMemo(() => (slashContext()?.mode === "models" ? "Models" : "Commands"))
  const suggestionQuery = createMemo(() => {
    const ctx = slashContext()
    if (!ctx) return undefined
    if (ctx.mode === "models") {
      return ctx.filter ? `/model ${ctx.filter}` : "/model"
    }
    return ctx.query ? `/${ctx.query}` : "/"
  })
  const suggestionWidth = createMemo(() => Math.max(1, Math.min(innerWidth(), 52)))
  const suggestionHeight = createMemo(() => {
    const itemCount = suggestionItems().length
    const contentLines = itemCount > 0 ? itemCount + 1 : 2
    return Math.min(innerHeight(), contentLines + 2)
  })
  const suggestionLeft = createMemo(() => (framed() ? 1 : 0))
  const suggestionTop = createMemo(() => {
    const bottomOffset = (framed() ? 1 : 0) + CHAT_INPUT_HEIGHT
    return Math.max(0, props.height - bottomOffset - suggestionHeight())
  })

  createEffect((prevMode: SuggestionMode | null) => {
    const ctx = slashContext()
    const mode = ctx?.mode ?? null
    const items = suggestionItems()
    if (mode !== prevMode) {
      if (mode === "models") {
        const currentKey = currentModelKey()
        const idx = items.findIndex((item) => item.kind === "model" && item.id === currentKey)
        setSuggestionIndex(idx >= 0 ? idx : 0)
      } else {
        setSuggestionIndex(0)
      }
    } else {
      setSuggestionIndex((current) => moveSelectionIndex(current, 0, items.length))
    }
    return mode
  }, null)

  createEffect(async () => {
    try {
      const providersRes = await client.provider.list({})
      const providerData = providersRes.data as ProviderListResponse | undefined
      const providers = providerData?.all || []
      setState((s) => ({ ...s, providers }))
      setDefaultModel(providers)
    } catch (err) {
      setState((s) => ({ ...s, error: String(err) }))
    }
  })

  createEffect(async () => {
    const sessionId = props.sessionId
    cancelPolling()
    if (!sessionId) {
      pendingParts.clear()
      setPartsStore(reconcile({}))
      setState((s) => ({
        ...s,
        id: "",
        session: null,
        messages: [],
        isLoading: false,
        error: null,
        sessionStatus: { type: "idle" },
      }))
      return
    }

    pendingParts.clear()
    setPartsStore(reconcile({}))

    try {
      const [sessionRes, messagesRes, statusRes] = await Promise.all([
        client.session.get({ sessionID: sessionId }),
        client.session.messages({ sessionID: sessionId }),
        client.session.status().catch(() => ({ data: {} })),
      ])
      const initialStatus = (statusRes.data as Record<string, SessionStatus>)?.[sessionId] || { type: "idle" }

      if (messagesRes.data) {
        const partsData: Record<string, Part[]> = {}
        for (const msg of messagesRes.data) {
          partsData[msg.info.id] = msg.parts
        }
        setPartsStore(reconcile(partsData))
      }
      setState((s) => ({
        ...s,
        id: sessionId,
        session: sessionRes.data || null,
        messages: messagesRes.data || [],
        error: null,
        sessionStatus: initialStatus,
        isLoading: initialStatus.type !== "idle",
      }))

      const lastAssistant = (messagesRes.data || []).findLast((m) => m.info.role === "assistant")
      if (lastAssistant && lastAssistant.info.role === "assistant") {
        const assistantInfo = lastAssistant.info as AssistantMessage
        setSelectedModel({ providerID: assistantInfo.providerID, modelID: assistantInfo.modelID })
      } else {
        setDefaultModel(state().providers)
      }
    } catch (err) {
      setState((s) => ({ ...s, error: String(err) }))
    }
  })

  // Subscribe to OpenCode events using our Bun-compatible SSE reader.
  // IMPORTANT: `@opencode-ai/sdk` event streaming can appear buffered under Bun, which breaks UI streaming.
  createEffect(() => {
    log("Starting event subscription...")
    const sub = subscribeToOpenCodeEvents(props.url, {
      onConnect: () => log("SSE connected"),
      onError: (err) => log(`SSE error: ${String(err)}`),
      onEvent: (evt) => handleEvent(evt as unknown as Event),
    })

    onCleanup(() => {
      sub.unsubscribe()
    })
  })

  const handleEvent = (event: Event) => {
    const sessionId = state().id
    if (!sessionId) return

    log(`EVENT: ${event.type}`)

    if (event.type === "message.updated") {
      const msg = event.properties.info
      if (msg.sessionID !== sessionId) return

      // Check for pending parts that arrived before this message
      const pending = pendingParts.get(msg.id) || []
      pendingParts.delete(msg.id)

      if (pending.length > 0) {
        setPartsStore(msg.id, pending)
      }

      setState((s) => {
        const existing = s.messages.findIndex((m) => m.info.id === msg.id)
        if (existing >= 0) {
          const newMessages = [...s.messages]
          // Update the info, keep parts from existing wrapper
          newMessages[existing] = { info: msg, parts: newMessages[existing].parts }
          return { ...s, messages: newMessages }
        }

        if (msg.role === "user") {
          const optimisticIdx = s.messages.findIndex(
            (m) => m.info.id.startsWith("pending_") && m.info.role === "user",
          )
          if (optimisticIdx >= 0) {
            const newMessages = [...s.messages]
            const realParts = pending.length > 0 ? pending : newMessages[optimisticIdx].parts
            newMessages[optimisticIdx] = { info: msg, parts: realParts }
            const optimisticMsgId = s.messages[optimisticIdx].info.id
            const optimisticParts = partsStore[optimisticMsgId] || []
            setPartsStore(produce((store) => {
              delete store[optimisticMsgId]
              store[msg.id] = realParts.length > 0 ? realParts : optimisticParts
            }))
            return { ...s, messages: newMessages }
          }
        }

        return {
          ...s,
          messages: [...s.messages, { info: msg, parts: pending }],
        }
      })
    } else if (event.type === "message.part.updated") {
      const part = event.properties.part
      if (part.sessionID !== sessionId) return

      const existing = partsStore[part.messageID] || []
      const partIdx = existing.findIndex((p) => p.id === part.id)

      if (partIdx >= 0) {
        setPartsStore(part.messageID, partIdx, reconcile(part))
      } else {
        if (!existing.length) {
          setPartsStore(part.messageID, [part])
        } else {
          setPartsStore(part.messageID, produce((parts) => parts.push(part)))
        }
      }
      renderer.requestRender()
    } else if (event.type === "message.removed") {
      if (event.properties.sessionID !== sessionId) return
      const messageID = event.properties.messageID
      setPartsStore(produce((store) => {
        delete store[messageID]
      }))
      setState((s) => ({
        ...s,
        messages: s.messages.filter((m) => m.info.id !== messageID),
      }))
    } else if (event.type === "message.part.removed") {
      if (event.properties.sessionID !== sessionId) return
      const { messageID, partID } = event.properties
      setPartsStore(messageID, produce((parts) => {
        if (!parts) return
        const idx = parts.findIndex((p) => p.id === partID)
        if (idx >= 0) parts.splice(idx, 1)
      }))
    } else if (event.type === "session.updated") {
      const updatedSession = event.properties.info
      if (updatedSession.id !== sessionId) return
      setState((s) => ({ ...s, session: updatedSession }))
    } else if (event.type === "session.status") {
      if (event.properties.sessionID !== sessionId) return
      const status = event.properties.status as SessionStatus
      setState((s) => ({
        ...s,
        sessionStatus: status,
        isLoading: status.type !== "idle",
      }))
      renderer.requestRender()
    } else if (event.type === "session.error") {
      if (event.properties.sessionID !== sessionId) return
      const err = event.properties.error
      let errorMsg = "OpenCode session error"
      if (typeof err === "string") {
        errorMsg = err
      } else if (err && "data" in err && typeof err.data === "object" && err.data && "message" in err.data) {
        errorMsg = String(err.data.message)
      }
      setState((s) => ({
        ...s,
        isLoading: false,
        error: errorMsg,
      }))
      renderer.requestRender()
    } else if (event.type === "session.idle") {
      const idleSessionId = event.properties?.sessionID
      if (idleSessionId && idleSessionId !== sessionId) return
      setState((s) => ({
        ...s,
        isLoading: false,
        sessionStatus: { type: "idle" },
      }))
      renderer.requestRender()
    } else if (event.type === "permission.asked") {
      const request = event.properties
      if (request?.sessionID !== sessionId) return
      void fetch(`${props.url}/permission/${request.id}/reply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reply: "once" }),
      }).catch((err) => {
        setState((s) => ({ ...s, error: String(err) }))
        renderer.requestRender()
      })
    }
  }

  const applySuggestionSelection = (item: SuggestionEntry): boolean => {
    if (item.kind === "command") {
      if (item.command === "model") {
        const next = "/model "
        if (inputController) {
          inputController.setInputText(next)
        } else {
          setInputText(next)
        }
        setSuggestionIndex(0)
        return false
      }
      if (item.command === "abort") {
        abortSession("command")
        return true
      }
      return false
    }

    if (item.kind === "model") {
      setSelectedModel(item.model)
      return true
    }

    return false
  }

  const submitMessage = (rawText: string): boolean => {
    const text = rawText.trim()
    if (!text) return false

    const ctx = getSlashContext(rawText)
    if (ctx) {
      const items = suggestionItems()
      if (!items.length) {
        if (ctx.mode === "commands") {
          const slashToken = getSlashToken(rawText)
          setState((s) => ({ ...s, error: `Unknown command: /${slashToken || ""}`.trim() }))
          renderer.requestRender()
        }
        return false
      }
      const index = moveSelectionIndex(suggestionIndex(), 0, items.length)
      return applySuggestionSelection(items[index])
    }
    if (getSlashToken(rawText) !== null) {
      const slashToken = getSlashToken(rawText)
      setState((s) => ({ ...s, error: `Unknown command: /${slashToken || ""}`.trim() }))
      renderer.requestRender()
      return false
    }

    if (state().isLoading || state().sessionStatus.type !== "idle") return false

    const sessionId = state().id
    if (!sessionId) {
      setState((s) => ({
        ...s,
        error: "No active session. Select one from the Sessions list.",
      }))
      renderer.requestRender()
      return false
    }

    void sendMessage(text, sessionId)
    return true
  }

  const sendMessage = async (text: string, sessionId: string) => {
    // Create optimistic user message and show immediately
    const optimisticMsgId = `pending_${Date.now()}`
    const now = Date.now()
    const optimisticUserMsg = {
      id: optimisticMsgId,
      sessionID: sessionId,
      role: "user" as const,
      time: { created: now },
    } as Message
    const optimisticPart = {
      id: `${optimisticMsgId}_part`,
      messageID: optimisticMsgId,
      sessionID: sessionId,
      type: "text" as const,
      text: text,
      time: { start: now },
    } as Part

    setState((s) => ({
      ...s,
      isLoading: true,
      error: null,
      messages: [...s.messages, { info: optimisticUserMsg, parts: [optimisticPart] }],
    }))
    setPartsStore(optimisticMsgId, [optimisticPart])

    await new Promise((resolve) => setTimeout(resolve, 0))

    const model = selectedModel()
    const directory = props.workingDir || state().session?.directory

    let stopPolling = false
    const promptStartedAt = Date.now()
    cancelPolling = () => {
      stopPolling = true
    }

    const poll = async () => {
      while (!stopPolling) {
        try {
          const messagesRes = await client.session.messages({ sessionID: sessionId })
          if (messagesRes.data) {
            const partsData: Record<string, Part[]> = {}
            for (const msg of messagesRes.data) {
              partsData[msg.info.id] = msg.parts
            }
            setPartsStore(produce((store) => Object.assign(store, partsData)))
            setState((s) => ({ ...s, messages: messagesRes.data! }))
            renderer.requestRender()
            const hasAssistant = messagesRes.data.some((msg) =>
              msg.info.role === "assistant" &&
              (msg.info.time?.created ?? 0) >= promptStartedAt,
            )
            if (state().sessionStatus.type === "idle" && hasAssistant) {
              stopPolling = true
              setState((s) => ({ ...s, isLoading: false }))
              renderer.requestRender()
              break
            }
          }
        } catch {
          // ignore polling errors
        }
        if (Date.now() - promptStartedAt > 120000) {
          stopPolling = true
          setState((s) => ({ ...s, isLoading: false, error: "Timed out waiting for a response." }))
          renderer.requestRender()
          break
        }
        await new Promise((r) => setTimeout(r, 200))
      }
      cancelPolling = () => {}
    }

    void poll()

    try {
      const response = await fetch(`${props.url}/session/${sessionId}/prompt_async`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          parts: [{ type: "text", text }],
          ...(model && { model }),
          ...(directory && { directory }),
        }),
      })
      if (!response.ok && response.status !== 204) {
        const body = await response.text().catch(() => "")
        stopPolling = true
        setState((s) => ({
          ...s,
          isLoading: false,
          error: `Send failed (${response.status}): ${body || response.statusText}`,
        }))
        renderer.requestRender()
        return
      }
    } catch (err) {
      stopPolling = true
      setState((s) => ({ ...s, isLoading: false, error: String(err) }))
      renderer.requestRender()
      return
    }
  }

  // Create a new session
  const createNewSession = async () => {
    try {
      const directory = props.workingDir || state().session?.directory
      const url = directory
        ? `${props.url}/session?directory=${encodeURIComponent(directory)}`
        : `${props.url}/session`
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      })
      if (!response.ok) {
        const text = await response.text().catch(() => "")
        setState((s) => ({ ...s, error: `Failed to create session: ${response.status} ${text}`.trim() }))
        return
      }
      const data = (await response.json().catch(() => null)) as { id?: string } | null
      const sessionId = data?.id
      if (!sessionId) {
        setState((s) => ({ ...s, error: "Failed to create session: missing session id" }))
        return
      }
      const sessionRes = await client.session.get({ sessionID: sessionId })
      pendingParts.clear()
      setPartsStore(reconcile({}))
      setState((s) => ({
        ...s,
        id: sessionId,
        session: sessionRes.data || null,
        messages: [],
        error: null,
        isLoading: false,
        sessionStatus: { type: "idle" },
      }))
      conversationController?.resetScroll()
    } catch (err) {
      setState((s) => ({ ...s, error: String(err) }))
    }
  }

  const abortSession = (reason: string) => {
    const sessionId = state().id
    if (!sessionId) return

    cancelPolling()
    setShowAbortedBanner(true)
    setState((s) => ({ ...s, isLoading: false }))
    setTimeout(() => {
      setShowAbortedBanner(false)
      renderer.requestRender()
    }, 3000)
    renderer.requestRender()

    void fetch(`${props.url}/session/${sessionId}/abort`, { method: "POST" })
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.text().catch(() => "")
          setState((s) => ({
            ...s,
            error: `Abort failed (${res.status}): ${body || res.statusText}`,
          }))
        }
      })
      .catch((err) => {
        log(`Abort error (${reason}): ${err}`)
        setState((s) => ({ ...s, error: String(err) }))
      })
      .finally(() => {
        renderer.requestRender()
      })
  }

  onCleanup(() => {
    cancelPolling()
  })

  const handleBack = () => {
    if (showSuggestions()) {
      if (inputController) {
        inputController.setInputText("")
      } else {
        setInputText("")
      }
      return true
    }
    return false
  }

  ;(ChatPane as any).handleBack = handleBack
  ;(ChatPane as any).getInputState = () =>
    inputController?.getInputState() ?? { inputText: "", inputTextLength: 0 }

  // Handle keyboard input
  useKeyboard((evt) => {
    // Don't process if another handler already handled this event
    if ((evt as any).defaultPrevented) return

    const scrollAction = matchAction(evt, "chat.scroll")
    if (
      scrollAction &&
      showSuggestions() &&
      (scrollAction === "nav.up" || scrollAction === "nav.down" || scrollAction === "nav.home" || scrollAction === "nav.end")
    ) {
      const items = suggestionItems()
      if (items.length) {
        if (scrollAction === "nav.up") {
          setSuggestionIndex((idx) => moveSelectionIndex(idx, -1, items.length))
        } else if (scrollAction === "nav.down") {
          setSuggestionIndex((idx) => moveSelectionIndex(idx, 1, items.length))
        } else if (scrollAction === "nav.home") {
          setSuggestionIndex(0)
        } else if (scrollAction === "nav.end") {
          setSuggestionIndex(items.length - 1)
        }
        evt.preventDefault?.()
        return
      }
    }

    if (scrollAction && conversationController?.handleScrollAction(scrollAction)) {
      evt.preventDefault?.()
      return
    }

    const action = matchAction(evt, "chat.normal")
    if (!action) {
      const text = getTextInput(evt)
      if (text && inputController?.handleTextInput(text)) {
        return
      }
      return
    }

    evt.preventDefault?.()
    if (action === "chat.newSession") {
      void createNewSession()
    } else if (action === "chat.send") {
      inputController?.handleAction(action)
    } else if (action === "chat.backspace") {
      inputController?.handleAction(action)
    } else if (action === "chat.abort") {
      if (state().isLoading || state().sessionStatus.type !== "idle") {
        abortSession("key")
      }
    }
  })

  const headerHeight = createMemo(() => (state().session?.directory ? 2 : 1))
  const frameProps = () =>
    framed() ? { border: true, borderColor: props.focused ? COLORS.textAccent : COLORS.border } : {}
  return (
    <box
      {...frameProps()}
      flexDirection="column"
      width={props.width}
      height={props.height}
    >
      {/* Header */}
      <box flexDirection="column" backgroundColor={COLORS.bgHeader} paddingLeft={1} paddingRight={1}>
        <box flexDirection="row" justifyContent="space-between">
          <Show
            when={state().session}
            fallback={
              <text fg={COLORS.text}>
                <Show when={showAbortedBanner()}>
                  <span style={{ fg: "#ef4444", bold: true }}>{"  ·  ABORTED"}</span>
                </Show>
                <Show when={state().sessionStatus.type !== "idle" && !showAbortedBanner()}>
                  <span style={{ fg: COLORS.textDim }}>{"  ·  " + getActionHint("chat.abort")}</span>
                </Show>
              </text>
            }
          >
            {(() => {
              const session = state().session!
              const isTimestampTitle = session.title && /^\d{4}-\d{2}-\d{2}T/.test(session.title)
              const displayTitle = (!session.title || isTimestampTitle) ? "New session" : session.title
              const timestamp = session.time?.created
                ? new Date(session.time.created).toISOString().slice(0, 19).replace("T", " ")
                : null
              return (
                <text fg={COLORS.text}>
                  <span style={{ fg: COLORS.textDim }}># </span>
                  <span style={{ bold: true }}>{displayTitle}</span>
                  <Show when={showAbortedBanner()}>
                    <span style={{ fg: "#ef4444", bold: true }}>{"  ·  ABORTED"}</span>
                  </Show>
                  <Show when={state().sessionStatus.type !== "idle" && !showAbortedBanner()}>
                    <span style={{ fg: COLORS.textDim }}>{"  ·  " + getActionHint("chat.abort")}</span>
                  </Show>
                  <Show when={timestamp}>
                    <span style={{ fg: COLORS.textDim }}>{" — " + timestamp}</span>
                  </Show>
                </text>
              )
            })()}
          </Show>
          <box flexDirection="row" gap={1} alignItems="center">
            <Show when={scrollFocused()}>
              <text fg={COLORS.textAccent}>[scroll]</text>
            </Show>
            <Show when={contextStats()}>
              <text fg={COLORS.textDim}>
                {contextStats()!.tokens} tokens
                <Show when={contextStats()!.percentUsed !== null}>
                  {" · "}{contextStats()!.percentUsed}%
                </Show>
                {" · "}{contextStats()!.cost}
              </text>
            </Show>
          </box>
        </box>
        <box flexDirection="row" gap={2}>
          <Show when={state().session?.directory}>
            <text fg={COLORS.textDim}>{state().session!.directory}</text>
          </Show>
          <Show when={state().sessionStatus.type !== "idle"}>
            <text>
              <span style={{ fg: COLORS.warning }}>{"● " + state().sessionStatus.type}</span>
            </text>
          </Show>
        </box>
      </box>

      <ChatConversation
        width={innerWidth()}
        height={innerHeight()}
        headerHeight={headerHeight()}
        sessionId={state().id}
        messages={state().messages}
        partsStore={partsStore}
        error={state().error}
        scrollFocused={scrollFocused()}
        isLoading={state().isLoading}
        register={(controller) => {
          conversationController = controller
        }}
      />

      <ChatInput
        focused={props.focused === true}
        width={innerWidth()}
        currentModelDisplay={currentModelDisplay()}
        onSubmit={submitMessage}
        onInputChange={(text) => {
          setInputText(text)
          setSuggestionIndex(0)
        }}
        register={(controller) => {
          inputController = controller
        }}
      />

      <Show when={showSuggestions()}>
        <SuggestionPopup
          title={suggestionTitle()}
          query={suggestionQuery()}
          items={suggestionItems()}
          selectedIndex={suggestionIndex()}
          width={suggestionWidth()}
          height={suggestionHeight()}
          left={suggestionLeft()}
          top={suggestionTop()}
        />
      </Show>
    </box>
  )
}
