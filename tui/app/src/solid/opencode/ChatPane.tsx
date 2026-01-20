/**
 * ChatPane - Main OpenCode chat component
 *
 * A thin SolidJS client for OpenCode that replaces the embedded TUI.
 */
import { createSignal, createEffect, createMemo, onCleanup, For, Show } from "solid-js"
import { createStore, reconcile, produce } from "solid-js/store"
import { useKeyboard, useRenderer } from "@opentui/solid"
import { getClient, type Message, type Part, type Event, type Session, type AssistantMessage } from "./client"
import { COLORS } from "../theme"
import { MessageBubble } from "./MessageBubble"
import { subscribeToOpenCodeEvents } from "../../api/opencode"
import { appState } from "../../state/app-state"

export type ChatPaneProps = {
  url: string
  sessionId?: string
  width: number
  height: number
  /** Working directory for OpenCode session execution */
  workingDir?: string
  onExit?: () => void
}

// The SDK returns messages in this wrapper format
type MessageWrapper = {
  info: Message
  parts: Part[]
}

type ProviderModel = {
  id: string
  name?: string
  limit: { context: number; output: number }
}

type Provider = {
  id: string
  name: string
  models: Record<string, ProviderModel>
}

type SelectedModel = {
  providerID: string
  modelID: string
}

type ProviderListResponse = {
  all: Provider[]
  connected: string[]
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
  // Debug log - disable in production by setting to empty function
  const [, setDebugLog] = createSignal<string[]>([])
  const log = (msg: string) => setDebugLog(logs => [...logs.slice(-5), msg])
  const [inputText, setInputText] = createSignal("")
  const [showModelSelector, setShowModelSelector] = createSignal(false)
  const [selectedModel, setSelectedModel] = createSignal<SelectedModel | null>(null)
  const [modelSelectorIndex, setModelSelectorIndex] = createSignal(0)
  // Interrupt UX.
  const [showAbortedBanner, setShowAbortedBanner] = createSignal(false)
  // Get renderer for explicit re-renders (opentui requires this for terminal updates)
  const renderer = useRenderer()
  // Buffer for parts that arrive before their message (mutable to avoid signal race conditions)
  const pendingParts = new Map<string, Part[]>()

  let client = getClient(props.url)
  let cancelPolling = () => {}

  createEffect(() => {
    client = getClient(props.url)
  })

  // Flatten models from connected providers (only show synth models)
  const availableModels = createMemo(() => {
    const models: { providerID: string; modelID: string; providerName: string; modelName: string }[] = []
    // Only include synth provider models
    const synthProvider = state().providers.find((p) => p.id === "synth")
    
    if (synthProvider) {
      for (const [modelId, model] of Object.entries(synthProvider.models)) {
        models.push({
          providerID: synthProvider.id,
          modelID: modelId,
          providerName: synthProvider.name,
          modelName: model.name || modelId,
        })
      }
    }

    return models
  })

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
      
      // Initialize partsStore for this message if we have pending parts
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
        } else {
          // If this is a user message and we have an optimistic one, replace it
          if (msg.role === "user") {
            const optimisticIdx = s.messages.findIndex(
              (m) => m.info.id.startsWith("pending_") && m.info.role === "user"
            )
            if (optimisticIdx >= 0) {
              const newMessages = [...s.messages]
              // Use parts from the real message or the optimistic if none provided
              const realParts = pending.length > 0 ? pending : newMessages[optimisticIdx].parts
              newMessages[optimisticIdx] = { info: msg, parts: realParts }
              // Also update partsStore - copy optimistic parts to real message ID
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
        }
      })
    } else if (event.type === "message.part.updated") {
      const part = event.properties.part
      if (part.sessionID !== sessionId) return

      // Update parts store directly - fine-grained update using SolidJS store
      const existing = partsStore[part.messageID] || []
      const partIdx = existing.findIndex((p) => p.id === part.id)
      
      if (partIdx >= 0) {
        // Update existing part in-place for fine-grained reactivity
        setPartsStore(part.messageID, partIdx, reconcile(part))
      } else {
        // Add new part - if message doesn't exist in store yet, create it
        if (!existing.length) {
          setPartsStore(part.messageID, [part])
        } else {
          setPartsStore(part.messageID, produce((parts) => parts.push(part)))
        }
      }
      // CRITICAL: Request terminal re-render for streaming updates
      renderer.requestRender()
    } else if (event.type === "message.removed") {
      if (event.properties.sessionID !== sessionId) return
      const messageID = event.properties.messageID
      // Clean up partsStore for removed message
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
      // Update partsStore using produce for fine-grained reactivity
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
      const errProp = event.properties.error
      const errMsg = typeof errProp === "string" ? errProp : (errProp as any)?.message || "OpenCode session error"
      setState((s) => ({
        ...s,
        isLoading: false,
        error: errMsg,
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

  const sendMessage = async () => {
    const text = inputText().trim()
    if (!text) return

    if (text === "/abort" || text === "/stop" || text === "/interrupt") {
      setInputText("")
      abortSession("command")
      return
    }

    if (state().isLoading) return

    const sessionId = state().id
    if (!sessionId) {
      setState((s) => ({
        ...s,
        error: "No active session. Press Shift+O to connect to OpenCode.",
      }))
      renderer.requestRender()
      return
    }

    // Clear input immediately
    setInputText("")
    
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
    
    // Add optimistic message to state immediately
    setState((s) => ({ 
      ...s, 
      isLoading: true, 
      error: null,
      messages: [...s.messages, { info: optimisticUserMsg, parts: [optimisticPart] }]
    }))
    // Also add to partsStore for consistency
    setPartsStore(optimisticMsgId, [optimisticPart])

    // Allow UI to render the optimistic message before making the API call
    await new Promise(resolve => setTimeout(resolve, 0))

    const model = selectedModel()
    // Pass directory to ensure tools run in the correct working directory
    const directory = props.workingDir || state().session?.directory
    
    // Fire off the prompt and poll for updates while it runs.
    // Rationale: OpenCode persists message parts as they stream; polling guarantees UI updates even if SSE delivery
    // or terminal redraws are unreliable in some environments.
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
              (msg.info.time?.created ?? 0) >= promptStartedAt
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

  createEffect(() => {
    const isBusy = state().isLoading || state().sessionStatus.type !== "idle"
    if (state().id && isBusy) {
      appState.openCodeAbort = () => abortSession("global")
    } else if (appState.openCodeAbort) {
      appState.openCodeAbort = null
    }
  })

  onCleanup(() => {
    if (appState.openCodeAbort) appState.openCodeAbort = null
  })

  // Handle keyboard input
  useKeyboard((evt) => {
    // Model selector mode
    if (showModelSelector()) {
      if (evt.name === "escape") {
        setShowModelSelector(false)
      } else if (evt.name === "return" || evt.name === "enter") {
        const models = availableModels()
        const idx = modelSelectorIndex()
        if (models[idx]) {
          setSelectedModel({ providerID: models[idx].providerID, modelID: models[idx].modelID })
          setShowModelSelector(false)
        }
      } else if (evt.name === "up" || (evt.ctrl && evt.name === "p")) {
        setModelSelectorIndex((i) => Math.max(0, i - 1))
      } else if (evt.name === "down" || (evt.ctrl && evt.name === "n")) {
        setModelSelectorIndex((i) => Math.min(availableModels().length - 1, i + 1))
      }
      return
    }

    const isBusy = state().isLoading || state().sessionStatus.type !== "idle"

    // Ctrl+G: fallback interrupt (not always delivered by terminals)
    if (evt.ctrl && evt.name === "g" && state().id) {
      abortSession("Ctrl+G")
      return
    }

    // Esc: immediate abort when busy (Shift+Esc is not distinguishable in most terminals).
    if (evt.name === "escape" && isBusy && state().id) {
      abortSession("Esc")
      return
    }

    // Ctrl+K to open model selector (Ctrl+M is same as Enter in terminals)
    if (evt.ctrl && evt.name === "k") {
      setModelSelectorIndex(0)
      setShowModelSelector(true)
    } else if (evt.name === "return" || evt.name === "enter") {
      sendMessage()
    } else if (evt.name === "escape") {
      props.onExit?.()
    } else if (evt.name === "backspace") {
      setInputText((t) => t.slice(0, -1))
    } else if (evt.sequence && evt.sequence.length === 1 && !evt.ctrl && !evt.meta) {
      // Single character input
      setInputText((t) => t + evt.sequence)
    }
  })


  const bubbleWidth = createMemo(() => {
    // Keep bubbles compact even on wide terminals (OpenCode-like).
    const hardMax = Math.min(72, Math.max(32, props.width - 6))
    const target = Math.floor(props.width * 0.62)
    return Math.max(32, Math.min(hardMax, target))
  })

  function _countWrappedLines(text: string, maxWidth: number): number {
    if (!text) return 0
    let lines = 0
    for (const para of String(text).split("\n")) {
      if (!para) {
        lines += 1
        continue
      }
      if (para.length <= maxWidth) {
        lines += 1
        continue
      }
      const words = para.split(" ")
      let current = ""
      for (const word of words) {
        if (!current) {
          if (word.length <= maxWidth) {
            current = word
          } else {
            lines += Math.ceil(word.length / maxWidth)
            current = ""
          }
          continue
        }
        if (current.length + 1 + word.length <= maxWidth) {
          current += " " + word
        } else {
          lines += 1
          if (word.length <= maxWidth) {
            current = word
          } else {
            lines += Math.ceil(word.length / maxWidth)
            current = ""
          }
        }
      }
      if (current) lines += 1
    }
    return lines
  }

  const headerHeight = createMemo(() => (state().session?.directory ? 2 : 1))
  const inputHeight = 3 // top border + 2 lines of content
  const innerHeight = createMemo(() => Math.max(0, props.height - 2)) // outer border consumes 2 rows
  const messageAreaHeight = createMemo(() => Math.max(0, innerHeight() - headerHeight() - inputHeight))

  const visibleMessages = createMemo(() => {
    // Window messages so we never emit more terminal rows than fit. This prevents terminal scroll
    // (which can visually “obscure” the fixed input bar after long sessions).
    const maxBubbleContentWidth = Math.max(10, bubbleWidth() - 4)
    const maxHeight = messageAreaHeight()
    const msgs = state().messages

    type Visible = { wrapper: MessageWrapper; maxLines?: number }
    const result: Visible[] = []
    let used = 0

    // Always reserve 1 line for the "Thinking..." indicator if present.
    const loadingReserve = state().isLoading ? 1 : 0
    const budget = Math.max(0, maxHeight - loadingReserve)

    for (let i = msgs.length - 1; i >= 0; i--) {
      const wrapper = msgs[i]
      const parts = partsStore[wrapper.info.id] || wrapper.parts || []
      let contentLines = 0
      for (const p of parts as any[]) {
        const text = (p?.text ?? p?.content ?? "").toString()
        if (!text) continue
        contentLines += _countWrappedLines(text.trim(), maxBubbleContentWidth)
      }
      // Bubble border adds 2 rows, and we also render a 1-row meta header above the bubble.
      // Wrapper has marginBottom={1} which effectively consumes one blank row.
      const bubbleHeight = Math.max(3, contentLines + 2) // minimum to show border/body
      const blockHeight = 1 + bubbleHeight + 1

      if (used + blockHeight <= budget) {
        result.push({ wrapper })
        used += blockHeight
        continue
      }

      // If we have no messages yet, include the last one but clamp its rendered lines to fit.
      if (result.length === 0 && budget > 0) {
        // Remaining for bubble content lines:
        // budget = 1(meta) + (content+2 border) + 1(margin)
        const remainingForBubble = Math.max(0, budget - 2) // strip meta + margin
        const remainingContent = Math.max(0, remainingForBubble - 2) // strip bubble borders
        result.push({ wrapper, maxLines: Math.max(1, remainingContent) })
      }
      break
    }

    return result.reverse()
  })

  return (
    <box
      flexDirection="column"
      width={props.width}
      height={props.height}
      border
      borderColor={COLORS.border}
    >
      {/* Header */}
      <box flexDirection="column" backgroundColor={COLORS.bgHeader} paddingLeft={1} paddingRight={1}>
        <box flexDirection="row" justifyContent="space-between">
          <Show
            when={state().session}
            fallback={
              <text fg="#e2e8f0">
                OpenCode Chat
                <span style={{ fg: COLORS.textDim }}>{"  (Shift+O sessions)"}</span>
                <Show when={showAbortedBanner()}>
                  <span style={{ fg: "#ef4444", bold: true }}>{"  ·  ABORTED"}</span>
                </Show>
                <Show when={state().sessionStatus.type !== "idle" && !showAbortedBanner()}>
                  <span style={{ fg: "#64748b" }}>{"  ·  Ctrl+X abort"}</span>
                </Show>
              </text>
            }
          >
            {(() => {
              const session = state().session!
              // Check if title is a timestamp (OpenCode uses timestamp as default title)
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
                    <span style={{ fg: "#64748b" }}>{"  ·  Ctrl+X abort"}</span>
                  </Show>
                  <Show when={timestamp}>
                    <span style={{ fg: "#64748b" }}>{" — " + timestamp}</span>
                  </Show>
                </text>
              )
            })()}
          </Show>
          <Show when={contextStats()}>
            <text fg="#64748b">
              {contextStats()!.tokens} tokens
              <Show when={contextStats()!.percentUsed !== null}>
                {" · "}{contextStats()!.percentUsed}%
              </Show>
              {" · "}{contextStats()!.cost}
            </text>
          </Show>
        </box>
        <box flexDirection="row" gap={2}>
          <Show when={state().session?.directory}>
            <text fg="#64748b">{state().session!.directory}</text>
          </Show>
          <Show when={state().sessionStatus.type !== "idle"}>
            <text>
              <span style={{ fg: "#fbbf24" }}>
                {"● " + state().sessionStatus.type}
              </span>
              <span style={{ fg: "#64748b" }}>
                {" (Esc abort, Ctrl+X abort)"}
              </span>
            </text>
          </Show>
        </box>
      </box>

      {/* Messages area */}
      <box flexDirection="column" flexGrow={1} overflow="hidden" paddingLeft={1} paddingRight={1}>
        <Show when={state().error}>
          <box backgroundColor="#7f1d1d" paddingLeft={1} paddingRight={1}>
            <text fg="#fca5a5">Error: {state().error}</text>
          </box>
        </Show>

        <Show when={!state().id && state().messages.length === 0}>
          <box paddingLeft={1} paddingRight={1} paddingTop={1}>
            <text fg="#94a3b8">
              No OpenCode session selected. Press Shift+O to connect.
            </text>
          </box>
        </Show>

        <For each={visibleMessages()}>
          {(item) => {
            const wrapper = item.wrapper
            // Look up parts from the separate store for fine-grained reactivity
            // With SolidJS store, access is direct and automatically reactive
            const parts = () => partsStore[wrapper.info.id] || wrapper.parts || []
            return (
              <box
                flexDirection="row"
                justifyContent={wrapper.info.role === "user" ? "flex-end" : "flex-start"}
                marginBottom={1}
              >
                <box flexDirection="column" width={bubbleWidth()}>
                  <box flexDirection="row" justifyContent="space-between" paddingLeft={1} paddingRight={1} marginBottom={0}>
                    <text fg={wrapper.info.role === "user" ? COLORS.textAccent : COLORS.success}>
                      <span style={{ bold: true }}>{wrapper.info.role === "user" ? "You" : "Assistant"}</span>
                    </text>
                    <Show when={wrapper.info.role === "assistant"}>
                      <text fg={COLORS.textDim}>
                        {(() => {
                          const a = wrapper.info as AssistantMessage
                          const duration = a.time?.completed ? `${((a.time.completed - a.time.created) / 1000).toFixed(1)}s` : null
                          return `${a.mode} · ${a.modelID}${duration ? ` · ${duration}` : ""}`
                        })()}
                      </text>
                    </Show>
                  </box>
                  <MessageBubble msg={wrapper.info} parts={parts} maxWidth={bubbleWidth()} maxLines={item.maxLines} />
                </box>
              </box>
            )
          }}
        </For>

        <Show when={state().isLoading}>
          <text fg="#94a3b8">Thinking...</text>
        </Show>
      </box>

      {/* Model Selector Overlay */}
      <Show when={showModelSelector()}>
        <box
          position="absolute"
          top={3}
          left={2}
          width={Math.min(props.width - 4, 50)}
          height={Math.min(props.height - 6, 15)}
          backgroundColor="#1e293b"
          border
          borderColor="#3b82f6"
          flexDirection="column"
        >
          <box paddingLeft={1} paddingRight={1} backgroundColor="#334155">
            <text fg="#e2e8f0">
              <span style={{ bold: true }}>Select Model</span>
              <span style={{ fg: "#64748b" }}> (Esc to close)</span>
            </text>
          </box>
          <box flexDirection="column" flexGrow={1} overflow="hidden" paddingLeft={1} paddingRight={1}>
            <For each={availableModels()}>
              {(model, index) => {
                const isSelected = () => index() === modelSelectorIndex()
                const isCurrent = () => {
                  const sel = selectedModel()
                  return sel?.providerID === model.providerID && sel?.modelID === model.modelID
                }
                return (
                  <box backgroundColor={isSelected() ? "#3b82f6" : undefined}>
                    <text fg={isSelected() ? "#e2e8f0" : "#94a3b8"}>
                      {isCurrent() ? "* " : "  "}
                      {model.modelName}
                      <span style={{ fg: isSelected() ? "#bfdbfe" : "#64748b" }}> ({model.providerName})</span>
                    </text>
                  </box>
                )
              }}
            </For>
          </box>
        </box>
      </Show>

      {/* Input area */}
      <box
        flexDirection="column"
        border={["top"]}
        borderColor={COLORS.border}
        backgroundColor="#0f172a"
      >
        <box paddingLeft={1} paddingRight={1}>
          <text fg="#64748b">{"> "}</text>
          <text fg="#e2e8f0">
            {inputText() || "_"}
          </text>
        </box>
        <box paddingLeft={1} paddingRight={1} flexDirection="row" justifyContent="space-between">
          <Show when={currentModelDisplay()} fallback={<text fg="#64748b">No model selected</text>}>
            <text fg="#64748b">
              {currentModelDisplay()!.modelName}
              <span style={{ fg: "#475569" }}> ({currentModelDisplay()!.providerName})</span>
            </text>
          </Show>
          <text fg="#475569">Shift+O sessions | Ctrl+K model | Ctrl+X abort | /abort</text>
        </box>
      </box>
    </box>
  )
}
