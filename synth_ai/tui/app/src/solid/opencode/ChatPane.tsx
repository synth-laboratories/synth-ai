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

type SessionListItem = {
  id: string
  title: string | undefined
  time: { updated: number }
}

type SessionState = {
  id: string
  session: Session | null
  messages: MessageWrapper[]
  providers: Provider[]
  isLoading: boolean
  error: string | null
  sessionList: SessionListItem[]
}

export function ChatPane(props: ChatPaneProps) {
  const [state, setState] = createSignal<SessionState>({
    id: props.sessionId || "",
    session: null,
    messages: [],
    providers: [],
    isLoading: false,
    error: null,
    sessionList: [],
  })
  // Use SolidJS store for parts - proper fine-grained reactivity like OpenCode's TUI
  const [partsStore, setPartsStore] = createStore<Record<string, Part[]>>({})
  // Debug log - disable in production by setting to empty function
  const [, setDebugLog] = createSignal<string[]>([])
  const log = (msg: string) => setDebugLog(logs => [...logs.slice(-5), msg])
  const [inputText, setInputText] = createSignal("")
  const [showModelSelector, setShowModelSelector] = createSignal(false)
  const [showSessionSelector, setShowSessionSelector] = createSignal(false)
  const [selectedModel, setSelectedModel] = createSignal<SelectedModel | null>(null)
  const [modelSelectorIndex, setModelSelectorIndex] = createSignal(0)
  const [sessionSelectorIndex, setSessionSelectorIndex] = createSignal(0)
  // Get renderer for explicit re-renders (opentui requires this for terminal updates)
  const renderer = useRenderer()
  // Buffer for parts that arrive before their message (mutable to avoid signal race conditions)
  const pendingParts = new Map<string, Part[]>()

  const client = getClient(props.url)

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

  // Initialize session and fetch providers
  createEffect(async () => {
    try {
      // Fetch providers for context limit info
      const providersRes = await client.provider.list({})
      const providerData = providersRes.data as ProviderListResponse | undefined
      const providers = providerData?.all || []

      // Set default model - use synth provider (routed through Synth backend)
      const setDefaultModel = () => {
        if (selectedModel()) return // Already have a model selected

        // Only use models from synth provider
        const synthProvider = providers.find((p) => p.id === "synth")
        if (!synthProvider) return

        // Preferred models in order of preference - synth-large-instant works best
        const preferredModels = ["synth-large-instant", "synth-large-thinking", "synth-medium", "synth-small"]

        // Try to find a preferred model
        for (const modelId of preferredModels) {
          if (synthProvider.models[modelId]) {
            setSelectedModel({ providerID: "synth", modelID: modelId })
            return
          }
        }

        // Fall back to first available model from synth
        const firstModelId = Object.keys(synthProvider.models)[0]
        if (firstModelId) {
          setSelectedModel({ providerID: "synth", modelID: firstModelId })
        }
      }

      if (props.sessionId) {
        // Load existing session
        const [sessionRes, messagesRes] = await Promise.all([
          client.session.get({ sessionID: props.sessionId }),
          client.session.messages({ sessionID: props.sessionId }),
        ])
        // Populate parts store from messages
        if (messagesRes.data) {
          const partsData: Record<string, Part[]> = {}
          for (const msg of messagesRes.data) {
            partsData[msg.info.id] = msg.parts
          }
          setPartsStore(reconcile(partsData))
        }
        setState((s) => ({
          ...s,
          id: props.sessionId!,
          session: sessionRes.data || null,
          messages: messagesRes.data || [],
          providers,
        }))
        // Set model from last assistant message if available
        const lastAssistant = (messagesRes.data || []).findLast((m) => m.info.role === "assistant")
        if (lastAssistant && lastAssistant.info.role === "assistant") {
          const assistantInfo = lastAssistant.info as AssistantMessage
          setSelectedModel({ providerID: assistantInfo.providerID, modelID: assistantInfo.modelID })
        } else {
          setDefaultModel()
        }
      } else {
        // Create new session
        const sessionRes = await client.session.create(props.workingDir ? { directory: props.workingDir } : {})
        if (sessionRes.data) {
          setState((s) => ({ ...s, id: sessionRes.data!.id, session: sessionRes.data!, providers }))
        }
        setDefaultModel()
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
      directory: props.workingDir,
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
    }
  }

  const sendMessage = async () => {
    const text = inputText().trim()
    if (!text || state().isLoading) return

    const sessionId = state().id
    if (!sessionId) return

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
          }
        } catch {
          // ignore polling errors
        }
        await new Promise((r) => setTimeout(r, 200))
      }
    }

    void poll()

    client.session
      .prompt({
        sessionID: sessionId,
        parts: [{ type: "text", text }],
        ...(model && { model }),
        ...(directory && { directory }),
      })
      .catch((err) => {
        setState((s) => ({ ...s, error: String(err) }))
      })
      .finally(async () => {
        stopPolling = true
        // One final refresh at the end for correctness
        try {
          const messagesRes = await client.session.messages({ sessionID: sessionId })
          if (messagesRes.data) {
            const partsData: Record<string, Part[]> = {}
            for (const msg of messagesRes.data) {
              partsData[msg.info.id] = msg.parts
            }
            setPartsStore(produce((store) => Object.assign(store, partsData)))
            setState((s) => ({ ...s, messages: messagesRes.data! }))
          }
        } catch {
          // ignore
        } finally {
          setState((s) => ({ ...s, isLoading: false }))
          renderer.requestRender()
        }
      })
  }

  // Create a new session
  const createNewSession = async () => {
    try {
      const sessionRes = await client.session.create(props.workingDir ? { directory: props.workingDir } : {})
      if (sessionRes.data) {
        // Clear pending parts and parts store for old session
        pendingParts.clear()
        setPartsStore(reconcile({}))
        setState((s) => ({
          ...s,
          id: sessionRes.data!.id,
          session: sessionRes.data!,
          messages: [],
          error: null,
        }))
      }
    } catch (err) {
      setState((s) => ({ ...s, error: String(err) }))
    }
  }

  // Load the list of sessions
  const loadSessionList = async () => {
    try {
      const sessionsRes = await client.session.list({ limit: 20 })
      if (sessionsRes.data) {
        const sessions = sessionsRes.data.map((s: any) => ({
          id: s.id,
          title: s.title,
          time: s.time,
        }))
        setState((s) => ({ ...s, sessionList: sessions }))
      }
    } catch (err) {
      console.error("Failed to load sessions:", err)
    }
  }

  // Switch to a different session
  const switchToSession = async (sessionId: string) => {
    try {
      const [sessionRes, messagesRes] = await Promise.all([
        client.session.get({ sessionID: sessionId }),
        client.session.messages({ sessionID: sessionId }),
      ])
      // Clear pending parts for old session
      pendingParts.clear()
      setState((s) => ({
        ...s,
        id: sessionId,
        session: sessionRes.data || null,
        messages: messagesRes.data || [],
        error: null,
      }))
      // Update model from last assistant message if available
      const lastAssistant = (messagesRes.data || []).findLast((m) => m.info.role === "assistant")
      if (lastAssistant && lastAssistant.info.role === "assistant") {
        const assistantInfo = lastAssistant.info as AssistantMessage
        setSelectedModel({ providerID: assistantInfo.providerID, modelID: assistantInfo.modelID })
      }
      setShowSessionSelector(false)
    } catch (err) {
      setState((s) => ({ ...s, error: String(err) }))
    }
  }

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

    // Session selector mode
    if (showSessionSelector()) {
      if (evt.name === "escape") {
        setShowSessionSelector(false)
      } else if (evt.name === "return" || evt.name === "enter") {
        const sessions = state().sessionList
        const idx = sessionSelectorIndex()
        if (sessions[idx]) {
          switchToSession(sessions[idx].id)
        }
      } else if (evt.name === "up" || (evt.ctrl && evt.name === "p")) {
        setSessionSelectorIndex((i) => Math.max(0, i - 1))
      } else if (evt.name === "down" || (evt.ctrl && evt.name === "n")) {
        setSessionSelectorIndex((i) => Math.min(state().sessionList.length - 1, i + 1))
      }
      return
    }

    // Normal mode
    
    // Shift+Escape: Abort/interrupt current request
    if (evt.shift && evt.name === "escape") {
      if (state().isLoading && state().id) {
        log("Aborting current request...")
        client.session.abort({ sessionID: state().id! }).catch((err) => {
          log(`Abort error: ${err}`)
        })
        setState((s) => ({ ...s, isLoading: false }))
      }
      return
    }
    
    // Ctrl+K to open model selector (Ctrl+M is same as Enter in terminals)
    if (evt.ctrl && evt.name === "k") {
      setModelSelectorIndex(0)
      setShowModelSelector(true)
    } else if (evt.ctrl && evt.name === "n") {
      // Ctrl+N: New session
      createNewSession()
    } else if (evt.ctrl && evt.name === "l") {
      // Ctrl+L: List/switch sessions
      setSessionSelectorIndex(0)
      loadSessionList()
      setShowSessionSelector(true)
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
          <Show when={state().session} fallback={<text fg="#e2e8f0">OpenCode Chat</text>}>
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
        <Show when={state().session?.directory}>
          <text fg="#64748b">{state().session!.directory}</text>
        </Show>
      </box>

      {/* Messages area */}
      <box flexDirection="column" flexGrow={1} overflow="hidden" paddingLeft={1} paddingRight={1}>
        <Show when={state().error}>
          <box backgroundColor="#7f1d1d" paddingLeft={1} paddingRight={1}>
            <text fg="#fca5a5">Error: {state().error}</text>
          </box>
        </Show>

        {/* Debug log - hidden in production */}

        <For each={state().messages}>
          {(wrapper) => {
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
                  <MessageBubble msg={wrapper.info} parts={parts} maxWidth={bubbleWidth()} />
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

      {/* Session Selector Overlay */}
      <Show when={showSessionSelector()}>
        <box
          position="absolute"
          top={3}
          left={2}
          width={Math.min(props.width - 4, 60)}
          height={Math.min(props.height - 6, 15)}
          backgroundColor="#1e293b"
          border
          borderColor="#10b981"
          flexDirection="column"
        >
          <box paddingLeft={1} paddingRight={1} backgroundColor="#334155">
            <text fg="#e2e8f0">
              <span style={{ bold: true }}>Switch Session</span>
              <span style={{ fg: "#64748b" }}> (Esc to close, Ctrl+N for new)</span>
            </text>
          </box>
          <box flexDirection="column" flexGrow={1} overflow="hidden" paddingLeft={1} paddingRight={1}>
            <Show when={state().sessionList.length === 0}>
              <text fg="#64748b">Loading sessions...</text>
            </Show>
            <For each={state().sessionList}>
              {(session, index) => {
                const isSelected = () => index() === sessionSelectorIndex()
                const isCurrent = () => session.id === state().id
                const title = session.title || "New session"
                const timeStr = new Date(session.time.updated).toLocaleDateString()
                return (
                  <box backgroundColor={isSelected() ? "#10b981" : undefined}>
                    <text fg={isSelected() ? "#e2e8f0" : "#94a3b8"}>
                      {isCurrent() ? "* " : "  "}
                      {title}
                      <span style={{ fg: isSelected() ? "#a7f3d0" : "#64748b" }}> ({timeStr})</span>
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
          <text fg="#475569">Ctrl+N new | Ctrl+L sessions | Ctrl+K model</text>
        </box>
      </box>
    </box>
  )
}
