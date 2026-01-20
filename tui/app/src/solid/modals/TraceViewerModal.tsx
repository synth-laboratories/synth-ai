import { Show, createEffect, createMemo, createSignal, onCleanup } from "solid-js"
import { useKeyboard, useRenderer } from "@opentui/solid"

import type { Snapshot } from "../../types"
import {
  fetchTracesList,
  fetchTraceJson,
  fetchTraceFromApi,
  extractImagesFromTrace,
  type TraceMetadata,
} from "../../api/traces"
import {
  loadImageFromBase64,
  loadImageFromUrl,
  scaleImageToFit,
  renderImageNative,
  getImageCellDimensionsNative,
  type ImageData,
} from "../../utils/image-renderer"

type TraceViewerModalProps = {
  visible: boolean
  snapshot: Snapshot
  width: number
  height: number
  onClose: () => void
  onStatus: (message: string) => void
}

type LoadingState = "idle" | "loading-list" | "loading-trace" | "loading-images" | "ready" | "error"

function formatReward(value: number | null): string {
  if (value == null) return "-"
  return Number.isFinite(value) ? value.toFixed(3).replace(/\.?0+$/, "") : "-"
}

function clampLine(text: string, width: number): string {
  if (text.length <= width) return text
  if (width <= 3) return text.slice(0, width)
  return `${text.slice(0, width - 3)}...`
}

export function TraceViewerModal(props: TraceViewerModalProps) {
  const renderer = useRenderer()
  const [traces, setTraces] = createSignal<TraceMetadata[]>([])
  const [selectedIndex, setSelectedIndex] = createSignal(0)
  const [traceContent, setTraceContent] = createSignal<Record<string, any> | null>(null)
  const [images, setImages] = createSignal<string[]>([])
  const [loadedImage, setLoadedImage] = createSignal<ImageData | null>(null)
  const [loadingState, setLoadingState] = createSignal<LoadingState>("idle")
  const [error, setError] = createSignal<string | null>(null)
  const [scrollOffset, setScrollOffset] = createSignal(0)
  const [showImage, setShowImage] = createSignal(true)

  // Fullscreen layout
  const modalWidth = createMemo(() => props.width - 2)
  const modalHeight = createMemo(() => props.height - 4)
  const contentHeight = createMemo(() => Math.max(10, modalHeight() - 4))

  // Two-column layout: left for trace, right for image
  const leftColumnWidth = createMemo(() => Math.floor(modalWidth() * 0.5))
  const rightColumnWidth = createMemo(() => modalWidth() - leftColumnWidth() - 2)

  const selectedTrace = createMemo(() => {
    const all = traces()
    if (all.length === 0) return null
    const idx = Math.max(0, Math.min(selectedIndex(), all.length - 1))
    return all[idx]
  })

  const clampIndex = (index: number) => {
    const total = traces().length
    if (total === 0) return 0
    return Math.max(0, Math.min(index, total - 1))
  }

  // Load traces when modal opens
  createEffect(() => {
    if (!props.visible) return
    const jobId = props.snapshot.selectedJob?.job_id
    if (!jobId) {
      setError("No job selected")
      setLoadingState("error")
      return
    }

    setLoadingState("loading-list")
    setError(null)
    setTraces([])
    setSelectedIndex(0)
    setTraceContent(null)
    setImages([])

    fetchTracesList(jobId, { limit: 100 })
      .then((result) => {
        setTraces(result)
        setLoadingState(result.length > 0 ? "ready" : "idle")
        if (result.length === 0) {
          setError("No traces found for this job")
        }
      })
      .catch((err) => {
        setError(`Failed to load traces: ${err?.message || "Unknown error"}`)
        setLoadingState("error")
      })
  })

  // Load trace content when selection changes
  createEffect(() => {
    const trace = selectedTrace()
    if (!trace) {
      setTraceContent(null)
      setImages([])
      return
    }

    setLoadingState("loading-trace")
    setScrollOffset(0)

    const jobId = props.snapshot.selectedJob?.job_id
    if (!jobId) {
      setError("No job selected")
      setLoadingState("error")
      return
    }

    fetchTraceFromApi(jobId, trace.seed, trace.candidate_id)
      .then((content) => {
        if (content) {
          setTraceContent(content)
          const extractedImages = extractImagesFromTrace(content)
          setImages(extractedImages)
          setLoadingState("ready")
          return
        }
        return fetchTraceJson(trace.trace_s3_url)
      })
      .then((fallbackContent) => {
        if (!fallbackContent) {
          if (!traceContent()) {
            setError("Failed to load trace (URL may have expired)")
            setLoadingState("error")
          }
          return
        }
        setTraceContent(fallbackContent)
        const extractedImages = extractImagesFromTrace(fallbackContent)
        setImages(extractedImages)
        setLoadingState("ready")
      })
      .catch((err) => {
        setError(`Failed to load trace: ${err?.message || "Unknown error"}`)
        setLoadingState("error")
      })
  })

  // Load first image when images change
  createEffect(() => {
    const imageUrls = images()
    if (imageUrls.length === 0) {
      setLoadedImage(null)
      return
    }

    // Load first image
    const firstImage = imageUrls[0]
    // For half-block rendering: 1 pixel = 1 cell width, 2 pixels = 1 cell height
    // Fit in right column
    const maxWidthPx = rightColumnWidth() - 4
    const maxHeightCells = contentHeight() - 4

    const loadImage = async () => {
      try {
        setLoadingState("loading-images")
        let imageData: ImageData | null = null

        if (firstImage.startsWith("data:")) {
          imageData = await loadImageFromBase64(firstImage)
        } else {
          imageData = await loadImageFromUrl(firstImage)
        }

        if (imageData) {
          // Scale to fit: width in pixels, height in pixels (2 per cell)
          const scaled = await scaleImageToFit(imageData, maxWidthPx, maxHeightCells * 2)
          setLoadedImage(scaled)
        }
        setLoadingState("ready")
      } catch (err) {
        console.error("Failed to load image:", err)
        setLoadingState("ready")
      }
    }

    loadImage()
  })

  // Render image using post-process function (right column)
  createEffect(() => {
    if (!props.visible || !showImage()) return

    const image = loadedImage()
    if (!image) return

    // Position in right column
    const imageX = leftColumnWidth() + 3
    const imageY = 4 // Below title

    const postProcessFn = (buffer: any) => {
      if (!props.visible || !showImage()) return
      const img = loadedImage()
      if (!img) return
      renderImageNative(buffer, img, imageX, imageY)
    }

    renderer.addPostProcessFn(postProcessFn)

    onCleanup(() => {
      renderer.removePostProcessFn(postProcessFn)
    })
  })

  // Build display content
  const displayContent = createMemo(() => {
    const trace = selectedTrace()
    const content = traceContent()
    const extractedImages = images()
    const height = contentHeight()

    const lines: string[] = []

    if (!trace) {
      lines.push("No trace selected")
      return { lines, maxOffset: 0 }
    }

    // Header
    lines.push(`Seed: ${trace.seed}`)
    if (trace.candidate_id) {
      lines.push(`Candidate: ${trace.candidate_id}`)
    }
    // Show best available reward: prefer reward_mean, fallback to outcome_reward
    const reward = trace.reward_mean ?? trace.outcome_reward
    lines.push(`Reward: ${formatReward(reward)}`)
    if (trace.events_score != null) {
      lines.push(`Events Score: ${formatReward(trace.events_score)}`)
    }
    lines.push("")

    // Loading state
    const state = loadingState()
    if (state === "loading-trace" || state === "loading-images") {
      lines.push("Loading trace content...")
      return { lines, maxOffset: 0 }
    }

    // Images info (image renders in right column)
    const currentImage = loadedImage()
    if (extractedImages.length > 0) {
      const dims = currentImage ? getImageCellDimensionsNative(currentImage) : null
      const imgStatus = currentImage
        ? (showImage() ? `${dims?.width}x${dims?.height} cells (→)` : "(hidden)")
        : "loading..."
      lines.push(`Images: ${extractedImages.length} | ${imgStatus}`)
    }
    lines.push("")

    // Trace content preview
    if (content) {
      lines.push("=== TRACE CONTENT ===")
      try {
        const json = JSON.stringify(content, null, 2)
        const jsonLines = json.split("\n")
        const maxJsonLines = 50
        if (jsonLines.length > maxJsonLines) {
          lines.push(...jsonLines.slice(0, maxJsonLines))
          lines.push(`... (${jsonLines.length - maxJsonLines} more lines)`)
        } else {
          lines.push(...jsonLines)
        }
      } catch {
        lines.push("(Unable to display trace content)")
      }
    }

    const maxOffset = Math.max(0, lines.length - height)
    return { lines, maxOffset }
  })

  const visibleLines = createMemo(() => {
    const { lines, maxOffset } = displayContent()
    const height = contentHeight()
    const offset = Math.min(scrollOffset(), maxOffset)
    return lines.slice(offset, offset + height)
  })

  // Keyboard handling
  const handleKey = (evt: any) => {
    if (!props.visible) return
    const name = typeof evt?.name === "string" ? evt.name : ""
    const key = name.toLowerCase()

    const total = traces().length
    const { maxOffset } = displayContent()
    const clampOffset = (value: number) => Math.max(0, Math.min(value, maxOffset))

    if (key === "q" || name === "escape") {
      evt.preventDefault?.()
      props.onClose()
      return
    }

    if (name === "left" || key === "h") {
      evt.preventDefault?.()
      if (total > 0) {
        setSelectedIndex((current) => clampIndex(current - 1))
      }
      return
    }

    if (name === "right" || key === "l") {
      evt.preventDefault?.()
      if (total > 0) {
        setSelectedIndex((current) => clampIndex(current + 1))
      }
      return
    }

    if (name === "up" || key === "k") {
      evt.preventDefault?.()
      setScrollOffset((current) => clampOffset(current - 1))
      return
    }

    if (name === "down" || key === "j") {
      evt.preventDefault?.()
      setScrollOffset((current) => clampOffset(current + 1))
      return
    }

    if (name === "pageup") {
      evt.preventDefault?.()
      setScrollOffset((current) => clampOffset(current - contentHeight() + 1))
      return
    }

    if (name === "pagedown") {
      evt.preventDefault?.()
      setScrollOffset((current) => clampOffset(current + contentHeight() - 1))
      return
    }

    if (name === "home") {
      evt.preventDefault?.()
      setScrollOffset(0)
      return
    }

    if (name === "end") {
      evt.preventDefault?.()
      setScrollOffset(maxOffset)
      return
    }

    // Refresh trace list
    if (key === "r") {
      evt.preventDefault?.()
      const jobId = props.snapshot.selectedJob?.job_id
      if (jobId) {
        setLoadingState("loading-list")
        props.onStatus("Refreshing traces...")
        fetchTracesList(jobId, { limit: 100 })
          .then((result) => {
            setTraces(result)
            setLoadingState(result.length > 0 ? "ready" : "idle")
            props.onStatus(`Loaded ${result.length} traces`)
          })
          .catch((err) => {
            setError(`Failed to refresh: ${err?.message || "Unknown error"}`)
            setLoadingState("error")
          })
      }
      return
    }

    // Toggle image display
    if (key === "i") {
      evt.preventDefault?.()
      setShowImage((prev) => !prev)
      return
    }
  }

  useKeyboard(handleKey)

  const hint = createMemo(() => {
    const total = traces().length
    const idx = selectedIndex()
    const { maxOffset } = displayContent()
    const scrollInfo = maxOffset > 0 ? ` [${scrollOffset() + 1}/${maxOffset + 1}]` : ""
    const imageHint = loadedImage() ? ` | i ${showImage() ? "hide" : "show"} image` : ""
    return `←/→ trace (${idx + 1}/${total}) | ↑/↓ scroll${scrollInfo}${imageHint} | r refresh | q close`
  })

  const title = createMemo(() => {
    const trace = selectedTrace()
    const total = traces().length
    if (!trace) return "Trace Viewer"
    return `Trace Viewer - Seed ${trace.seed} (${selectedIndex() + 1}/${total})`
  })

  return (
    <Show when={props.visible}>
      <box
        position="absolute"
        left={1}
        top={2}
        width={modalWidth()}
        height={modalHeight()}
        backgroundColor="#0b1220"
        border
        borderStyle="single"
        borderColor="#3b82f6"
        zIndex={30}
        flexDirection="column"
        paddingLeft={1}
        paddingRight={1}
        paddingTop={0}
        paddingBottom={0}
      >
        {/* Title bar */}
        <text fg="#3b82f6">
          {clampLine(title(), Math.max(10, modalWidth() - 4))}
        </text>

        {/* Two-column layout */}
        <box flexDirection="row" height={contentHeight()}>
          {/* Left column: trace content */}
          <box
            flexDirection="column"
            width={leftColumnWidth()}
            height={contentHeight()}
            overflow="hidden"
            paddingRight={1}
          >
            {/* Loading/Error state */}
            <Show when={loadingState() === "loading-list"}>
              <text fg="#94a3b8">Loading traces...</text>
            </Show>

            <Show when={loadingState() === "error"}>
              <text fg="#ef4444">{error() || "An error occurred"}</text>
            </Show>

            {/* Trace content */}
            <Show when={loadingState() !== "loading-list" && loadingState() !== "error"}>
              <text fg="#e2e8f0">{visibleLines().join("\n")}</text>
            </Show>
          </box>

          {/* Right column: image area (rendered via post-process) */}
          <box
            flexDirection="column"
            width={rightColumnWidth()}
            height={contentHeight()}
            border
            borderStyle="single"
            borderColor="#1e3a5f"
          >
            <Show when={!loadedImage() && images().length > 0}>
              <text fg="#94a3b8">Loading image...</text>
            </Show>
            <Show when={loadedImage() && !showImage()}>
              <text fg="#94a3b8">Image hidden (press 'i' to show)</text>
            </Show>
            <Show when={!images().length}>
              <text fg="#64748b">No images in trace</text>
            </Show>
            {/* Image renders here via post-process */}
          </box>
        </box>

        {/* Hint bar */}
        <text fg="#94a3b8">{hint()}</text>
      </box>
    </Show>
  )
}
