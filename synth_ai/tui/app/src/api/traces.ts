/**
 * API layer for fetching rollout traces.
 */

import { apiGet } from "./client"

export type TraceMetadata = {
  seed: number
  candidate_id: string | null
  trace_id: string
  trace_s3_key: string
  trace_s3_url: string
  reward_mean: number | null
  outcome_reward: number | null
  events_score: number | null
  verifier_score: number | null
}

export type TraceListResponse = {
  traces: TraceMetadata[]
}

/**
 * Fetch list of traces for a job.
 * Optionally filter by candidate_id or seed.
 */
export async function fetchTracesList(
  jobId: string,
  filters?: { candidateId?: string; seed?: number; limit?: number },
  options: { signal?: AbortSignal } = {}
): Promise<TraceMetadata[]> {
  const params = new URLSearchParams()
  if (filters?.candidateId) params.set("candidate_id", filters.candidateId)
  if (filters?.seed !== undefined) params.set("seed", String(filters.seed))
  if (filters?.limit) params.set("limit", String(filters.limit))

  const queryString = params.toString()
  const path = `/prompt-learning/online/jobs/${jobId}/traces/list${queryString ? `?${queryString}` : ""}`

  const response = await apiGet(path, options)
  // API returns array directly, not wrapped in { traces: [...] }
  return Array.isArray(response) ? response : (response.traces ?? [])
}

/**
 * Fetch trace JSON content from a presigned S3 URL.
 * Returns null if the fetch fails (e.g., URL expired).
 */
export async function fetchTraceJson(presignedUrl: string): Promise<Record<string, any> | null> {
  try {
    const response = await fetch(presignedUrl)
    if (!response.ok) {
      if (response.status === 403) {
        // Presigned URL likely expired
        return null
      }
      throw new Error(`Failed to fetch trace: HTTP ${response.status}`)
    }
    return await response.json()
  } catch (error) {
    console.error("Error fetching trace:", error)
    return null
  }
}

/**
 * Fetch trace JSON content from backend API with image URLs resolved.
 */
export async function fetchTraceFromApi(
  jobId: string,
  seed: number,
  candidateId?: string | null
): Promise<Record<string, any> | null> {
  const params = new URLSearchParams()
  params.set("format", "json")
  params.set("mode", "urls")
  if (candidateId) params.set("candidate_id", candidateId)
  const path = `/prompt-learning/online/jobs/${jobId}/traces/${seed}?${params.toString()}`
  try {
    const response = await apiGet(path)
    if (response && typeof response === "object" && "trace" in response) {
      return response.trace as Record<string, any>
    }
    return response as Record<string, any>
  } catch (error) {
    console.error("Error fetching trace from API:", error)
    return null
  }
}

/**
 * Extract images from a v3/v4 trace.
 * Images may be in various locations depending on the task type.
 * Returns array of image data URLs (base64) or presigned URLs.
 */
export function extractImagesFromTrace(trace: Record<string, any>): string[] {
  const images: string[] = []
  const imageUrls = typeof trace.image_urls === "object" && trace.image_urls ? trace.image_urls : {}
  const resolveImageRef = (ref: string): string => {
    const filename = ref.split("/").pop() ?? ref
    return imageUrls[filename] ?? imageUrls[ref] ?? ref
  }

  for (const url of Object.values(imageUrls)) {
    if (typeof url === "string") {
      images.push(url)
    }
  }

  // Check artifacts array
  if (Array.isArray(trace.artifacts)) {
    for (const artifact of trace.artifacts) {
      if (artifact.type === "image" && artifact.data) {
        images.push(artifact.data)
      }
      if (artifact.url) {
        images.push(artifact.url)
      }
    }
  }

  // Check event_history for LLM outputs with images
  if (Array.isArray(trace.event_history)) {
    for (const event of trace.event_history) {
      // Check output_messages for image content
      if (Array.isArray(event.output_messages)) {
        for (const message of event.output_messages) {
          if (Array.isArray(message.content)) {
            for (const part of message.content) {
              // Handle various image formats
              if (part.type === "image" && part.uri) {
                images.push(part.uri)
              }
              if (part.type === "image_url" && part.image_url?.url) {
                const url = part.image_url.url
                images.push(url.startsWith("images/") ? resolveImageRef(url) : url)
              }
              if (part.image) {
                images.push(part.image)
              }
              if (part.image_ref) {
                images.push(resolveImageRef(part.image_ref))
              }
            }
          }
          // Direct image field
          if (message.image) {
            images.push(message.image)
          }
        }
      }

      // Check for images in event data
      if (event.data?.image) {
        images.push(event.data.image)
      }
      if (event.data?.generated_image) {
        images.push(event.data.generated_image)
      }

      // Check llm_response.message.content for image_url (web design traces)
      const llmMessage = event.llm_response?.message
      if (llmMessage && Array.isArray(llmMessage.content)) {
        for (const part of llmMessage.content) {
          if (part.type === "image_url" && part.image_url?.url) {
            const url = part.image_url.url
            images.push(url.startsWith("images/") ? resolveImageRef(url) : url)
          }
          if (part.image_ref) {
            images.push(resolveImageRef(part.image_ref))
          }
        }
      }
    }
  }

  // Check root-level fields
  if (trace.generated_image) {
    images.push(trace.generated_image)
  }
  if (trace.output_image) {
    images.push(trace.output_image)
  }

  const collectImageRefs = (obj: any) => {
    if (obj && typeof obj === "object") {
      if (Array.isArray(obj)) {
        for (const item of obj) {
          collectImageRefs(item)
        }
        return
      }
      if (typeof obj.image_ref === "string") {
        images.push(resolveImageRef(obj.image_ref))
      }
      for (const value of Object.values(obj)) {
        collectImageRefs(value)
      }
    }
  }

  collectImageRefs(trace)

  return images
}
