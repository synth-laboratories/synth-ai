/**
 * HTTP API client for backend communication.
 *
 * URLs come from launcher.py (which gets them from urls.py).
 * API key comes from process.env.SYNTH_API_KEY.
 */

function sanitizeErrorBody(text: string, maxLen: number): string {
  const raw = (text ?? "").toString()
  if (!raw) return ""

  // Strip HTML tags and collapse whitespace/control chars.
  const noTags = raw.replace(/<[^>]+>/g, " ")
  const collapsed = noTags.replace(/[\r\n\t]+/g, " ").replace(/\s+/g, " ").trim()
  const safe = collapsed.replace(/[^\x20-\x7E]/g, "") // keep printable ASCII
  return safe.length > maxLen ? safe.slice(0, maxLen) : safe
}

async function parseJsonOrThrow(res: Response, label: string): Promise<any> {
  const contentType = res.headers.get("content-type") || ""
  const isJson = contentType.includes("application/json") || contentType.includes("application/problem+json")
  const text = await res.text().catch(() => "")

  if (!isJson) {
    const snippet = sanitizeErrorBody(text, 200)
    const suffix = snippet ? ` - ${snippet}` : ""
    throw new Error(`${label}: expected JSON but got ${contentType || "unknown content-type"}${suffix}`)
  }

  try {
    return text ? JSON.parse(text) : {}
  } catch {
    const snippet = sanitizeErrorBody(text, 200)
    const suffix = snippet ? ` - ${snippet}` : ""
    throw new Error(`${label}: invalid JSON response${suffix}`)
  }
}

type RequestOptions = {
  signal?: AbortSignal
}

export async function apiGet(path: string, options: RequestOptions = {}): Promise<any> {
  if (!process.env.SYNTH_API_KEY) {
    throw new Error("Missing API key")
  }
  const res = await fetch(`${process.env.SYNTH_BACKEND_URL}/api${path}`, {
    headers: { Authorization: `Bearer ${process.env.SYNTH_API_KEY}` },
    signal: options.signal,
  })
  if (!res.ok) {
    const body = await res.text().catch(() => "")
    const snippet = sanitizeErrorBody(body, 200)
    const suffix = snippet ? ` - ${snippet}` : ""
    throw new Error(`GET ${path}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  return await parseJsonOrThrow(res, `GET ${path}`)
}

export async function apiGetV1(path: string, options: RequestOptions = {}): Promise<any> {
  if (!process.env.SYNTH_API_KEY) {
    throw new Error("Missing API key")
  }
  const res = await fetch(`${process.env.SYNTH_BACKEND_URL}/api/v1${path}`, {
    headers: { Authorization: `Bearer ${process.env.SYNTH_API_KEY}` },
    signal: options.signal,
  })
  if (!res.ok) {
    const body = await res.text().catch(() => "")
    const snippet = sanitizeErrorBody(body, 200)
    const suffix = snippet ? ` - ${snippet}` : ""
    throw new Error(`GET /api/v1${path}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  return await parseJsonOrThrow(res, `GET /api/v1${path}`)
}

export async function apiPost(path: string, body: any, options: RequestOptions = {}): Promise<any> {
  if (!process.env.SYNTH_API_KEY) {
    throw new Error("Missing API key")
  }
  const res = await fetch(`${process.env.SYNTH_BACKEND_URL}/api${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${process.env.SYNTH_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    signal: options.signal,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    const snippet = sanitizeErrorBody(text, 200)
    const suffix = snippet ? ` - ${snippet}` : ""
    throw new Error(`POST ${path}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  return await parseJsonOrThrow(res, `POST ${path}`)
}

export async function checkBackendHealth(): Promise<string> {
  try {
    const res = await fetch(`${process.env.SYNTH_BACKEND_URL}/health`)
    return res.ok ? "ok" : `bad(${res.status})`
  } catch (err: any) {
    return `err(${err?.message || "unknown"})`
  }
}
