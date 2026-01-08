/**
 * HTTP API client for backend communication.
 *
 * URLs come from launcher.py (which gets them from urls.py).
 * API key comes from process.env.SYNTH_API_KEY.
 */

export async function apiGet(path: string): Promise<any> {
  if (!process.env.SYNTH_API_KEY) {
    throw new Error("Missing API key")
  }
  const res = await fetch(`${process.env.SYNTH_BACKEND_URL}/api${path}`, {
    headers: { Authorization: `Bearer ${process.env.SYNTH_API_KEY}` },
  })
  if (!res.ok) {
    const body = await res.text().catch(() => "")
    const suffix = body ? ` - ${body.slice(0, 160)}` : ""
    throw new Error(`GET ${path}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  return res.json()
}

export async function apiGetV1(path: string): Promise<any> {
  if (!process.env.SYNTH_API_KEY) {
    throw new Error("Missing API key")
  }
  const res = await fetch(`${process.env.SYNTH_BACKEND_URL}/api/v1${path}`, {
    headers: { Authorization: `Bearer ${process.env.SYNTH_API_KEY}` },
  })
  if (!res.ok) {
    const body = await res.text().catch(() => "")
    const suffix = body ? ` - ${body.slice(0, 160)}` : ""
    throw new Error(`GET /api/v1${path}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  return res.json()
}

export async function apiPost(path: string, body: any): Promise<any> {
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
  })
  if (!res.ok) {
    const text = await res.text().catch(() => "")
    const suffix = text ? ` - ${text.slice(0, 160)}` : ""
    throw new Error(`POST ${path}: HTTP ${res.status} ${res.statusText}${suffix}`)
  }
  return res.json().catch(() => ({}))
}

export async function checkBackendHealth(): Promise<string> {
  try {
    const res = await fetch(`${process.env.SYNTH_BACKEND_URL}/health`)
    return res.ok ? "ok" : `bad(${res.status})`
  } catch (err: any) {
    return `err(${err?.message || "unknown"})`
  }
}
