import { AsyncLocalStorage } from "node:async_hooks"
import { getAbortSignal as getShutdownSignal } from "../lifecycle/shutdown"

export type RequestSignalOptions = {
  signal?: AbortSignal
  includeScope?: boolean
}

type RequestScope = {
  signal: AbortSignal | null
}

const requestScope = new AsyncLocalStorage<RequestScope>()

export function mergeAbortSignals(
  signals: Array<AbortSignal | null | undefined>,
): AbortSignal | undefined {
  const filtered = signals.filter(Boolean) as AbortSignal[]
  if (filtered.length === 0) return undefined
  if (filtered.length === 1) return filtered[0]

  const any = (AbortSignal as { any?: (signals: AbortSignal[]) => AbortSignal }).any
  if (any) return any(filtered)

  const controller = new AbortController()
  const onAbort = () => controller.abort()
  for (const signal of filtered) {
    if (signal.aborted) {
      controller.abort()
      break
    }
    signal.addEventListener("abort", onAbort, { once: true })
  }
  return controller.signal
}

export function withAbortScope<T>(
  signal: AbortSignal,
  task: () => Promise<T> | T,
): Promise<T> | T {
  return requestScope.run({ signal }, task)
}

export function getRequestSignal(options: RequestSignalOptions = {}): AbortSignal {
  const includeScope = options.includeScope !== false
  const scopeSignal = includeScope ? requestScope.getStore()?.signal ?? null : null
  return (
    mergeAbortSignals([getShutdownSignal(), scopeSignal, options.signal]) ??
    getShutdownSignal()
  )
}

export function isAborted(signal?: AbortSignal): boolean {
  return getRequestSignal({ signal }).aborted
}

function createAbortError(message: string = "Aborted"): Error {
  const error = new Error(message)
  ;(error as { name: string }).name = "AbortError"
  return error
}

export async function sleep(ms: number, options: RequestSignalOptions = {}): Promise<void> {
  const signal = getRequestSignal(options)
  if (signal.aborted) {
    throw createAbortError()
  }

  return await new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timeoutId)
      signal.removeEventListener("abort", onAbort)
      reject(createAbortError())
    }
    const timeoutId = setTimeout(() => {
      signal.removeEventListener("abort", onAbort)
      resolve()
    }, ms)

    signal.addEventListener("abort", onAbort, { once: true })
  })
}

export async function fetchWithTimeout(
  url: string,
  init: RequestInit & {
    timeoutMs?: number
    signal?: AbortSignal
    includeScope?: boolean
  } = {},
): Promise<Response> {
  const { timeoutMs, signal, includeScope, ...rest } = init
  if (!timeoutMs || timeoutMs <= 0) {
    return await fetch(url, {
      ...rest,
      signal: getRequestSignal({ signal, includeScope }),
    })
  }

  const controller = new AbortController()
  let timedOut = false
  const timeoutId = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  const baseSignal = getRequestSignal({ signal, includeScope })
  const mergedSignal = mergeAbortSignals([baseSignal, controller.signal]) ?? baseSignal

  try {
    return await fetch(url, {
      ...rest,
      signal: mergedSignal,
    })
  } catch (err) {
    if (timedOut && (err as { name?: string })?.name === "AbortError") {
      const timeoutError = new Error(`Timeout after ${timeoutMs}ms`)
      ;(timeoutError as { name: string }).name = "TimeoutError"
      throw timeoutError
    }
    throw err
  } finally {
    clearTimeout(timeoutId)
  }
}
