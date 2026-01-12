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

export type ManagedSignal = {
  signal: AbortSignal
  dispose: () => void
}

const noopDispose = () => {}

export function mergeAbortSignals(
  signals: Array<AbortSignal | null | undefined>,
): ManagedSignal | undefined {
  const filtered = signals.filter(Boolean) as AbortSignal[]
  if (filtered.length === 0) return undefined
  if (filtered.length === 1) return { signal: filtered[0], dispose: noopDispose }

  const any = (AbortSignal as { any?: (signals: AbortSignal[]) => AbortSignal }).any
  if (any) return { signal: any(filtered), dispose: noopDispose }

  // Polyfill: manually wire up abort propagation with proper cleanup
  const controller = new AbortController()
  let disposed = false

  const dispose = () => {
    if (disposed) return
    disposed = true
    for (const signal of filtered) {
      signal.removeEventListener("abort", onAbort)
    }
  }

  const onAbort = () => {
    controller.abort()
    dispose() // Clean up all listeners when any signal aborts
  }

  for (const signal of filtered) {
    if (signal.aborted) {
      controller.abort()
      dispose() // Clean up any listeners added so far
      return { signal: controller.signal, dispose: noopDispose }
    }
    signal.addEventListener("abort", onAbort)
  }

  return { signal: controller.signal, dispose }
}

export function withAbortScope<T>(
  signal: AbortSignal,
  task: () => Promise<T> | T,
): Promise<T> | T {
  return requestScope.run({ signal }, task)
}

export function getRequestSignal(options: RequestSignalOptions = {}): ManagedSignal {
  const includeScope = options.includeScope !== false
  const scopeSignal = includeScope ? requestScope.getStore()?.signal ?? null : null
  return (
    mergeAbortSignals([getShutdownSignal(), scopeSignal, options.signal]) ?? {
      signal: getShutdownSignal(),
      dispose: noopDispose,
    }
  )
}

export function isAborted(signal?: AbortSignal): boolean {
  const managed = getRequestSignal({ signal })
  const result = managed.signal.aborted
  managed.dispose()
  return result
}

function createAbortError(message: string = "Aborted"): Error {
  const error = new Error(message)
  ;(error as { name: string }).name = "AbortError"
  return error
}

export async function sleep(ms: number, options: RequestSignalOptions = {}): Promise<void> {
  const managed = getRequestSignal(options)
  const signal = managed.signal

  if (signal.aborted) {
    managed.dispose()
    throw createAbortError()
  }

  try {
    return await new Promise<void>((resolve, reject) => {
      const onAbort = () => {
        clearTimeout(timeoutId)
        reject(createAbortError())
      }
      const timeoutId = setTimeout(() => {
        resolve()
      }, ms)

      signal.addEventListener("abort", onAbort, { once: true })
    })
  } finally {
    managed.dispose()
  }
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

  // Fast path: no timeout needed
  if (!timeoutMs || timeoutMs <= 0) {
    const managed = getRequestSignal({ signal, includeScope })
    try {
      return await fetch(url, {
        ...rest,
        signal: managed.signal,
      })
    } finally {
      managed.dispose()
    }
  }

  // Timeout path: merge base signal with timeout controller
  const baseManaged = getRequestSignal({ signal, includeScope })
  const controller = new AbortController()
  let timedOut = false

  const timeoutId = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  const mergedManaged = mergeAbortSignals([baseManaged.signal, controller.signal]) ?? baseManaged

  try {
    return await fetch(url, {
      ...rest,
      signal: mergedManaged.signal,
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
    mergedManaged.dispose()
    baseManaged.dispose()
  }
}
