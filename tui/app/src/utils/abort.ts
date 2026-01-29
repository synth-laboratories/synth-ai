type AbortableTask<T> = (signal: AbortSignal) => Promise<T> | T

export function isAbortError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false
  const name = (err as { name?: string }).name
  return name === "AbortError" || name === "AbortSignal" || name === "AbortController"
}

export function createAbortControllerRegistry() {
  const controllers = new Map<string, AbortController>()

  function abort(key: string): void {
    const controller = controllers.get(key)
    if (!controller) return
    controller.abort()
    controllers.delete(key)
  }

  function abortAll(): void {
    for (const key of controllers.keys()) {
      abort(key)
    }
  }

  async function run<T>(key: string, task: AbortableTask<T>): Promise<T> {
    abort(key)
    const controller = new AbortController()
    controllers.set(key, controller)
    try {
      return await task(controller.signal)
    } finally {
      if (controllers.get(key) === controller) {
        controllers.delete(key)
      }
    }
  }

  return {
    abort,
    abortAll,
    run,
  }
}
