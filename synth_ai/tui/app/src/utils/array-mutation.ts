import { log } from "./log"

type ArrayMutationWatch = {
  label: string
  getId?: (item: any) => string | null | undefined
}

const watchMap = new WeakMap<object, ArrayMutationWatch>()
const proxyMap = new WeakMap<object, any>()
let methodsPatched = false

function describeArray(items: any[], getId?: (item: any) => string | null | undefined) {
  if (!getId) {
    return { length: items.length }
  }
  const ids = items.map((item) => getId(item) ?? "")
  return { length: items.length, ids }
}

function describeArg(arg: any, getId?: (item: any) => string | null | undefined) {
  if (Array.isArray(arg)) {
    return {
      type: "array",
      ...describeArray(arg, getId),
    }
  }
  if (arg && typeof arg === "object") {
    const id = getId ? getId(arg) : null
    if (id) {
      return { type: "object", id }
    }
    if ("job_id" in arg) {
      return { type: "object", id: String((arg as any).job_id) }
    }
    if ("id" in arg) {
      return { type: "object", id: String((arg as any).id) }
    }
    return { type: "object" }
  }
  if (typeof arg === "function") {
    return { type: "function", name: arg.name || "anonymous" }
  }
  return arg
}

function summarizeArgs(args: any[], getId?: (item: any) => string | null | undefined) {
  return args.map((arg) => describeArg(arg, getId))
}

function isNumericKey(prop: PropertyKey): boolean {
  if (typeof prop === "number") return Number.isInteger(prop) && prop >= 0
  if (typeof prop !== "string") return false
  if (prop === "") return false
  const num = Number(prop)
  return Number.isInteger(num) && String(num) === prop
}

function ensurePatched(): void {
  if (methodsPatched) return
  methodsPatched = true
  const methods = [
    "push",
    "pop",
    "shift",
    "unshift",
    "splice",
    "sort",
    "reverse",
    "copyWithin",
    "fill",
  ] as const
  for (const method of methods) {
    const original = (Array.prototype as any)[method]
    if (typeof original !== "function") continue
    Object.defineProperty(Array.prototype, method, {
      value: function mutationLogger(this: any[], ...args: any[]) {
        const watch = watchMap.get(this as any)
        if (!watch) {
          return original.apply(this, args)
        }
        const before = describeArray(this, watch.getId)
        const stack = new Error().stack || ""
        const result = original.apply(this, args)
        const after = describeArray(this, watch.getId)
        log("state", "array mutation", {
          label: watch.label,
          method,
          args: summarizeArgs(args, watch.getId),
          before,
          after,
          stack,
        })
        return result
      },
      writable: true,
      configurable: true,
    })
  }
  log("state", "array mutation hooks installed", { methods })
}

export function wrapArrayWrites<T>(
  items: T[],
  options: { label: string; getId?: (item: T) => string | null | undefined },
): T[] {
  if (!Array.isArray(items)) return items
  const existing = proxyMap.get(items as any)
  if (existing) return existing

  const proxy = new Proxy(items as any, {
    set(target, prop, value, receiver) {
      const watch = watchMap.get(target)
      const shouldLog = Boolean(watch) && (prop === "length" || isNumericKey(prop))
      const before = shouldLog ? describeArray(target, watch?.getId) : null
      const stack = shouldLog ? new Error().stack || "" : ""
      const result = Reflect.set(target, prop, value, receiver)
      if (shouldLog) {
        const after = describeArray(target, watch?.getId)
        log("state", "array write", {
          label: watch?.label ?? options.label,
          prop: String(prop),
          value: describeArg(value, watch?.getId),
          before,
          after,
          stack,
        })
      }
      return result
    },
  })
  proxyMap.set(items as any, proxy)
  proxyMap.set(proxy as any, proxy)
  watchMap.set(items as any, {
    label: options.label,
    getId: options.getId as any,
  })
  watchMap.set(proxy as any, {
    label: options.label,
    getId: options.getId as any,
  })
  return proxy as T[]
}

export function watchArrayMutations<T>(
  items: T[],
  options: { label: string; getId?: (item: T) => string | null | undefined },
): T[] {
  if (!Array.isArray(items)) return items
  ensurePatched()
  if (!watchMap.has(items as any)) {
    log("state", "array mutation watch", {
      label: options.label,
      ...describeArray(items as any[], options.getId as any),
    })
  }
  watchMap.set(items as any, {
    label: options.label,
    getId: options.getId as any,
  })
  return items
}
