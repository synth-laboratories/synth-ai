/**
 * Job detail panel formatting.
 */
import type { AppData } from "../types"
import type { JobEvent, JobSummary } from "../tui_data"
import { isEvalJob, normalizeJobType, num } from "../tui_data"

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === "object" && !Array.isArray(value)
}

function pickString(...values: unknown[]): string | null {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim()
  }
  return null
}

function shorten(value: string, max = 36): string {
  if (value.length <= max) return value
  return value.slice(0, Math.max(0, max - 3)) + "..."
}

function formatHost(value: string | null): string | null {
  if (!value) return null
  try {
    return new URL(value).host || null
  } catch {
    return value.replace(/^https?:\/\//, "").split("/")[0] || null
  }
}

function formatCompactCount(value: number): string {
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}m`
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`
  return `${Math.round(value)}`
}

function formatCost(value: number): string {
  const decimals = value >= 1 ? 2 : value >= 0.01 ? 4 : 6
  return `$${value.toFixed(decimals)}`
}

function formatDurationMs(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(1)}s`
  return `${Math.round(value)}ms`
}

function extractPromptLearningConfig(meta: Record<string, any>): Record<string, any> | null {
  const config =
    meta.prompt_initial_snapshot?.raw_config?.prompt_learning ||
    meta.config?.prompt_learning ||
    meta.job_config?.prompt_learning ||
    meta.prompt_learning ||
    meta.config ||
    meta.job_config
  return isRecord(config) ? config : null
}

function extractOptimizerConfig(meta: Record<string, any>): Record<string, any> | null {
  const config = meta.prompt_initial_snapshot?.optimizer_config || meta.optimizer_config
  return isRecord(config) ? config : null
}

function extractPolicyConfig(
  meta: Record<string, any>,
  rawConfig: Record<string, any> | null,
  optimizerConfig: Record<string, any> | null,
): Record<string, any> | null {
  const policy =
    (rawConfig && isRecord(rawConfig.policy) ? rawConfig.policy : null) ||
    (optimizerConfig && isRecord(optimizerConfig.policy_config) ? optimizerConfig.policy_config : null) ||
    (isRecord(meta.policy) ? meta.policy : null) ||
    (isRecord(meta.policy_config) ? meta.policy_config : null)
  return policy ?? null
}

function formatPolicyLine(policy: Record<string, any> | null): string | null {
  if (!policy) return null
  const model = pickString(policy.model)
  const provider = pickString(policy.provider)
  const inference = pickString(policy.inference_mode)
  if (!model && !provider) return null
  let label = model || provider || ""
  if (model && provider) {
    label = `${model} (${provider})`
  }
  if (inference && inference !== provider) {
    label = `${label} ${inference}`.trim()
  }
  return label ? `Model: ${label}` : null
}

function buildTaskLine(
  meta: Record<string, any>,
  rawConfig: Record<string, any> | null,
): string | null {
  const envName = pickString(rawConfig?.env_name, meta.env_name)
  const taskId = pickString(
    rawConfig?.task_app_id,
    rawConfig?.app_id,
    meta.task_app_id,
    meta.app_id,
  )
  const host = formatHost(
    pickString(rawConfig?.localapi_url, meta.localapi_url, rawConfig?.task_app_url, meta.endpoint_base_url),
  )
  const parts = [envName, taskId].filter((value, index, array) => value && array.indexOf(value) === index) as string[]
  let label = parts.join(" / ")
  if (!label && host) label = host
  if (label && host && !label.includes(host)) {
    label = `${label} @ ${host}`
  }
  return label ? `Task: ${shorten(label)}` : null
}

function countSeeds(value: unknown): number | null {
  return Array.isArray(value) ? value.length : null
}

function buildPromptSeedsLine(
  meta: Record<string, any>,
  rawConfig: Record<string, any> | null,
  optimizerConfig: Record<string, any> | null,
): string | null {
  const gepaEval = isRecord(rawConfig?.gepa?.evaluation) ? rawConfig?.gepa?.evaluation : null
  const miproEval = isRecord(rawConfig?.mipro?.evaluation) ? rawConfig?.mipro?.evaluation : null
  const evalConfig = (
    gepaEval ||
    miproEval ||
    (isRecord(rawConfig?.evaluation) ? rawConfig.evaluation : null) ||
    (isRecord(optimizerConfig?.evaluation) ? optimizerConfig.evaluation : null)
  ) as Record<string, any> | null

  const trainSeeds = countSeeds(
    evalConfig?.seeds ??
      evalConfig?.train_seeds ??
      rawConfig?.seeds ??
      rawConfig?.train_seeds ??
      meta.seeds ??
      meta.train_seeds,
  )
  const valSeeds = countSeeds(
    evalConfig?.val_seeds ??
      evalConfig?.validation_seeds ??
      rawConfig?.val_seeds ??
      rawConfig?.validation_seeds ??
      meta.val_seeds ??
      meta.validation_seeds,
  )
  if (trainSeeds == null && valSeeds == null) return null
  const parts: string[] = []
  if (trainSeeds != null) parts.push(`${trainSeeds} train`)
  if (valSeeds != null) parts.push(`${valSeeds} val`)
  return parts.length > 0 ? `Seeds: ${parts.join(", ")}` : null
}

function buildOptimizerLine(
  job: JobSummary,
  meta: Record<string, any>,
  rawConfig: Record<string, any> | null,
  optimizerConfig: Record<string, any> | null,
): string | null {
  const algoRaw = pickString(job.algorithm, rawConfig?.algorithm, meta.algorithm, job.training_type)
  if (!algoRaw) return null
  const algo = algoRaw.toLowerCase()

  if (algo.includes("gepa") || job.training_type === "graph_gepa" || job.training_type === "gepa") {
    const gepa =
      (isRecord(rawConfig?.gepa) ? rawConfig?.gepa : null) ||
      (isRecord(optimizerConfig?.gepa) ? optimizerConfig?.gepa : null) ||
      (isRecord(meta.gepa) ? meta.gepa : null)
    const generations = num(gepa?.num_generations ?? gepa?.generations)
    const children = num(gepa?.children_per_generation ?? gepa?.children_per_gen)
    const population = num(gepa?.population_size ?? gepa?.population)
    const parts: string[] = []
    if (generations != null) parts.push(`gen ${generations}`)
    if (children != null) parts.push(`kids ${children}`)
    if (population != null) parts.push(`pop ${population}`)
    return parts.length > 0 ? `Optimizer: GEPA (${parts.join(", ")})` : "Optimizer: GEPA"
  }

  if (algo.includes("mipro")) {
    const mipro =
      (isRecord(rawConfig?.mipro) ? rawConfig?.mipro : null) ||
      (isRecord(optimizerConfig?.mipro) ? optimizerConfig?.mipro : null) ||
      (isRecord(meta.mipro) ? meta.mipro : null)
    const iterations = num(mipro?.num_iterations ?? mipro?.iterations)
    const evals = num(mipro?.num_evaluations_per_iteration ?? mipro?.num_evals_per_iteration)
    const batchSize = num(mipro?.batch_size)
    const parts: string[] = []
    if (iterations != null) parts.push(`iters ${iterations}`)
    if (evals != null) parts.push(`evals ${evals}`)
    if (batchSize != null) parts.push(`batch ${batchSize}`)
    return parts.length > 0 ? `Optimizer: MIPRO (${parts.join(", ")})` : "Optimizer: MIPRO"
  }

  return `Algorithm: ${algoRaw}`
}

function buildEvalSeedsLine(meta: Record<string, any>, rows: Array<Record<string, any>>): string | null {
  const totalSeeds = Array.isArray(meta.seeds) ? meta.seeds.length : rows.length
  if (!totalSeeds) return null
  const scored = rows.filter((row) => num(row.reward ?? row.local_api_reward ?? row.verifier_reward ?? row.event_reward) != null)
  const errored = rows.filter((row) => row.error != null)
  const parts: string[] = []
  if (scored.length > 0) parts.push(`${scored.length} scored`)
  if (errored.length > 0) parts.push(`${errored.length} err`)
  const suffix = parts.length > 0 ? ` (${parts.join(", ")})` : ""
  return `Seeds: ${totalSeeds}${suffix}`
}

function buildEvalUsageLine(rows: Array<Record<string, any>>): string | null {
  if (!rows.length) return null
  const totalTokens = rows.reduce((acc, row) => acc + (num(row.tokens) ?? 0), 0)
  const totalCost = rows.reduce((acc, row) => acc + (num(row.cost_usd) ?? 0), 0)
  const latencies = rows.map((row) => num(row.latency_ms)).filter((value): value is number => value != null)
  const meanLatency = latencies.length > 0 ? latencies.reduce((acc, val) => acc + val, 0) / latencies.length : null
  const parts: string[] = []
  if (totalTokens > 0) parts.push(`${formatCompactCount(totalTokens)} tok`)
  if (totalCost > 0) parts.push(formatCost(totalCost))
  if (meanLatency != null) parts.push(`avg ${formatDurationMs(meanLatency)}`)
  return parts.length > 0 ? `Usage: ${parts.join(" | ")}` : null
}

function buildVerifierLine(meta: Record<string, any>): string | null {
  const verifier = isRecord(meta.verifier_config) ? meta.verifier_config : null
  if (!verifier) return null
  const rewardSource = pickString(verifier.reward_source)
  const graph = pickString(verifier.verifier_graph_id)
  const model = pickString(verifier.backend_model)
  const parts: string[] = []
  if (rewardSource && rewardSource !== "task_app") parts.push(rewardSource)
  if (graph) parts.push(`graph ${graph}`)
  if (model) parts.push(model)
  return parts.length > 0 ? `Verifier: ${shorten(parts.join(", "), 48)}` : null
}

function formatLearningRate(value: number): string {
  if (value === 0) return "0"
  if (Math.abs(value) < 0.001) return value.toExponential(1)
  if (Math.abs(value) < 1) return value.toFixed(4)
  return value.toFixed(2)
}

function buildTrainingLine(job: JobSummary, meta: Record<string, any>): string | null {
  const trainKind = pickString(job.hyperparameters?.train_kind, meta.train_kind)
  const trainingFile = pickString(meta.training_file, job.training_file_id)
  if (!trainKind && !trainingFile) return null
  const parts: string[] = []
  if (trainKind) parts.push(trainKind)
  if (trainingFile) {
    const label = trainingFile.split("/").pop() || trainingFile
    parts.push(shorten(label, 28))
  }
  return `Training: ${parts.join(" ")}`
}

function buildHyperparamsLine(job: JobSummary, meta: Record<string, any>): string | null {
  const hyper = isRecord(job.hyperparameters) ? job.hyperparameters : isRecord(meta.hyperparameters) ? meta.hyperparameters : null
  if (!hyper) return null
  const epochs = num(hyper.n_epochs ?? hyper.epochs)
  const batch = num(hyper.batch_size ?? hyper.micro_batch_size)
  const lr = num(hyper.learning_rate ?? hyper.lr)
  const parts: string[] = []
  if (epochs != null) parts.push(`epochs ${epochs}`)
  if (batch != null) parts.push(`batch ${batch}`)
  if (lr != null) parts.push(`lr ${formatLearningRate(lr)}`)
  return parts.length > 0 ? `Hyper: ${parts.join(", ")}` : null
}

function buildComputeLine(job: JobSummary): string | null {
  const costInfo = isRecord(job.cost_information) ? job.cost_information : null
  if (!costInfo) return null
  const gpuType = pickString(costInfo.gpu_type, costInfo.gpu_variant)
  const gpuCount = num(costInfo.container_count ?? costInfo.total_gpus ?? costInfo.gpu_count)
  const nodes = num(costInfo.nodes)
  if (!gpuType && gpuCount == null) return null
  let label = gpuType || "GPU"
  if (gpuCount != null) label = `${label} x${gpuCount}`
  if (nodes != null && nodes > 1) label = `${label} (${nodes} nodes)`
  return `Compute: ${label}`
}

function buildOutputLine(job: JobSummary, meta: Record<string, any>): string | null {
  const output = pickString(job.fine_tuned_model, meta.fine_tuned_model)
  return output ? `Output: ${shorten(output, 42)}` : null
}

export function formatDetails(data: AppData): string {
  const job = data.selectedJob
  if (!job) return "No job selected."

  if (isEvalJob(job)) {
    return formatEvalDetails(data, job)
  }
  if (job.job_source === "learning") {
    return formatLearningDetails(job)
  }
  return formatPromptLearningDetails(data, job)
}

export function formatEvalDetails(data: AppData, job: JobSummary): string {
  const meta = isRecord(job.metadata) ? job.metadata : {}
  const rawConfig = extractPromptLearningConfig(meta)
  const policy = extractPolicyConfig(meta, rawConfig, null)
  const rows = Array.isArray(data.evalResultRows) ? data.evalResultRows : []
  const lines: string[] = [`Job: ${job.job_id}`, `Type: ${normalizeJobType(job)}`]

  const taskLine = buildTaskLine(meta, rawConfig)
  if (taskLine) lines.push(taskLine)

  const policyLine = formatPolicyLine(policy)
  if (policyLine) lines.push(policyLine)

  const seedLine = buildEvalSeedsLine(meta, rows)
  if (seedLine) lines.push(seedLine)

  const usageLine = buildEvalUsageLine(rows)
  if (usageLine) lines.push(usageLine)

  const verifierLine = buildVerifierLine(meta)
  if (verifierLine) lines.push(verifierLine)

  return lines.join("\n")
}

export function formatLearningDetails(job: JobSummary): string {
  const meta = isRecord(job.metadata) ? job.metadata : {}
  const lines: string[] = [`Job: ${job.job_id}`, `Type: ${normalizeJobType(job)}`]

  const model = pickString(job.model_id, meta.model_id, meta.model)
  if (model) lines.push(`Model: ${shorten(model, 48)}`)

  const trainingLine = buildTrainingLine(job, meta)
  if (trainingLine) lines.push(trainingLine)

  const hyperLine = buildHyperparamsLine(job, meta)
  if (hyperLine) lines.push(hyperLine)

  const computeLine = buildComputeLine(job)
  if (computeLine) lines.push(computeLine)

  const outputLine = buildOutputLine(job, meta)
  if (outputLine) lines.push(outputLine)

  if (job.linked_job_id) {
    lines.push(`Linked: ${shorten(job.linked_job_id, 42)}`)
  }

  return lines.join("\n")
}

export function formatPromptLearningDetails(_data: AppData, job: JobSummary): string {
  const meta = isRecord(job.metadata) ? job.metadata : {}
  const rawConfig = extractPromptLearningConfig(meta)
  const optimizerConfig = extractOptimizerConfig(meta)
  const policy = extractPolicyConfig(meta, rawConfig, optimizerConfig)
  const lines: string[] = [`Job: ${job.job_id}`, `Type: ${normalizeJobType(job)}`]

  const taskLine = buildTaskLine(meta, rawConfig)
  if (taskLine) lines.push(taskLine)

  const policyLine = formatPolicyLine(policy)
  if (policyLine) lines.push(policyLine)

  const optimizerLine = buildOptimizerLine(job, meta, rawConfig, optimizerConfig)
  if (optimizerLine) lines.push(optimizerLine)

  const seedLine = buildPromptSeedsLine(meta, rawConfig, optimizerConfig)
  if (seedLine) lines.push(seedLine)

  return lines.join("\n")
}

export function calculateTotalTokensFromEvents(events: JobEvent[]): number {
  let total = 0
  for (const event of events) {
    const data: any = event.data as any
    if (!data) continue
    // Sum up token fields from various event types
    if (typeof data.prompt_tokens === "number") total += data.prompt_tokens
    if (typeof data.completion_tokens === "number") total += data.completion_tokens
    if (typeof data.reasoning_tokens === "number") total += data.reasoning_tokens
    // Also check for total_tokens directly
    if (typeof data.total_tokens === "number") total += data.total_tokens
  }
  return total
}
