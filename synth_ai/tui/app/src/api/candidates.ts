/**
 * Candidates API client for fetching candidates from the unified candidates table.
 *
 * This replaces the event-parsing approach for getting candidate data,
 * providing direct access to persisted candidates via the candidates API.
 */

import { apiGet } from "./client"

// Type definitions for candidates API responses
export interface Candidate {
  candidate_id: string
  job_id: string
  org_id: string
  task_id: string | null
  candidate_type: string
  content: Record<string, any>
  generation: number | null
  parent_id: string | null
  mutation_type: string | null
  mutation_params: Record<string, any> | null
  status: string | null
  is_pareto: boolean
  objective: number | null
  reward: number | null
  seed_rewards: Array<{ seed: number; reward: number }> | null
  token_usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number } | null
  cost_usd: number | null
  graph_complexity: number | null
  graph_execution_time_ms: number | null
  graph_text_export: string | null
  created_at: string | null
  updated_at: string | null
}

export interface CandidateListResponse {
  candidates: Candidate[]
  total: number
  limit: number
  offset: number
}

export interface Task {
  task_id: string
  org_id: string
  task_kind: string
  app_id: string | null
  env_name: string | null
  dataset_id: string | null
  dataset_split: string | null
  dataset_revision: string | null
  display_name: string | null
  metadata: Record<string, any> | null
  created_at: string | null
  updated_at: string | null
}

export interface TaskListResponse {
  tasks: Task[]
  total: number
}

/**
 * Fetch candidates for a job.
 */
export async function fetchCandidatesForJob(
  jobId: string,
  options: {
    status?: string
    isPareto?: boolean
    limit?: number
    offset?: number
  } = {},
): Promise<CandidateListResponse> {
  const params = new URLSearchParams()
  params.set("job_id", jobId)
  if (options.status) params.set("status", options.status)
  if (options.isPareto !== undefined) params.set("is_pareto", String(options.isPareto))
  if (options.limit) params.set("limit", String(options.limit))
  if (options.offset) params.set("offset", String(options.offset))

  const response = await apiGet(`/candidates?${params.toString()}`)
  return response as CandidateListResponse
}

/**
 * Fetch Pareto frontier candidates for a job.
 */
export async function fetchParetoCandidates(jobId: string): Promise<CandidateListResponse> {
  const response = await apiGet(`/candidates/by-job/${jobId}/pareto`)
  return response as CandidateListResponse
}

/**
 * Fetch a single candidate by ID.
 */
export async function fetchCandidate(candidateId: string): Promise<Candidate> {
  const response = await apiGet(`/candidates/${candidateId}`)
  return response as Candidate
}

/**
 * Fetch candidate lineage (ancestors).
 */
export async function fetchCandidateLineage(jobId: string, candidateId: string): Promise<Candidate[]> {
  const response = await apiGet(`/candidates/by-job/${jobId}/lineage/${candidateId}`)
  return response as Candidate[]
}

/**
 * Fetch candidates for a task (across all jobs).
 */
export async function fetchCandidatesForTask(
  taskId: string,
  options: {
    status?: string
    isPareto?: boolean
    limit?: number
    offset?: number
  } = {},
): Promise<CandidateListResponse> {
  const params = new URLSearchParams()
  if (options.status) params.set("status", options.status)
  if (options.isPareto !== undefined) params.set("is_pareto", String(options.isPareto))
  if (options.limit) params.set("limit", String(options.limit))
  if (options.offset) params.set("offset", String(options.offset))

  const response = await apiGet(`/tasks/${taskId}/candidates?${params.toString()}`)
  return response as CandidateListResponse
}

/**
 * Fetch all tasks.
 */
export async function fetchTasks(options: {
  taskKind?: string
  appId?: string
  envName?: string
  limit?: number
} = {}): Promise<TaskListResponse> {
  const params = new URLSearchParams()
  if (options.taskKind) params.set("task_kind", options.taskKind)
  if (options.appId) params.set("app_id", options.appId)
  if (options.envName) params.set("env_name", options.envName)
  if (options.limit) params.set("limit", String(options.limit))

  const queryString = params.toString()
  const path = queryString ? `/tasks?${queryString}` : "/tasks"
  const response = await apiGet(path)
  return response as TaskListResponse
}

/**
 * Fetch a single task by ID.
 */
export async function fetchTask(taskId: string): Promise<Task> {
  const response = await apiGet(`/tasks/${taskId}`)
  return response as Task
}

/**
 * Convert API candidate to the format expected by TUI formatters.
 *
 * This provides compatibility with existing TUI code that expects
 * candidates in a specific format.
 */
export function candidateToLegacyFormat(candidate: Candidate): Record<string, any> {
  return {
    candidate_id: candidate.candidate_id,
    id: candidate.candidate_id,
    name: candidate.content?.name || candidate.content?.candidate_name,
    reward: candidate.reward,
    is_pareto: candidate.is_pareto,
    isPareto: candidate.is_pareto,
    generation: candidate.generation,
    parent_id: candidate.parent_id,
    status: candidate.status,
    // Extract stages/content from the content field
    stages: candidate.content?.stages,
    prompt_text: candidate.content?.prompt_text,
    messages: candidate.content?.messages,
    // Score object for compatibility
    score: {
      reward: candidate.reward,
      instance_scores: candidate.seed_rewards?.map((sr) => sr.reward),
    },
    // Instance scores for mean calculation
    instance_scores: candidate.seed_rewards?.map((sr) => sr.reward),
    instance_rewards: candidate.seed_rewards?.map((sr) => sr.reward),
    // Graph-specific fields
    graph_text_export: candidate.graph_text_export,
    graph_complexity: candidate.graph_complexity,
    // Token usage
    token_usage: candidate.token_usage,
    cost_usd: candidate.cost_usd,
    // Timestamps
    created_at: candidate.created_at,
    updated_at: candidate.updated_at,
    // Full content for inspection
    _content: candidate.content,
  }
}

/**
 * Convert a list of API candidates to legacy format.
 */
export function candidatesToLegacyFormat(candidates: Candidate[]): Record<string, any>[] {
  return candidates.map(candidateToLegacyFormat)
}
