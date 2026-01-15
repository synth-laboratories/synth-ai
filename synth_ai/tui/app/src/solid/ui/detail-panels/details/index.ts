/**
 * Details panel exports and registration.
 */
import { registerDetailsPanel } from "../registries/details-registry"
import { EvalDetails } from "./EvalDetails"
import { LearningDetails } from "./LearningDetails"
import { PromptLearningDetails } from "./PromptLearningDetails"
import { DefaultDetails } from "./DefaultDetails"

// Register panels with priority order
registerDetailsPanel("eval:*", EvalDetails)
registerDetailsPanel("learning:*", LearningDetails)
// Prompt learning types
registerDetailsPanel("gepa:*", PromptLearningDetails)
registerDetailsPanel("graph_gepa:*", PromptLearningDetails)
registerDetailsPanel("mipro:*", PromptLearningDetails)
registerDetailsPanel("graph_evolve:*", PromptLearningDetails)
// Fallback
registerDetailsPanel("*:*", DefaultDetails)

export { EvalDetails } from "./EvalDetails"
export { LearningDetails } from "./LearningDetails"
export { PromptLearningDetails } from "./PromptLearningDetails"
export { DefaultDetails } from "./DefaultDetails"
export type { DetailsPanelProps } from "./types"
