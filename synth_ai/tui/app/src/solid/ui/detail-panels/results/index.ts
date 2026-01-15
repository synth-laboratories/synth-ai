/**
 * Results panel exports and registration.
 */
import { registerResultsPanel } from "../registries/results-registry"
import { GraphEvolveResultsPanel } from "./GraphEvolveResultsPanel"
import { DefaultResultsPanel } from "./DefaultResultsPanel"
import { TrainingGraphsPanel } from "./TrainingGraphsPanel"

// Register panels with priority order
registerResultsPanel("graph_evolve:verifier", GraphEvolveResultsPanel)
registerResultsPanel("graph_evolve:*", TrainingGraphsPanel)
registerResultsPanel("gepa:*", TrainingGraphsPanel)
registerResultsPanel("graph_gepa:*", TrainingGraphsPanel)
registerResultsPanel("mipro:*", TrainingGraphsPanel)
registerResultsPanel("prompt_learning:*", TrainingGraphsPanel)
registerResultsPanel("prompt-learning:*", TrainingGraphsPanel)
registerResultsPanel("prompt_opt:*", TrainingGraphsPanel)
registerResultsPanel("sft_offline:*", TrainingGraphsPanel)
registerResultsPanel("sft_online:*", TrainingGraphsPanel)
registerResultsPanel("sft:*", TrainingGraphsPanel)
registerResultsPanel("fine_tune:*", TrainingGraphsPanel)
registerResultsPanel("fine_tuning:*", TrainingGraphsPanel)
registerResultsPanel("dpo:*", TrainingGraphsPanel)
registerResultsPanel("rl_online:*", TrainingGraphsPanel)
registerResultsPanel("rl_offline:*", TrainingGraphsPanel)
registerResultsPanel("rl_test:*", TrainingGraphsPanel)
registerResultsPanel("rl:*", TrainingGraphsPanel)
registerResultsPanel("context_learning:*", TrainingGraphsPanel)
registerResultsPanel("learning:*", TrainingGraphsPanel)
registerResultsPanel("eval:*", TrainingGraphsPanel)
registerResultsPanel("*:*", DefaultResultsPanel)

export { GraphEvolveResultsPanel } from "./GraphEvolveResultsPanel"
export { GraphEvolveGenerationGraph } from "./GraphEvolveGenerationGraph"
export { EvalResultsPanel } from "./EvalResultsPanel"
export { DefaultResultsPanel } from "./DefaultResultsPanel"
export { TrainingGraphsPanel } from "./TrainingGraphsPanel"
export * from "./graphs"
export type { ResultsPanelProps } from "./types"
