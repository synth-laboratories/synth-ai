import { For, Show, createMemo, type Component } from "solid-js"

import type { ResultsPanelProps } from "./types"
import type { TrainingGraphProps } from "./graphs"
import {
  CandidateCountGraph,
  CostGraph,
  ErrorRateGraph,
  EvalSuccessRateGraph,
  FrontierDensityGraph,
  FrontierSeedsSolvedGraph,
  GradientNormGraph,
  GpuUtilizationGraph,
  LatencyGraph,
  LearningRateGraph,
  LossTrendGraph,
  MemoryUsageGraph,
  PerplexityGraph,
  RLEntropyKlGraph,
  RLEpisodeLengthGraph,
  RLEpisodeRewardGraph,
  RLClipFractionGraph,
  RLPolicyValueLossGraph,
  RewardDistributionGraph,
  RewardTrendGraph,
  ThroughputGraph,
  TokenUsageGraph,
  VerifierAgreementGraph,
  VerifierRewardGraph,
} from "./graphs"
import { TEXT, PANEL, getPanelBorderColor } from "../../../theme"

type GraphComponent = Component<TrainingGraphProps>

const SYSTEM_GRAPHS: GraphComponent[] = [
  ThroughputGraph,
  TokenUsageGraph,
  CostGraph,
  GpuUtilizationGraph,
  MemoryUsageGraph,
]

function normalize(value: unknown): string {
  return typeof value === "string" ? value.toLowerCase() : ""
}

function uniqueGraphs(list: GraphComponent[]): GraphComponent[] {
  const seen = new Set<GraphComponent>()
  const out: GraphComponent[] = []
  for (const item of list) {
    if (seen.has(item)) continue
    seen.add(item)
    out.push(item)
  }
  return out
}

function resolveGraphs(props: ResultsPanelProps): GraphComponent[] {
  const job = props.data.selectedJob
  if (!job) return [RewardTrendGraph]

  const trainingType = normalize(job.training_type)
  const algorithm = normalize(job.algorithm ?? job.metadata?.algorithm)
  const graphType = normalize(job.metadata?.graph_type)
  const evalMode = normalize(job.metadata?.eval_mode)
  const jobSource = normalize(job.job_source)

  const isEval = trainingType === "eval" || jobSource === "eval"
  const isGraphEvolve = trainingType === "graph_evolve"
  const isVerifier = trainingType.includes("verifier") || graphType === "verifier" || evalMode === "verifier"
  const isGepa = trainingType.includes("gepa") || algorithm.includes("gepa")
  const isMipro = trainingType.includes("mipro") || algorithm.includes("mipro")
  const isPromptOpt =
    trainingType.includes("prompt") ||
    trainingType === "prompt_learning" ||
    trainingType === "prompt_opt" ||
    jobSource === "prompt-learning"
  const isSft =
    trainingType.startsWith("sft") ||
    trainingType === "sft" ||
    trainingType === "fine_tune" ||
    trainingType === "fine_tuning" ||
    trainingType === "dpo"
  const isRl =
    trainingType.startsWith("rl") ||
    trainingType === "rl" ||
    trainingType === "rl_test"

  if (isEval) {
    const evalGraphs = isVerifier
      ? [VerifierRewardGraph, VerifierAgreementGraph, EvalSuccessRateGraph, LatencyGraph, ErrorRateGraph]
      : [EvalSuccessRateGraph, LatencyGraph, ErrorRateGraph, RewardTrendGraph]
    return uniqueGraphs([...evalGraphs, ...SYSTEM_GRAPHS])
  }

  if (isGraphEvolve) {
    const evolveGraphs = graphType === "policy" || graphType === "rlm"
      ? [RewardTrendGraph, CandidateCountGraph, FrontierDensityGraph]
      : [RewardTrendGraph, CandidateCountGraph, FrontierDensityGraph, RewardDistributionGraph]
    return uniqueGraphs([...evolveGraphs, ...SYSTEM_GRAPHS])
  }

  if (isGepa) {
    const gepaGraphs = [
      RewardTrendGraph,
      FrontierDensityGraph,
      FrontierSeedsSolvedGraph,
      CandidateCountGraph,
      RewardDistributionGraph,
    ]
    return uniqueGraphs([...gepaGraphs, ...SYSTEM_GRAPHS])
  }

  if (isMipro) {
    const miproGraphs = [
      RewardTrendGraph,
      CandidateCountGraph,
      RewardDistributionGraph,
      ErrorRateGraph,
    ]
    return uniqueGraphs([...miproGraphs, ...SYSTEM_GRAPHS])
  }

  if (isPromptOpt) {
    const promptGraphs = [
      RewardTrendGraph,
      RewardDistributionGraph,
      CandidateCountGraph,
      ErrorRateGraph,
    ]
    return uniqueGraphs([...promptGraphs, ...SYSTEM_GRAPHS])
  }

  if (isSft) {
    const sftGraphs = [
      LossTrendGraph,
      PerplexityGraph,
      LearningRateGraph,
      GradientNormGraph,
      ThroughputGraph,
      TokenUsageGraph,
      CostGraph,
      GpuUtilizationGraph,
      MemoryUsageGraph,
    ]
    return uniqueGraphs(sftGraphs)
  }

  if (isRl) {
    const rlGraphs = [
      RLEpisodeRewardGraph,
      RLEpisodeLengthGraph,
      RLPolicyValueLossGraph,
      RLEntropyKlGraph,
      RLClipFractionGraph,
      ThroughputGraph,
      CostGraph,
    ]
    return uniqueGraphs([...rlGraphs, ...SYSTEM_GRAPHS])
  }

  if (trainingType === "context_learning") {
    const contextGraphs = [RewardTrendGraph, LossTrendGraph, CandidateCountGraph]
    return uniqueGraphs([...contextGraphs, ...SYSTEM_GRAPHS])
  }

  if (trainingType === "learning" || jobSource === "learning") {
    const learningGraphs = [LossTrendGraph, RewardTrendGraph, ThroughputGraph, CostGraph]
    return uniqueGraphs([...learningGraphs, ...SYSTEM_GRAPHS])
  }

  if (isVerifier) {
    const verifierGraphs = [VerifierRewardGraph, VerifierAgreementGraph, RewardTrendGraph]
    return uniqueGraphs([...verifierGraphs, ...SYSTEM_GRAPHS])
  }

  return uniqueGraphs([RewardTrendGraph, ThroughputGraph, CostGraph, TokenUsageGraph])
}

function pickGraphCount(height: number, total: number): number {
  if (total <= 1) return total
  if (height >= 20) return Math.min(3, total)
  if (height >= 12) return Math.min(2, total)
  return 1
}

export function TrainingGraphsPanel(props: ResultsPanelProps) {
  const job = createMemo(() => props.data.selectedJob)
  const graphs = createMemo(() => resolveGraphs(props))
  const title = createMemo(() => {
    const current = job()
    const trainingType = normalize(current?.training_type)
    const jobSource = normalize(current?.job_source)
    return trainingType === "eval" || jobSource === "eval" ? "Results" : "Graphs"
  })

  const innerWidth = createMemo(() => Math.max(18, props.width - 6))
  const innerHeight = createMemo(() => Math.max(3, props.height - 2))
  const graphCount = createMemo(() => pickGraphCount(innerHeight(), graphs().length))
  const graphGap = createMemo(() => (graphCount() > 1 ? 1 : 0))
  const graphHeight = createMemo(() => {
    const count = graphCount()
    const available = innerHeight() - graphGap() * Math.max(0, count - 1)
    return Math.max(3, Math.floor(available / Math.max(1, count)))
  })

  return (
    <box
      border={PANEL.border}
      borderStyle={PANEL.borderStyle}
      borderColor={getPanelBorderColor(props.focused)}
      title={title()}
      titleAlignment={PANEL.titleAlignment}
      paddingLeft={PANEL.paddingLeft}
      height={props.height}
    >
      <Show
        when={job()}
        fallback={<text fg={TEXT.fg}>No job selected.</text>}
      >
        <box flexDirection="column" gap={graphGap()}>
          <For each={graphs().slice(0, graphCount())}>
            {(GraphComponent) => (
              <GraphComponent
                data={props.data}
                width={innerWidth()}
                height={graphHeight()}
                focused={props.focused}
              />
            )}
          </For>
        </box>
      </Show>
    </box>
  )
}
