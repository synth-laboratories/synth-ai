import { createLineGraph } from "./graph-utils"

export const RLEpisodeRewardGraph = createLineGraph({
  title: "episode reward",
  metricNames: [
    "rl.reward",
    "epoch.reward_mean",
    "eval.reward_mean",
    "rl.episode_reward",
    "episode_reward",
    "rollout.reward",
    "episode_return",
    "return",
  ],
  xLabel: "episode",
  decimals: 3,
})
