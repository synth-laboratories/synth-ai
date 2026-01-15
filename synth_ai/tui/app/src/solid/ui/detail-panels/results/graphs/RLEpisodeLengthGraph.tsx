import { createLineGraph } from "./graph-utils"

export const RLEpisodeLengthGraph = createLineGraph({
  title: "episode length",
  metricNames: [
    "episode_length",
    "episode_steps",
    "rollout_length",
    "trajectory_length",
  ],
  xLabel: "episode",
  integerValues: true,
})
