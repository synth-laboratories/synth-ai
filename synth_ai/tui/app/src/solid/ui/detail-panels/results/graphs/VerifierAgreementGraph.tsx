import { createLineGraph } from "./graph-utils"

export const VerifierAgreementGraph = createLineGraph({
  title: "verifier agreement",
  metricNames: [
    "verifier.agreement_rate",
    "verifier.agreement",
    "verifier_alignment",
    "verifier.match_rate",
    "verifier_match_rate",
  ],
  xLabel: "step",
  decimals: 3,
})
