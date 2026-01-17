import type { AvailableModel, Provider } from "./model-types"

export function buildAvailableModels(providers: Provider[]): AvailableModel[] {
  const models: AvailableModel[] = []
  if (!providers.length) return models

  const ordered = [...providers].sort((a, b) => {
    if (a.id === "synth" && b.id !== "synth") return -1
    if (b.id === "synth" && a.id !== "synth") return 1
    return a.id.localeCompare(b.id)
  })

  for (const provider of ordered) {
    for (const [modelId, model] of Object.entries(provider.models)) {
      models.push({
        providerID: provider.id,
        modelID: modelId,
        providerName: provider.name,
        modelName: model.name || modelId,
      })
    }
  }

  return models
}
