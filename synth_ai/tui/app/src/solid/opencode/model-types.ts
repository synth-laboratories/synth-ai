export type ProviderModel = {
  id: string
  name?: string
  limit: { context: number; output: number }
}

export type Provider = {
  id: string
  name: string
  models: Record<string, ProviderModel>
}

export type SelectedModel = {
  providerID: string
  modelID: string
}

export type ProviderListResponse = {
  all: Provider[]
  connected: string[]
}

export type AvailableModel = {
  providerID: string
  modelID: string
  providerName: string
  modelName: string
}
