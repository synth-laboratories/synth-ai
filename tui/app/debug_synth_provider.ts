/**
 * Debug script to inspect synth provider configuration
 */
import { createOpencodeClient } from "@opencode-ai/sdk/v2"

const OPENCODE_URL = "http://127.0.0.1:59083"

async function main() {
  console.log("Connecting to OpenCode at:", OPENCODE_URL)

  const client = createOpencodeClient({ baseUrl: OPENCODE_URL })

  const providers = await client.provider.list({})

  // Find synth provider specifically
  const synthProvider = providers.data?.all?.find((p: any) => p.id === "synth")

  if (!synthProvider) {
    console.log("\n=== SYNTH PROVIDER NOT FOUND ===")
    console.log("Available providers:", providers.data?.all?.map((p: any) => p.id))
    return
  }

  console.log("\n=== SYNTH PROVIDER DETAILS ===")
  console.log("ID:", synthProvider.id)
  console.log("Name:", synthProvider.name)
  console.log("Models:", Object.keys(synthProvider.models))

  console.log("\n=== SYNTH MODEL DETAILS ===")
  for (const [modelId, model] of Object.entries(synthProvider.models as Record<string, any>)) {
    console.log(`\nModel: ${modelId}`)
    console.log("  Name:", model.name)
    console.log("  API ID:", model.api?.id)
    console.log("  API URL:", model.api?.url)
    console.log("  Provider ID:", model.providerID)
  }

  // Test making a request with synth-small
  console.log("\n=== TESTING SYNTH-SMALL REQUEST ===")
  const session = await client.session.create()
  console.log("Session ID:", session.data?.id)

  if (!session.data?.id) {
    console.error("Failed to create session")
    return
  }

  // Explicitly use synth/synth-small
  console.log("\nSending prompt with model: synth/synth-small")
  const promptResult = await client.session.prompt({
    sessionID: session.data.id,
    parts: [{ type: "text", text: "Say hello" }],
    model: { providerID: "synth", modelID: "synth-small" }
  })

  console.log("Prompt error:", (promptResult.data as any)?.info?.error)

  await new Promise(r => setTimeout(r, 3000))

  const messages = await client.session.messages({ sessionID: session.data.id })
  console.log("\nMessages:", messages.data?.length)

  const lastMsg = messages.data?.at(-1)
  if (lastMsg) {
    console.log("Last message role:", lastMsg.info.role)
    console.log("Last message parts:", lastMsg.parts.length)
    if ((lastMsg.info as any).error) {
      console.log("Last message error:", JSON.stringify((lastMsg.info as any).error, null, 2))
    }
    if (lastMsg.parts.length > 0) {
      console.log("First part:", JSON.stringify(lastMsg.parts[0]).slice(0, 300))
    }
  }
}

main().catch(console.error)
