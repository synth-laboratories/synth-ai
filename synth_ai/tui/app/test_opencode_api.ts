/**
 * Test script to debug OpenCode API - run with: bun test_opencode_api.ts
 */
import { createOpencodeClient } from "@opencode-ai/sdk/v2"

const OPENCODE_URL = "http://127.0.0.1:56542"

async function main() {
  console.log("Connecting to OpenCode at:", OPENCODE_URL)

  const client = createOpencodeClient({ baseUrl: OPENCODE_URL })

  // List all providers and their exact model IDs
  console.log("\n=== All Providers and Models ===")
  const providers = await client.provider.list({})

  for (const provider of providers.data?.all || []) {
    console.log(`\nProvider: ${provider.id} (${provider.name})`)
    console.log("Models:")
    for (const [modelId, model] of Object.entries(provider.models)) {
      console.log(`  - ${modelId}: ${model.name || '(no name)'}`)
    }
  }

  console.log("\n=== Connected providers ===")
  console.log(providers.data?.connected)

  // Try OpenAI provider instead
  console.log("\n=== Creating Session ===")
  const session = await client.session.create()
  console.log("Session ID:", session.data?.id)

  if (!session.data?.id) {
    console.error("Failed to create session")
    return
  }

  // Don't specify model - let OpenCode use default
  console.log("\n=== Sending Prompt (default model) ===")
  const promptResult = await client.session.prompt({
    sessionID: session.data.id,
    parts: [{ type: "text", text: "Say hello in exactly 5 words" }],
  })
  console.log("Prompt result error:", (promptResult.data as any)?.info?.error)
  console.log("Prompt model used:", (promptResult.data as any)?.info?.modelID)
  console.log("Prompt provider used:", (promptResult.data as any)?.info?.providerID)

  await new Promise(r => setTimeout(r, 3000))

  const messages = await client.session.messages({ sessionID: session.data.id })
  console.log("\nMessages:", messages.data?.length)

  const lastMsg = messages.data?.at(-1)
  if (lastMsg) {
    console.log("Last message role:", lastMsg.info.role)
    console.log("Last message parts:", lastMsg.parts.length)
    console.log("Last message error:", (lastMsg.info as any).error?.name)
    if (lastMsg.parts.length > 0) {
      console.log("First part:", JSON.stringify(lastMsg.parts[0]).slice(0, 200))
    }
  }
}

main().catch(console.error)
