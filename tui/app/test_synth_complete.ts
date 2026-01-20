/**
 * Complete test of synth provider integration
 */
import { createOpencodeClient } from "@opencode-ai/sdk/v2"

const OPENCODE_URL = "http://127.0.0.1:61160"

async function main() {
  console.log("=== Testing Synth Provider Integration ===\n")

  const client = createOpencodeClient({ baseUrl: OPENCODE_URL })

  // Create session
  const session = await client.session.create()
  if (!session.data?.id) {
    console.error("Failed to create session")
    return
  }
  console.log("Session created:", session.data.id)

  // Send prompt with synth-small
  console.log("\nSending prompt: 'Say hello in exactly 5 words'")
  console.log("Model: synth/synth-large-instant")

  const startTime = Date.now()
  await client.session.prompt({
    sessionID: session.data.id,
    parts: [{ type: "text", text: "Say hello in exactly 5 words" }],
    model: { providerID: "synth", modelID: "synth-large-instant" }
  })

  // Poll for response
  console.log("\nWaiting for response...")
  for (let i = 0; i < 15; i++) {
    await new Promise(r => setTimeout(r, 1000))

    const messages = await client.session.messages({ sessionID: session.data.id })
    const lastMsg = messages.data?.at(-1)

    if (lastMsg && lastMsg.info.role === "assistant") {
      const assistantInfo = lastMsg.info as any

      console.log(`\n=== Response (${(Date.now() - startTime) / 1000}s) ===`)
      console.log("Provider:", assistantInfo.providerID)
      console.log("Model:", assistantInfo.modelID)
      console.log("Status:", assistantInfo.status)

      if (assistantInfo.error) {
        console.log("\nERROR:", JSON.stringify(assistantInfo.error, null, 2))
        return
      }

      console.log("Parts count:", lastMsg.parts.length)

      if (lastMsg.parts.length > 0) {
        for (const part of lastMsg.parts) {
          const p = part as any
          if (p.type === "text" && p.text) {
            console.log("\n=== Assistant Response ===")
            console.log(p.text)
          }
        }
        console.log("\n=== SUCCESS! ===")
        return
      }

      if (assistantInfo.status === "completed" || assistantInfo.status === "error") {
        console.log("\nNo parts received but status is:", assistantInfo.status)
        return
      }
    }
    process.stdout.write(".")
  }

  console.log("\nTimeout waiting for response")
}

main().catch(console.error)
