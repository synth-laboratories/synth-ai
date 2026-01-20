import { createOpencodeClient } from "@opencode-ai/sdk/v2"

const client = createOpencodeClient({ baseUrl: "http://127.0.0.1:61160" })

async function test() {
  console.log("Creating session...")
  const session = await client.session.create()
  console.log("Session:", session.data?.id)

  console.log("\nSending prompt: 'Say hello in exactly 3 words'")
  console.log("Model: synth/synth-large-instant")

  await client.session.prompt({
    sessionID: session.data!.id,
    parts: [{ type: "text", text: "Say hello in exactly 3 words" }],
    model: { providerID: "synth", modelID: "synth-large-instant" }
  })

  // Poll for response
  console.log("\nWaiting for response...")
  const startTime = Date.now()

  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 1000))

    const messages = await client.session.messages({ sessionID: session.data!.id })
    const assistantMsgs = (messages.data || []).filter((m: any) => m.info?.role === "assistant")

    if (assistantMsgs.length > 0) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2)
      console.log(`\n=== Response (${elapsed}s) ===`)

      for (const msg of assistantMsgs) {
        const info = msg.info as any
        console.log("Provider:", info.providerID)
        console.log("Model:", info.modelID)
        console.log("Tokens:", JSON.stringify(info.tokens))
        console.log("Finish:", info.finish)

        for (const p of msg.parts) {
          if ((p as any).text) {
            console.log("\nText:", (p as any).text)
          }
        }

        if (info.error) {
          console.log("\nERROR:", JSON.stringify(info.error, null, 2))
        }
      }

      break
    }

    if (i % 5 === 0) {
      console.log(`  ...waiting (${i}s)`)
    }
  }
}

test().catch(e => {
  console.error("Error:", e.message)
  process.exit(1)
})
