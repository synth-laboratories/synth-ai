import { createOpencodeClient } from "@opencode-ai/sdk/v2"

const client = createOpencodeClient({ baseUrl: "http://127.0.0.1:61160" })

async function test() {
  const session = await client.session.create()
  console.log("Session:", session.data?.id)

  await client.session.prompt({
    sessionID: session.data!.id,
    parts: [{ type: "text", text: "Say hi" }],
    model: { providerID: "synth", modelID: "synth-large-instant" }
  })

  // Wait a bit and check messages
  await new Promise(r => setTimeout(r, 5000))

  const messages = await client.session.messages({ sessionID: session.data!.id })
  console.log("Messages:", messages.data?.length)

  const last = messages.data?.at(-1)
  if (last) {
    const info = last.info as any
    console.log("Role:", info.role)
    console.log("Provider:", info.providerID)
    console.log("Model:", info.modelID)
    console.log("Tokens:", JSON.stringify(info.tokens))
    console.log("Finish:", info.finish)
    console.log("Parts:", last.parts.length)
    for (const p of last.parts) {
      console.log("  Part:", (p as any).type)
      if ((p as any).text) console.log("  Text:", (p as any).text.slice(0, 100))
    }
  }
}

test().catch(e => {
  console.error("Error:", e.message)
  process.exit(1)
})
