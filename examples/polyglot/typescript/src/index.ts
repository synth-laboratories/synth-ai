/**
 * Synth Task App Example - TypeScript Implementation
 *
 * A minimal but complete Task App implementing the Synth contract for prompt optimization.
 *
 * ## Running
 * ```bash
 * npm install
 * npm run dev  # Development with hot reload
 * npm run build && npm start  # Production
 * ```
 *
 * ## Deploying to Cloudflare Workers
 * See README.md for Cloudflare Workers deployment instructions.
 */

import { Hono } from "hono";
import { serve } from "@hono/node-server";

// =============================================================================
// Types (matching OpenAPI contract)
// =============================================================================

interface Sample {
  text: string;
  label: string;
}

interface RolloutRequest {
  run_id: string;
  env: {
    seed?: number;
    config?: Record<string, unknown>;
  };
  policy: {
    policy_id?: string;
    policy_name?: string;
    config: {
      model?: string;
      inference_url?: string;
      api_base?: string;
      base_url?: string;
      prompt_template?: PromptTemplate;
      [key: string]: unknown;
    };
  };
  mode?: string;
}

interface PromptTemplate {
  prompt_template_id?: string;
  id?: string;
  prompt_sections?: PromptSection[];
  sections?: PromptSection[];
  [key: string]: unknown;
}

interface PromptSection {
  role: string;
  content?: string;
  pattern?: string;
  order?: number;
}

interface RolloutResponse {
  run_id: string;
  trajectories: Trajectory[];
  metrics: Metrics;
  aborted: boolean;
  ops_executed: number;
}

interface Trajectory {
  env_id: string;
  policy_id: string;
  steps: Step[];
  length: number;
  inference_url: string;
}

interface Step {
  obs: Record<string, unknown>;
  tool_calls: ToolCall[];
  reward: number;
  done: boolean;
  info: Record<string, unknown>;
}

interface ToolCall {
  id: string;
  type: string;
  function: {
    name: string;
    arguments: string;
  };
}

interface Metrics {
  episode_returns: number[];
  mean_return: number;
  num_steps: number;
  num_episodes: number;
  outcome_score: number;
}

// =============================================================================
// Dataset
// =============================================================================

// Load dataset from JSON file
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

interface Dataset {
  samples: Sample[];
  labels: string[];
}

function loadDataset(): Dataset {
  const dataPath = join(__dirname, "../../data/banking77.json");
  try {
    const data = JSON.parse(readFileSync(dataPath, "utf-8"));
    console.log(`Loaded ${data.samples.length} samples from ${dataPath}`);
    return data;
  } catch {
    // Fallback to embedded samples if file not found
    console.warn("Dataset file not found, using embedded samples");
    return {
      samples: [
        { text: "How do I reset my PIN?", label: "change_pin" },
        { text: "My card hasn't arrived yet", label: "card_arrival" },
        { text: "I want to cancel my card", label: "terminate_account" },
        { text: "How do I activate my new card?", label: "activate_my_card" },
        { text: "I need to dispute a transaction", label: "transaction_charged_twice" },
        { text: "Can I get a refund?", label: "request_refund" },
        { text: "How do I transfer money?", label: "transfer_into_account" },
        { text: "I lost my card", label: "lost_or_stolen_card" },
        { text: "Is there a fee for this?", label: "transfer_fee_charged" },
      ],
      labels: ["change_pin", "card_arrival", "terminate_account", "activate_my_card",
               "transaction_charged_twice", "request_refund", "transfer_into_account",
               "lost_or_stolen_card", "transfer_fee_charged"],
    };
  }
}

const dataset = loadDataset();
const samples = dataset.samples;
const labels = dataset.labels;

function getSample(seed: number): Sample {
  return samples[seed % samples.length];
}

// =============================================================================
// Prompt Rendering
// =============================================================================

function renderTemplate(
  template: string,
  placeholders: Record<string, string>
): string {
  let result = template;
  for (const [key, value] of Object.entries(placeholders)) {
    result = result.replace(new RegExp(`\\{${key}\\}`, "g"), value);
  }
  return result;
}

function buildMessages(
  policyConfig: RolloutRequest["policy"]["config"],
  sample: Sample
): { role: string; content: string }[] {
  const placeholders = {
    query: sample.text,
    intents: labels.join(", "),
  };

  const promptTemplate = policyConfig.prompt_template;
  if (promptTemplate) {
    const sections =
      promptTemplate.prompt_sections || promptTemplate.sections || [];
    const sorted = [...sections].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));

    return sorted.map((section) => {
      const template = section.content || section.pattern || "";
      return {
        role: section.role,
        content: renderTemplate(template, placeholders),
      };
    });
  }

  // Default messages
  return [
    {
      role: "system",
      content:
        "You are an expert banking assistant. Classify queries using the classify tool.",
    },
    {
      role: "user",
      content: `Query: ${sample.text}\nIntents: ${labels.join(", ")}\nClassify this query.`,
    },
  ];
}

// =============================================================================
// LLM Client
// =============================================================================

async function callLlm(
  inferenceUrl: string,
  model: string,
  messages: { role: string; content: string }[],
  apiKey?: string,
  llmApiKey?: string
): Promise<{ predicted: string | null; toolCalls: ToolCall[] }> {
  // Build URL - handle query params correctly
  // inference_url may be "http://host/path?query" - we need "http://host/path/chat/completions?query"
  let url: string;
  const queryIndex = inferenceUrl.indexOf("?");
  if (queryIndex !== -1) {
    const base = inferenceUrl.slice(0, queryIndex).replace(/\/$/, "");
    const query = inferenceUrl.slice(queryIndex);
    url = `${base}/chat/completions${query}`;
  } else {
    url = `${inferenceUrl.replace(/\/$/, "")}/chat/completions`;
  }

  console.log(`LLM call: inference_url=${inferenceUrl} full_url=${url} model=${model}`);

  const tool = {
    type: "function",
    function: {
      name: "classify",
      description: "Classify the customer query into an intent category",
      parameters: {
        type: "object",
        properties: {
          intent: { type: "string", description: "The classified intent" },
        },
        required: ["intent"],
      },
    },
  };

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  // Add Bearer auth for OpenAI-compatible APIs
  if (llmApiKey) {
    headers["Authorization"] = `Bearer ${llmApiKey}`;
  }

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model,
      messages,
      tools: [tool],
      tool_choice: "required",
      temperature: 0,
      max_tokens: 100,
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`LLM request failed: ${response.status} - ${body}`);
  }

  const data = await response.json();
  const toolCalls: ToolCall[] = [];
  let predicted: string | null = null;

  const choice = data.choices?.[0];
  if (choice?.message?.tool_calls) {
    for (const call of choice.message.tool_calls) {
      toolCalls.push({
        id: call.id,
        type: "function",
        function: {
          name: call.function.name,
          arguments: call.function.arguments,
        },
      });

      if (call.function.name === "classify") {
        try {
          const args = JSON.parse(call.function.arguments);
          predicted = args.intent;
        } catch {}
      }
    }
  }

  // Fallback to content
  if (!predicted && choice?.message?.content) {
    predicted = choice.message.content.trim();
  }

  return { predicted, toolCalls };
}

// =============================================================================
// App
// =============================================================================

const app = new Hono();

const API_KEY = process.env.ENVIRONMENT_API_KEY;
const LLM_API_KEY = process.env.GROQ_API_KEY || process.env.OPENAI_API_KEY;

if (API_KEY) {
  console.log("API key authentication enabled");
} else {
  console.warn("No ENVIRONMENT_API_KEY set - running without authentication");
}

if (LLM_API_KEY) {
  console.log("LLM API key configured");
} else {
  console.warn("No GROQ_API_KEY or OPENAI_API_KEY set - LLM calls may fail");
}

// Health endpoint (unauthenticated)
app.get("/health", (c) => {
  return c.json({ healthy: true });
});

// Task info endpoint (authenticated)
app.get("/task_info", (c) => {
  // Check authentication
  if (API_KEY) {
    const providedKey = c.req.header("x-api-key");
    if (providedKey !== API_KEY) {
      return c.json({ detail: "Invalid or missing API key" }, 401);
    }
  }

  // Parse seeds from query string - handles both ?seed=0&seed=1 and ?seeds=0&seeds=1
  const url = new URL(c.req.url);
  const seedParams = url.searchParams.getAll("seed");
  const seedsParams = url.searchParams.getAll("seeds");
  const requestedSeeds = [...seedParams, ...seedsParams]
    .map((s) => parseInt(s, 10))
    .filter((n) => !isNaN(n));

  const datasetSize = samples.length;
  const allSeeds = Array.from({ length: datasetSize }, (_, i) => i);

  // If seeds specified, return one TaskInfo per seed; otherwise return all
  const seedsToReturn = requestedSeeds.length > 0
    ? requestedSeeds.map((s) => [s])
    : [allSeeds];

  const infos = seedsToReturn.map((seeds) => ({
    task: {
      task_id: "banking77-typescript",
      name: "Banking77 Intent Classification (TypeScript)",
      description: "Classify banking customer queries into intent categories",
      version: "1.0.0",
    },
    environment: "banking77",
    dataset: {
      seeds,
      train_count: datasetSize,
      val_count: 0,
      test_count: 0,
    },
    rubric: {
      scoring_criteria: "exact_match",
      metric_primary: "accuracy",
      metric_range: [0.0, 1.0],
    },
    inference: {
      mode: "tool_call",
      supported_tools: ["classify"],
    },
    limits: {
      max_response_tokens: 100,
      timeout_seconds: 30,
    },
  }));

  return c.json(infos);
});

// Rollout endpoint (authenticated)
app.post("/rollout", async (c) => {
  // Check authentication
  if (API_KEY) {
    const providedKey = c.req.header("x-api-key");
    if (providedKey !== API_KEY) {
      return c.json({ detail: "Invalid or missing API key" }, 401);
    }
  }

  const request: RolloutRequest = await c.req.json();

  // Get sample
  const seed = request.env.seed ?? 0;
  const sample = getSample(seed);

  console.log(
    `Rollout: run_id=${request.run_id} seed=${seed} query=${sample.text}`
  );

  // Get inference URL
  const inferenceUrl =
    request.policy.config.inference_url ||
    request.policy.config.api_base ||
    request.policy.config.base_url;

  if (!inferenceUrl) {
    return c.json({ detail: "Missing inference_url in policy.config" }, 400);
  }

  const model = (request.policy.config.model as string) || "gpt-4o-mini";

  // Build messages and call LLM
  const messages = buildMessages(request.policy.config, sample);

  let predicted: string | null = null;
  let toolCalls: ToolCall[] = [];

  try {
    const providedKey = c.req.header("x-api-key");
    const result = await callLlm(inferenceUrl, model, messages, providedKey, LLM_API_KEY);
    predicted = result.predicted;
    toolCalls = result.toolCalls;
  } catch (error) {
    console.warn("LLM call failed:", error);
    return c.json({ detail: `LLM call failed: ${error}` }, 502);
  }

  // Compute reward
  const isCorrect =
    predicted?.toLowerCase() === sample.label.toLowerCase();
  const reward = isCorrect ? 1.0 : 0.0;

  console.log(
    `Result: expected=${sample.label} predicted=${predicted} correct=${isCorrect} reward=${reward}`
  );

  // Build response
  const response: RolloutResponse = {
    run_id: request.run_id,
    trajectories: [
      {
        env_id: `task::train::${seed}`,
        policy_id:
          request.policy.policy_id || request.policy.policy_name || "policy",
        steps: [
          {
            obs: { query: sample.text, index: seed },
            tool_calls: toolCalls,
            reward,
            done: true,
            info: {
              expected: sample.label,
              predicted,
              correct: isCorrect,
            },
          },
        ],
        length: 1,
        inference_url: inferenceUrl,
      },
    ],
    metrics: {
      episode_returns: [reward],
      mean_return: reward,
      num_steps: 1,
      num_episodes: 1,
      outcome_score: reward,
    },
    aborted: false,
    ops_executed: 1,
  };

  return c.json(response);
});

// =============================================================================
// Server
// =============================================================================

const port = parseInt(process.env.PORT || "8001", 10);

console.log(`Dataset loaded: ${samples.length} samples`);
console.log(`Starting task app on port ${port}`);

serve({
  fetch: app.fetch,
  port,
});
