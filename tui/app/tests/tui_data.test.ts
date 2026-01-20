import { describe, expect, test } from "bun:test"
import {
  coerceJob,
  extractEvents,
  extractJobs,
  mergeJobs,
} from "../src/tui_data.ts"

describe("tui_data extraction", () => {
  test("extractJobs handles list payloads and training type", () => {
    const payload = {
      jobs: [
        {
          job_id: "pl_123",
          status: "succeeded",
          created_at: "2026-01-03T00:00:00Z",
          best_score: "0.7",
          metadata: { config: { algorithm: "gepa" } },
        },
      ],
    }
    const jobs = extractJobs(payload, "prompt-learning")
    expect(jobs).toHaveLength(1)
    expect(jobs[0]).toMatchObject({
      job_id: "pl_123",
      status: "succeeded",
      job_type: "gepa",
      job_source: "prompt-learning",
      best_reward: 0.7,
    })
  })

  test("coerceJob marks eval jobs and preserves source", () => {
    const job = coerceJob({ job_id: "eval_abcd", status: "completed" }, "learning")
    expect(job.job_type).toBe("eval")
    expect(job.job_source).toBe("eval")
  })

  test("extractEvents maps variants and next_seq", () => {
    const payload = {
      events: [
        {
          seq: 10,
          type: "prompt.learning.started",
          message: "started",
          data: { foo: "bar" },
          timestamp: "2026-01-03T00:00:00Z",
        },
        {
          sequence: "11",
          event_type: "prompt.learning.progress",
          payload: { pct: 50 },
          created_at: "2026-01-03T00:00:10Z",
        },
        {
          id: "evt_12",
          message: "fallback type",
        },
      ],
      next_seq: "13",
    }
    const { events, nextSeq } = extractEvents(payload)
    expect(nextSeq).toBe(13)
    expect(events[0]).toMatchObject({
      seq: 10,
      type: "prompt.learning.started",
      message: "started",
      timestamp: "2026-01-03T00:00:00Z",
    })
    expect(events[1]).toMatchObject({
      seq: 11,
      type: "prompt.learning.progress",
      timestamp: "2026-01-03T00:00:10Z",
    })
    expect(events[2].type).toBe("event")
    expect(events[2].seq).toBe(2)
  })

  test("mergeJobs de-duplicates and sorts by created_at desc", () => {
    const primary = [
      { job_id: "job_a", status: "running", created_at: "2026-01-03T01:00:00Z" },
      { job_id: "job_b", status: "succeeded", created_at: "2026-01-03T02:00:00Z" },
    ]
    const secondary = [
      { job_id: "job_b", status: "failed", created_at: "2026-01-03T03:00:00Z" },
      { job_id: "job_c", status: "queued", created_at: "2026-01-02T00:00:00Z" },
    ]
    const merged = mergeJobs(primary, secondary)
    expect(merged.map((job) => job.job_id)).toEqual(["job_b", "job_a", "job_c"])
    expect(merged.find((job) => job.job_id === "job_b")?.status).toBe("succeeded")
  })
})
