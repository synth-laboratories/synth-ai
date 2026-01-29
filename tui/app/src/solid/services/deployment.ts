/**
 * Deployment service for LocalAPI files.
 * Handles spawning deploy processes and parsing NDJSON status streams.
 */
import { spawn, type ChildProcess } from "child_process"
import * as readline from "readline"
import * as path from "path"
import { formatTimestampForFilename } from "../utils/files"

export interface DeployResult {
  success: boolean
  url?: string
  proc?: ChildProcess
  error?: string
  deploymentId?: string
}

export interface Deployment {
  id: string
  localApiPath: string
  url: string | null
  status: "deploying" | "ready" | "error"
  error?: string
  logs: string[]
  proc: ChildProcess | null
  startedAt: Date
}

export type DeploymentStatusHandler = (deployment: Deployment) => void
export type DeploymentLogHandler = (line: string) => void

/**
 * Deploy a LocalAPI file and return a promise that resolves when deployment completes.
 * Handles NDJSON stream parsing for status updates.
 */
export function deployLocalApi(
  filePath: string,
  onStatus?: DeploymentStatusHandler,
  onLog?: DeploymentLogHandler,
): Promise<DeployResult> {
  return new Promise((resolve) => {
    // Generate deployment ID
    const fileName = path.basename(filePath, ".py")
    const timestamp = formatTimestampForFilename(new Date())
    const deploymentId = `${fileName}_${timestamp}`

    // Initialize deployment state
    const deployment: Deployment = {
      id: deploymentId,
      localApiPath: filePath,
      url: null,
      status: "deploying",
      logs: [],
      proc: null,
      startedAt: new Date(),
    }

    // Spawn the deploy process
    const proc = spawn("python", ["-m", "tui.deploy", filePath, "--deployment-id", deploymentId], {
      stdio: ["ignore", "pipe", "pipe"],
    })
    deployment.proc = proc
    onStatus?.(deployment)

    let resolved = false
    const finalize = (result: DeployResult): void => {
      if (resolved) return
      resolved = true
      resolve({ deploymentId, ...result })
    }

    // Read NDJSON stream line by line from stdout
    const rl = readline.createInterface({ input: proc.stdout })

    rl.on("line", (line: string) => {
      if (resolved) return
      
      // Log all lines
      deployment.logs.push(line)
      onLog?.(line)

      try {
        const result = JSON.parse(line)
        
        // Handle status updates
        if (result.type === "status") {
          if (result.status === "ready") {
            deployment.status = "ready"
            deployment.url = result.url
            onStatus?.(deployment)
            finalize({ success: true, url: result.url, proc })
          } else if (result.status === "error") {
            deployment.status = "error"
            deployment.error = result.error || "Deployment failed"
            onStatus?.(deployment)
            finalize({ success: false, error: result.error || "Deployment failed" })
          }
          // Ignore "starting" status - keep waiting
        }
      } catch {
        // Ignore non-JSON lines (regular log output)
      }
    })

    // Capture stderr for error messages
    let stderrBuffer = ""
    proc.stderr.on("data", (data: Buffer) => {
      const text = data.toString()
      stderrBuffer += text
      deployment.logs.push(`[stderr] ${text}`)
      onLog?.(`[stderr] ${text}`)
    })

    proc.on("error", (err) => {
      deployment.status = "error"
      deployment.error = err.message
      onStatus?.(deployment)
      finalize({ success: false, error: err.message, deploymentId })
    })

    proc.on("close", (code) => {
      if (resolved) return
      
      if (code !== 0) {
        const errorMsg = stderrBuffer.trim() || `Process exited with code ${code}`
        deployment.status = "error"
        deployment.error = errorMsg
        onStatus?.(deployment)
        finalize({ success: false, error: errorMsg })
      } else {
        // If we get here without a ready status, something went wrong
        if (deployment.status !== "ready") {
          finalize({ success: false, error: "Deployment completed but no URL received" })
        } else {
          finalize({ success: true, url: deployment.url ?? undefined, proc })
        }
      }
    })
  })
}

/**
 * Submit an eval job for a deployed LocalAPI.
 */
export function submitEvalJob(
  deployedUrl: string,
  split: string = "default",
): { success: boolean; error?: string } {
  try {
    // Fire-and-forget: spawn eval job process
    const proc = spawn("python", ["-m", "tui.eval_job", deployedUrl, split], {
      stdio: "ignore",
      detached: true,
    })
    proc.on("error", () => {
      // Ignore errors - fire and forget
    })
    proc.unref()
    
    return { success: true }
  } catch (err) {
    return { success: false, error: err instanceof Error ? err.message : String(err) }
  }
}

/**
 * Submit a learning job for a deployed LocalAPI.
 */
export function submitLearningJob(
  deployedUrl: string,
  jobType: "prompt_learning" | "learning" = "prompt_learning",
): { success: boolean; error?: string } {
  try {
    // Fire-and-forget: spawn learning job process
    const module = jobType === "prompt_learning" 
      ? "tui.prompt_learning_job"
      : "tui.learning_job"
    
    const proc = spawn("python", ["-m", module, deployedUrl], {
      stdio: "ignore",
      detached: true,
    })
    proc.on("error", () => {
      // Ignore errors - fire and forget
    })
    proc.unref()
    
    return { success: true }
  } catch (err) {
    return { success: false, error: err instanceof Error ? err.message : String(err) }
  }
}

