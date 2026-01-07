import { createModalUI, clamp, type ModalController } from "./base"
import type { AppContext } from "../context"
import { JobType } from "../types"
import { focusManager } from "../focus"
import { LOCALAPI_TEMPLATE } from "../templates/localapi"
import { getUniqueFilename, toDisplayPath, expandPath } from "../utils/files"
import { formatErrorMessage } from "../utils/truncate"
import { scanForLocalAPIs, type ScannedLocalAPI } from "../utils/localapi-scanner"
import * as fs from "fs"
import * as path from "path"
import * as readline from "readline"
import { spawn, type ChildProcess } from "child_process"

/** Result from deploy attempt */
type DeployResult = {
	success: boolean
	url?: string
	proc?: ChildProcess
	error?: string
}

/** Deploy a LocalAPI file and wait for the result (handles NDJSON stream) */
function deployLocalApi(filePath: string): Promise<DeployResult> {
	return new Promise((resolve) => {
		const proc = spawn("python", ["-m", "synth_ai.tui.deploy", filePath], {
			stdio: ["ignore", "pipe", "pipe"],
		})

		let resolved = false

		// Read NDJSON stream line by line
		const rl = readline.createInterface({ input: proc.stdout })

		rl.on("line", (line: string) => {
			if (resolved) return
			try {
				const result = JSON.parse(line)
				// Wait for terminal status (ready or error)
				if (result.type === "status") {
					if (result.status === "ready") {
						resolved = true
						resolve({ success: true, url: result.url, proc })
					} else if (result.status === "error") {
						resolved = true
						resolve({ success: false, error: result.error || "Deployment failed" })
					}
					// Ignore "starting" status - keep waiting
				}
			} catch {
				// Ignore non-JSON lines
			}
		})

		// Capture stderr for error messages
		let stderrBuffer = ""
		proc.stderr.on("data", (data: Buffer) => {
			stderrBuffer += data.toString()
		})

		proc.on("error", (err) => {
			if (resolved) return
			resolved = true
			resolve({ success: false, error: err.message })
		})

		proc.on("close", (code) => {
			if (resolved) return
			resolved = true
			if (code !== 0) {
				const errorMsg = stderrBuffer.trim() || `Process exited with code ${code}`
				resolve({ success: false, error: errorMsg })
			}
		})
	})
}

/** Step definition for the wizard */
type StepConfig = {
	id: string
	label: string
	prompt: string
	getOptions: () => string[]
	/** If true, don't show in summary */
	hideFromSummary?: boolean
	/** Called when this step's selection is made. Return false to prevent advancing. */
	onSelect?: (value: string) => Promise<boolean> | boolean
}

/** Tracks selection for a completed step */
type StepSelection = {
	stepId: string
	value: string
	displayLabel: string
}

const CREATE_NEW_OPTION = "Create new"
const ENTER_MANUAL_PATH = "Enter manual path"

export function createCreateJobModal(ctx: AppContext): ModalController & {
	open: () => void
	move: (delta: number) => void
	select: () => void
} {
	const { renderer } = ctx

	const modal = createModalUI(renderer, {
		id: "create-job-modal",
		width: 70,
		height: 24,
		borderColor: "#10b981",
		titleColor: "#10b981",
		zIndex: 10,
	})

	// Track created file path for review step
	let createdFilePath: string | null = null

	// Track deployed tunnel URL
	let deployedUrl: string | null = null

	// Deploy in-progress state - prevents double submission
	let isDeploying = false

	// Text input mode state
	let isInputMode = false
	let inputBuffer = ""

	// Deploy error state - shown in red in modal
	let deployError: string | null = null
	let deployErrorLines: string[] = []  // All wrapped error lines
	let deployErrorOffset = 0            // Current scroll offset

	// Cache scanned LocalAPIs (refreshed on modal open)
	let scannedLocalAPIs: ScannedLocalAPI[] = []

	// Dynamic steps - changes based on selections
	function getSteps(): StepConfig[] {
		// Build LocalAPI options: found files first (with full ~/ path), then "Create new"
		const localApiOptions = [
			...scannedLocalAPIs.map((api) => toDisplayPath(api.filepath)),
			CREATE_NEW_OPTION,
		]

		const baseSteps: StepConfig[] = [
			{
				id: "localApi",
				label: "LocalAPI file",
				prompt: "Select or create LocalAPI file:",
				getOptions: () => localApiOptions,
			},
		]

		const localApiSelection = getSelectionForStep("localApi")

		// If user selected an existing LocalAPI file, add mode selection then deploy
		const selectedExistingApi = scannedLocalAPIs.find(
			(api) => toDisplayPath(api.filepath) === localApiSelection?.value
		)
		if (selectedExistingApi) {
			// Set createdFilePath to the selected file for deploy step
			createdFilePath = selectedExistingApi.filepath
			const fileName = selectedExistingApi.filename

			baseSteps.push({
				id: "deployLocalApi",
				label: "Deploy",
				prompt: `Deploy ${fileName}?`,
				getOptions: () => ["Yes", "No"],
				hideFromSummary: true,
				onSelect: async (value) => {
					deployError = null
					deployErrorOffset = 0

					if (value === "Yes" && createdFilePath) {
						isDeploying = true
						ctx.state.snapshot.status = `Deploying ${toDisplayPath(createdFilePath)}...`
						updateContent()
						ctx.render()

						const result = await deployLocalApi(createdFilePath)
						isDeploying = false

						if (result.success) {
							deployedUrl = result.url!
							ctx.state.snapshot.status = `Deployed: ${result.url}`
							ctx.state.appState.deployedUrl = result.url!
							ctx.state.appState.deployProc = result.proc!
							updateContent()
							ctx.render()
							return true
						} else {
							deployError = result.error || "Unknown error"
							deployErrorOffset = 0
							ctx.state.snapshot.status = `Deploy failed: ${deployError}`
							updateContent()
							ctx.render()
							return false
						}
					}
					return true
				},
			})
		}

		// If "create new" was selected, add directory selection and creation steps
		if (localApiSelection?.value === CREATE_NEW_OPTION) {
			// Directory selection step
			const cwdDisplay = `CWD: ${toDisplayPath(process.cwd())}`
			baseSteps.push({
				id: "selectDirectory",
				label: "Directory",
				prompt: "Where to save localapi.py?",
				getOptions: () => [cwdDisplay, ENTER_MANUAL_PATH],
			})

			// Get selected directory for subsequent steps
			const dirSelection = getSelectionForStep("selectDirectory")
			if (dirSelection) {
				// Strip "CWD: " prefix if present
				const dirValue = dirSelection.value.startsWith("CWD: ")
					? dirSelection.value.slice(5)
					: dirSelection.value
				const selectedDir = expandPath(dirValue)

				// Use created file if exists, otherwise propose new filename
				// This ensures filesystem is source of truth after creation
				const actualFilePath = createdFilePath ?? getUniqueFilename(selectedDir, "localapi", ".py")
				const actualFileName = path.basename(actualFilePath)
				const displayDir = toDisplayPath(selectedDir)

				baseSteps.push({
					id: "confirmCreate",
					label: "Confirm",
					prompt: `Create ${actualFileName} in ${displayDir}?`,
					getOptions: () => ["Yes, create it", "No, go back"],
					hideFromSummary: true,
					onSelect: async (value) => {
						if (value === "No, go back") {
							goBack()
							return false
						}
						// Create the file (and directory if needed)
						try {
							fs.mkdirSync(selectedDir, { recursive: true })
							fs.writeFileSync(actualFilePath, LOCALAPI_TEMPLATE, "utf-8")
							createdFilePath = actualFilePath
							ctx.state.snapshot.status = `Created ${toDisplayPath(actualFilePath)}`
							ctx.render()
							return true
						} catch (err) {
							ctx.state.snapshot.status = `Error creating file: ${err}`
							ctx.render()
							return false
						}
					},
				})
				baseSteps.push({
					id: "openForReview",
					label: "Review",
					prompt: `Open ${actualFileName} for review?`,
					getOptions: () => ["Yes, open in editor", "No, skip"],
					hideFromSummary: true,
					onSelect: async (value) => {
						if (value === "Yes, open in editor" && createdFilePath) {
							// Open with system default or EDITOR
							const editor = process.env.EDITOR
							if (editor) {
								spawn(editor, [createdFilePath], {
									detached: true,
									stdio: "ignore",
								}).unref()
							} else {
								// Use 'open' on macOS to open with default app
								spawn("open", [createdFilePath], {
									detached: true,
									stdio: "ignore",
								}).unref()
							}
						}
						// Always continue to next step
						return true
					},
				})
				baseSteps.push({
					id: "deployLocalApi",
					label: "Deploy",
					prompt: `Deploy ${actualFileName}?`,
					getOptions: () => ["Yes, deploy now", "No, deploy later"],
					hideFromSummary: true,
					onSelect: async (value) => {
						// Clear any previous error
						deployError = null
						deployErrorOffset = 0

						if (value === "Yes, deploy now" && createdFilePath) {
							isDeploying = true
							ctx.state.snapshot.status = `Deploying ${toDisplayPath(createdFilePath)}...`
							updateContent()
							ctx.render()

							const result = await deployLocalApi(createdFilePath)
							isDeploying = false

							if (result.success) {
								deployedUrl = result.url!
								ctx.state.snapshot.status = `Deployed: ${result.url}`
								ctx.state.appState.deployedUrl = result.url!
								ctx.state.appState.deployProc = result.proc!
								updateContent()
								ctx.render()
								return true // Advance to next step
							} else {
								// Show error in modal - don't advance
								deployError = result.error || "Unknown error"
								deployErrorOffset = 0
								ctx.state.snapshot.status = `Deploy failed: ${deployError}`
								updateContent()
								ctx.render()
								return false
							}
						}
						return true // "No, deploy later" - skip
					},
				})
			}
		}

		// Job type is always last step
		baseSteps.push({
			id: "jobType",
			label: "Job Type",
			prompt: "Select Job Type:",
			getOptions: () => Object.values(JobType),
			onSelect: async (value) => {
				if (value === JobType.Eval && deployedUrl) {
					// Submit eval job and close
					ctx.state.snapshot.status = "Submitting eval job..."
					ctx.render()

					// Fire-and-forget: spawn eval job process
					spawn("python", ["-m", "synth_ai.tui.eval_job", deployedUrl, "default"], {
						stdio: "ignore",
						detached: true,
					}).unref()

					ctx.state.snapshot.status = "Eval job submitted"
					toggle(false)
					ctx.render()
				} else {
					// Non-eval job type or no deploy - just close for now
					toggle(false)
				}
				return true
			},
		})

		return baseSteps
	}

	// Modal state
	let currentStepIndex = 0
	let cursor = 0
	let selections: StepSelection[] = []

	function getCurrentStep(): StepConfig {
		return getSteps()[currentStepIndex]
	}

	function getSelectionForStep(stepId: string): StepSelection | undefined {
		return selections.find((s) => s.stepId === stepId)
	}

	function renderSummary(): string[] {
		const lines: string[] = []
		const steps = getSteps()

		// Show completed selections with ability to go back
		for (let i = 0; i < currentStepIndex; i++) {
			const step = steps[i]
			if (step.hideFromSummary) continue
			const selection = getSelectionForStep(step.id)

			// For localApi step, show filepath if existing file was selected
			if (step.id === "localApi" && selection) {
				const selectedExistingApi = scannedLocalAPIs.find(
					(api) => toDisplayPath(api.filepath) === selection.value
				)
				if (selectedExistingApi) {
					lines.push(`  LocalAPI file: ${toDisplayPath(selectedExistingApi.filepath)}`)
					continue
				}
				// "Create new" was selected - don't show in summary (directory step will show it)
				continue
			}

			// For directory step, show the full localapi filepath instead
			if (step.id === "selectDirectory" && selection) {
				// Use created file path if available (source of truth after creation)
				if (createdFilePath) {
					lines.push(`  LocalAPI file: ${toDisplayPath(createdFilePath)}`)
					continue
				}
				// Otherwise compute proposed path (before creation)
				const dirValue = selection.value.startsWith("CWD: ")
					? selection.value.slice(5)
					: selection.value
				const selectedDir = expandPath(dirValue)
				const filePath = getUniqueFilename(selectedDir, "localapi", ".py")
				lines.push(`  LocalAPI file: ${toDisplayPath(filePath)}`)
				continue
			}

			const value = selection?.value ?? "(not set)"
			lines.push(`  ${step.label}: ${value}`)
		}

		// Show deployed URL after file path
		if (deployedUrl) {
			lines.push(`  LocalAPI URL: ${deployedUrl}`)
		}

		return lines
	}

	function updateContent(): void {
		const step = getCurrentStep()
		const options = step.getOptions()
		const lines: string[] = []

		// Show summary of previous selections
		const summary = renderSummary()
		if (summary.length > 0) {
			lines.push(...summary)
			lines.push("")
		}

		// Current step prompt
		lines.push(`  ${step.prompt}`)
		lines.push("")

		if (isDeploying) {
			// Show deploying message instead of options
			lines.push(`  Deploying...`)
		} else if (isInputMode) {
			// Text input mode - ~/ is fixed prefix, inputBuffer is editable part
			lines.push(`  Path: ~/${inputBuffer}█`)
			lines.push("")
			lines.push("  (enter to confirm, escape to cancel)")
		} else {
			// Options for current step
			for (let idx = 0; idx < options.length; idx++) {
				const option = options[idx]
				const isCursor = idx === cursor
				const cursorChar = isCursor ? "›" : " "
				lines.push(`  ${cursorChar} ${option}`)
			}
		}

		// Show deploy error in red if present (scrollable, 2 lines visible)
		if (deployError) {
			lines.push("")
			// Wrap error without line limit
			deployErrorLines = formatErrorMessage(deployError, 64, Infinity)
			const visibleLines = deployErrorLines.slice(deployErrorOffset, deployErrorOffset + 2)
			for (const errLine of visibleLines) {
				lines.push(`  \x1b[31m${errLine}\x1b[0m`)
			}
			// Show scroll hint if more content exists
			if (deployErrorLines.length > 2) {
				const position = `[${deployErrorOffset + 1}-${Math.min(deployErrorOffset + 2, deployErrorLines.length)}/${deployErrorLines.length}]`
				lines.push(`  \x1b[90m${position} Shift+j/k scroll\x1b[0m`)
			}
		}

		// Padding to fill modal
		while (lines.length < 18) {
			lines.push("")
		}

		modal.setContent(lines.join("\n"))
	}

	function updateHint(): void {
		if (isInputMode) {
			modal.setHint("enter confirm | escape cancel")
			return
		}
		const hints = ["j/k navigate", "enter select"]
		if (currentStepIndex > 0) {
			hints.push("backspace back")
		}
		hints.push("q close")
		modal.setHint(hints.join(" | "))
	}

	function reset(): void {
		currentStepIndex = 0
		cursor = 0
		selections = []
		createdFilePath = null
		deployedUrl = null
		isDeploying = false
		isInputMode = false
		inputBuffer = ""
		deployError = null
		deployErrorLines = []
		deployErrorOffset = 0
		// Scan CWD for existing LocalAPI files
		scannedLocalAPIs = scanForLocalAPIs(process.cwd())
	}

	function toggle(visible: boolean): void {
		if (visible) {
			focusManager.push({
				id: "create-job-modal",
				handleKey,
			})
			reset()
			modal.center()
			updateContent()
			updateHint()
		} else {
			focusManager.pop("create-job-modal")
		}
		modal.setVisible(visible)
	}

	function open(): void {
		modal.setTitle("Create New Job")
		toggle(true)
	}

	function move(delta: number): void {
		const step = getCurrentStep()
		const options = step.getOptions()
		const max = options.length - 1
		cursor = clamp(cursor + delta, 0, max)
		updateContent()
	}

	function goBack(): void {
		if (currentStepIndex > 0) {
			currentStepIndex--
			// Restore cursor to previous selection
			const step = getCurrentStep()
			const prevSelection = getSelectionForStep(step.id)
			if (prevSelection) {
				const options = step.getOptions()
				cursor = options.indexOf(prevSelection.value)
				if (cursor < 0) cursor = 0
				// Remove this selection so it can be re-selected
				selections = selections.filter((s) => s.stepId !== step.id)
			} else {
				cursor = 0
			}
			updateContent()
			updateHint()
		}
	}

	async function select(): Promise<void> {
		const step = getCurrentStep()
		const options = step.getOptions()
		const selected = options[cursor]
		if (!selected) return

		// Handle "Enter manual path" option
		if (selected === ENTER_MANUAL_PATH) {
			isInputMode = true
			inputBuffer = "" // ~/ prefix is fixed, not part of buffer
			updateContent()
			updateHint()
			return
		}

		// Call onSelect hook if present
		if (step.onSelect) {
			const shouldContinue = await step.onSelect(selected)
			if (!shouldContinue) {
				updateContent()
				updateHint()
				return
			}
		}

		// Store selection
		selections = selections.filter((s) => s.stepId !== step.id)
		selections.push({
			stepId: step.id,
			value: selected,
			displayLabel: step.label,
		})

		// Check if there are more steps (re-evaluate since steps can change dynamically)
		const steps = getSteps()
		if (currentStepIndex < steps.length - 1) {
			currentStepIndex++
			cursor = 0
			updateContent()
			updateHint()
		} else {
			// All steps complete - create the job
			const summary = selections
				.filter((s) => {
					const step = steps.find((st) => st.id === s.stepId)
					return !step?.hideFromSummary
				})
				.map((s) => `${s.displayLabel}: ${s.value}`)
				.join(", ")
			toggle(false)
			ctx.state.snapshot.status = `Creating job: ${summary}`
			ctx.render()
			// TODO: Actually create the job via API
		}
	}

	function handleKey(key: any): boolean {
		if (!modal.visible) return false

		// Block all input while deploying (except escape to close)
		if (isDeploying) {
			if (key.name === "q" || key.name === "escape") {
				toggle(false)
				return true
			}
			return true // Consume but ignore other keys
		}

		// Text input mode handling
		if (isInputMode) {
			if (key.name === "escape") {
				// Cancel input mode
				isInputMode = false
				inputBuffer = ""
				updateContent()
				updateHint()
				return true
			}
			if (key.name === "return" || key.name === "enter") {
				// Confirm input - store as selection with ~/ prefix and advance
				const step = getCurrentStep()
				const inputValue = inputBuffer.trim()
				// Always prepend ~/ to the input value
				const fullPath = `~/${inputValue}`
				isInputMode = false
				selections = selections.filter((s) => s.stepId !== step.id)
				selections.push({
					stepId: step.id,
					value: fullPath,
					displayLabel: step.label,
				})
				inputBuffer = ""
				currentStepIndex++
				cursor = 0
				updateContent()
				updateHint()
				return true
			}
			if (key.name === "backspace") {
				// Delete last character
				if (inputBuffer.length > 0) {
					inputBuffer = inputBuffer.slice(0, -1)
					updateContent()
				}
				return true
			}
			// Handle printable characters
			if (key.sequence && key.sequence.length === 1 && !key.ctrl && !key.meta) {
				inputBuffer += key.sequence
				updateContent()
				return true
			}
			return true
		}

		// Scroll error with Shift+j/k when error is visible
		if (deployError && deployErrorLines.length > 2 && key.shift) {
			if (key.name === "j" || key.name === "down") {
				deployErrorOffset = Math.min(deployErrorOffset + 1, deployErrorLines.length - 2)
				updateContent()
				return true
			}
			if (key.name === "k" || key.name === "up") {
				deployErrorOffset = Math.max(0, deployErrorOffset - 1)
				updateContent()
				return true
			}
		}

		// Normal mode handling
		if (key.name === "up" || key.name === "k") {
			move(-1)
			return true
		}
		if (key.name === "down" || key.name === "j") {
			move(1)
			return true
		}
		if (key.name === "return" || key.name === "enter") {
			void select()
			return true
		}
		if (key.name === "backspace" && currentStepIndex > 0) {
			goBack()
			return true
		}
		if (key.name === "q" || key.name === "escape") {
			toggle(false)
			return true
		}
		return true
	}

	return {
		get isVisible() {
			return modal.visible
		},
		toggle,
		open,
		move,
		select,
		handleKey,
	}
}
