/**
 * Create Job modal for SolidJS TUI.
 * 
 * Multi-step wizard for creating new jobs:
 * 1. Select LocalAPI file (or create new)
 * 2. (If creating new) Select directory and create file
 * 3. Select job type
 * 4. Deploy and submit job
 */
import { For, Show, createMemo, createSignal, createEffect } from "solid-js"
import { COLORS } from "../theme"
import { toDisplayPath, expandPath, getUniqueFilename } from "../utils/files"
import { createLocalApiFile, openInEditor } from "../services/file-creation"
import { deployLocalApi, submitEvalJob, submitLearningJob } from "../services/deployment"

export interface CreateJobModalProps {
  visible: boolean
  onClose: () => void
  onJobCreated: (jobInfo: JobCreatedInfo) => void
  onStatusUpdate: (status: string) => void
  onError: (error: string) => void
  localApiFiles: string[]
  width: number
  height: number
}

export interface JobCreatedInfo {
  localApiPath: string
  trainingType: "prompt_learning" | "eval" | "learning"
  deployedUrl?: string
  jobSubmitted: boolean
}

type Step = 
  | "selectFile" 
  | "selectDirectory" 
  | "confirmCreate" 
  | "selectType" 
  | "deploying"
  | "confirm"

const CREATE_NEW_OPTION = "+ Create new LocalAPI file"
const CWD_OPTION_PREFIX = "CWD: "

const JOB_TYPES = [
  { id: "eval" as const, label: "Evaluation", description: "Run evaluation on your LocalAPI" },
  { id: "prompt_learning" as const, label: "Prompt Learning", description: "Optimize prompts for better performance" },
  { id: "learning" as const, label: "Learning", description: "Train and optimize models" },
]

export function CreateJobModal(props: CreateJobModalProps) {
  const [step, setStep] = createSignal<Step>("selectFile")
  const [selectedFileIndex, setSelectedFileIndex] = createSignal(0)
  const [selectedTypeIndex, setSelectedTypeIndex] = createSignal(0)
  const [selectedDirIndex, setSelectedDirIndex] = createSignal(0)
  const [selectedFile, setSelectedFile] = createSignal<string | null>(null)
  const [selectedType, setSelectedType] = createSignal<typeof JOB_TYPES[number] | null>(null)
  const [createdFilePath, setCreatedFilePath] = createSignal<string | null>(null)
  const [isDeploying, setIsDeploying] = createSignal(false)
  const [deployError, setDeployError] = createSignal<string | null>(null)
  const [deployedUrl, setDeployedUrl] = createSignal<string | null>(null)
  const [confirmIndex, setConfirmIndex] = createSignal(0)

  // Reset state when modal opens
  createEffect(() => {
    if (props.visible) {
      setStep("selectFile")
      setSelectedFileIndex(0)
      setSelectedTypeIndex(0)
      setSelectedDirIndex(0)
      setSelectedFile(null)
      setSelectedType(null)
      setCreatedFilePath(null)
      setIsDeploying(false)
      setDeployError(null)
      setDeployedUrl(null)
      setConfirmIndex(0)
    }
  })

  const fileOptions = createMemo(() => {
    const files = props.localApiFiles
    if (files.length === 0) {
      return [CREATE_NEW_OPTION]
    }
    return [...files, CREATE_NEW_OPTION]
  })

  const directoryOptions = createMemo(() => {
    const cwd = process.cwd()
    return [
      `${CWD_OPTION_PREFIX}${toDisplayPath(cwd)}`,
    ]
  })

  // Get the actual file path (either selected or created)
  const effectiveFilePath = createMemo(() => {
    return createdFilePath() ?? selectedFile()
  })

  async function handleDeploy(): Promise<void> {
    const filePath = effectiveFilePath()
    if (!filePath) return

    setIsDeploying(true)
    setStep("deploying")
    setDeployError(null)
    props.onStatusUpdate(`Deploying ${toDisplayPath(filePath)}...`)

    try {
      const result = await deployLocalApi(
        filePath,
        (deployment) => {
          if (deployment.status === "ready" && deployment.url) {
            props.onStatusUpdate(`Deployed: ${deployment.url}`)
          } else if (deployment.status === "error") {
            props.onStatusUpdate(`Deploy failed: ${deployment.error}`)
          }
        },
      )

      setIsDeploying(false)

      if (result.success && result.url) {
        setDeployedUrl(result.url)
        setStep("confirm")
      } else {
        setDeployError(result.error ?? "Deployment failed")
        setStep("selectType") // Go back to type selection
      }
    } catch (err) {
      setIsDeploying(false)
      const errorMsg = err instanceof Error ? err.message : String(err)
      setDeployError(errorMsg)
      props.onError(errorMsg)
      setStep("selectType")
    }
  }

  function handleCreateFile(): boolean {
    const dirOption = directoryOptions()[selectedDirIndex()]
    const dirPath = dirOption.startsWith(CWD_OPTION_PREFIX)
      ? expandPath(dirOption.slice(CWD_OPTION_PREFIX.length))
      : expandPath(dirOption)

    const result = createLocalApiFile(dirPath)
    
    if (result.success && result.filePath) {
      setCreatedFilePath(result.filePath)
      props.onStatusUpdate(`Created ${result.displayPath}`)
      return true
    } else {
      setDeployError(result.error ?? "Failed to create file")
      return false
    }
  }

  function handleSubmitJob(): void {
    const type = selectedType()
    const url = deployedUrl()
    const filePath = effectiveFilePath()
    
    if (!type || !filePath) return

    let submitted = false
    
    if (url) {
      // We have a deployed URL, submit the job
      if (type.id === "eval") {
        const result = submitEvalJob(url)
        submitted = result.success
        if (!result.success) {
          props.onError(result.error ?? "Failed to submit eval job")
        }
      } else {
        const result = submitLearningJob(url, type.id)
        submitted = result.success
        if (!result.success) {
          props.onError(result.error ?? "Failed to submit learning job")
        }
      }
      
      if (submitted) {
        props.onStatusUpdate(`${type.label} job submitted`)
      }
    } else {
      props.onStatusUpdate(`Job ready to deploy: ${toDisplayPath(filePath)}`)
    }

    props.onJobCreated({
      localApiPath: filePath,
      trainingType: type.id,
      deployedUrl: url ?? undefined,
      jobSubmitted: submitted,
    })
    props.onClose()
  }

  function handleKeyPress(evt: { name: string; shift?: boolean; ctrl?: boolean }): boolean {
    if (!props.visible) return false
    if (isDeploying()) return true // Block input while deploying

    const currentStep = step()

    // Navigation
    if (evt.name === "j" || evt.name === "down") {
      if (currentStep === "selectFile") {
        setSelectedFileIndex(i => Math.min(i + 1, fileOptions().length - 1))
      } else if (currentStep === "selectDirectory") {
        setSelectedDirIndex(i => Math.min(i + 1, directoryOptions().length - 1))
      } else if (currentStep === "selectType") {
        setSelectedTypeIndex(i => Math.min(i + 1, JOB_TYPES.length - 1))
      } else if (currentStep === "confirm") {
        setConfirmIndex(i => Math.min(i + 1, 1))
      }
      return true
    }
    if (evt.name === "k" || evt.name === "up") {
      if (currentStep === "selectFile") {
        setSelectedFileIndex(i => Math.max(i - 1, 0))
      } else if (currentStep === "selectDirectory") {
        setSelectedDirIndex(i => Math.max(i - 1, 0))
      } else if (currentStep === "selectType") {
        setSelectedTypeIndex(i => Math.max(i - 1, 0))
      } else if (currentStep === "confirm") {
        setConfirmIndex(i => Math.max(i - 1, 0))
      }
      return true
    }

    // Selection
    if (evt.name === "return" || evt.name === "enter") {
      if (currentStep === "selectFile") {
        const option = fileOptions()[selectedFileIndex()]
        if (option === CREATE_NEW_OPTION) {
          setStep("selectDirectory")
          return true
        }
        // Expand the path from display format
        setSelectedFile(expandPath(option))
        setStep("selectType")
        return true
      }
      
      if (currentStep === "selectDirectory") {
        setStep("confirmCreate")
        return true
      }
      
      if (currentStep === "confirmCreate") {
        if (handleCreateFile()) {
          // Optionally open in editor
          const created = createdFilePath()
          if (created) {
            openInEditor(created)
          }
          setStep("selectType")
        }
        return true
      }
      
      if (currentStep === "selectType") {
        setSelectedType(JOB_TYPES[selectedTypeIndex()])
        // Start deployment
        void handleDeploy()
        return true
      }
      
      if (currentStep === "confirm") {
        if (confirmIndex() === 0) {
          handleSubmitJob()
        } else {
          props.onClose()
        }
        return true
      }
    }

    // Go back
    if (evt.name === "escape" || evt.name === "q") {
      if (currentStep === "selectFile") {
        props.onClose()
      } else if (currentStep === "selectDirectory") {
        setStep("selectFile")
        setDeployError(null)
      } else if (currentStep === "confirmCreate") {
        setStep("selectDirectory")
      } else if (currentStep === "selectType") {
        if (createdFilePath()) {
          setStep("confirmCreate")
        } else {
          setStep("selectFile")
        }
        setDeployError(null)
      } else if (currentStep === "confirm") {
        setStep("selectType")
      }
      return true
    }

    return false
  }

  // Export the key handler for the parent to use
  ;(CreateJobModal as any).handleKeyPress = handleKeyPress

  const stepTitle = createMemo(() => {
    const currentStep = step()
    if (currentStep === "selectFile") return "Create New Job - Select LocalAPI"
    if (currentStep === "selectDirectory") return "Create New Job - Select Directory"
    if (currentStep === "confirmCreate") return "Create New Job - Create File"
    if (currentStep === "selectType") return "Create New Job - Select Type"
    if (currentStep === "deploying") return "Create New Job - Deploying..."
    return "Create New Job - Confirm"
  })

  const stepHint = createMemo(() => {
    const currentStep = step()
    if (isDeploying()) return "Deploying..."
    if (currentStep === "selectFile") return "j/k navigate | Enter select | q cancel"
    if (currentStep === "selectDirectory") return "j/k navigate | Enter select | Esc back"
    if (currentStep === "confirmCreate") return "Enter to create | Esc back"
    if (currentStep === "selectType") return "j/k navigate | Enter deploy | Esc back"
    return "j/k navigate | Enter confirm | Esc back"
  })

  const stepNumber = createMemo(() => {
    const currentStep = step()
    const isCreating = createdFilePath() !== null || step() === "selectDirectory" || step() === "confirmCreate"
    if (currentStep === "selectFile") return isCreating ? "1/5" : "1/4"
    if (currentStep === "selectDirectory") return "2/5"
    if (currentStep === "confirmCreate") return "3/5"
    if (currentStep === "selectType") return isCreating ? "4/5" : "2/4"
    if (currentStep === "deploying") return isCreating ? "4/5" : "3/4"
    return isCreating ? "5/5" : "4/4"
  })

  // Get proposed file path for create confirmation
  const proposedFilePath = createMemo(() => {
    const dirOption = directoryOptions()[selectedDirIndex()]
    const dirPath = dirOption.startsWith(CWD_OPTION_PREFIX)
      ? expandPath(dirOption.slice(CWD_OPTION_PREFIX.length))
      : expandPath(dirOption)
    return getUniqueFilename(dirPath, "localapi", ".py")
  })

  return (
    <Show when={props.visible}>
      <box
        position="absolute"
        width={props.width}
        height={props.height}
        left={Math.floor((process.stdout.columns - props.width) / 2)}
        top={Math.floor((process.stdout.rows - props.height) / 2)}
        border
        borderStyle="single"
        borderColor={COLORS.success}
        backgroundColor={COLORS.bg}
        flexDirection="column"
        paddingLeft={2}
        paddingRight={2}
        paddingTop={1}
        paddingBottom={1}
        zIndex={100}
      >
        {/* Header */}
        <box flexDirection="row">
          <text fg={COLORS.success}>{stepTitle()}</text>
          <box flexGrow={1} />
          <text fg={COLORS.textDim}>[{stepNumber()}]</text>
        </box>
        <box height={1} />

        {/* Step 1: Select File */}
        <Show when={step() === "selectFile"}>
          <text fg={COLORS.text}>Select LocalAPI file:</text>
          <box height={1} />
          <For each={fileOptions()}>
            {(option, idx) => {
              const isSelected = idx() === selectedFileIndex()
              const isCreateNew = option === CREATE_NEW_OPTION
              return (
                <text 
                  fg={isSelected ? COLORS.textSelected : (isCreateNew ? COLORS.textAccent : COLORS.text)}
                  bg={isSelected ? COLORS.bgSelection : undefined}
                >
                  {isSelected ? "> " : "  "}{option}
                </text>
              )
            }}
          </For>
        </Show>

        {/* Step 2: Select Directory (for new file) */}
        <Show when={step() === "selectDirectory"}>
          <text fg={COLORS.text}>Where to save the new LocalAPI file?</text>
          <box height={1} />
          <For each={directoryOptions()}>
            {(option, idx) => {
              const isSelected = idx() === selectedDirIndex()
              return (
                <text 
                  fg={isSelected ? COLORS.textSelected : COLORS.text}
                  bg={isSelected ? COLORS.bgSelection : undefined}
                >
                  {isSelected ? "> " : "  "}{option}
                </text>
              )
            }}
          </For>
        </Show>

        {/* Step 3: Confirm Create */}
        <Show when={step() === "confirmCreate"}>
          <text fg={COLORS.text}>Create new LocalAPI file?</text>
          <box height={1} />
          <text fg={COLORS.textDim}>  File: {toDisplayPath(proposedFilePath())}</text>
          <box height={1} />
          <text fg={COLORS.success}>Press Enter to create and open in editor</text>
        </Show>

        {/* Step 4: Select Type */}
        <Show when={step() === "selectType"}>
          <text fg={COLORS.text}>
            Select job type for: {toDisplayPath(effectiveFilePath() ?? "")}
          </text>
          <box height={1} />
          <For each={JOB_TYPES}>
            {(type, idx) => {
              const isSelected = idx() === selectedTypeIndex()
              return (
                <box flexDirection="column">
                  <text 
                    fg={isSelected ? COLORS.textSelected : COLORS.text}
                    bg={isSelected ? COLORS.bgSelection : undefined}
                  >
                    {isSelected ? "> " : "  "}{type.label}
                  </text>
                  <Show when={isSelected}>
                    <text fg={COLORS.textDim}>    {type.description}</text>
                  </Show>
                </box>
              )
            }}
          </For>
          <Show when={deployError()}>
            <box height={1} />
            <text fg={COLORS.error}>Error: {deployError()}</text>
          </Show>
        </Show>

        {/* Deploying */}
        <Show when={step() === "deploying"}>
          <text fg={COLORS.warning}>
            Deploying {toDisplayPath(effectiveFilePath() ?? "")}...
          </text>
          <box height={1} />
          <text fg={COLORS.textDim}>Please wait while the LocalAPI is deployed.</text>
        </Show>

        {/* Step 5: Confirm */}
        <Show when={step() === "confirm"}>
          <text fg={COLORS.text}>Ready to submit job:</text>
          <box height={1} />
          <text fg={COLORS.textDim}>  File: {toDisplayPath(effectiveFilePath() ?? "")}</text>
          <text fg={COLORS.textDim}>  Type: {selectedType()?.label}</text>
          <Show when={deployedUrl()}>
            <text fg={COLORS.textDim}>  URL: {deployedUrl()}</text>
          </Show>
          <box height={1} />
          <text 
            fg={confirmIndex() === 0 ? COLORS.textSelected : COLORS.text}
            bg={confirmIndex() === 0 ? COLORS.bgSelection : undefined}
          >
            {confirmIndex() === 0 ? "> " : "  "}Submit Job
          </text>
          <text 
            fg={confirmIndex() === 1 ? COLORS.textSelected : COLORS.text}
            bg={confirmIndex() === 1 ? COLORS.bgSelection : undefined}
          >
            {confirmIndex() === 1 ? "> " : "  "}Cancel
          </text>
        </Show>

        <box flexGrow={1} />
        <text fg={COLORS.textDim}>{stepHint()}</text>
      </box>
    </Show>
  )
}
