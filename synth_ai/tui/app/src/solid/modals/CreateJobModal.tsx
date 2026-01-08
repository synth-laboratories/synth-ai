/**
 * Create Job modal for SolidJS TUI.
 * 
 * Multi-step wizard for creating new jobs:
 * 1. Select LocalAPI file (or create new)
 * 2. Select job type
 * 3. Confirm and create
 */
import { For, Show, createMemo, createSignal, createEffect } from "solid-js"
import { COLORS } from "../theme"

export interface CreateJobModalProps {
  visible: boolean
  onClose: () => void
  onCreateJob: (options: CreateJobOptions) => void
  onDeploy?: (filePath: string) => Promise<{ success: boolean; url?: string; error?: string }>
  localApiFiles: string[]
  width: number
  height: number
}

export interface CreateJobOptions {
  localApiPath: string
  trainingType: "prompt_learning" | "eval" | "learning"
  deployedUrl?: string
}

type Step = "selectFile" | "selectType" | "deploy" | "confirm"

const CREATE_NEW_OPTION = "+ Create new LocalAPI file"

const TRAINING_TYPES = [
  { id: "eval" as const, label: "Evaluation", description: "Run evaluation on your LocalAPI" },
  { id: "prompt_learning" as const, label: "Prompt Learning", description: "Optimize prompts for better performance" },
  { id: "learning" as const, label: "Learning", description: "Train and optimize models" },
]

export function CreateJobModal(props: CreateJobModalProps) {
  const [step, setStep] = createSignal<Step>("selectFile")
  const [selectedFileIndex, setSelectedFileIndex] = createSignal(0)
  const [selectedTypeIndex, setSelectedTypeIndex] = createSignal(0)
  const [selectedFile, setSelectedFile] = createSignal<string | null>(null)
  const [selectedType, setSelectedType] = createSignal<typeof TRAINING_TYPES[number] | null>(null)
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
      setSelectedFile(null)
      setSelectedType(null)
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

  function handleKeyPress(evt: { name: string; shift?: boolean; ctrl?: boolean }): boolean {
    if (!props.visible) return false
    if (isDeploying()) return true // Block input while deploying

    const currentStep = step()

    // Navigation
    if (evt.name === "j" || evt.name === "down") {
      if (currentStep === "selectFile") {
        setSelectedFileIndex(i => Math.min(i + 1, fileOptions().length - 1))
      } else if (currentStep === "selectType") {
        setSelectedTypeIndex(i => Math.min(i + 1, TRAINING_TYPES.length - 1))
      } else if (currentStep === "confirm") {
        setConfirmIndex(i => Math.min(i + 1, 1))
      }
      return true
    }
    if (evt.name === "k" || evt.name === "up") {
      if (currentStep === "selectFile") {
        setSelectedFileIndex(i => Math.max(i - 1, 0))
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
          // TODO: Implement file creation flow
          setDeployError("File creation not yet implemented. Please create a LocalAPI file manually.")
          return true
        }
        setSelectedFile(option)
        setStep("selectType")
        return true
      }
      if (currentStep === "selectType") {
        setSelectedType(TRAINING_TYPES[selectedTypeIndex()])
        setStep("confirm")
        return true
      }
      if (currentStep === "confirm") {
        if (confirmIndex() === 0) {
          // Create job
          const file = selectedFile()
          const type = selectedType()
          if (file && type) {
            props.onCreateJob({
              localApiPath: file,
              trainingType: type.id,
              deployedUrl: deployedUrl() ?? undefined,
            })
          }
          props.onClose()
        } else {
          // Cancel
          props.onClose()
        }
        return true
      }
    }

    // Go back
    if (evt.name === "escape" || evt.name === "q") {
      if (currentStep === "selectFile") {
        props.onClose()
      } else if (currentStep === "selectType") {
        setStep("selectFile")
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
    if (currentStep === "selectType") return "Create New Job - Select Type"
    if (currentStep === "deploy") return "Create New Job - Deploying..."
    return "Create New Job - Confirm"
  })

  const stepHint = createMemo(() => {
    const currentStep = step()
    if (isDeploying()) return "Deploying..."
    if (currentStep === "selectFile") return "j/k navigate | Enter select | q cancel"
    if (currentStep === "selectType") return "j/k navigate | Enter select | Esc back"
    return "j/k navigate | Enter confirm | Esc back"
  })

  const stepNumber = createMemo(() => {
    const currentStep = step()
    if (currentStep === "selectFile") return "1/3"
    if (currentStep === "selectType") return "2/3"
    return "3/3"
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
          <Show when={deployError()}>
            <box height={1} />
            <text fg={COLORS.error}>{deployError()}</text>
          </Show>
        </Show>

        {/* Step 2: Select Type */}
        <Show when={step() === "selectType"}>
          <text fg={COLORS.text}>Select job type for: {selectedFile()}</text>
          <box height={1} />
          <For each={TRAINING_TYPES}>
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
        </Show>

        {/* Step 3: Confirm */}
        <Show when={step() === "confirm"}>
          <text fg={COLORS.text}>Confirm job creation:</text>
          <box height={1} />
          <text fg={COLORS.textDim}>  File: {selectedFile()}</text>
          <text fg={COLORS.textDim}>  Type: {selectedType()?.label}</text>
          <box height={1} />
          <text 
            fg={confirmIndex() === 0 ? COLORS.textSelected : COLORS.text}
            bg={confirmIndex() === 0 ? COLORS.bgSelection : undefined}
          >
            {confirmIndex() === 0 ? "> " : "  "}Create Job
          </text>
          <text 
            fg={confirmIndex() === 1 ? COLORS.textSelected : COLORS.text}
            bg={confirmIndex() === 1 ? COLORS.bgSelection : undefined}
          >
            {confirmIndex() === 1 ? "> " : "  "}Cancel
          </text>
        </Show>

        {/* Deploying state */}
        <Show when={step() === "deploy"}>
          <text fg={COLORS.warning}>Deploying {selectedFile()}...</text>
          <box height={1} />
          <text fg={COLORS.textDim}>Please wait...</text>
        </Show>

        <box flexGrow={1} />
        <text fg={COLORS.textDim}>{stepHint()}</text>
      </box>
    </Show>
  )
}
