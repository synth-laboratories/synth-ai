/**
 * Create Job modal for SolidJS TUI.
 * 
 * Simplified version of the original wizard - provides basic job creation flow.
 */
import { For, Show, createMemo, createSignal } from "solid-js"
import { COLORS } from "../theme"

export interface CreateJobModalProps {
  visible: boolean
  onClose: () => void
  onCreateJob: (options: CreateJobOptions) => void
  localApiFiles: string[]
  width: number
  height: number
}

export interface CreateJobOptions {
  localApiPath: string
  trainingType: "prompt_learning" | "eval" | "learning"
}

type Step = "selectFile" | "selectType" | "confirm"

const TRAINING_TYPES = [
  { id: "prompt_learning" as const, label: "Prompt Learning", description: "Optimize prompts for better performance" },
  { id: "eval" as const, label: "Evaluation", description: "Evaluate model performance" },
  { id: "learning" as const, label: "Learning", description: "Train and optimize models" },
]

export function CreateJobModal(props: CreateJobModalProps) {
  const [step, setStep] = createSignal<Step>("selectFile")
  const [selectedFileIndex, setSelectedFileIndex] = createSignal(0)
  const [selectedTypeIndex, setSelectedTypeIndex] = createSignal(0)
  const [selectedFile, setSelectedFile] = createSignal<string | null>(null)
  const [selectedType, setSelectedType] = createSignal<typeof TRAINING_TYPES[number] | null>(null)

  const currentOptions = createMemo(() => {
    const currentStep = step()
    if (currentStep === "selectFile") {
      return props.localApiFiles.length > 0 
        ? props.localApiFiles 
        : ["No LocalAPI files found"]
    }
    if (currentStep === "selectType") {
      return TRAINING_TYPES.map(t => t.label)
    }
    return ["Create Job", "Cancel"]
  })

  const currentIndex = createMemo(() => {
    const currentStep = step()
    if (currentStep === "selectFile") return selectedFileIndex()
    if (currentStep === "selectType") return selectedTypeIndex()
    return 0
  })

  function handleKeyPress(evt: { name: string; shift?: boolean; ctrl?: boolean }) {
    if (!props.visible) return false

    // Navigation
    if (evt.name === "j" || evt.name === "down") {
      const options = currentOptions()
      if (step() === "selectFile") {
        setSelectedFileIndex(i => Math.min(i + 1, options.length - 1))
      } else if (step() === "selectType") {
        setSelectedTypeIndex(i => Math.min(i + 1, options.length - 1))
      }
      return true
    }
    if (evt.name === "k" || evt.name === "up") {
      if (step() === "selectFile") {
        setSelectedFileIndex(i => Math.max(i - 1, 0))
      } else if (step() === "selectType") {
        setSelectedTypeIndex(i => Math.max(i - 1, 0))
      }
      return true
    }

    // Selection
    if (evt.name === "return" || evt.name === "enter") {
      const currentStep = step()
      if (currentStep === "selectFile") {
        if (props.localApiFiles.length === 0) {
          props.onClose()
          return true
        }
        setSelectedFile(props.localApiFiles[selectedFileIndex()])
        setStep("selectType")
        return true
      }
      if (currentStep === "selectType") {
        setSelectedType(TRAINING_TYPES[selectedTypeIndex()])
        setStep("confirm")
        return true
      }
      if (currentStep === "confirm") {
        const file = selectedFile()
        const type = selectedType()
        if (file && type) {
          props.onCreateJob({
            localApiPath: file,
            trainingType: type.id,
          })
        }
        props.onClose()
        return true
      }
    }

    // Go back
    if (evt.name === "escape" || evt.name === "q") {
      const currentStep = step()
      if (currentStep === "selectFile") {
        props.onClose()
      } else if (currentStep === "selectType") {
        setStep("selectFile")
      } else {
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
    return "Create New Job - Confirm"
  })

  const stepHint = createMemo(() => {
    const currentStep = step()
    if (currentStep === "selectFile") return "j/k to navigate, Enter to select, q to cancel"
    if (currentStep === "selectType") return "j/k to navigate, Enter to select, Esc to go back"
    return "Enter to create, Esc to go back"
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
        <text fg={COLORS.success}>{stepTitle()}</text>
        <box height={1} />

        <Show when={step() === "selectFile"}>
          <text fg={COLORS.text}>Select LocalAPI file:</text>
          <box height={1} />
          <For each={currentOptions()}>
            {(option, idx) => {
              const isSelected = idx() === currentIndex()
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

        <Show when={step() === "selectType"}>
          <text fg={COLORS.text}>Select job type:</text>
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

        <Show when={step() === "confirm"}>
          <text fg={COLORS.text}>Confirm job creation:</text>
          <box height={1} />
          <text fg={COLORS.textDim}>File: {selectedFile()}</text>
          <text fg={COLORS.textDim}>Type: {selectedType()?.label}</text>
          <box height={1} />
          <text fg={COLORS.success}>Press Enter to create job, Esc to go back</text>
        </Show>

        <box flexGrow={1} />
        <text fg={COLORS.textDim}>{stepHint()}</text>
      </box>
    </Show>
  )
}


