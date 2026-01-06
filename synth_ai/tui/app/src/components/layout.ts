/**
 * Main UI layout component.
 */

import {
	BoxRenderable,
	TextRenderable,
	SelectRenderable,
	InputRenderable,
	type CliRenderer,
} from "@opentui/core"
import { createKeyHint } from "./key-hint"

export type UI = ReturnType<typeof buildLayout>

export function buildLayout(renderer: CliRenderer, getFooterText: () => string) {
	const root = new BoxRenderable(renderer, {
		id: "root",
		width: "auto",
		height: "auto",
		flexGrow: 1,
		flexShrink: 1,
		flexDirection: "column",
		backgroundColor: "#0b1120",
		border: false,
	})
	renderer.root.add(root)

	const headerBox = new BoxRenderable(renderer, {
		id: "header-box",
		width: "auto",
		height: 3,
		backgroundColor: "#1e293b",
		borderStyle: "single",
		borderColor: "#334155",
		flexGrow: 0,
		flexShrink: 0,
		flexDirection: "row",
		border: true,
	})
	const headerText = new TextRenderable(renderer, {
		id: "header-text",
		content: "Synth AI",
		fg: "#e2e8f0",
	})
	headerBox.add(headerText)
	root.add(headerBox)

	const tabsBox = new BoxRenderable(renderer, {
		id: "tabs-box",
		width: "auto",
		height: 2,
		backgroundColor: "#111827",
		borderStyle: "single",
		borderColor: "#1f2937",
		flexDirection: "row",
		gap: 2,
		border: true,
	})
	const newJobTabText = createKeyHint(renderer, {
		id: "tabs-new-job",
		description: "Create New Job",
		key: "n"
	})
	const jobsTabText = createKeyHint(renderer, {
		id: "tabs-jobs",
		description: "View Jobs",
		key: "b",
		active: true
	})
	const eventsTabText = createKeyHint(renderer, {
		id: "tabs-events",
		description: "View Job's Events",
		key: "e"
	})
	tabsBox.add(newJobTabText)
	tabsBox.add(jobsTabText)
	tabsBox.add(eventsTabText)
	root.add(tabsBox)

	const main = new BoxRenderable(renderer, {
		id: "main",
		width: "auto",
		height: "auto",
		flexDirection: "row",
		flexGrow: 1,
		flexShrink: 1,
		border: false,
	})
	root.add(main)

	const jobsBox = new BoxRenderable(renderer, {
		id: "jobs-box",
		width: 36,
		height: "auto",
		minWidth: 36,
		flexGrow: 0,
		flexShrink: 0,
		borderStyle: "single",
		borderColor: "#334155",
		title: "Jobs",
		titleAlignment: "left",
		border: true,
	})
	const jobsSelect = new SelectRenderable(renderer, {
		id: "jobs-select",
		width: "auto",
		height: "auto",
		options: [],
		backgroundColor: "#0f172a",
		focusedBackgroundColor: "#1e293b",
		textColor: "#e2e8f0",
		focusedTextColor: "#f8fafc",
		selectedBackgroundColor: "#2563eb",
		selectedTextColor: "#ffffff",
		descriptionColor: "#94a3b8",
		selectedDescriptionColor: "#e2e8f0",
		showScrollIndicator: true,
		wrapSelection: true,
		showDescription: true,
		flexGrow: 1,
		flexShrink: 1,
	})
	jobsBox.add(jobsSelect)
	main.add(jobsBox)

	const detailColumn = new BoxRenderable(renderer, {
		id: "detail-column",
		width: "auto",
		height: "auto",
		flexDirection: "column",
		flexGrow: 2,
		flexShrink: 1,
		border: false,
	})
	main.add(detailColumn)

	const detailBox = new BoxRenderable(renderer, {
		id: "detail-box",
		width: "auto",
		height: 12,
		borderStyle: "single",
		borderColor: "#334155",
		title: "Details",
		titleAlignment: "left",
		border: true,
	})
	const detailText = new TextRenderable(renderer, {
		id: "detail-text",
		content: "No job selected.",
		fg: "#e2e8f0",
	})
	detailBox.add(detailText)
	detailColumn.add(detailBox)

	const resultsBox = new BoxRenderable(renderer, {
		id: "results-box",
		width: "auto",
		height: 6,
		borderStyle: "single",
		borderColor: "#334155",
		title: "Results",
		titleAlignment: "left",
		backgroundColor: "#0b1220",
		border: true,
	})
	const resultsText = new TextRenderable(renderer, {
		id: "results-text",
		content: "Results: -",
		fg: "#e2e8f0",
	})
	resultsBox.add(resultsText)
	detailColumn.add(resultsBox)

	const metricsBox = new BoxRenderable(renderer, {
		id: "metrics-box",
		width: "auto",
		height: 5,
		borderStyle: "single",
		borderColor: "#334155",
		title: "Metrics",
		titleAlignment: "left",
		border: true,
	})
	const metricsText = new TextRenderable(renderer, {
		id: "metrics-text",
		content: "Metrics: -",
		fg: "#cbd5f5",
	})
	metricsBox.add(metricsText)
	detailColumn.add(metricsBox)

	const eventsBox = new BoxRenderable(renderer, {
		id: "events-box",
		width: "auto",
		height: "auto",
		flexGrow: 1,
		flexShrink: 1,
		borderStyle: "single",
		borderColor: "#334155",
		title: "Events",
		titleAlignment: "left",
		border: true,
	})
	const eventsList = new BoxRenderable(renderer, {
		id: "events-list",
		width: "auto",
		height: "auto",
		flexDirection: "column",
		flexGrow: 1,
		flexShrink: 1,
		gap: 1,
		border: false,
	})
	const eventsEmptyText = new TextRenderable(renderer, {
		id: "events-empty-text",
		content: "No events yet.",
		fg: "#e2e8f0",
	})
	eventsBox.add(eventsList)
	eventsBox.add(eventsEmptyText)
	detailColumn.add(eventsBox)

	const statusBox = new BoxRenderable(renderer, {
		id: "status-box",
		width: "auto",
		height: 3,
		backgroundColor: "#0f172a",
		borderStyle: "single",
		borderColor: "#334155",
		flexGrow: 0,
		flexShrink: 0,
		border: true,
	})
	const statusText = new TextRenderable(renderer, {
		id: "status-text",
		content: "Ready.",
		fg: "#e2e8f0",
	})
	statusBox.add(statusText)
	root.add(statusBox)

	const footerBox = new BoxRenderable(renderer, {
		id: "footer-box",
		width: "auto",
		height: 2,
		backgroundColor: "#111827",
		flexGrow: 0,
		flexShrink: 0,
	})
	const footerTextNode = new TextRenderable(renderer, {
		id: "footer-text",
		content: getFooterText(),
		fg: "#94a3b8",
	})
	footerBox.add(footerTextNode)
	root.add(footerBox)

	// Snapshot modal
	const modalBox = new BoxRenderable(renderer, {
		id: "modal-box",
		width: 50,
		height: 5,
		position: "absolute",
		left: 4,
		top: 4,
		backgroundColor: "#0f172a",
		borderStyle: "single",
		borderColor: "#94a3b8",
		border: true,
		zIndex: 5,
	})
	const modalLabel = new TextRenderable(renderer, {
		id: "modal-label",
		content: "Snapshot ID:",
		fg: "#e2e8f0",
		position: "absolute",
		left: 6,
		top: 5,
		zIndex: 6,
	})
	const modalInput = new InputRenderable(renderer, {
		id: "modal-input",
		width: 44,
		height: 1,
		position: "absolute",
		left: 6,
		top: 6,
		placeholder: "Enter snapshot id",
		backgroundColor: "#111827",
		focusedBackgroundColor: "#1f2937",
		textColor: "#e2e8f0",
		focusedTextColor: "#ffffff",
	})
	modalBox.visible = false
	modalLabel.visible = false
	modalInput.visible = false
	renderer.root.add(modalBox)
	renderer.root.add(modalLabel)
	renderer.root.add(modalInput)

	// Event filter modal
	const filterBox = new BoxRenderable(renderer, {
		id: "filter-box",
		width: 52,
		height: 5,
		position: "absolute",
		left: 6,
		top: 6,
		backgroundColor: "#0f172a",
		borderStyle: "single",
		borderColor: "#60a5fa",
		border: true,
		zIndex: 5,
	})
	const filterLabel = new TextRenderable(renderer, {
		id: "filter-label",
		content: "Event filter:",
		fg: "#e2e8f0",
		position: "absolute",
		left: 8,
		top: 7,
		zIndex: 6,
	})
	const filterInput = new InputRenderable(renderer, {
		id: "filter-input",
		width: 46,
		height: 1,
		position: "absolute",
		left: 8,
		top: 8,
		placeholder: "Type to filter events",
		backgroundColor: "#111827",
		focusedBackgroundColor: "#1f2937",
		textColor: "#e2e8f0",
		focusedTextColor: "#ffffff",
	})
	filterBox.visible = false
	filterLabel.visible = false
	filterInput.visible = false
	renderer.root.add(filterBox)
	renderer.root.add(filterLabel)
	renderer.root.add(filterInput)

	// Job filter modal
	const jobFilterBox = new BoxRenderable(renderer, {
		id: "job-filter-box",
		width: 52,
		height: 11,
		position: "absolute",
		left: 6,
		top: 6,
		backgroundColor: "#0f172a",
		borderStyle: "single",
		borderColor: "#60a5fa",
		border: true,
		zIndex: 5,
	})
	const jobFilterLabel = new TextRenderable(renderer, {
		id: "job-filter-label",
		content: "Job filter (status: all)",
		fg: "#e2e8f0",
		position: "absolute",
		left: 8,
		top: 7,
		zIndex: 6,
	})
	const jobFilterHelp = new TextRenderable(renderer, {
		id: "job-filter-help",
		content: "Enter/space toggle | a select all | x clear | q close",
		fg: "#94a3b8",
		position: "absolute",
		left: 8,
		top: 8,
		zIndex: 6,
	})
	const jobFilterListText = new TextRenderable(renderer, {
		id: "job-filter-list",
		content: "",
		fg: "#e2e8f0",
		position: "absolute",
		left: 8,
		top: 9,
		zIndex: 6,
	})
	jobFilterBox.visible = false
	jobFilterLabel.visible = false
	jobFilterHelp.visible = false
	jobFilterListText.visible = false
	renderer.root.add(jobFilterBox)
	renderer.root.add(jobFilterLabel)
	renderer.root.add(jobFilterHelp)
	renderer.root.add(jobFilterListText)

	// Event detail modal
	const eventModalBox = new BoxRenderable(renderer, {
		id: "event-modal-box",
		width: 80,
		height: 16,
		position: "absolute",
		left: 6,
		top: 6,
		backgroundColor: "#0b1220",
		borderStyle: "single",
		borderColor: "#60a5fa",
		border: true,
		zIndex: 6,
	})
	const eventModalTitle = new TextRenderable(renderer, {
		id: "event-modal-title",
		content: "Event details",
		fg: "#e2e8f0",
		position: "absolute",
		left: 8,
		top: 7,
		zIndex: 7,
	})
	const eventModalText = new TextRenderable(renderer, {
		id: "event-modal-text",
		content: "",
		fg: "#e2e8f0",
		position: "absolute",
		left: 8,
		top: 8,
		zIndex: 7,
	})
	const eventModalHint = new TextRenderable(renderer, {
		id: "event-modal-hint",
		content: "Event details",
		fg: "#94a3b8",
		position: "absolute",
		left: 8,
		top: 9,
		zIndex: 7,
	})
	eventModalBox.visible = false
	eventModalTitle.visible = false
	eventModalText.visible = false
	eventModalHint.visible = false
	renderer.root.add(eventModalBox)
	renderer.root.add(eventModalTitle)
	renderer.root.add(eventModalText)
	renderer.root.add(eventModalHint)

	// Results modal
	const resultsModalBox = new BoxRenderable(renderer, {
		id: "results-modal-box",
		width: 100,
		height: 24,
		position: "absolute",
		left: 6,
		top: 4,
		backgroundColor: "#0b1220",
		borderStyle: "single",
		borderColor: "#22c55e",
		border: true,
		zIndex: 8,
	})
	const resultsModalTitle = new TextRenderable(renderer, {
		id: "results-modal-title",
		content: "Results - Best Prompt",
		fg: "#22c55e",
		position: "absolute",
		left: 8,
		top: 5,
		zIndex: 9,
	})
	const resultsModalText = new TextRenderable(renderer, {
		id: "results-modal-text",
		content: "",
		fg: "#e2e8f0",
		position: "absolute",
		left: 8,
		top: 6,
		zIndex: 9,
	})
	const resultsModalHint = new TextRenderable(renderer, {
		id: "results-modal-hint",
		content: "Results | j/k scroll | esc/q/enter close",
		fg: "#94a3b8",
		position: "absolute",
		left: 8,
		top: 26,
		zIndex: 9,
	})
	resultsModalBox.visible = false
	resultsModalTitle.visible = false
	resultsModalText.visible = false
	resultsModalHint.visible = false
	renderer.root.add(resultsModalBox)
	renderer.root.add(resultsModalTitle)
	renderer.root.add(resultsModalText)
	renderer.root.add(resultsModalHint)

	// Config modal
	const configModalBox = new BoxRenderable(renderer, {
		id: "config-modal-box",
		width: 100,
		height: 24,
		position: "absolute",
		left: 6,
		top: 4,
		backgroundColor: "#0b1220",
		borderStyle: "single",
		borderColor: "#f59e0b",
		border: true,
		zIndex: 8,
	})
	const configModalTitle = new TextRenderable(renderer, {
		id: "config-modal-title",
		content: "Job Configuration",
		fg: "#f59e0b",
		position: "absolute",
		left: 8,
		top: 5,
		zIndex: 9,
	})
	const configModalText = new TextRenderable(renderer, {
		id: "config-modal-text",
		content: "",
		fg: "#e2e8f0",
		position: "absolute",
		left: 8,
		top: 6,
		zIndex: 9,
	})
	const configModalHint = new TextRenderable(renderer, {
		id: "config-modal-hint",
		content: "Config | j/k scroll | esc/q/enter close",
		fg: "#94a3b8",
		position: "absolute",
		left: 8,
		top: 26,
		zIndex: 9,
	})
	configModalBox.visible = false
	configModalTitle.visible = false
	configModalText.visible = false
	configModalHint.visible = false
	renderer.root.add(configModalBox)
	renderer.root.add(configModalTitle)
	renderer.root.add(configModalText)
	renderer.root.add(configModalHint)

	// Prompt browser modal
	const promptBrowserBox = new BoxRenderable(renderer, {
		id: "prompt-browser-box",
		width: 100,
		height: 24,
		position: "absolute",
		left: 6,
		top: 4,
		backgroundColor: "#0b1220",
		borderStyle: "single",
		borderColor: "#a855f7",
		border: true,
		zIndex: 10,
	})
	const promptBrowserTitle = new TextRenderable(renderer, {
		id: "prompt-browser-title",
		content: "Prompt Browser",
		fg: "#a855f7",
		position: "absolute",
		left: 8,
		top: 5,
		zIndex: 11,
	})
	const promptBrowserText = new TextRenderable(renderer, {
		id: "prompt-browser-text",
		content: "",
		fg: "#e2e8f0",
		position: "absolute",
		left: 8,
		top: 6,
		zIndex: 11,
	})
	const promptBrowserHint = new TextRenderable(renderer, {
		id: "prompt-browser-hint",
		content: "Prompts | h/l prev/next | j/k scroll | y copy | esc close",
		fg: "#94a3b8",
		position: "absolute",
		left: 8,
		top: 26,
		zIndex: 11,
	})
	promptBrowserBox.visible = false
	promptBrowserTitle.visible = false
	promptBrowserText.visible = false
	promptBrowserHint.visible = false
	renderer.root.add(promptBrowserBox)
	renderer.root.add(promptBrowserTitle)
	renderer.root.add(promptBrowserText)
	renderer.root.add(promptBrowserHint)

	// Key input modal
	const keyModalBox = new BoxRenderable(renderer, {
		id: "key-modal-box",
		width: 70,
		height: 7,
		position: "absolute",
		left: 8,
		top: 8,
		backgroundColor: "#0b1220",
		borderStyle: "single",
		borderColor: "#7dd3fc",
		border: true,
		zIndex: 10,
	})
	const keyModalLabel = new TextRenderable(renderer, {
		id: "key-modal-label",
		content: "Set API key (saved for this session only)",
		fg: "#7dd3fc",
		position: "absolute",
		left: 10,
		top: 9,
		zIndex: 11,
	})
	const keyModalInput = new InputRenderable(renderer, {
		id: "key-modal-input",
		width: 62,
		height: 1,
		position: "absolute",
		left: 10,
		top: 10,
		backgroundColor: "#0f172a",
		borderStyle: "single",
		borderColor: "#1d4ed8",
		border: true,
		fg: "#e2e8f0",
		zIndex: 11,
	})
	const keyModalHelp = new TextRenderable(renderer, {
		id: "key-modal-help",
		content: "Paste any way | enter save | q close | empty clears",
		fg: "#94a3b8",
		position: "absolute",
		left: 10,
		top: 12,
		zIndex: 11,
	})
	keyModalBox.visible = false
	keyModalLabel.visible = false
	keyModalInput.visible = false
	keyModalHelp.visible = false
	renderer.root.add(keyModalBox)
	renderer.root.add(keyModalLabel)
	renderer.root.add(keyModalInput)
	renderer.root.add(keyModalHelp)

	return {
		// Main layout elements
		jobsBox,
		eventsBox,
		jobsSelect,
		detailText,
		resultsText,
		metricsText,
		eventsList,
		eventsEmptyText,
		jobsTabText,
		eventsTabText,
		statusText,
		footerText: footerTextNode,

		// Snapshot modal
		modalBox,
		modalLabel,
		modalInput,
		modalVisible: false,

		// Event filter modal
		filterBox,
		filterLabel,
		filterInput,
		filterModalVisible: false,

		// Job filter modal
		jobFilterBox,
		jobFilterLabel,
		jobFilterHelp,
		jobFilterListText,
		jobFilterModalVisible: false,

		// Event detail modal
		eventModalBox,
		eventModalTitle,
		eventModalText,
		eventModalHint,
		eventModalVisible: false,
		eventModalPayload: "",

		// Results modal
		resultsModalBox,
		resultsModalTitle,
		resultsModalText,
		resultsModalHint,
		resultsModalVisible: false,
		resultsModalPayload: "",

		// Config modal
		configModalBox,
		configModalTitle,
		configModalText,
		configModalHint,
		configModalVisible: false,
		configModalPayload: "",

		// Prompt browser modal
		promptBrowserBox,
		promptBrowserTitle,
		promptBrowserText,
		promptBrowserHint,
		promptBrowserVisible: false,

		// Key modal
		keyModalBox,
		keyModalLabel,
		keyModalInput,
		keyModalHelp,
		keyModalVisible: false,

		// Event cards (dynamically created)
		eventCards: [] as Array<{ box: BoxRenderable; text: TextRenderable }>,
	}
}
