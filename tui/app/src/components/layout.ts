/**
 * Main UI layout component.
 */

import {
	BoxRenderable,
	TextRenderable,
	SelectRenderable,
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
	const logsTabText = createKeyHint(renderer, {
		id: "tabs-logs",
		description: "View Logs",
		key: "g"
	})
	tabsBox.add(newJobTabText)
	tabsBox.add(jobsTabText)
	tabsBox.add(eventsTabText)
	tabsBox.add(logsTabText)
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

	// Task Apps panel - shows active tunnels with health status
	const taskAppsBox = new BoxRenderable(renderer, {
		id: "task-apps-box",
		width: "auto",
		height: 6,
		borderStyle: "single",
		borderColor: "#334155",
		title: "Task Apps",
		titleAlignment: "left",
		border: true,
	})
	const taskAppsText = new TextRenderable(renderer, {
		id: "task-apps-text",
		content: "Loading task apps...",
		fg: "#e2e8f0",
	})
	taskAppsBox.add(taskAppsText)
	detailColumn.add(taskAppsBox)

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

	// Logs panel - shows streaming deployment logs
	const logsBox = new BoxRenderable(renderer, {
		id: "logs-box",
		width: "auto",
		height: "auto",
		flexGrow: 1,
		flexShrink: 1,
		borderStyle: "single",
		borderColor: "#334155",
		title: "Logs",
		titleAlignment: "left",
		border: true,
		visible: false, // Hidden by default, shown when logs pane is active
	})
	const logsContent = new BoxRenderable(renderer, {
		id: "logs-content",
		width: "auto",
		height: "auto",
		flexDirection: "column",
		flexGrow: 1,
		flexShrink: 1,
		gap: 0,
		border: false,
	})
	const logsEmptyText = new TextRenderable(renderer, {
		id: "logs-empty-text",
		content: "No log files found.",
		fg: "#94a3b8",
	})
	logsBox.add(logsContent)
	logsBox.add(logsEmptyText)
	detailColumn.add(logsBox)

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

	// Sessions modal (for OpenCode/Daytona connections)
	const sessionsModalBox = new BoxRenderable(renderer, {
		id: "sessions-modal-box",
		width: 70,
		height: 20,
		backgroundColor: "#1e293b",
		borderStyle: "single",
		borderColor: "#60a5fa",
		border: true,
		visible: false,
		zIndex: 10,
	})
	const sessionsModalTitle = new TextRenderable(renderer, {
		id: "sessions-modal-title",
		content: "OpenCode Sessions",
		fg: "#60a5fa",
		visible: false,
	})
	const sessionsModalText = new TextRenderable(renderer, {
		id: "sessions-modal-text",
		content: "",
		fg: "#e2e8f0",
		visible: false,
	})
	const sessionsModalHint = new TextRenderable(renderer, {
		id: "sessions-modal-hint",
		content: "j/k select | c connect | C custom URL | d disconnect | q close",
		fg: "#94a3b8",
		visible: false,
	})
	sessionsModalBox.add(sessionsModalTitle)
	sessionsModalBox.add(sessionsModalText)
	sessionsModalBox.add(sessionsModalHint)
	renderer.root.add(sessionsModalBox)

	return {
		// Main layout elements
		jobsBox,
		eventsBox,
		jobsSelect,
		detailBox,
		detailText,
		resultsBox,
		resultsText,
		metricsBox,
		metricsText,
		eventsList,
		eventsEmptyText,
		jobsTabText,
		eventsTabText,
		logsTabText,
		statusText,
		footerText: footerTextNode,

		// Task Apps panel
		taskAppsBox,
		taskAppsText,

		// Logs panel
		logsBox,
		logsContent,
		logsEmptyText,

		// Event cards (dynamically created)
		eventCards: [] as Array<{ box: BoxRenderable; text: TextRenderable }>,

		// Log entries (dynamically created)
		logEntries: [] as Array<{ text: TextRenderable }>,

		// Sessions modal
		sessionsModalBox,
		sessionsModalTitle,
		sessionsModalText,
		sessionsModalHint,
		sessionsModalVisible: false,

		// Detail column (for principal pane switching)
		detailColumn,
	}
}
