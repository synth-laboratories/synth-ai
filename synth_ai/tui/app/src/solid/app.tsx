import { render, useRenderer, useTerminalDimensions } from "@opentui/solid"
import { createEffect, createMemo, createSignal, onMount, type Component } from "solid-js"

import { computeLayoutMetrics } from "./layout"
import { useSolidData } from "./data"
import { AppBody } from "./ui/AppBody"
import { AppHeader } from "./ui/chrome/AppHeader"
import { AppTabs } from "./ui/chrome/AppTabs"
import { StatusBar } from "./ui/chrome/StatusBar"
import { KeyFooter, type JobsLoadMoreState } from "./ui/chrome/KeyFooter"
import { AppModals } from "./ui/modals/AppModals"
import { toDisplayPath } from "./utils/files"
import { useActionRunner } from "./hooks/useActionRunner"
import { useAppKeybindings } from "./hooks/useAppKeybindings"
import { useAuthFlow } from "./hooks/useAuthFlow"
import { useDetailModal } from "./hooks/useDetailModal"
import { useFocusBindings } from "./hooks/useFocusBindings"
import { useJobEvents } from "./hooks/useJobEvents"
import { useJobsStream } from "./hooks/useJobsStream"
import { useJobsDetailState } from "./hooks/useJobsDetailState"
import { useJobsListState } from "./hooks/useJobsListState"
import { useLogsDetailState } from "./hooks/useLogsDetailState"
import { useLogsListState } from "./hooks/useLogsListState"
import { useModalComponents } from "./hooks/useModalComponents"
import { useModalStack } from "./hooks/useModalStack"
import { useOverlayModals } from "./hooks/useOverlayModals"
import { useStatusText } from "./hooks/useStatusText"

import { cancelSelected, fetchArtifacts, fetchMetrics, loadMoreJobs } from "../api/jobs"
import type { JobSummary } from "../tui_data"
import { bindFocusState } from "../focus"
import { installSignalHandlers, registerRenderer, shutdown } from "../lifecycle"
import { log, installGlobalErrorHandlers } from "../utils/log"

function wireShutdown(renderer: { stop: () => void; destroy: () => void }): void {
	registerRenderer(renderer)
	installSignalHandlers() // Safe to call multiple times
}

function getJobTimestamp(job: JobSummary): number {
	const value = job.created_at || job.started_at || job.finished_at
	if (!value) return 0
	const parsed = Date.parse(value)
	return Number.isFinite(parsed) ? parsed : 0
}

function countJobsCacheRemaining(jobs: JobSummary[], cache: JobSummary[]): number {
	if (!cache.length) return 0
	const seen = new Set(jobs.map((job) => job.job_id))
	const oldest = jobs[jobs.length - 1]
	const oldestTs = oldest ? getJobTimestamp(oldest) : 0
	let count = 0
	for (const job of cache) {
		if (!job.job_id || seen.has(job.job_id)) continue
		if (oldestTs && getJobTimestamp(job) > oldestTs) continue
		count += 1
	}
	return count
}

export async function runSolidApp(): Promise<void> {
	installGlobalErrorHandlers()
	log("lifecycle", "runSolidApp starting")
	return new Promise<void>((resolve) => {
		render(
			() => <SolidShell onExit={resolve} />,
			{
				targetFps: 30,
				exitOnCtrlC: false,
				useKittyKeyboard: {},
			},
		)
	})
}

function SolidShell(props: { onExit?: () => void }) {
	log("lifecycle", "SolidShell init")
	const { onExit } = props
	const dimensions = useTerminalDimensions()
	const renderer = useRenderer()
	wireShutdown(renderer)
	// Set global renderer for OpenCode embed to find
	; (globalThis as any).__OPENCODE_EMBED_RENDERER__ = renderer
	const layout = createMemo(() =>
		computeLayoutMetrics(dimensions().width, dimensions().height),
	)
	const data = useSolidData()
	const [chatPaneComponent, setChatPaneComponent] = createSignal<Component<any> | null>(null)
	let chatPaneLoading = false
	const modalComponents = useModalComponents()

	async function ensureChatPane(): Promise<void> {
		if (chatPaneComponent() || chatPaneLoading) return
		chatPaneLoading = true
		const mod = await import("./opencode")
		setChatPaneComponent(() => mod.ChatPane)
		chatPaneLoading = false
	}
	const appState = data.ctx.state.ui
	const appData = data.ctx.state.data
	const { setData, setUi } = data.ctx
	const actionRunner = useActionRunner({ setData })
	const { actions, runAction } = actionRunner
	bindFocusState({
		getFocusTarget: () => appState.focusTarget,
		setFocusTarget: (target) => setUi("focusTarget", target),
		getActivePane: () => appState.activePane,
		setActivePane: (pane) => setUi("activePane", pane),
	})
	onMount(() => {
		if (!process.env.SYNTH_TUI_BENCH) return
		const start = (globalThis as any).__TUI_BENCH_START
		const elapsed = typeof start === "number" ? Date.now() - start : null
		setTimeout(() => {
			const suffix = elapsed == null ? "" : ` ${elapsed}ms`
			process.stderr.write(`tui_first_render${suffix}\n`)
			onExit?.()
			void shutdown(0)
		}, 0)
	})
	const activePane = createMemo(() => appState.activePane)
	const focusTarget = createMemo(() => appState.focusTarget)
	const verifierEvolveGenerationIndex = createMemo(() => appState.verifierEvolveGenerationIndex)
	const principalPane = createMemo(() => appState.principalPane)
	const jobsStreamEnabled = createMemo(() => Boolean(appData.userId))
	const jobsList = useJobsListState({
		data: appData,
		ui: appState,
		activePane,
		height: () => layout().contentHeight,
		onSelectJob: (jobId) => {
			runAction("select-job", () => data.select(jobId))
		},
	})
	const jobsDetail = useJobsDetailState({
		data: appData,
		ui: appState,
		setUi,
		layoutHeight: () => layout().contentHeight,
	})
	const activeOpenCodeSession = createMemo(() => {
		const sessionId = appState.openCodeSessionId
		if (!sessionId) return null
		return appData.sessions.find((session) => session.session_id === sessionId) || null
	})
	const opencodeUrl = createMemo(() => {
		const session = activeOpenCodeSession()
		return (
			session?.opencode_url ||
			session?.access_url ||
			appState.openCodeUrl ||
			process.env.OPENCODE_URL ||
			"http://localhost:3000"
		)
	})
	const opencodeWorkingDir = createMemo(() => {
		const raw = (
			process.env.SYNTH_TUI_LAUNCH_CWD ||
			process.env.OPENCODE_WORKING_DIR ||
			process.env.INIT_CWD ||
			process.env.PWD ||
			process.cwd()
		) as string
		const trimmed = raw.trim()
		return trimmed || undefined
	})
	const opencodeSessionId = createMemo(() => appState.openCodeSessionId ?? undefined)
	createEffect(() => {
		const sessionID = opencodeSessionId()
		process.env.OPENCODE_ROUTE = JSON.stringify(
			sessionID ? { type: "session", sessionID } : { type: "home" },
		)
	})
	const opencodeDimensions = createMemo(() => ({
		width: Math.max(1, layout().detailWidth),
		height: Math.max(1, layout().contentHeight),
	}))
	createEffect(() => {
		if (principalPane() === "opencode") {
			void ensureChatPane()
		}
	})
	const committedJobId = createMemo(() => appData.selectedJob?.job_id ?? null)
	useJobEvents({
		ctx: data.ctx,
		selectedJobId: committedJobId,
		activePane,
		principalPane,
	})
	useJobsStream({
		ctx: data.ctx,
		enabled: jobsStreamEnabled,
	})
	const logsList = useLogsListState({
		activePane,
		principalPane,
		height: () => layout().contentHeight,
		ui: appState,
	})
	const logsDetail = useLogsDetailState({
		selectedFile: logsList.selectedFile,
		lines: logsList.liveLogs.lines,
		height: () => layout().contentHeight,
		ui: appState,
		setUi,
	})
	const logsDetailView = logsDetail.view
	const logsDetailTail = createMemo(() => appState.logsDetailTail)
	const metricsView = createMemo(() => appState.metricsView)
	const jobsCacheRemaining = createMemo(() =>
		countJobsCacheRemaining(appData.jobs, appData.jobsCache),
	)
	const jobsLoadMoreState = createMemo<JobsLoadMoreState>(() => {
		if (appState.jobsListLoadingMore) return "loading"
		if (appState.jobsListHasMore) return "server"
		return jobsCacheRemaining() > 0 ? "cache" : "none"
	})
	const hasSelectedJob = createMemo(() => Boolean(appData.selectedJob))

	const statusText = useStatusText({ data: appData, ui: appState })
	const lastError = createMemo(() => {
		return appData.lastError
	})
	const detailModal = useDetailModal({ layout })
	const modalStack = useModalStack({ abortAction: actions.abort })
	const authFlow = useAuthFlow({
		ui: appState,
		data: appData,
		setData,
		actions,
		openOverlayModal: modalStack.openOverlayModal,
		closeActiveModal: modalStack.closeActiveModal,
		refreshData: data.refresh,
	})
	const overlayModals = useOverlayModals({
		ctx: data.ctx,
		data: appData,
		ui: appState,
		setData,
		setUi,
		modalInputValue: modalStack.modalInputValue,
		setModalInputValue: modalStack.setModalInputValue,
		openOverlayModal: modalStack.openOverlayModal,
		closeActiveModal: modalStack.closeActiveModal,
		runAction,
		ensureCandidatesModal: modalComponents.ensureCandidatesModal,
		ensureGraphEvolveGenerationsModal: modalComponents.ensureGraphEvolveGenerationsModal,
		ensureTraceViewerModal: modalComponents.ensureTraceViewerModal,
		promptLogin: authFlow.promptLogin,
		refreshData: data.refresh,
		logFiles: logsList.liveLogs.files,
	})
	useFocusBindings({
		ctx: data.ctx,
		ui: appState,
		jobsList,
		logsList,
		jobsDetail,
		logsDetail,
		openEventModal: detailModal.openEventModal,
		openLogModal: detailModal.openLogModal,
		openMetricsAndFetch: overlayModals.openMetricsAndFetch,
		openCandidatesForGeneration: overlayModals.openCandidatesForGeneration,
	})

	createEffect(() => {
		const opencodeUrl = appState.openCodeUrl
		if (!opencodeUrl) {
			setUi("openCodeAutoConnectAttempted", false)
			return
		}
		if (appState.openCodeSessionId) return
		if (appState.openCodeAutoConnectAttempted) return
		setUi("openCodeAutoConnectAttempted", true)
		runAction("opencode-connect", () => overlayModals.connectLocalSession())
	})

	useAppKeybindings({
		onExit,
		data: appData,
		ui: appState,
		setUi,
		setData,
		modal: detailModal.modal,
		setModal: detailModal.setModal,
		activeModal: modalStack.activeModal,
		showCreateJobModal: modalComponents.showCreateJobModal,
		setShowCreateJobModal: modalComponents.setShowCreateJobModal,
		createJobModalComponent: modalComponents.createJobModalComponent,
		chatPaneComponent,
		closeActiveModal: modalStack.closeActiveModal,
		applyFilterModal: overlayModals.applyFilterModal,
		applySnapshotModal: overlayModals.applySnapshotModal,
		moveSettingsCursor: overlayModals.moveSettingsCursor,
		selectSettingsBackend: overlayModals.selectSettingsBackend,
		openUsageBilling: overlayModals.openUsageBilling,
		moveTaskAppsSelection: overlayModals.moveTaskAppsSelection,
		copySelectedTunnelUrl: overlayModals.copySelectedTunnelUrl,
		moveSessionsSelection: overlayModals.moveSessionsSelection,
		copySelectedSessionUrl: overlayModals.copySelectedSessionUrl,
		connectLocalSession: overlayModals.connectLocalSession,
		disconnectSelectedSession: overlayModals.disconnectSelectedSession,
		refreshSessionsModal: overlayModals.refreshSessionsModal,
		selectSession: overlayModals.selectSession,
		moveListFilter: overlayModals.moveListFilter,
		toggleListFilterSelection: overlayModals.toggleListFilterSelection,
		selectAllListFilterSelection: overlayModals.selectAllListFilterSelection,
		startLoginAuth: authFlow.startLoginAuth,
		openFilterModal: overlayModals.openFilterModal,
		openConfigModal: overlayModals.openConfigModal,
		openProfileModal: overlayModals.openProfileModal,
		openResultsModal: overlayModals.openResultsModal,
		openListFilterModal: overlayModals.openListFilterModal,
		openSnapshotModal: overlayModals.openSnapshotModal,
		openUrlsModal: overlayModals.openUrlsModal,
		openSettingsModal: overlayModals.openSettingsModal,
		openUsageModal: overlayModals.openUsageModal,
		openTaskAppsModal: overlayModals.openTaskAppsModal,
		openSessionsModal: overlayModals.openSessionsModal,
		openMetricsAndFetch: overlayModals.openMetricsAndFetch,
		openTracesModal: overlayModals.openTracesModal,
		ensureCreateJobModal: modalComponents.ensureCreateJobModal,
		logout: authFlow.logout,
		runAction,
		refreshData: data.refresh,
		ensureOpenCodeServer: data.ensureOpenCodeServer,
		cancelSelectedJob: async () => {
			await cancelSelected(data.ctx)
		},
		fetchArtifacts: async () => {
			await fetchArtifacts(data.ctx)
		},
		refreshMetrics: async () => {
			await fetchMetrics(data.ctx)
		},
		loadMoreJobs: async () => {
			await loadMoreJobs(data.ctx)
		},
	})

	return (
		<box
			width={layout().totalWidth}
			height={layout().totalHeight}
			flexDirection="column"
			backgroundColor="#0b1120"
		>
			<AppHeader />
			<AppTabs activePane={activePane} principalPane={principalPane} />

			<AppBody
				layout={layout}
				activePane={activePane}
				principalPane={principalPane}
				focusTarget={focusTarget}
				jobsList={jobsList}
				logsList={logsList}
				jobsDetail={jobsDetail}
				logsDetailView={logsDetailView}
				logsDetailTail={logsDetailTail}
				metricsView={metricsView}
				data={appData}
				lastError={lastError}
				verifierEvolveGenerationIndex={verifierEvolveGenerationIndex}
				chatPaneComponent={chatPaneComponent}
				opencodeUrl={opencodeUrl}
				opencodeSessionId={opencodeSessionId}
				opencodeDimensions={opencodeDimensions}
				opencodeWorkingDir={opencodeWorkingDir}
				onExitOpenCode={() => {
					setUi("principalPane", "jobs")
				}}
			/>

			<AppModals
				detail={{
					modal: detailModal.modal,
					modalLayout: detailModal.modalLayout,
					modalView: detailModal.modalView,
					modalHint: detailModal.modalHint,
				}}
				overlay={{
					activeModal: modalStack.activeModal,
					dimensions,
					ui: appState,
					setModalInputValue: modalStack.setModalInputValue,
					setModalInputRef: modalStack.setModalInputRef,
					settingsCursor: overlayModals.settingsCursor,
					usageData: overlayModals.usageData,
					sessionsCache: overlayModals.sessionsCache,
					sessionsHealthCache: overlayModals.sessionsHealthCache,
					sessionsSelectedIndex: overlayModals.sessionsSelectedIndex,
					sessionsScrollOffset: overlayModals.sessionsScrollOffset,
					loginStatus: authFlow.loginStatus,
					candidatesModalComponent: modalComponents.candidatesModalComponent,
					graphEvolveGenerationsModalComponent: modalComponents.graphEvolveGenerationsModalComponent,
					traceViewerModalComponent: modalComponents.traceViewerModalComponent,
					closeActiveModal: modalStack.closeActiveModal,
					onStatusUpdate: (message: string) => {
						setData("status", message)
					},
					openCandidatesForGeneration: overlayModals.openCandidatesForGeneration,
					data: appData,
				}}
				createJob={{
					createJobModalComponent: modalComponents.createJobModalComponent,
					showCreateJobModal: modalComponents.showCreateJobModal,
					onClose: () => modalComponents.setShowCreateJobModal(false),
					onJobCreated: (info) => {
						if (info.jobSubmitted) {
							setData("status", `${info.trainingType} job submitted for ${toDisplayPath(info.localApiPath)}`)
						} else if (info.deployedUrl) {
							setData("status", `Deployed: ${info.deployedUrl}`)
						} else {
							setData("status", `Ready to deploy: ${toDisplayPath(info.localApiPath)}`)
						}
						setData("lastError", null)
						runAction("refresh", () => data.refresh())
					},
					onStatusUpdate: (status: string) => {
						setData("status", status)
					},
					onError: (error: string) => {
						setData("lastError", error)
					},
					localApiFiles: modalComponents.localApiFiles,
					width: () => Math.min(70, layout().totalWidth - 4),
					height: () => Math.min(24, layout().totalHeight - 4),
					dimensions,
				}}
			/>

			<StatusBar statusText={statusText} />
			<KeyFooter
				principalPane={principalPane}
				activePane={activePane}
				focusTarget={focusTarget}
				jobsLoadMoreState={jobsLoadMoreState}
				hasSelectedJob={hasSelectedJob}
			/>
		</box>
	)
}
