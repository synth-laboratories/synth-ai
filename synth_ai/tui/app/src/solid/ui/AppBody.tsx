import { ErrorBoundary, Show, type Accessor, type Component } from "solid-js"
import { Dynamic } from "solid-js/web"

import type { AppData, FocusTarget, PrimaryView } from "../../types"
import type { LayoutMetrics } from "../layout"
import { JobsList } from "./list-panels/JobsList"
import { LogsList } from "./list-panels/LogsList"
import { SessionsList } from "./list-panels/SessionsList"
import { JobsDetail } from "./detail-panels/JobsDetail"
import { LogsDetail } from "./detail-panels/LogsDetail"
import { COLORS } from "../theme"
import type { JobsDetailLayout } from "../hooks/useJobsDetailLayout"
import type { JobsDetailState } from "../hooks/useJobsDetailState"
import type { JobsListState } from "../hooks/useJobsListState"
import type { LogsDetailView } from "../hooks/useLogsDetailState"
import type { LogsListState } from "../hooks/useLogsListState"
import type { SessionsListState } from "../hooks/useSessionsListState"
import { logError } from "../../utils/log"

type AppBodyProps = {
	layout: Accessor<LayoutMetrics>
	primaryView: Accessor<PrimaryView>
	focusTarget: Accessor<FocusTarget>
	principalWidth: Accessor<number>
	principalHeight: Accessor<number>
	principalInnerWidth: Accessor<number>
	principalInnerHeight: Accessor<number>
	jobsList: JobsListState
	logsList: LogsListState
	sessionsList: SessionsListState
	jobsDetail: JobsDetailState
	jobsDetailLayout: Accessor<JobsDetailLayout>
	jobsDetailScrollOffset: Accessor<number>
	logsDetailView: Accessor<LogsDetailView>
	logsDetailTail: Accessor<boolean>
	metricsView: Accessor<"latest" | "charts">
	data: AppData
	lastError: Accessor<string | null>
	verifierEvolveGenerationIndex: Accessor<number>
	chatPaneComponent: Accessor<Component<any> | null>
	opencodeUrl: Accessor<string>
	opencodeSessionId: Accessor<string | undefined>
	opencodeDimensions: Accessor<{ width: number; height: number }>
	opencodeWorkingDir: Accessor<string | undefined>
	onExitOpenCode: () => void
}

export function AppBody(props: AppBodyProps) {
	const principalBorderColor = () => {
		if (props.primaryView() === "agent") {
			return COLORS.border
		}
		// Only highlight principal border when "principal" is focused (logs view)
		// For jobs view, child sections handle their own focus highlighting
		return props.focusTarget() === "principal" ? COLORS.borderAccent : COLORS.border
	}

	const principalTitle = () => {
		if (props.primaryView() === "agent") return "Agent"
		if (props.primaryView() !== "logs") return ""
		const view = props.logsDetailView()
		if (!view.lines.length) return "Logs"
		const current = view.offset + 1
		const end = Math.min(view.offset + view.visibleLines.length, view.lines.length)
		const tailIndicator = props.logsDetailTail() ? " [TAIL]" : ""
		return `Logs [${current}-${end}/${view.lines.length}]${tailIndicator}`
	}

	const principalTitleValue = () => principalTitle()

	return (
		<box
			flexDirection="row"
			height={props.layout().contentHeight}
			flexGrow={1}
			border={false}
		>
			<Show
				when={props.primaryView() === "agent"}
				fallback={
					<Show
						when={props.primaryView() === "logs"}
						fallback={
							<JobsList
								items={props.jobsList.listWindow.visibleItems()}
								selectedIndex={props.jobsList.selectedIndex()}
								focused={props.focusTarget() === "list"}
								width={props.layout().jobsWidth}
								height={props.layout().contentHeight}
								title={props.jobsList.title()}
								totalCount={props.jobsList.totalCount()}
								loadMoreHint={props.jobsList.loadMoreHint()}
							/>
						}
					>
						<LogsList
							items={props.logsList.listWindow.visibleItems()}
							selectedIndex={props.logsList.selectedIndex()}
							focused={props.focusTarget() === "list"}
							width={props.layout().jobsWidth}
							height={props.layout().contentHeight}
							title={props.logsList.listTitle()}
							totalCount={props.logsList.totalCount()}
						/>
					</Show>
				}
			>
				<SessionsList
					items={props.sessionsList.listWindow.visibleItems()}
					selectedIndex={props.sessionsList.selectedIndex()}
					focused={props.focusTarget() === "list"}
					width={props.layout().jobsWidth}
					height={props.layout().contentHeight}
					title={props.sessionsList.listTitle()}
					totalCount={props.sessionsList.totalCount()}
				/>
			</Show>

			<box
				flexDirection="column"
				border
				borderStyle="single"
				borderColor={principalBorderColor()}
				width={props.principalWidth()}
				height={props.principalHeight()}
				title={principalTitleValue()}
				titleAlignment="left"
			>
				<box
					flexDirection="column"
					width={props.principalInnerWidth()}
					height={props.principalInnerHeight()}
				>
					<Show
						when={props.primaryView() === "agent"}
						fallback={
							<Show
								when={props.primaryView() === "logs"}
								fallback={
									<JobsDetail
										data={props.data}
										eventItems={props.jobsDetail.listWindow.visibleItems()}
										totalEvents={props.jobsDetail.listWindow.total()}
										selectedIndex={props.jobsDetail.selectedIndex()}
										lastError={props.lastError()}
										detailWidth={props.principalInnerWidth()}
										detailHeight={props.principalInnerHeight()}
										scrollOffset={props.jobsDetailScrollOffset()}
										layout={props.jobsDetailLayout()}
										detailsFocused={props.focusTarget() === "details"}
										resultsFocused={props.focusTarget() === "results"}
										promptDiffFocused={props.focusTarget() === "promptDiff"}
										eventsFocused={props.focusTarget() === "events"}
										metricsFocused={props.focusTarget() === "metrics"}
										metricsView={props.metricsView()}
										verifierEvolveGenerationIndex={props.verifierEvolveGenerationIndex()}
									/>
								}
							>
								<LogsDetail
									title="Logs"
									filePath={props.logsList.selectedFile()?.path ?? null}
									lines={props.logsDetailView().lines}
									visibleLines={props.logsDetailView().visibleLines}
									focused={props.focusTarget() === "principal"}
									tail={props.logsDetailTail()}
									offset={props.logsDetailView().offset}
									maxOffset={props.logsDetailView().maxOffset}
									framed={false}
								/>
							</Show>
						}
					>
						<box flexDirection="column" flexGrow={1} border={false}>
							<ErrorBoundary
								fallback={(err) => {
									logError("OpenCode embed render failed", err)
									return (
										<box flexDirection="column" paddingLeft={2} paddingTop={1} gap={1}>
											<text fg={COLORS.error}>OpenCode embed failed to render.</text>
											<text fg={COLORS.textDim}>{String(err)}</text>
											<text fg={COLORS.textDim}>Try restarting the TUI or running opencode-synth tui standalone.</text>
										</box>
									)
								}}
							>
								<Show
									when={props.chatPaneComponent()}
									fallback={
										<box flexDirection="column" paddingLeft={2} paddingTop={1}>
											<text fg={COLORS.textDim}>Loading agent...</text>
										</box>
									}
								>
									<Dynamic
										component={props.chatPaneComponent() as Component<any>}
										url={props.opencodeUrl()}
										sessionId={props.opencodeSessionId()}
										width={props.opencodeDimensions().width}
										height={props.opencodeDimensions().height}
										workingDir={props.opencodeWorkingDir()}
										onExit={props.onExitOpenCode}
										focused={props.focusTarget() === "agent"}
										scrollFocused={props.focusTarget() === "conversation"}
										framed={false}
									/>
								</Show>
							</ErrorBoundary>
						</box>
					</Show>
				</box>
			</box>
		</box>
	)
}
