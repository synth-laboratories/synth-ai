import { ErrorBoundary, Show, type Accessor, type Component } from "solid-js"
import { Dynamic } from "solid-js/web"

import type { AppData, ActivePane, FocusTarget, PrincipalPane } from "../../types"
import type { LayoutMetrics } from "../layout"
import { ListPane } from "../../types"
import { JobsList } from "./list-panels/JobsList"
import { LogsList } from "./list-panels/LogsList"
import { JobsDetail } from "./detail-panels/JobsDetail"
import { LogsDetail } from "./detail-panels/LogsDetail"
import { COLORS } from "../theme"
import type { JobsDetailState } from "../hooks/useJobsDetailState"
import type { JobsListState } from "../hooks/useJobsListState"
import type { LogsDetailView } from "../hooks/useLogsDetailState"
import type { LogsListState } from "../hooks/useLogsListState"
import { logError } from "../../utils/log"

type AppBodyProps = {
	layout: Accessor<LayoutMetrics>
	activePane: Accessor<ActivePane>
	principalPane: Accessor<PrincipalPane>
	focusTarget: Accessor<FocusTarget>
	jobsList: JobsListState
	logsList: LogsListState
	jobsDetail: JobsDetailState
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
	onExitOpenCode: () => void
}

export function AppBody(props: AppBodyProps) {
	return (
		<box
			flexDirection="row"
			height={props.layout().contentHeight}
			flexGrow={1}
			border={false}
		>
			<Show
				when={props.activePane() === ListPane.Jobs}
				fallback={
					<LogsList
						items={props.logsList.listWindow.visibleItems()}
						selectedIndex={props.logsList.selectedIndex()}
						focused={props.focusTarget() === "list"}
						width={props.layout().jobsWidth}
						height={props.layout().contentHeight}
						title={props.logsList.listTitle()}
						totalCount={props.logsList.totalCount()}
					/>
				}
			>
				<JobsList
					items={props.jobsList.listWindow.visibleItems()}
					selectedIndex={props.jobsList.selectedIndex()}
					focused={props.focusTarget() === "list"}
					width={props.layout().jobsWidth}
					height={props.layout().contentHeight}
					title={props.jobsList.title()}
					totalCount={props.jobsList.totalCount()}
				/>
			</Show>

			<Show
				when={props.principalPane() === "jobs"}
				fallback={
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
									onExit={props.onExitOpenCode}
								/>
							</Show>
						</ErrorBoundary>
					</box>
				}
			>
      <Show
        when={props.activePane() !== ListPane.Logs}
					fallback={
						<LogsDetail
							title={props.logsList.filesTitle()}
							filePath={props.logsList.selectedFile()?.path ?? null}
							lines={props.logsDetailView().lines}
							visibleLines={props.logsDetailView().visibleLines}
							focused={props.focusTarget() === "logs-detail"}
							tail={props.logsDetailTail()}
							offset={props.logsDetailView().offset}
							maxOffset={props.logsDetailView().maxOffset}
						/>
					}
				>
					<JobsDetail
						data={props.data}
						events={props.jobsDetail.events()}
						eventWindow={props.jobsDetail.eventWindow()}
						lastError={props.lastError()}
						detailWidth={props.layout().detailWidth}
						detailHeight={props.layout().contentHeight}
						resultsFocused={props.focusTarget() === "results"}
						eventsFocused={props.focusTarget() === "events"}
						metricsFocused={props.focusTarget() === "metrics"}
						metricsView={props.metricsView()}
						verifierEvolveGenerationIndex={props.verifierEvolveGenerationIndex()}
					/>
				</Show>
			</Show>
		</box>
	)
}
