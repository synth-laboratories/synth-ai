import { createEffect, type Accessor, type Setter } from "solid-js"

import type { ActivePane, Snapshot } from "../../types"
import type { JobSummary } from "../../tui_data"

type UseJobSelectionOptions = {
  jobs: Accessor<JobSummary[]>
  selectedIndex: Accessor<number>
  setSelectedIndex: Setter<number>
  activePane: Accessor<ActivePane>
  snapshot: Accessor<Snapshot>
  onSelectJob: (jobId: string) => void
}

export function useJobSelection(options: UseJobSelectionOptions): void {
  createEffect(() => {
    const count = options.jobs().length
    if (count === 0) {
      options.setSelectedIndex(0)
      return
    }
    if (options.selectedIndex() >= count) {
      options.setSelectedIndex(count - 1)
    }
  })

  // Auto-select job when highlighted index changes (only in jobs pane).
  createEffect(() => {
    if (options.activePane() !== "jobs") return

    const index = options.selectedIndex()
    const jobsList = options.jobs()
    const currentSnapshot = options.snapshot()
    const currentSelected = currentSnapshot.selectedJob

    if (jobsList.length === 0 || index < 0 || index >= jobsList.length) {
      return
    }

    const job = jobsList[index]
    if (!job?.job_id) {
      return
    }

    if (!currentSelected || currentSelected.job_id !== job.job_id) {
      options.onSelectJob(job.job_id)
    }
  })
}
