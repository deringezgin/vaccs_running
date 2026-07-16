from __future__ import annotations

import curses

from .summaries import filter_running_jobs
from ..slurm import (
    Job,
    JobRecordGroup,
    Node,
    PriorityQueueEntry,
    group_job_records,
    record_from_job,
)


class NavigationMixin:
    def _filter_priority_entries(
        self,
        entries: tuple[PriorityQueueEntry, ...] | list[PriorityQueueEntry],
    ) -> list[PriorityQueueEntry]:
        partitions = self.state.priority_partitions
        if not partitions:
            return list(entries)
        return [entry for entry in entries if entry.job.partition in partitions]

    def _visible_priority_grouped_entries(self) -> list[PriorityQueueEntry]:
        snapshot = self.state.priority_queue
        if snapshot is None:
            return []
        # Fall back to my_jobs for compatibility with snapshots created before
        # cluster-wide packed groups were added.
        grouped = snapshot.grouped_entries or snapshot.my_jobs
        return self._filter_priority_entries(grouped)

    def _visible_priority_all_entries(self) -> list[PriorityQueueEntry]:
        snapshot = self.state.priority_queue
        if snapshot is None:
            return []
        return self._filter_priority_entries(snapshot.all_entries)

    def _visible_priority_entries(self) -> list[PriorityQueueEntry]:
        if self.state.priority_extended:
            return self._visible_priority_all_entries()
        return self._visible_priority_grouped_entries()

    def _visible_jobs(self) -> list[Job]:
        if self._jobs_filter_active():
            return self.state.jobs

        return filter_running_jobs(self.state.jobs)

    def _visible_job_groups(self) -> list[JobRecordGroup]:
        if self._jobs_filter_active() and not self.state.job_records:
            return group_job_records(record_from_job(job) for job in self.state.jobs)
        return group_job_records(self.state.job_records)

    def _visible_history_groups(self) -> list[JobRecordGroup]:
        return group_job_records(self.state.history)

    def _visible_nodes(self) -> list[Node]:
        visible = self.state.nodes
        if self.state.gpu_nodes_only:
            visible = [node for node in visible if node.has_gpus]
        if self.state.free_gpu_only:
            visible = [node for node in visible if node.gpu_free > 0]
        return visible

    def _visible_count(self) -> int:
        if self.state.view in {"leaderboard", "info"}:
            return 0
        if self.state.view == "priority":
            return len(self._visible_priority_entries())
        if self.state.view == "nodes":
            return len(self._visible_nodes())
        if self.state.view == "history":
            return len(self._visible_history_groups())
        if self.state.jobs_grouped:
            return len(self._visible_job_groups())
        return len(self._visible_jobs())

    def _page_size(self, stdscr: curses.window) -> int:
        height, _ = stdscr.getmaxyx()
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        return max(1, table_height - 3)

    def _jump_page(self, stdscr: curses.window, direction: int) -> None:
        count = self._visible_count()
        if count == 0:
            self.state.selected = 0
            self.state.scroll = 0
            return

        page_size = self._page_size(stdscr)
        current_page_top = (self.state.selected // page_size) * page_size
        next_page_top = current_page_top + direction * page_size
        last_page_top = ((count - 1) // page_size) * page_size
        next_page_top = max(0, min(next_page_top, last_page_top))
        self.state.selected = next_page_top
        self.state.scroll = next_page_top

    def _switch_view(self, view: str) -> None:
        if self.state.view == view:
            return
        self.state.view = view
        self.state.selected = 0
        self.state.scroll = 0
        self.state.info_scroll = 0
        self._refresh_current()
        self._clamp_selection()

    def _clamp_selection(self) -> None:
        count = self._visible_count()
        if count == 0:
            self.state.selected = 0
            self.state.scroll = 0
            return
        self.state.selected = max(0, min(self.state.selected, count - 1))

    def _selected_job(self) -> Job | None:
        if self.state.jobs_grouped:
            return None
        visible = self._visible_jobs()
        if not visible:
            return None
        self._clamp_selection()
        return visible[self.state.selected]

    def _selected_job_group(self) -> JobRecordGroup | None:
        if not self.state.jobs_grouped:
            return None
        visible = self._visible_job_groups()
        if not visible:
            return None
        self._clamp_selection()
        return visible[self.state.selected]

    def _selected_history_group(self) -> JobRecordGroup | None:
        visible = self._visible_history_groups()
        if not visible:
            return None
        self._clamp_selection()
        return visible[self.state.selected]

    def _selected_node(self) -> Node | None:
        visible = self._visible_nodes()
        if not visible:
            return None
        self._clamp_selection()
        return visible[self.state.selected]

    def _selected_priority_entry(self) -> PriorityQueueEntry | None:
        visible = self._visible_priority_entries()
        if not visible:
            return None
        self._clamp_selection()
        return visible[self.state.selected]
