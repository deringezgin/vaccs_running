from __future__ import annotations

import time

from ..slurm import SlurmError


class RefreshMixin:
    def _refresh_current(self) -> None:
        if self.state.view == "leaderboard":
            # First visit kicks off the background load; afterwards the cached
            # results are kept until the user presses 'r'.
            self._ensure_leaderboard_loaded()
            self.state.last_refresh = time.monotonic()
            return
        if self.state.view == "info":
            self._ensure_info_loaded()
            self.state.last_refresh = time.monotonic()
            return
        if self.state.view == "history":
            message = self._refresh_history()
        elif self.state.view == "nodes":
            message = self._refresh_nodes()
        elif self.state.view == "priority":
            message = self._refresh_priority()
        else:
            message = self._refresh_jobs()
        self.state.last_refresh = time.monotonic()
        self.state.message = f"refreshed {message}"
        self._clamp_selection()

    def _refresh_jobs(self) -> str:
        try:
            self.state.jobs, self.state.job_records = (
                self.client.fetch_active_job_records()
            )
            return f"{len(self.state.jobs)} jobs"
        except SlurmError as exc:
            return f"jobs: {exc}"

    def _refresh_nodes(self) -> str:
        try:
            self.state.nodes = self.client.fetch_nodes()
            return f"{len(self.state.nodes)} nodes"
        except SlurmError as exc:
            return f"nodes: {exc}"

    def _refresh_history(self) -> str:
        try:
            self.state.history = self.client.fetch_job_history(
                self.state.history_window
            )
            return f"{len(self.state.history)} tasks in {self.state.history_window}"
        except SlurmError as exc:
            return f"history: {exc}"

    def _refresh_priority(self) -> str:
        selected_key: tuple[str, str, str, str] | None = None
        selected_task_ids: set[str] = set()
        selected = self._selected_priority_entry()
        if selected is not None:
            selected_key = (
                selected.job.job_id,
                selected.job.array_parent,
                selected.job.partition,
                selected.job.normalized_reservation,
            )
            selected_task_ids = set(selected.task_job_ids) or {selected.job.job_id}
        try:
            snapshot = self.client.fetch_priority_queue()
        except SlurmError as exc:
            # Keep the last successful snapshot visible during a transient RPC
            # failure; the status message still exposes the error.
            return f"priority: {exc}"

        self.state.priority_queue = snapshot
        if selected_key is not None:
            job_id, parent, partition, reservation = selected_key
            fallback_index: int | None = None
            for index, entry in enumerate(self._visible_priority_entries()):
                if self.state.priority_extended:
                    matches = (
                        entry.job.job_id == job_id
                        and entry.job.partition == partition
                        and entry.job.normalized_reservation == reservation
                    )
                else:
                    entry_task_ids = set(entry.task_job_ids) or {entry.job.job_id}
                    matches = (
                        bool(selected_task_ids & entry_task_ids)
                        and entry.job.partition == partition
                        and entry.job.normalized_reservation == reservation
                    )
                    if fallback_index is None and (
                        entry.job.array_parent == parent
                        and entry.job.partition == partition
                        and entry.job.normalized_reservation == reservation
                    ):
                        fallback_index = index
                if matches:
                    self.state.selected = index
                    break
            else:
                if not self.state.priority_extended and fallback_index is not None:
                    self.state.selected = fallback_index
        suffix = ""
        if snapshot.factors_error:
            suffix = " (priority breakdown unavailable)"
        groups = len(self._visible_priority_grouped_entries())
        entries = len(self._visible_priority_all_entries())
        return f"{groups} packed rows / {entries} rank slots{suffix}"
