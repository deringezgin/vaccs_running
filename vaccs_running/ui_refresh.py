from __future__ import annotations

import time

from .slurm import SlurmError


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
