from __future__ import annotations

import threading

from ..slurm import (
    LEADERBOARD_WINDOWS,
    LeaderboardRow,
    SlurmError,
    build_group_leaderboard,
    build_user_leaderboard,
    sort_leaderboard,
)


class LeaderboardDataMixin:
    def _ensure_leaderboard_loaded(self) -> None:
        if not self._lb_started:
            self._start_leaderboard_refresh()

    def _leaderboard_loading(self) -> bool:
        with self._lb_lock:
            return any(
                window["status"] == "loading"
                for window in self._lb_windows.values()
            )

    def _start_leaderboard_refresh(self) -> bool:
        """Spawn a fetch thread per window. No-op while one is already running."""
        if self._leaderboard_loading():
            return False
        self._lb_started = True
        with self._lb_lock:
            self._lb_generation += 1
            generation = self._lb_generation
            # Drop the old fairshare so refreshed usage never renders against a
            # previous generation's scores (or stale scores if sshare later fails).
            self._lb_fairshare = {}
            for window, _label in LEADERBOARD_WINDOWS:
                self._lb_windows[window] = {
                    "status": "loading",
                    "usage": [],
                    "error": "",
                }
        self._lb_threads = []
        for window, _label in LEADERBOARD_WINDOWS:
            thread = threading.Thread(
                target=self._fetch_leaderboard_window,
                args=(generation, window),
                daemon=True,
            )
            thread.start()
            self._lb_threads.append(thread)
        fairshare_thread = threading.Thread(
            target=self._fetch_leaderboard_fairshare,
            args=(generation,),
            daemon=True,
        )
        fairshare_thread.start()
        self._lb_threads.append(fairshare_thread)
        return True

    def _fetch_leaderboard_window(self, generation: int, window: str) -> None:
        try:
            usage = self.client.fetch_usage_window(window)
            result = {"status": "ready", "usage": usage, "error": ""}
        except SlurmError as exc:
            result = {"status": "error", "usage": [], "error": str(exc)}
        except Exception as exc:  # never let a daemon thread die silently
            result = {"status": "error", "usage": [], "error": str(exc)}
        with self._lb_lock:
            if generation == self._lb_generation:
                self._lb_windows[window] = result

    def _fetch_leaderboard_fairshare(self, generation: int) -> None:
        try:
            fairshare = self.client.fetch_fairshare()
        except Exception:
            fairshare = None
        with self._lb_lock:
            if generation == self._lb_generation and fairshare is not None:
                self._lb_fairshare = fairshare

    def _leaderboard_snapshot(self) -> dict[str, dict[str, object]]:
        """Build the ranked rows for each window from the cached raw data."""
        with self._lb_lock:
            windows = {
                window: dict(info) for window, info in self._lb_windows.items()
            }
            fairshare = dict(self._lb_fairshare)
        build = (
            build_group_leaderboard
            if self.state.leaderboard_group_mode
            else build_user_leaderboard
        )
        needle = self.state.leaderboard_filter.strip().lower()
        snapshot: dict[str, dict[str, object]] = {}
        for window, _label in LEADERBOARD_WINDOWS:
            info = windows.get(
                window, {"status": "idle", "usage": [], "error": ""}
            )
            # Rows are (rank, row): the rank is the 1-based position in the full
            # sorted list and is assigned BEFORE filtering, so a filtered row
            # keeps its overall standing (e.g. the 32nd user still shows 32).
            rows: list[tuple[int, LeaderboardRow]] = []
            if info["status"] == "ready":
                ranked = list(
                    enumerate(
                        sort_leaderboard(
                            build(info["usage"], fairshare),
                            self.state.leaderboard_sort,
                            descending=not self.state.leaderboard_ascending,
                        ),
                        start=1,
                    )
                )
                if needle:
                    ranked = [
                        pair for pair in ranked if needle in pair[1].name.lower()
                    ]
                rows = ranked
            snapshot[window] = {
                "status": info["status"],
                "rows": rows,
                "error": info.get("error", ""),
            }
        return snapshot
