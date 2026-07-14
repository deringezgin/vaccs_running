from __future__ import annotations

import curses
import threading
import time
from dataclasses import dataclass

# --- re-exported leaf helper modules (facade) ---
from .constants import *  # noqa: F401,F403
from .widgets import *  # noqa: F401,F403
from .text_layout import *  # noqa: F401,F403
from .table_layout import *  # noqa: F401,F403
from .summaries import *  # noqa: F401,F403
from .info_panel import *  # noqa: F401,F403
from .curses_compat import *  # noqa: F401,F403
from .filter_menu import JobFilterMenuMixin
from .keys import KeyHandlingMixin
from .popups import PopupMixin, command_text  # noqa: F401 (re-export)
from .render_frame import RenderFrameMixin
from .render_tables import RenderTablesMixin
from .render_detail import RenderDetailMixin
from .render_leaderboard import RenderLeaderboardMixin
from .leaderboard_data import LeaderboardDataMixin
from .info_data import InfoDataMixin
from .filter_state import JobFilterStateMixin
from .navigation import NavigationMixin
from .refresh import RefreshMixin
from .curses_draw import CursesMixin
from .colors import ColorMixin

from ..slurm import (
    LEADERBOARD_WINDOWS,
    Job,
    JobRecord,
    Node,
    SlurmClient,
)




# Constants, pure helpers, and the mixins that make up VaccsRunningApp all live
# in sibling modules (ui_constants, widgets, table_layout, ui_render_*, etc.);
# this module keeps the composition root and re-exports the public helper API.
# Info content starts on row 5, matching the gap the other tables leave above
# them (below the header + controls rows).


@dataclass
class AppState:
    jobs: list[Job]
    job_records: list[JobRecord]
    nodes: list[Node]
    history: list[JobRecord]
    view: str = "jobs"
    selected: int = 0
    scroll: int = 0
    message: str = ""
    last_refresh: float = 0.0
    gpu_nodes_only: bool = False
    free_gpu_only: bool = False
    jobs_grouped: bool = False
    history_window: str = "24h"
    leaderboard_group_mode: bool = False
    leaderboard_sort: str = "gpu"
    leaderboard_ascending: bool = False
    leaderboard_scroll: int = 0
    leaderboard_filter: str = ""
    leaderboard_filter_editing: bool = False
    info_scroll: int = 0


class VaccsRunningApp(
    JobFilterMenuMixin,
    KeyHandlingMixin,
    PopupMixin,
    RenderFrameMixin,
    RenderTablesMixin,
    RenderDetailMixin,
    RenderLeaderboardMixin,
    LeaderboardDataMixin,
    InfoDataMixin,
    JobFilterStateMixin,
    NavigationMixin,
    RefreshMixin,
    CursesMixin,
    ColorMixin,
):

    def __init__(
        self,
        client: SlurmClient,
        refresh_seconds: float,
        initial_view: str = "jobs",
    ):
        self.client = client
        self.refresh_seconds = refresh_seconds
        self.state = AppState(
            jobs=[],
            job_records=[],
            nodes=[],
            history=[],
            view=(
                initial_view
                if initial_view in {"jobs", "history", "nodes", "leaderboard", "info"}
                else "jobs"
            ),
        )
        self.colors_enabled = False

        # Usage data is fetched off the UI thread because the sreport queries
        # can take several seconds. Each window is fetched by its own daemon
        # thread; results land in _lb_windows under _lb_lock and the draw loop
        # picks them up on its next frame. The tab never auto-refreshes -- only
        # pressing 'r' kicks off a new generation.
        self._lb_lock = threading.Lock()
        self._lb_generation = 0
        self._lb_started = False
        self._lb_threads: list[threading.Thread] = []
        self._lb_windows: dict[str, dict[str, object]] = {
            window: {"status": "idle", "usage": [], "error": ""}
            for window, _label in LEADERBOARD_WINDOWS
        }
        self._lb_fairshare: dict[tuple[str, str], float] = {}
        self._lb_level_fairshare: dict[str, float] = {}
        self._lb_default_accounts: dict[str, str] = {}

        # The Info tab loads its user card (accounts, fairshare, per-window
        # compute usage, GPFS quota) in a single background thread so switching
        # to the tab never blocks. Manual refresh only (press 'r').
        self._info_lock = threading.Lock()
        self._info_generation = 0
        self._info_started = False
        self._info_data: dict[str, object] = {"status": "idle"}

    def run(self) -> None:
        curses.wrapper(self._main)

    def _active_refresh_seconds(self) -> float:
        if self.state.view in {"leaderboard", "info"}:
            # Manual refresh only: usage queries are heavy, so never auto-run them.
            return 0.0
        if not self.refresh_seconds:
            return self.refresh_seconds
        if self.state.view == "history" and self.refresh_seconds:
            return HISTORY_REFRESH_SECONDS
        if (
            self.state.view == "jobs"
            and len(self.state.jobs) > BUSY_JOBS_REFRESH_THRESHOLD
        ):
            return max(self.refresh_seconds, BUSY_JOBS_REFRESH_SECONDS)
        return self.refresh_seconds

    def _main(self, stdscr: curses.window) -> None:
        safe_curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)
        self._init_colors()
        self._refresh_current()

        while True:
            self._draw(stdscr)
            key = stdscr.getch()
            if key != -1 and not self._handle_key(stdscr, key):
                return

            now = time.monotonic()
            refresh_seconds = self._active_refresh_seconds()
            if (
                refresh_seconds
                and now - self.state.last_refresh >= refresh_seconds
            ):
                self._refresh_current()

            if key == -1:
                time.sleep(0.05)
