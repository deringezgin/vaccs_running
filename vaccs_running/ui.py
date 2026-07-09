from __future__ import annotations

import curses
from collections.abc import Callable
import textwrap
import time
from dataclasses import dataclass

from .slurm import (
    HISTORY_WINDOWS,
    Job,
    JobFilterChoices,
    JobRecord,
    JobRecordGroup,
    Node,
    SlurmClient,
    SlurmError,
    VACC_PARTITIONS,
    group_job_records,
    plural_label,
    record_from_job,
    summarize_job_records,
    summarize_jobs,
    summarize_nodes,
)


STATE_COLORS = {
    "RUNNING": 1,
    "PENDING": 2,
    "COMPLETED": 3,
    "FAILED": 4,
    "CANCELLED": 4,
    "NODE_FAIL": 4,
    "OUT_OF_MEMORY": 4,
    "PREEMPTED": 4,
    "TIMEOUT": 4,
}

NODE_COLORS = {
    "IDLE": 1,
    "MIXED": 2,
    "ALLOCATED": 3,
    "DOWN": 4,
    "DRAIN": 4,
    "DRAINED": 4,
}

BORDER_PAIR = 9
TEXT_PAIR = 10
ACTIVE_TAB_PAIR = 11
TITLE_PAIR = 12
MUTED_PAIR = 13
SURFACE_PAIR = 14
MIN_TERMINAL_WIDTH = 70
MIN_TERMINAL_HEIGHT = 16
HISTORY_REFRESH_SECONDS = 10.0
BUSY_JOBS_REFRESH_SECONDS = 5.0
BUSY_JOBS_REFRESH_THRESHOLD = 50
HISTORY_FILTER_OPTIONS = [
    ("1h", "last 1 hour"),
    ("3h", "last 3 hours"),
    ("24h", "last 24 hours"),
    ("3d", "last 3 days"),
    ("7d", "last 7 days"),
]
JOB_STATE_FILTER_OPTIONS = [
    ("BF", "BOOT_FAIL"),
    ("CA", "CANCELLED"),
    ("CD", "COMPLETED"),
    ("CF", "CONFIGURING"),
    ("CG", "COMPLETING"),
    ("DL", "DEADLINE"),
    ("F", "FAILED"),
    ("NF", "NODE_FAIL"),
    ("OOM", "OUT_OF_MEMORY"),
    ("PD", "PENDING"),
    ("PR", "PREEMPTED"),
    ("R", "RUNNING"),
    ("RD", "RESV_DEL_HOLD"),
    ("RF", "REQUEUE_FED"),
    ("RH", "REQUEUE_HOLD"),
    ("RQ", "REQUEUED"),
    ("RS", "RESIZING"),
    ("RV", "REVOKED"),
    ("SI", "SIGNALING"),
    ("SE", "SPECIAL_EXIT"),
    ("SO", "STAGE_OUT"),
    ("ST", "STOPPED"),
    ("S", "SUSPENDED"),
    ("TO", "TIMEOUT"),
]
JOB_STATE_CODES = [state for state, _ in JOB_STATE_FILTER_OPTIONS]


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


class VaccsRunningApp:
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
            view=initial_view if initial_view in {"jobs", "history", "nodes"} else "jobs",
        )
        self.colors_enabled = False

    def run(self) -> None:
        curses.wrapper(self._main)

    def _active_refresh_seconds(self) -> float:
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

    def _init_colors(self) -> None:
        try:
            curses.start_color()
            curses.use_default_colors()
            orange = self._orange_color()
            grid = self._grid_color()
            title = self._title_color()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_CYAN, -1)
            curses.init_pair(4, curses.COLOR_RED, -1)
            curses.init_pair(5, orange, -1)
            curses.init_pair(6, orange, -1)
            curses.init_pair(7, curses.COLOR_BLACK, orange)
            curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_RED)
            curses.init_pair(BORDER_PAIR, grid, -1)
            curses.init_pair(TEXT_PAIR, curses.COLOR_WHITE, -1)
            curses.init_pair(ACTIVE_TAB_PAIR, curses.COLOR_BLACK, orange)
            curses.init_pair(TITLE_PAIR, title, -1)
            curses.init_pair(MUTED_PAIR, curses.COLOR_WHITE, -1)
            curses.init_pair(SURFACE_PAIR, curses.COLOR_WHITE, -1)
            self.colors_enabled = True
        except curses.error:
            self.colors_enabled = False

    def _custom_color(self, slot: int, red: int, green: int, blue: int) -> int | None:
        if curses.COLORS <= slot or not curses.can_change_color():
            return None
        try:
            curses.init_color(slot, red, green, blue)
            return slot
        except curses.error:
            return None

    def _orange_color(self) -> int:
        custom = self._custom_color(16, 863, 345, 165)  # #DC582A
        if custom is not None:
            return custom
        return 173 if curses.COLORS > 173 else curses.COLOR_YELLOW

    def _grid_color(self) -> int:
        custom = self._custom_color(17, 863, 345, 165)  # #DC582A
        if custom is not None:
            return custom
        return curses.COLOR_WHITE

    def _title_color(self) -> int:
        custom = self._custom_color(18, 969, 969, 969)  # #F7F7F7
        if custom is not None:
            return custom
        return curses.COLOR_WHITE

    def _refresh_current(self) -> None:
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

    def _squeue_state_filter(self) -> str:
        return str(getattr(self.client, "squeue_states", "all") or "all")

    def _squeue_state_filter_active(self) -> bool:
        active = getattr(self.client, "state_filter_active", None)
        if active is not None:
            return bool(active)
        return self._squeue_state_filter().lower() != "all"

    def _job_user_filter_active(self) -> bool:
        active = getattr(self.client, "job_user_filter_active", None)
        return bool(active) if active is not None else False

    def _job_partition_filter_active(self) -> bool:
        active = getattr(self.client, "job_partition_filter_active", None)
        if active is not None:
            return bool(active)
        return bool(self._selected_job_partitions())

    def _jobs_filter_active(self) -> bool:
        return (
            self._squeue_state_filter_active()
            or self._job_user_filter_active()
            or self._job_partition_filter_active()
        )

    def _show_job_principal_columns(self) -> bool:
        return (
            self._job_all_principals()
            or len(self._selected_job_users()) > 1
            or bool(self._selected_job_groups())
        )

    def _visible_nodes(self) -> list[Node]:
        visible = self.state.nodes
        if self.state.gpu_nodes_only:
            visible = [node for node in visible if node.has_gpus]
        if self.state.free_gpu_only:
            visible = [node for node in visible if node.gpu_free > 0]
        return visible

    def _visible_count(self) -> int:
        if self.state.view == "nodes":
            return len(self._visible_nodes())
        if self.state.view == "history":
            return len(self._visible_history_groups())
        if self.state.jobs_grouped:
            return len(self._visible_job_groups())
        return len(self._visible_jobs())

    def _handle_key(self, stdscr: curses.window, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == curses.KEY_DOWN:
            self.state.selected += 1
        elif key in (curses.KEY_UP, ord("k")):
            self.state.selected -= 1
        elif key == curses.KEY_NPAGE:
            self.state.selected += 10
        elif key == curses.KEY_PPAGE:
            self.state.selected -= 10
        elif key == curses.KEY_RIGHT:
            self._jump_page(stdscr, 1)
        elif key == curses.KEY_LEFT:
            self._jump_page(stdscr, -1)
        elif key == curses.KEY_HOME:
            self.state.selected = 0
        elif key == curses.KEY_END:
            self.state.selected = self._visible_count() - 1
        elif key == ord("n"):
            self._switch_view("nodes")
        elif key == ord("h"):
            self._switch_view("history")
        elif key == ord("j"):
            self._switch_view("jobs")
        elif key == ord("g"):
            if self.state.view == "nodes":
                enabled = not self.state.gpu_nodes_only
                self.state.gpu_nodes_only = enabled
                if enabled:
                    self.state.free_gpu_only = False
                self.state.selected = 0
                self.state.scroll = 0
                state = "on" if self.state.gpu_nodes_only else "off"
                self.state.message = f"GPU node filter {state}"
            elif self.state.view == "jobs":
                self.state.jobs_grouped = not self.state.jobs_grouped
                self.state.selected = 0
                self.state.scroll = 0
                state = "on" if self.state.jobs_grouped else "off"
                self.state.message = f"job grouping {state}"
        elif key == ord("f"):
            if self.state.view == "nodes":
                enabled = not self.state.free_gpu_only
                self.state.free_gpu_only = enabled
                if enabled:
                    self.state.gpu_nodes_only = False
                self.state.selected = 0
                self.state.scroll = 0
                state = "on" if self.state.free_gpu_only else "off"
                self.state.message = f"free GPU filter {state}"
            elif self.state.view == "jobs":
                self._show_jobs_filter(stdscr)
            elif self.state.view == "history":
                self._show_history_filter(stdscr)
        elif key == ord("d"):
            if self.state.view in {"jobs", "nodes"}:
                self._show_detail(stdscr)
        elif key == ord("p"):
            if self.state.view == "nodes":
                self._show_node_jobs(stdscr)
        elif key == ord("i"):
            if self.state.view == "nodes":
                self._show_node_usage(stdscr)

        self._clamp_selection()
        return True

    def _set_history_window(self, window: str) -> None:
        if window not in HISTORY_WINDOWS:
            return
        self.state.history_window = window
        self.state.selected = 0
        self.state.scroll = 0
        self._refresh_history()
        self._clamp_selection()

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

    def _draw(self, stdscr: curses.window) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if terminal_too_small(width, height):
            self._draw_terminal_too_small(stdscr, width, height)
            stdscr.refresh()
            return
        self._draw_header(stdscr, width)
        if self.state.view == "nodes":
            self._draw_nodes_table(stdscr, self._visible_nodes(), height, width)
            self._draw_node_detail(stdscr, height, width)
        elif self.state.view == "history":
            self._draw_history_groups_table(
                stdscr,
                self._visible_history_groups(),
                height,
                width,
            )
            self._draw_history_group_detail(stdscr, height, width)
        elif self.state.jobs_grouped:
            self._draw_job_groups_table(
                stdscr,
                self._visible_job_groups(),
                height,
                width,
            )
            self._draw_job_group_detail(stdscr, height, width)
        else:
            self._draw_jobs_table(stdscr, self._visible_jobs(), height, width)
            self._draw_job_detail(stdscr, height, width)
        stdscr.refresh()

    def _draw_terminal_too_small(
        self,
        stdscr: curses.window,
        width: int,
        height: int,
    ) -> None:
        lines = [
            ("Terminal size too small:", curses.A_BOLD),
            (f"Width = {width} Height = {height}", curses.A_BOLD),
            ("", 0),
            ("Needed for current config:", curses.A_BOLD),
            (
                f"Width = {MIN_TERMINAL_WIDTH} Height = {MIN_TERMINAL_HEIGHT}",
                curses.A_BOLD,
            ),
        ]
        top = max(0, (height - len(lines)) // 2)
        for offset, (line, attr) in enumerate(lines):
            if not line:
                continue
            x = max(0, (width - len(line)) // 2)
            self._addstr(stdscr, top + offset, x, line, self._pair(MUTED_PAIR) | attr)

    def _draw_header(self, stdscr: curses.window, width: int) -> None:
        title = " VACC's Running? "
        right = time.strftime("%H:%M:%S")
        self._draw_box(stdscr, 0, 0, 3, width)

        x = 2
        for view, label in [
            ("jobs", " j Jobs "),
            ("nodes", " n Nodes "),
            ("history", " h History "),
        ]:
            attr = (
                self._pair(ACTIVE_TAB_PAIR) | curses.A_BOLD
                if self.state.view == view
                else self._pair(MUTED_PAIR)
            )
            self._addstr(stdscr, 1, x, label, attr)
            x += len(label) + 1

        title_x = max(x, (width - len(title)) // 2)
        right_x = width - len(right) - 2
        if title_x + len(title) < right_x:
            self._addstr(
                stdscr,
                1,
                title_x,
                title,
                self._pair(TITLE_PAIR) | curses.A_BOLD,
            )
        if width > len(right) + 2:
            self._addstr(stdscr, 1, right_x, right, self._pair(MUTED_PAIR))
        if self.state.view == "nodes":
            x = 1
            gpu_filter_text = " g gpu-nodes "
            self._addstr(
                stdscr,
                3,
                x,
                gpu_filter_text,
                self._pair(ACTIVE_TAB_PAIR if self.state.gpu_nodes_only else MUTED_PAIR),
            )
            x += len(gpu_filter_text) + 1
            free_filter_text = " f free-gpu "
            self._addstr(
                stdscr,
                3,
                x,
                free_filter_text,
                self._pair(ACTIVE_TAB_PAIR if self.state.free_gpu_only else MUTED_PAIR),
            )
            x += len(free_filter_text) + 1
            self._addstr(stdscr, 3, x, " d detail ", self._pair(MUTED_PAIR))
            x += len(" d detail ") + 1
            self._addstr(stdscr, 3, x, " p peek ", self._pair(MUTED_PAIR))
            x += len(" p peek ") + 1
            self._addstr(stdscr, 3, x, " i usage ", self._pair(MUTED_PAIR))
            x += len(" i usage ") + 1
            self._addstr(stdscr, 3, x, " q quit", self._pair(MUTED_PAIR))
        elif self.state.view == "history":
            x = 1
            filter_text = f" f filter: {history_window_short_label(self.state.history_window)} "
            self._addstr(stdscr, 3, x, filter_text, self._pair(MUTED_PAIR))
            x += len(filter_text) + 1
            self._addstr(stdscr, 3, x, " q quit", self._pair(MUTED_PAIR))
        else:
            x = 1
            group_text = " g group "
            self._addstr(
                stdscr,
                3,
                x,
                group_text,
                self._pair(ACTIVE_TAB_PAIR if self.state.jobs_grouped else MUTED_PAIR),
            )
            x += len(group_text) + 1
            filter_text = " f filter "
            self._addstr(
                stdscr,
                3,
                x,
                filter_text,
                self._pair(ACTIVE_TAB_PAIR if self._jobs_filter_active() else MUTED_PAIR),
            )
            x += len(filter_text) + 1
            if self._squeue_state_filter_active():
                state_text = f" state: {job_state_filter_label(self._squeue_state_filter())} "
                self._addstr(stdscr, 3, x, state_text, self._pair(ACTIVE_TAB_PAIR))
                x += len(state_text) + 1
            if self._job_user_filter_active():
                user_summary = self._job_user_summary()
                group_summary = self._job_group_summary()
                if user_summary != "me":
                    user_text = f" user: {user_summary} "
                    self._addstr(stdscr, 3, x, user_text, self._pair(ACTIVE_TAB_PAIR))
                    x += len(user_text) + 1
                if group_summary != "none":
                    group_text = f" group: {group_summary} "
                    self._addstr(stdscr, 3, x, group_text, self._pair(ACTIVE_TAB_PAIR))
                    x += len(group_text) + 1
            if self._job_partition_filter_active():
                partition_text = f" partition: {self._job_partition_summary()} "
                self._addstr(stdscr, 3, x, partition_text, self._pair(ACTIVE_TAB_PAIR))
                x += len(partition_text) + 1
            self._addstr(stdscr, 3, x, " d detail ", self._pair(MUTED_PAIR))
            x += len(" d detail ") + 1
            self._addstr(stdscr, 3, x, " q quit", self._pair(MUTED_PAIR))

    def _draw_jobs_table(
        self,
        stdscr: curses.window,
        visible: list[Job],
        height: int,
        width: int,
    ) -> None:
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        title = summary_title(
            summarize_jobs(visible),
            ["RUNNING", "PENDING", "COMPLETED"],
        )
        self._draw_box(stdscr, table_top, 0, table_height, width, title)
        header_y = table_top + 1
        first_row = table_top + 2
        rows = max(0, table_height - 3)
        available_width = max(1, width - 4)
        job_specs = responsive_job_specs(
            available_width,
            show_principals=self._show_job_principal_columns(),
        )
        row_values = [
            [value_fn(job) for _, _, _, value_fn in job_specs]
            for job in visible
        ]
        columns = fit_columns(
            [
                (label, min_width, max_width)
                for label, min_width, max_width, _ in job_specs
            ],
            row_values,
            available_width,
        )
        headers = [
            (label, column_width)
            for (label, _, _, _), column_width in zip(job_specs, columns)
        ]
        x = 2
        for label, size in headers:
            self._addstr(stdscr, header_y, x, label[:size].ljust(size), self._pair(MUTED_PAIR) | curses.A_BOLD)
            x += size + 1

        if self.state.selected < self.state.scroll:
            self.state.scroll = self.state.selected
        if self.state.selected >= self.state.scroll + rows:
            self.state.scroll = self.state.selected - rows + 1
        page_label = page_status(self.state.selected, len(visible), rows)
        if page_label:
            footer = f" {page_label} "
            footer_x = max(2, width - len(footer) - 2)
            self._addstr(
                stdscr,
                table_top + table_height - 1,
                footer_x,
                footer,
                self._pair(5) | curses.A_BOLD,
            )

        for screen_row, job in enumerate(
            visible[self.state.scroll : self.state.scroll + rows],
            start=first_row,
        ):
            index = self.state.scroll + screen_row - first_row
            attr = self._state_attr(job.state)
            if index == self.state.selected:
                attr |= curses.A_REVERSE
            cells = [
                (value_fn(job), column_width)
                for (_, _, _, value_fn), column_width in zip(job_specs, columns)
            ]
            x = 2
            for value, size in cells:
                text = value[:size].ljust(size)
                if x < width:
                    self._addstr(stdscr, screen_row, x, text[: max(0, width - x - 1)], attr)
                x += size + 1

    def _draw_job_groups_table(
        self,
        stdscr: curses.window,
        visible: list[JobRecordGroup],
        height: int,
        width: int,
    ) -> None:
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        title = status_title(
            "Running Groups",
            summarize_jobs(self._visible_jobs()),
            ["RUNNING", "PENDING", "COMPLETED"],
        )
        self._draw_box(stdscr, table_top, 0, table_height, width, title)
        header_y = table_top + 1
        first_row = table_top + 2
        rows = max(0, table_height - 3)
        available_width = max(1, width - 4)
        group_specs = responsive_job_group_specs(
            available_width,
            show_principals=self._show_job_principal_columns(),
        )
        row_values = [
            [value_fn(group) for _, _, _, value_fn in group_specs]
            for group in visible
        ]
        columns = fit_columns(
            [
                (label, min_width, max_width)
                for label, min_width, max_width, _ in group_specs
            ],
            row_values,
            available_width,
        )
        headers = [
            (label, column_width)
            for (label, _, _, _), column_width in zip(group_specs, columns)
        ]
        x = 2
        for label, size in headers:
            self._addstr(
                stdscr,
                header_y,
                x,
                label[:size].ljust(size),
                self._pair(MUTED_PAIR) | curses.A_BOLD,
            )
            x += size + 1

        if self.state.selected < self.state.scroll:
            self.state.scroll = self.state.selected
        if self.state.selected >= self.state.scroll + rows:
            self.state.scroll = self.state.selected - rows + 1
        page_label = page_status(self.state.selected, len(visible), rows)
        if page_label:
            footer = f" {page_label} "
            footer_x = max(2, width - len(footer) - 2)
            self._addstr(
                stdscr,
                table_top + table_height - 1,
                footer_x,
                footer,
                self._pair(5) | curses.A_BOLD,
            )

        for screen_row, group in enumerate(
            visible[self.state.scroll : self.state.scroll + rows],
            start=first_row,
        ):
            index = self.state.scroll + screen_row - first_row
            attr = self._state_attr(group.dominant_state)
            if index == self.state.selected:
                attr |= curses.A_REVERSE
            cells = [
                (value_fn(group), column_width)
                for (_, _, _, value_fn), column_width in zip(group_specs, columns)
            ]
            x = 2
            for value, size in cells:
                text = value[:size].ljust(size)
                if x < width:
                    self._addstr(
                        stdscr,
                        screen_row,
                        x,
                        text[: max(0, width - x - 1)],
                        attr,
                    )
                x += size + 1

    def _draw_history_groups_table(
        self,
        stdscr: curses.window,
        visible: list[JobRecordGroup],
        height: int,
        width: int,
    ) -> None:
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        title = summary_title(
            summarize_job_records(self.state.history),
            ["RUNNING", "PENDING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"],
        )
        self._draw_box(stdscr, table_top, 0, table_height, width, title)
        header_y = table_top + 1
        first_row = table_top + 2
        rows = max(0, table_height - 3)
        available_width = max(1, width - 4)
        specs = responsive_history_group_specs(available_width)
        row_values = [
            [value_fn(group) for _, _, _, value_fn in specs]
            for group in visible
        ]
        columns = fit_columns(label_widths(specs), row_values, available_width)
        x = 2
        for (label, _, _, _), size in zip(specs, columns):
            self._addstr(
                stdscr,
                header_y,
                x,
                label[:size].ljust(size),
                self._pair(MUTED_PAIR) | curses.A_BOLD,
            )
            x += size + 1

        if self.state.selected < self.state.scroll:
            self.state.scroll = self.state.selected
        if self.state.selected >= self.state.scroll + rows:
            self.state.scroll = self.state.selected - rows + 1
        page_label = page_status(self.state.selected, len(visible), rows)
        if page_label:
            footer = f" {page_label} "
            footer_x = max(2, width - len(footer) - 2)
            self._addstr(
                stdscr,
                table_top + table_height - 1,
                footer_x,
                footer,
                self._pair(5) | curses.A_BOLD,
            )

        for screen_row, group in enumerate(
            visible[self.state.scroll : self.state.scroll + rows],
            start=first_row,
        ):
            index = self.state.scroll + screen_row - first_row
            attr = self._state_attr(group.dominant_state)
            if index == self.state.selected:
                attr |= curses.A_REVERSE
            x = 2
            for (_, _, _, value_fn), size in zip(specs, columns):
                text = value_fn(group)[:size].ljust(size)
                if x < width:
                    self._addstr(
                        stdscr,
                        screen_row,
                        x,
                        text[: max(0, width - x - 1)],
                        attr,
                    )
                x += size + 1

    def _draw_nodes_table(
        self,
        stdscr: curses.window,
        visible: list[Node],
        height: int,
        width: int,
    ) -> None:
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        title = summary_title(
            summarize_nodes(self.state.nodes),
            ["IDLE", "MIXED", "ALLOCATED", "DOWN"],
        )
        self._draw_box(stdscr, table_top, 0, table_height, width, title)
        header_y = table_top + 1
        first_row = table_top + 2
        rows = max(0, table_height - 3)
        available_width = max(1, width - 4)
        cpu_count_width = resource_count_width(
            [(node.cpu_alloc, node.cpu_total) for node in visible]
        )
        gpu_count_width = resource_text_width([node.gpu_text for node in visible])
        memory_count_width = resource_text_width(
            [node.memory_text for node in visible]
        )
        show_resource_bars = available_width >= minimum_table_width(
            [
                ("NODE", 10, 22),
                ("STATE", 8, 18),
                ("PARTITION", 10, 22),
                ("CPU", 24, 38),
                ("MEM", 24, 38),
                ("GPU", 18, 30),
                ("GRES", 12, 48),
            ]
        )
        node_specs = responsive_node_specs(
            show_resource_bars,
            cpu_count_width,
            memory_count_width,
            gpu_count_width,
        )
        row_values = []
        for node in visible:
            gpu_percent = pct(node.gpu_alloc, node.gpu_total)
            if show_resource_bars:
                row_values.append(
                    [
                        node.name,
                        node.state,
                        node.partitions,
                        resource_meter(
                            node.cpu_alloc,
                            node.cpu_total,
                            node.cpu_percent,
                            meter_width=16,
                            count_width=cpu_count_width,
                        ),
                        resource_text_meter(
                            node.memory_text,
                            node.memory_percent,
                            meter_width=14,
                            count_width=memory_count_width,
                        ),
                        resource_text_meter(
                            node.gpu_text,
                            gpu_percent,
                            meter_width=12,
                            count_width=gpu_count_width,
                        ),
                        node.gres,
                    ]
                )
            else:
                row_values.append(
                    [
                        node.name,
                        node.state,
                        node.partitions,
                        f"{node.cpu_alloc}/{node.cpu_total}",
                        node.memory_text,
                        node.gpu_text,
                        node.gres,
                    ]
                )
        columns = fit_columns(
            node_specs,
            row_values,
            available_width,
        )
        headers = [
            (label, column_width)
            for (label, _, _), column_width in zip(node_specs, columns)
        ]
        x = 2
        for label, size in headers:
            self._addstr(stdscr, header_y, x, label[:size].ljust(size), self._pair(MUTED_PAIR) | curses.A_BOLD)
            x += size + 1

        if self.state.selected < self.state.scroll:
            self.state.scroll = self.state.selected
        if self.state.selected >= self.state.scroll + rows:
            self.state.scroll = self.state.selected - rows + 1
        page_label = page_status(self.state.selected, len(visible), rows)
        if page_label:
            footer = f" {page_label} "
            footer_x = max(2, width - len(footer) - 2)
            self._addstr(
                stdscr,
                table_top + table_height - 1,
                footer_x,
                footer,
                self._pair(5) | curses.A_BOLD,
            )

        for screen_row, node in enumerate(
            visible[self.state.scroll : self.state.scroll + rows],
            start=first_row,
        ):
            index = self.state.scroll + screen_row - first_row
            attr = self._node_attr(node)
            if index == self.state.selected:
                attr |= curses.A_REVERSE
            gpu_percent = pct(node.gpu_alloc, node.gpu_total)
            if show_resource_bars:
                values = [
                    node.name,
                    node.state,
                    node.partitions,
                    resource_meter(
                        node.cpu_alloc,
                        node.cpu_total,
                        node.cpu_percent,
                        meter_width=16,
                        count_width=cpu_count_width,
                    ),
                    resource_text_meter(
                        node.memory_text,
                        node.memory_percent,
                        meter_width=14,
                        count_width=memory_count_width,
                    ),
                    resource_text_meter(
                        node.gpu_text,
                        gpu_percent,
                        meter_width=12,
                        count_width=gpu_count_width,
                    ),
                    node.gres,
                ]
            else:
                values = [
                    node.name,
                    node.state,
                    node.partitions,
                    f"{node.cpu_alloc}/{node.cpu_total}",
                    node.memory_text,
                    node.gpu_text,
                    node.gres,
                ]
            cells = list(zip(values, columns))
            x = 2
            for value, size in cells:
                text = value[:size].ljust(size)
                if x < width:
                    self._addstr(stdscr, screen_row, x, text[: max(0, width - x - 1)], attr)
                x += size + 1

    def _draw_job_detail(self, stdscr: curses.window, height: int, width: int) -> None:
        panel_height = min(8, max(4, height // 4))
        top = max(4, height - panel_height)
        job = self._selected_job()
        self._draw_box(stdscr, top, 0, panel_height, width, " selected job ")
        if not job:
            self._addstr(stdscr, top + 1, 2, "No jobs found.", self._pair(2))
            return

        lines = [
            f"{job.name}  job={job.job_id}  array-parent={job.array_parent}",
            f"state={job.state}  partition={job.partition}  nodes={job.nodes or '-'}",
            f"submitted={job.submit_time}  started={job.start_time}  reason={job.reason}",
            f"resources: nodes={job.node_count}  cpus={job.cpus}  gres={job.gres}",
        ]
        body_rows = max(0, min(height - 1, top + panel_height - 1) - top - 1)
        wrapped = wrap_detail_lines(lines, max(1, width - 4))
        for offset, line in enumerate(wrapped[:body_rows]):
            self._addstr(
                stdscr,
                top + 1 + offset,
                2,
                line,
                self._state_attr(job.state),
            )

    def _draw_job_group_detail(
        self,
        stdscr: curses.window,
        height: int,
        width: int,
    ) -> None:
        panel_height = min(8, max(4, height // 4))
        top = max(4, height - panel_height)
        group = self._selected_job_group()
        self._draw_box(stdscr, top, 0, panel_height, width, " selected job group ")
        if not group:
            self._addstr(stdscr, top + 1, 2, "No job groups found.", self._pair(2))
            return

        lines = [
            f"{group.name}  array-parent={group.array_parent}",
            (
                f"requested={group.total}  done={group.completed} "
                f"running={group.running}  pending={group.pending}  failed={group.failed}"
            ),
            (
                f"longest-running={group.longest_running_elapsed}  "
                f"limit={group.limit}  other={group.other}"
            ),
        ]
        body_rows = max(0, min(height - 1, top + panel_height - 1) - top - 1)
        wrapped = wrap_detail_lines(lines, max(1, width - 4))
        for offset, line in enumerate(wrapped[:body_rows]):
            self._addstr(
                stdscr,
                top + 1 + offset,
                2,
                line,
                self._state_attr(group.dominant_state),
            )

    def _draw_history_group_detail(
        self,
        stdscr: curses.window,
        height: int,
        width: int,
    ) -> None:
        panel_height = min(8, max(4, height // 4))
        top = max(4, height - panel_height)
        group = self._selected_history_group()
        self._draw_box(stdscr, top, 0, panel_height, width, " selected history group ")
        if not group:
            self._addstr(stdscr, top + 1, 2, "No history groups found.", self._pair(2))
            return

        lines = [
            f"{group.name}  array-parent={group.array_parent}",
            (
                f"requested={group.total}  done={group.completed}  running={group.running} "
                f"pending={group.pending}  failed={group.failed}  other={group.other}"
            ),
            (
                f"resources: cpus={group.cpus}  gpus={group.gpus} "
                f"limit={group.limit}"
            ),
            f"submitted={group.submit_time}  latest-end={group.end_time or '-'}",
        ]
        body_rows = max(0, min(height - 1, top + panel_height - 1) - top - 1)
        wrapped = wrap_detail_lines(lines, max(1, width - 4))
        for offset, line in enumerate(wrapped[:body_rows]):
            self._addstr(
                stdscr,
                top + 1 + offset,
                2,
                line,
                self._state_attr(group.dominant_state),
            )

    def _draw_node_detail(self, stdscr: curses.window, height: int, width: int) -> None:
        panel_height = min(8, max(4, height // 4))
        top = max(4, height - panel_height)
        node = self._selected_node()
        self._draw_box(stdscr, top, 0, panel_height, width, " selected node ")
        if not node:
            self._addstr(stdscr, top + 1, 2, "No nodes found.", self._pair(2))
            return

        gpu_percent = pct(node.gpu_alloc, node.gpu_total)
        lines = [
            f"{node.name}  state={node.state}  partition={node.partitions}",
            (
                f"cpu alloc={node.cpu_alloc}/{node.cpu_total} "
                f"free={node.free_cpus}  {meter(node.cpu_percent, 18)}  live-load={node.cpu_load:.2f}"
            ),
            (
                f"mem alloc={node.memory_text} ({node.memory_percent:.1f}%) "
                f"{meter(node.memory_percent, 18)}  free-os={node.free_memory_mb // 1024}G"
            ),
            f"gpu alloc={node.gpu_text} free={node.gpu_free}  {meter(gpu_percent, 18)}  tres={node.alloc_tres or '-'}",
            f"features={node.features}",
        ]
        body_rows = max(0, min(height - 1, top + panel_height - 1) - top - 1)
        wrapped = wrap_detail_lines(lines, max(1, width - 4))
        for offset, line in enumerate(wrapped[:body_rows]):
            self._addstr(
                stdscr,
                top + 1 + offset,
                2,
                line,
                self._node_attr(node),
            )

    def _state_attr(self, state: str) -> int:
        return self._pair(STATE_COLORS.get(state.upper(), 3))

    def _node_attr(self, node: Node) -> int:
        return self._pair(NODE_COLORS.get(node.base_state, 3))

    def _pair(self, pair_id: int) -> int:
        if not self.colors_enabled:
            return 0
        return curses.color_pair(pair_id)

    def _addstr(
        self,
        win: curses.window,
        y: int,
        x: int,
        text: str,
        attr: int = 0,
    ) -> None:
        max_y, max_x = win.getmaxyx()
        if y < 0 or y >= max_y or x < 0 or x >= max_x:
            return
        width = max_x - x
        if width <= 0:
            return
        try:
            win.addstr(y, x, text[:width], attr)
        except curses.error:
            pass

    def _draw_box(
        self,
        win: curses.window,
        top: int,
        left: int,
        height: int,
        width: int,
        title: str = "",
    ) -> None:
        if height < 2 or width < 2:
            return
        attr = self._pair(BORDER_PAIR) | curses.A_DIM
        right = left + width - 1
        bottom = top + height - 1
        self._addstr(win, top, left, "╭", attr)
        self._addstr(win, top, right, "╮", attr)
        self._addstr(win, bottom, left, "╰", attr)
        self._addstr(win, bottom, right, "╯", attr)
        for x in range(left + 1, right):
            self._addstr(win, top, x, "─", attr)
            self._addstr(win, bottom, x, "─", attr)
        for y in range(top + 1, bottom):
            self._addstr(win, y, left, "│", attr)
            self._addstr(win, y, right, "│", attr)
        if title:
            self._addstr(win, top, left + 2, title[: max(0, width - 4)], self._pair(5) | curses.A_BOLD)

    def _show_detail(self, stdscr: curses.window) -> None:
        if self.state.view == "nodes":
            node = self._selected_node()
            if not node:
                return
            self._popup_command(
                stdscr,
                f"scontrol show node {node.name}",
                self.client.show_node,
                node.name,
                close_keys=(ord("d"),),
            )
            return
        job = self._selected_job()
        if not job:
            return
        self._popup_command(
            stdscr,
            f"scontrol show job {job.job_id}",
            self.client.show_job,
            job.job_id,
            close_keys=(ord("d"),),
        )

    def _show_node_jobs(self, stdscr: curses.window) -> None:
        node = self._selected_node()
        if not node:
            return
        self._popup_command(
            stdscr,
            f"squeue -a -w {node.name}",
            self.client.node_jobs,
            node.name,
            close_keys=(ord("p"),),
        )

    def _show_node_usage(self, stdscr: curses.window) -> None:
        self._popup(
            stdscr,
            "running usage by user",
            command_text(lambda _: self.client.cluster_usage(), ""),
            close_keys=(ord("i"),),
        )

    def _show_jobs_filter(self, stdscr: curses.window) -> None:
        choices = self._fetch_job_filter_choices()
        self._run_jobs_filter_menu(
            stdscr,
            "running filter",
            lambda: self._jobs_filter_home_items(),
            lambda win, height, width, item: self._activate_jobs_filter_home_item(
                stdscr,
                choices,
                item,
            ),
            " enter/click open  c clear  q close ",
            close_keys=(ord("f"),),
        )

    def _fetch_job_filter_choices(self) -> JobFilterChoices:
        fetch = getattr(self.client, "fetch_running_filter_choices", None)
        if not fetch:
            return JobFilterChoices(users=[], groups=[], partitions=list(VACC_PARTITIONS))
        try:
            return fetch()
        except SlurmError:
            return JobFilterChoices(users=[], groups=[], partitions=list(VACC_PARTITIONS))

    def _run_jobs_filter_menu(
        self,
        stdscr: curses.window,
        title: str,
        items_fn,
        activate_fn,
        footer: str,
        close_keys: tuple[int, ...] = (),
    ) -> None:
        selected = 0
        scroll = 0
        height, width = stdscr.getmaxyx()
        content_width = max(len(title) + 4, len(footer), 58)
        box_width = min(max(44, content_width + 4), max(20, width - 8))
        box_height = min(max(8, height - 4), 28)
        body_height = max(1, box_height - 4)
        top = max(1, (height - box_height) // 2)
        left = max(1, (width - box_width) // 2)
        win = curses.newwin(box_height, box_width, top, left)
        win.keypad(True)
        win.nodelay(False)
        safe_mousemask()

        while True:
            items = items_fn()
            selectable_indexes = [
                index for index, item in enumerate(items) if item["kind"] != "separator"
            ]
            if not selectable_indexes:
                return
            if selected not in selectable_indexes:
                selected = selectable_indexes[0]
            selected_position = selectable_indexes.index(selected)
            selected_position = max(0, min(selected_position, len(selectable_indexes) - 1))
            selected = selectable_indexes[selected_position]
            if selected < scroll:
                scroll = selected
            if selected >= scroll + body_height:
                scroll = selected - body_height + 1

            win.erase()
            win.border()
            self._addstr(win, 0, 2, f" {title} ", self._pair(6) | curses.A_BOLD)
            hitboxes = []
            for offset, item in enumerate(items[scroll : scroll + body_height], start=1):
                item_index = scroll + offset - 1
                if item["kind"] == "separator":
                    self._addstr(
                        win,
                        offset,
                        2,
                        str(item["label"])[: box_width - 4],
                        self._pair(MUTED_PAIR) | curses.A_BOLD,
                    )
                    continue
                marker = ">" if item_index == selected else " "
                prefix = ""
                if "checked" in item:
                    prefix = "[x] " if item.get("checked") else "[ ] "
                text = f"{marker} {prefix}{item['label']}"
                attr = (
                    self._pair(ACTIVE_TAB_PAIR) | curses.A_BOLD
                    if item_index == selected
                    else self._pair(MUTED_PAIR)
                )
                self._addstr(win, offset, 2, text[: box_width - 4], attr)
                hitboxes.append((top + offset, left + 1, left + box_width - 2, item_index))
            self._addstr(win, box_height - 1, 2, footer[: box_width - 4], self._pair(5))
            win.refresh()

            key = win.getch()
            if key in (ord("q"), 27, *close_keys):
                return
            if key in (curses.KEY_DOWN, ord("j")):
                selected_position = min(len(selectable_indexes) - 1, selected_position + 1)
                selected = selectable_indexes[selected_position]
            elif key in (curses.KEY_UP, ord("k")):
                selected_position = max(0, selected_position - 1)
                selected = selectable_indexes[selected_position]
            elif key == curses.KEY_NPAGE:
                selected_position = min(
                    len(selectable_indexes) - 1,
                    selected_position + body_height,
                )
                selected = selectable_indexes[selected_position]
            elif key == curses.KEY_PPAGE:
                selected_position = max(0, selected_position - body_height)
                selected = selectable_indexes[selected_position]
            elif key in (ord("\n"), curses.KEY_ENTER, ord(" ")):
                activate_fn(win, box_height, box_width, items[selected])
            elif key == ord("c"):
                self._clear_job_filters()
            elif key in (ord("s"), ord("u"), ord("g"), ord("p")):
                shortcut_actions = {
                    ord("s"): ("status",),
                    ord("u"): ("custom_user", "user"),
                    ord("g"): ("custom_group", "group"),
                    ord("p"): ("custom_partition", "partition"),
                }
                for action in shortcut_actions[key]:
                    match = next(
                        (
                            index
                            for index in selectable_indexes
                            if items[index].get("action") == action
                        ),
                        None,
                    )
                    if match is not None:
                        selected = match
                        activate_fn(win, box_height, box_width, items[match])
                        break
            elif key == curses.KEY_MOUSE:
                mouse = safe_getmouse()
                if mouse:
                    _, mouse_x, mouse_y, _, button_state = mouse
                    if button_state & (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED):
                        for y, x_min, x_max, item_index in hitboxes:
                            if mouse_y == y and x_min <= mouse_x <= x_max:
                                selected = item_index
                                activate_fn(win, box_height, box_width, items[item_index])
                                break

    def _jobs_filter_home_items(self) -> list[dict[str, object]]:
        return [
            {
                "kind": "submenu",
                "action": "status",
                "label": f"Filter by status: {job_state_filter_label(self._squeue_state_filter())}",
            },
            {
                "kind": "submenu",
                "action": "user",
                "label": f"Filter by user: {self._job_user_summary()}",
            },
            {
                "kind": "submenu",
                "action": "group",
                "label": f"Filter by group: {self._job_group_summary()}",
            },
            {
                "kind": "submenu",
                "action": "partition",
                "label": f"Filter by partition: {self._job_partition_summary()}",
            },
        ]

    def _activate_jobs_filter_home_item(
        self,
        stdscr: curses.window,
        choices: JobFilterChoices,
        item: dict[str, object],
    ) -> None:
        action = item.get("action")
        if action == "status":
            self._show_jobs_status_filter(stdscr)
        elif action == "user":
            self._show_jobs_user_filter(stdscr, choices)
        elif action == "group":
            self._show_jobs_group_filter(stdscr, choices)
        elif action == "partition":
            self._show_jobs_partition_filter(stdscr, choices)

    def _show_jobs_status_filter(self, stdscr: curses.window) -> None:
        self._run_jobs_filter_menu(
            stdscr,
            "filter by status",
            self._jobs_status_filter_items,
            self._activate_jobs_status_filter_item,
            " enter/space toggle  c clear  q back ",
            close_keys=(ord("f"),),
        )

    def _show_jobs_user_filter(
        self,
        stdscr: curses.window,
        choices: JobFilterChoices,
    ) -> None:
        self._run_jobs_filter_menu(
            stdscr,
            "filter by user",
            lambda: self._jobs_user_filter_items(choices),
            lambda win, height, width, item: self._activate_jobs_user_filter_item(
                win,
                height,
                width,
                choices,
                item,
            ),
            " enter/click toggle  c clear  q back ",
            close_keys=(ord("f"),),
        )

    def _show_jobs_group_filter(
        self,
        stdscr: curses.window,
        choices: JobFilterChoices,
    ) -> None:
        self._run_jobs_filter_menu(
            stdscr,
            "filter by group",
            lambda: self._jobs_group_filter_items(choices),
            lambda win, height, width, item: self._activate_jobs_group_filter_item(
                win,
                height,
                width,
                choices,
                item,
            ),
            " enter/click select  c clear  q back ",
            close_keys=(ord("f"),),
        )

    def _show_jobs_partition_filter(
        self,
        stdscr: curses.window,
        choices: JobFilterChoices,
    ) -> None:
        self._run_jobs_filter_menu(
            stdscr,
            "filter by partition",
            lambda: self._jobs_partition_filter_items(choices),
            lambda win, height, width, item: self._activate_jobs_partition_filter_item(
                win,
                height,
                width,
                choices,
                item,
            ),
            " enter/click select  c clear  q back ",
            close_keys=(ord("f"),),
        )

    def _jobs_status_filter_items(self) -> list[dict[str, object]]:
        selected_states = self._selected_job_state_codes()
        items: list[dict[str, object]] = []
        for code, label in JOB_STATE_FILTER_OPTIONS:
            items.append(
                {
                    "kind": "state",
                    "value": code,
                    "label": f"{code:<3} {label}",
                    "checked": code in selected_states,
                }
            )
        return items

    def _jobs_user_filter_items(self, choices: JobFilterChoices) -> list[dict[str, object]]:
        selected_users = self._selected_job_users()
        all_principals = self._job_all_principals()
        default_user = str(getattr(self.client, "user", "me") or "me")
        items: list[dict[str, object]] = [
            {
                "kind": "action",
                "action": "users_all",
                "label": "Select all",
                "checked": bool(choices.users) and selected_users == set(choices.users),
            },
            {
                "kind": "action",
                "action": "users_clear",
                "label": f"Clear all (only {default_user})",
                "checked": not all_principals and not selected_users,
            },
            {
                "kind": "action",
                "action": "custom_user",
                "label": "Enter user name...",
            },
        ]
        for user in choices.users:
            items.append(
                {
                    "kind": "user",
                    "value": user,
                    "label": user,
                    "checked": not all_principals and user in selected_users,
                }
            )
        return items

    def _jobs_group_filter_items(self, choices: JobFilterChoices) -> list[dict[str, object]]:
        selected_groups = self._selected_job_groups()
        items: list[dict[str, object]] = [
            {
                "kind": "action",
                "action": "groups_all",
                "label": "Select all",
                "checked": bool(choices.groups) and selected_groups == set(choices.groups),
            },
            {
                "kind": "action",
                "action": "groups_clear",
                "label": "Clear all",
                "checked": not selected_groups,
            },
            {
                "kind": "action",
                "action": "custom_group",
                "label": "Enter group name...",
            },
        ]
        for group in choices.groups:
            items.append(
                {
                    "kind": "group",
                    "value": group,
                    "label": group,
                    "checked": group in selected_groups,
                }
            )
        return items

    def _jobs_partition_filter_items(
        self,
        choices: JobFilterChoices,
    ) -> list[dict[str, object]]:
        selected_partitions = self._selected_job_partitions()
        items: list[dict[str, object]] = [
            {
                "kind": "action",
                "action": "partitions_all",
                "label": "Select all",
                "checked": (
                    bool(choices.partitions)
                    and selected_partitions == set(choices.partitions)
                ),
            },
            {
                "kind": "action",
                "action": "partitions_clear",
                "label": "Clear all",
                "checked": not selected_partitions,
            },
            {
                "kind": "action",
                "action": "custom_partition",
                "label": "Enter partition name...",
            },
        ]
        for partition in choices.partitions:
            items.append(
                {
                    "kind": "partition",
                    "value": partition,
                    "label": partition,
                    "checked": partition in selected_partitions,
                }
            )
        return items

    def _activate_jobs_status_filter_item(
        self,
        win: curses.window,
        box_height: int,
        box_width: int,
        item: dict[str, object],
    ) -> None:
        kind = item["kind"]
        if kind == "state":
            states = self._selected_job_state_codes()
            value = str(item["value"])
            if value in states:
                states.remove(value)
            else:
                states.add(value)
            self._set_job_state_codes(states)
        self._refresh_jobs_after_filter_change()

    def _activate_jobs_user_filter_item(
        self,
        win: curses.window,
        box_height: int,
        box_width: int,
        choices: JobFilterChoices,
        item: dict[str, object],
    ) -> None:
        kind = item["kind"]
        action = item.get("action")
        if kind == "user":
            users = self._selected_job_users()
            value = str(item["value"])
            if value in users:
                users.remove(value)
            else:
                users.add(value)
            self._set_job_principal_filters(users, self._selected_job_groups())
        elif action == "users_all":
            self._set_job_principal_filters(
                set(choices.users),
                self._selected_job_groups(),
            )
        elif action == "users_clear":
            self._set_job_principal_filters(set(), self._selected_job_groups())
        elif action == "custom_user":
            value = self._read_jobs_filter_choice(
                win,
                box_height,
                box_width,
                "user",
                choices.users,
            )
            if value:
                if value not in choices.users:
                    choices.users.append(value)
                    choices.users.sort()
                self._set_job_principal_filters({value}, set())
        self._refresh_jobs_after_filter_change()

    def _activate_jobs_group_filter_item(
        self,
        win: curses.window,
        box_height: int,
        box_width: int,
        choices: JobFilterChoices,
        item: dict[str, object],
    ) -> None:
        kind = item["kind"]
        action = item.get("action")
        if kind == "group":
            groups = self._selected_job_groups()
            value = str(item["value"])
            if value in groups:
                groups.remove(value)
            else:
                groups.add(value)
            self._set_job_principal_filters(self._selected_job_users(), groups)
        elif action == "groups_all":
            self._set_job_principal_filters(
                self._selected_job_users(),
                set(choices.groups),
            )
        elif action == "groups_clear":
            self._set_job_principal_filters(self._selected_job_users(), set())
        elif action == "custom_group":
            value = self._read_jobs_filter_choice(
                win,
                box_height,
                box_width,
                "group",
                choices.groups,
            )
            if value:
                if value not in choices.groups:
                    choices.groups.append(value)
                    choices.groups.sort()
                self._set_job_principal_filters(set(), {value})
        self._refresh_jobs_after_filter_change()

    def _activate_jobs_partition_filter_item(
        self,
        win: curses.window,
        box_height: int,
        box_width: int,
        choices: JobFilterChoices,
        item: dict[str, object],
    ) -> None:
        kind = item["kind"]
        action = item.get("action")
        if kind == "partition":
            partitions = self._selected_job_partitions()
            value = str(item["value"])
            if value in partitions:
                partitions.remove(value)
            else:
                partitions.add(value)
            self._set_job_partition_filters(partitions)
        elif action == "partitions_all":
            self._set_job_partition_filters(set(choices.partitions))
        elif action == "partitions_clear":
            self._set_job_partition_filters(set())
        elif action == "custom_partition":
            value = self._read_jobs_filter_choice(
                win,
                box_height,
                box_width,
                "partition",
                choices.partitions,
            )
            if value:
                if value not in choices.partitions:
                    choices.partitions.append(value)
                    choices.partitions.sort()
                self._set_job_partition_filters({value})
        self._refresh_jobs_after_filter_change()

    def _clear_job_filters(self) -> None:
        clear = getattr(self.client, "clear_job_filters", None)
        if clear:
            clear()
        else:
            self.client.squeue_states = "all"
        self._refresh_jobs_after_filter_change()

    def _refresh_jobs_after_filter_change(self) -> None:
        self._refresh_jobs()
        self.state.selected = 0
        self.state.scroll = 0
        self._clamp_selection()

    def _set_job_state_filter(self, states: str) -> None:
        setter = getattr(self.client, "set_job_state_filter", None)
        if setter:
            setter(states)
        else:
            self.client.squeue_states = states

    def _set_job_state_codes(self, states: set[str]) -> None:
        ordered = [state for state in JOB_STATE_CODES if state in states]
        self._set_job_state_filter(",".join(ordered) if ordered else "all")

    def _set_job_principal_filters(
        self,
        users: set[str],
        groups: set[str],
        *,
        all_principals: bool = False,
    ) -> None:
        setter = getattr(self.client, "set_job_principal_filters", None)
        if setter:
            setter(users, groups, all_principals=all_principals)

    def _set_job_partition_filters(self, partitions: set[str]) -> None:
        setter = getattr(self.client, "set_job_partition_filters", None)
        if setter:
            setter(partitions)
        else:
            setattr(self.client, "job_partitions", set(partitions))

    def _selected_job_state_codes(self) -> set[str]:
        states = self._squeue_state_filter()
        if states.lower() == "all":
            return set()
        return {
            state.strip().upper()
            for state in states.split(",")
            if state.strip()
        }

    def _selected_job_users(self) -> set[str]:
        users = set(getattr(self.client, "job_users", set()) or set())
        groups = self._selected_job_groups()
        all_principals = self._job_all_principals()
        default_user = str(getattr(self.client, "user", ""))
        if not all_principals and not groups and users == {default_user}:
            return set()
        return users

    def _selected_job_groups(self) -> set[str]:
        return set(getattr(self.client, "job_groups", set()) or set())

    def _selected_job_partitions(self) -> set[str]:
        return set(getattr(self.client, "job_partitions", set()) or set())

    def _job_all_principals(self) -> bool:
        return bool(getattr(self.client, "job_all_principals", False))

    def _job_user_summary(self) -> str:
        if self._job_all_principals():
            return "all users"
        users = self._selected_job_users()
        if not users:
            return "me"
        return plural_label(len(users), "user")

    def _job_group_summary(self) -> str:
        groups = self._selected_job_groups()
        if not groups:
            return "none"
        return plural_label(len(groups), "group")

    def _job_partition_summary(self) -> str:
        partitions = self._selected_job_partitions()
        if not partitions:
            return "all"
        if len(partitions) == 1:
            return next(iter(partitions))
        return plural_label(len(partitions), "partition")

    def _read_jobs_filter_choice(
        self,
        win: curses.window,
        box_height: int,
        box_width: int,
        label: str,
        options: list[str],
    ) -> str | None:
        query = ""
        selected = 0
        footer = " type to filter  enter select/add  esc cancel "
        safe_curs_set(1)
        try:
            while True:
                matches = filter_choice_options(options, query)
                if matches:
                    selected = max(0, min(selected, len(matches) - 1))
                else:
                    selected = 0

                win.erase()
                win.border()
                self._addstr(
                    win,
                    0,
                    2,
                    f" enter {label} ",
                    self._pair(6) | curses.A_BOLD,
                )
                prompt = f"{label}: {query}"
                self._addstr(win, 1, 2, prompt[: box_width - 4], self._pair(MUTED_PAIR))
                body_height = max(1, box_height - 4)
                if matches:
                    first_match = max(0, selected - body_height + 1)
                    visible_matches = matches[first_match : first_match + body_height]
                    for row, value in enumerate(visible_matches, start=2):
                        match_index = first_match + row - 2
                        marker = ">" if match_index == selected else " "
                        attr = (
                            self._pair(ACTIVE_TAB_PAIR) | curses.A_BOLD
                            if match_index == selected
                            else self._pair(MUTED_PAIR)
                        )
                        self._addstr(
                            win,
                            row,
                            2,
                            f"{marker} {value}"[: box_width - 4],
                            attr,
                        )
                elif query.strip():
                    self._addstr(
                        win,
                        2,
                        2,
                        f'Add "{query.strip()}"'[: box_width - 4],
                        self._pair(ACTIVE_TAB_PAIR) | curses.A_BOLD,
                    )
                else:
                    self._addstr(
                        win,
                        2,
                        2,
                        "No choices.",
                        self._pair(MUTED_PAIR),
                    )
                self._addstr(win, box_height - 1, 2, footer[: box_width - 4], self._pair(5))
                win.refresh()

                key = win.getch()
                if key == 27:
                    return None
                if key in (ord("\n"), curses.KEY_ENTER):
                    if matches:
                        return matches[selected]
                    value = query.strip()
                    return value or None
                if key in (curses.KEY_DOWN, ord("j")) and matches:
                    selected = min(len(matches) - 1, selected + 1)
                elif key in (curses.KEY_UP, ord("k")) and matches:
                    selected = max(0, selected - 1)
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    query = query[:-1]
                    selected = 0
                elif key in (21,):
                    query = ""
                    selected = 0
                elif 32 <= key <= 126:
                    query += chr(key)
                    selected = 0
        finally:
            safe_curs_set(0)

    def _show_history_filter(self, stdscr: curses.window) -> None:
        current_windows = [window for window, _ in HISTORY_FILTER_OPTIONS]
        selected = max(0, current_windows.index(self.state.history_window))
        height, width = stdscr.getmaxyx()
        footer = " up/down select  enter apply  q close "
        content_width = max(
            len("history filter") + 4,
            len(footer),
            *(len(label) + len(window) + 6 for window, label in HISTORY_FILTER_OPTIONS),
        )
        box_width = min(max(36, content_width + 4), max(20, width - 8))
        box_height = len(HISTORY_FILTER_OPTIONS) + 4
        top = max(1, (height - box_height) // 2)
        left = max(1, (width - box_width) // 2)
        win = curses.newwin(box_height, box_width, top, left)
        win.keypad(True)
        win.nodelay(False)

        while True:
            win.erase()
            win.border()
            self._addstr(win, 0, 2, " history filter ", self._pair(6) | curses.A_BOLD)
            for row, (window, label) in enumerate(HISTORY_FILTER_OPTIONS, start=2):
                marker = ">" if row - 2 == selected else " "
                current = "*" if window == self.state.history_window else " "
                text = f"{marker} {label:<14} {window:>3} {current}"
                attr = self._pair(ACTIVE_TAB_PAIR) | curses.A_BOLD if row - 2 == selected else self._pair(MUTED_PAIR)
                self._addstr(win, row, 2, text[: box_width - 4], attr)
            self._addstr(win, box_height - 1, 2, footer[: box_width - 4], self._pair(5))
            win.refresh()

            key = win.getch()
            if key in (ord("q"), 27, ord("f")):
                return
            if key in (ord("\n"), curses.KEY_ENTER):
                self._set_history_window(HISTORY_FILTER_OPTIONS[selected][0])
                return
            if key in (curses.KEY_DOWN, ord("j")):
                selected = min(len(HISTORY_FILTER_OPTIONS) - 1, selected + 1)
            elif key in (curses.KEY_UP, ord("k")):
                selected = max(0, selected - 1)

    def _popup_command(
        self,
        stdscr: curses.window,
        title: str,
        fn,
        job_id: str,
        close_keys: tuple[int, ...] = (),
    ) -> None:
        self._popup(
            stdscr,
            title,
            lambda: command_text(fn, job_id),
            close_keys=close_keys,
        )

    def _popup(
        self,
        stdscr: curses.window,
        title: str,
        text: str | Callable[[], str],
        close_keys: tuple[int, ...] = (),
        refresh_while_open: bool = True,
    ) -> None:
        get_text = text if callable(text) else lambda: text
        current_text = get_text()
        last_refresh = time.monotonic()
        height, width = stdscr.getmaxyx()
        top, left, box_height, box_width = popup_geometry(height, width, title, current_text)
        win = curses.newwin(box_height, box_width, top, left)
        win.nodelay(True)
        win.keypad(True)
        scroll = 0
        wrapped = wrap_lines(current_text, box_width - 4)
        while True:
            now = time.monotonic()
            refresh_seconds = self._active_refresh_seconds()
            if (
                refresh_while_open
                and refresh_seconds
                and now - last_refresh >= refresh_seconds
            ):
                self._refresh_current()
                self._draw(stdscr)
                height, width = stdscr.getmaxyx()
                current_text = get_text()
                top, left, box_height, box_width = popup_geometry(height, width, title, current_text)
                win.resize(box_height, box_width)
                win.mvwin(top, left)
                wrapped = wrap_lines(current_text, box_width - 4)
                body_height = box_height - 4
                scroll = min(scroll, max(0, len(wrapped) - body_height))
                last_refresh = now

            win.erase()
            win.border()
            self._addstr(win, 0, 2, f" {title} ", self._pair(6) | curses.A_BOLD)
            body_height = box_height - 4
            for idx, line in enumerate(wrapped[scroll : scroll + body_height], start=2):
                self._addstr(win, idx, 2, line[: box_width - 4])
            footer = " up/down scroll  q/esc close "
            self._addstr(win, box_height - 1, 2, footer[: box_width - 4], self._pair(5))
            win.refresh()
            key = win.getch()
            if key == -1:
                time.sleep(0.05)
                continue
            if key in (ord("q"), 27, ord("\n"), *close_keys):
                return
            if key in (curses.KEY_DOWN, ord("j")):
                scroll = min(max(0, len(wrapped) - body_height), scroll + 1)
            elif key in (curses.KEY_UP, ord("k")):
                scroll = max(0, scroll - 1)
            elif key == curses.KEY_NPAGE:
                scroll = min(max(0, len(wrapped) - body_height), scroll + body_height)
            elif key == curses.KEY_PPAGE:
                scroll = max(0, scroll - body_height)


def command_text(fn, job_id: str) -> str:
    try:
        return fn(job_id).strip() or "No output."
    except SlurmError as exc:
        return str(exc)


def filter_choice_options(options: list[str], query: str) -> list[str]:
    stripped = query.strip().lower()
    if not stripped:
        return list(options)
    return [option for option in options if stripped in option.lower()]


def wrap_lines(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for line in text.splitlines():
        if not line:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(line, width=width, replace_whitespace=False) or [""])
    return lines


def wrap_detail_lines(lines: list[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(
            textwrap.wrap(
                line,
                width=max(1, width),
                subsequent_indent="  ",
                replace_whitespace=False,
            )
            or [""]
        )
    return wrapped


def popup_geometry(
    screen_height: int,
    screen_width: int,
    title: str,
    text: str,
) -> tuple[int, int, int, int]:
    footer = " up/down scroll  q/esc close "
    max_box_width = max(20, screen_width - 8)
    max_box_height = max(6, screen_height - 4)
    longest_line = max((len(line) for line in text.splitlines()), default=0)
    content_width = max(len(title) + 4, len(footer), longest_line)
    box_width = min(max_box_width, max(40, content_width + 4))
    body_width = max(1, box_width - 4)
    wrapped = wrap_lines(text, body_width)
    box_height = min(max_box_height, max(8, len(wrapped) + 4))
    top = max(1, (screen_height - box_height) // 2)
    left = max(1, (screen_width - box_width) // 2)
    return top, left, box_height, box_width


def status_title(label: str, summary: dict[str, int], preferred: list[str]) -> str:
    suffix = state_summary_text(summary, preferred)
    return f" {label}: {suffix} "


def summary_title(summary: dict[str, int], preferred: list[str]) -> str:
    return f" {state_summary_text(summary, preferred)} "


def state_summary_text(summary: dict[str, int], preferred: list[str]) -> str:
    bits: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if summary.get(key):
            bits.append(f"{key}:{summary[key]}")
            seen.add(key)
    for key, value in sorted(summary.items()):
        if key not in seen and value:
            bits.append(f"{key}:{value}")
    return " ".join(bits) if bits else "none"


def meter(percent: float, width: int) -> str:
    bounded = max(0.0, min(100.0, percent))
    inner = max(1, width)
    filled = round(inner * bounded / 100.0)
    return "[" + "|" * filled + "." * (inner - filled) + "]"


def resource_count_width(pairs: list[tuple[int, int]]) -> int:
    return max((len(f"{used}/{total}") for used, total in pairs), default=0)


def resource_text_width(values: list[str]) -> int:
    return max((len(value) for value in values), default=0)


def resource_meter(
    used: int,
    total: int,
    percent: float,
    *,
    meter_width: int,
    count_width: int,
) -> str:
    count = f"{used}/{total}".rjust(count_width)
    return f"{count} {meter(percent, meter_width)}"


def resource_text_meter(
    text: str,
    percent: float,
    *,
    meter_width: int,
    count_width: int,
) -> str:
    return f"{text.rjust(count_width)} {meter(percent, meter_width)}"


def pct(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * value / total


def page_status(selected: int, total_items: int, page_size: int) -> str:
    if total_items <= 0 or page_size <= 0:
        return "0/0"
    page_count = (total_items + page_size - 1) // page_size
    current_page = min(page_count, max(0, selected) // page_size + 1)
    return f"{current_page}/{page_count}"


def terminal_too_small(width: int, height: int) -> bool:
    return width < MIN_TERMINAL_WIDTH or height < MIN_TERMINAL_HEIGHT


def history_window_short_label(window: str) -> str:
    return window


def job_state_filter_label(states: str) -> str:
    if states.lower() == "all":
        return "all"
    selected = {
        state.strip().upper()
        for state in states.split(",")
        if state.strip()
    }
    if selected == set(JOB_STATE_CODES):
        return "all selected"
    if len(selected) > 4:
        return f"{len(selected)} states"
    for value, label in JOB_STATE_FILTER_OPTIONS:
        if states == value:
            return f"{label} ({value})"
    return states


def filter_running_jobs(jobs: list[Job]) -> list[Job]:
    return [
        job
        for job in jobs
        if job.state.upper() in {"RUNNING", "PENDING"}
    ]


def job_group_key(job: Job) -> tuple[str, str]:
    return (job.array_parent, job.name)


def responsive_job_specs(
    available_width: int,
    *,
    show_principals: bool = False,
) -> list[tuple[str, int, int, Callable[[Job], str]]]:
    specs: list[tuple[str, int, int, Callable[[Job], str]]] = [
        ("JOBID", 10, 22, lambda job: job.job_id),
    ]
    if show_principals:
        specs.extend(
            [
                ("USER", 6, 16, lambda job: job.user or "-"),
                ("GROUP", 6, 18, lambda job: job.group or "-"),
            ]
        )
    specs.extend(
        [
        ("STATE", 8, 14, lambda job: job.state),
        ("PARTITION", 10, 22, lambda job: job.partition),
        ("ELAPSED", 8, 14, lambda job: job.elapsed),
        ("LIMIT", 8, 14, lambda job: job.limit),
        ("CPUS", 4, 8, lambda job: job.cpus),
        ("WHERE / WHY", 18, 56, lambda job: job.location),
        ]
    )
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs

    specs = [spec for spec in specs if spec[0] != "LIMIT"]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs

    specs = [spec for spec in specs if spec[0] != "CPUS"]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs

    specs = [spec for spec in specs if spec[0] != "GROUP"]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs

    return [spec for spec in specs if spec[0] != "USER"]


def responsive_job_group_specs(
    available_width: int,
    *,
    show_principals: bool = False,
) -> list[tuple[str, int, int, Callable[[JobRecordGroup], str]]]:
    specs: list[tuple[str, int, int, Callable[[JobRecordGroup], str]]] = [
        ("JOBID", 10, 16, lambda group: group.array_parent),
        ("JOB", 12, 28, lambda group: group.name),
    ]
    if show_principals:
        specs.extend(
            [
                ("USER", 6, 16, lambda group: group.user or "-"),
                ("GROUP", 6, 18, lambda group: group.group or "-"),
            ]
        )
    specs.extend(
        [
        ("REQ", 3, 5, lambda group: str(group.total)),
        ("DONE", 4, 5, lambda group: str(group.completed)),
        ("RUN", 3, 5, lambda group: str(group.running)),
        ("PEND", 4, 5, lambda group: str(group.pending)),
        ("FAIL", 4, 5, lambda group: str(group.failed)),
        ("RUN_FOR", 7, 12, lambda group: group.longest_running_elapsed),
        ("LIMIT", 8, 14, lambda group: group.limit),
        ]
    )
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs
    specs = [spec for spec in specs if spec[0] != "LIMIT"]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs
    specs = [spec for spec in specs if spec[0] != "GROUP"]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs
    return [spec for spec in specs if spec[0] != "USER"]


def responsive_history_group_specs(
    available_width: int,
) -> list[tuple[str, int, int, Callable[[JobRecordGroup], str]]]:
    specs: list[tuple[str, int, int, Callable[[JobRecordGroup], str]]] = [
        ("JOBID", 10, 16, lambda group: group.array_parent),
        ("JOB", 12, 28, lambda group: group.name),
        ("REQ", 3, 5, lambda group: str(group.total)),
        ("DONE", 4, 5, lambda group: str(group.completed)),
        ("RUN", 3, 5, lambda group: str(group.running)),
        ("PEND", 4, 5, lambda group: str(group.pending)),
        ("FAIL", 4, 5, lambda group: str(group.failed)),
        ("CPUS", 4, 6, lambda group: str(group.cpus)),
        ("GPUS", 4, 6, lambda group: str(group.gpus)),
        ("RUN_FOR", 7, 12, lambda group: group.longest_running_elapsed),
        ("LIMIT", 8, 14, lambda group: group.limit),
    ]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs

    specs = [spec for spec in specs if spec[0] != "LIMIT"]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs

    return [spec for spec in specs if spec[0] not in {"CPUS", "GPUS"}]


def responsive_node_specs(
    show_resource_bars: bool,
    cpu_count_width: int,
    memory_count_width: int,
    gpu_count_width: int,
) -> list[tuple[str, int, int]]:
    if show_resource_bars:
        return [
            ("NODE", 10, 22),
            ("STATE", 8, 18),
            ("PARTITION", 10, 22),
            ("CPU", 24, 38),
            ("MEM", 24, 38),
            ("GPU", 18, 30),
            ("GRES", 12, 48),
        ]

    cpu_width = max(4, cpu_count_width)
    memory_width = max(4, memory_count_width)
    gpu_width = max(3, gpu_count_width)
    return [
        ("NODE", 10, 22),
        ("STATE", 8, 18),
        ("PARTITION", 10, 22),
        ("CPU", cpu_width, max(cpu_width, 8)),
        ("MEM", memory_width, max(memory_width, 12)),
        ("GPU", gpu_width, max(gpu_width, 8)),
        ("GRES", 12, 48),
    ]


def label_widths(
    specs: list[tuple[str, int, int, Callable[..., str]]],
) -> list[tuple[str, int, int]]:
    return [(label, min_width, max_width) for label, min_width, max_width, _ in specs]


def minimum_table_width(specs: list[tuple[str, int, int]]) -> int:
    if not specs:
        return 0
    return sum(min_width for _, min_width, _ in specs) + len(specs) - 1


def fit_columns(
    specs: list[tuple[str, int, int]],
    rows: list[list[str]],
    available_width: int,
) -> list[int]:
    widths: list[int] = []
    for index, (label, min_width, max_width) in enumerate(specs):
        content_width = len(label)
        for row in rows:
            if index < len(row):
                content_width = max(content_width, len(row[index]))
        widths.append(min(max(content_width, min_width), max_width))

    gaps = max(0, len(widths) - 1)
    target = max(1, available_width - gaps)
    while sum(widths) > target:
        candidates = [
            index
            for index, width in enumerate(widths)
            if width > specs[index][1]
        ]
        if not candidates:
            break
        widest = max(candidates, key=lambda index: widths[index])
        widths[widest] -= 1
    return widths


def safe_curs_set(visibility: int) -> None:
    try:
        curses.curs_set(visibility)
    except curses.error:
        pass


def safe_mousemask() -> None:
    try:
        curses.mousemask(curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED)
    except curses.error:
        pass


def safe_getmouse() -> tuple[int, int, int, int, int] | None:
    try:
        return curses.getmouse()
    except curses.error:
        return None
