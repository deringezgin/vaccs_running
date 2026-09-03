from __future__ import annotations

import curses

from .curses_compat import safe_getmouse
from .constants import (
    HISTORY_FILTER_OPTIONS,
    JOB_SORT_LABELS,
    JOB_SORTS,
    LEADERBOARD_PAGE,
    TOP_TABS,
)
from ..slurm import (
    HISTORY_WINDOWS,
    LEADERBOARD_SORTS,
    LEADERBOARD_SORT_LABELS,
)


class KeyHandlingMixin:
    def _handle_key(self, stdscr: curses.window, key: int) -> bool:
        if key == curses.KEY_MOUSE:
            self._handle_mouse(stdscr)
            return True
        # While the Usage find box is open, every keystroke edits the query
        # (so typing 'q', 'j', etc. filters instead of quitting/switching).
        if self.state.view == "leaderboard" and self.state.leaderboard_filter_editing:
            self._handle_leaderboard_filter_key(key)
            return True
        if key in (ord("q"), 27):
            return False
        # View shortcuts are global. Handle them before view-local keys so a
        # tab cannot accidentally consume its own escape route (for example,
        # Info previously treated "j" as scroll-down instead of Jobs).
        view_shortcuts = {ord(key_name): view for view, key_name, _ in TOP_TABS}
        if key in view_shortcuts:
            self._switch_view(view_shortcuts[key])
            return True
        if self.state.view == "leaderboard" and self._handle_leaderboard_key(key):
            return True
        if self.state.view == "info" and self._handle_info_key(key):
            return True
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
            elif self.state.view == "priority":
                self._toggle_priority_gpu_filter()
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
            elif self.state.view == "priority":
                self._show_priority_filter(stdscr)
            elif self.state.view == "history":
                self._cycle_history_window()
        elif key == ord("d"):
            if self.state.view in {"jobs", "nodes"}:
                self._show_detail(stdscr)
        elif key == ord("p"):
            if self.state.view == "nodes":
                self._show_node_jobs(stdscr)
        elif key == ord("a"):
            if self.state.view == "nodes":
                self._show_node_usage(stdscr)
        elif key == ord("e"):
            if self.state.view == "history":
                self._show_job_efficiency(stdscr)
            elif self.state.view == "priority":
                self._toggle_priority_extended()
        elif key == ord("s"):
            if self.state.view == "jobs":
                self._cycle_jobs_sort()
        elif key == ord("o"):
            if self.state.view == "jobs":
                self.state.jobs_ascending = not self.state.jobs_ascending
                self.state.selected = 0
                self.state.scroll = 0
                order = "ascending" if self.state.jobs_ascending else "descending"
                self.state.message = f"jobs order: {order}"
        self._clamp_selection()
        return True

    def _handle_mouse(self, stdscr: curses.window) -> None:
        mouse = safe_getmouse()
        if not mouse:
            return
        _, mouse_x, mouse_y, _, button_state = mouse
        left_click = curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED
        if self.state.view != "priority" or mouse_y != 3:
            return
        if not button_state & left_click:
            return

        extend_x = 1
        extend_text = " e extend "
        if extend_x <= mouse_x < extend_x + len(extend_text):
            self._toggle_priority_extended()
            return

        filter_x = extend_x + len(extend_text) + 1
        filter_text = " f filter "
        if filter_x <= mouse_x < filter_x + len(filter_text):
            self._show_priority_filter(stdscr)
            return

        gpu_x = filter_x + len(filter_text) + 1
        gpu_text = " g gpu-queue "
        if gpu_x <= mouse_x < gpu_x + len(gpu_text):
            self._toggle_priority_gpu_filter()

    def _toggle_priority_extended(self) -> None:
        selected = self._selected_priority_entry()
        selected_job_id = selected.job.job_id if selected is not None else ""
        selected_task_ids = (
            set(selected.task_job_ids) if selected is not None else set()
        )
        if selected is not None and not selected_task_ids:
            selected_task_ids.add(selected.job.job_id)
        selected_parent = selected.job.array_parent if selected is not None else ""
        selected_partition = selected.job.partition if selected is not None else ""
        selected_reservation = (
            selected.job.normalized_reservation if selected is not None else ""
        )

        self.state.priority_extended = not self.state.priority_extended
        self.state.selected = 0
        self.state.scroll = 0
        visible = self._visible_priority_entries()
        if self.state.priority_extended:
            for index, entry in enumerate(visible):
                if (
                    entry.job.job_id == selected_job_id
                    and entry.job.partition == selected_partition
                    and entry.job.normalized_reservation == selected_reservation
                ):
                    self.state.selected = index
                    break
        else:
            fallback_index: int | None = None
            for index, entry in enumerate(visible):
                same_partition = entry.job.partition == selected_partition
                same_scope = (
                    same_partition
                    and entry.job.normalized_reservation == selected_reservation
                )
                entry_task_ids = set(entry.task_job_ids) or {entry.job.job_id}
                if bool(selected_task_ids & entry_task_ids) and same_scope:
                    self.state.selected = index
                    break
                if fallback_index is None and (
                    entry.job.array_parent == selected_parent
                    and same_scope
                ):
                    fallback_index = index
            else:
                if fallback_index is not None:
                    self.state.selected = fallback_index

        mode = "extended" if self.state.priority_extended else "packed"
        self.state.message = f"priority queue {mode}"

    def _handle_info_key(self, key: int) -> bool:
        """Info-tab keys: refresh + scroll. Returns False so tab keys pass through."""
        if key == ord("r"):
            if self._start_info_refresh():
                self.state.message = "info refreshing"
            else:
                self.state.message = "info still loading"
            return True
        if key == curses.KEY_DOWN:
            self.state.info_scroll += 1
            return True
        if key == curses.KEY_UP:
            self.state.info_scroll = max(0, self.state.info_scroll - 1)
            return True
        if key == curses.KEY_NPAGE:
            self.state.info_scroll += 10
            return True
        if key == curses.KEY_PPAGE:
            self.state.info_scroll = max(0, self.state.info_scroll - 10)
            return True
        if key == curses.KEY_HOME:
            self.state.info_scroll = 0
            return True
        return False

    def _handle_leaderboard_key(self, key: int) -> bool:
        """Handle leaderboard-only keys; return False so tab keys still switch views."""
        if key == ord("r"):
            if self._start_leaderboard_refresh():
                self.state.message = "usage refreshing"
            else:
                self.state.message = "usage still loading"
            return True
        if key == ord("m"):
            self.state.leaderboard_group_mode = not self.state.leaderboard_group_mode
            self.state.leaderboard_scroll = 0
            mode = "group" if self.state.leaderboard_group_mode else "user"
            self.state.message = f"usage mode: {mode}"
            return True
        if key == ord("f"):
            self.state.leaderboard_filter_editing = True
            self.state.message = "usage find: type to filter by name"
            return True
        if key == ord("s"):
            self._cycle_leaderboard_sort()
            return True
        if key == ord("o"):
            self.state.leaderboard_ascending = not self.state.leaderboard_ascending
            self.state.leaderboard_scroll = 0
            order = "ascending" if self.state.leaderboard_ascending else "descending"
            self.state.message = f"usage order: {order}"
            return True
        if key == curses.KEY_DOWN:
            self.state.leaderboard_scroll += 1
            return True
        if key == curses.KEY_UP:
            self.state.leaderboard_scroll = max(0, self.state.leaderboard_scroll - 1)
            return True
        if key == curses.KEY_NPAGE:
            self.state.leaderboard_scroll += LEADERBOARD_PAGE
            return True
        if key == curses.KEY_PPAGE:
            self.state.leaderboard_scroll = max(
                0, self.state.leaderboard_scroll - LEADERBOARD_PAGE
            )
            return True
        if key == curses.KEY_HOME:
            self.state.leaderboard_scroll = 0
            return True
        if key == curses.KEY_END:
            # A very large value; _draw_leaderboard clamps it to the last page.
            self.state.leaderboard_scroll = 10 ** 9
            return True
        return False

    def _handle_leaderboard_filter_key(self, key: int) -> None:
        """Edit the live find query. Rows filter as each character is typed."""
        if key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            self.state.leaderboard_filter_editing = False  # keep the filter
            return
        if key == 27:  # Esc: clear the filter and close the find box
            self.state.leaderboard_filter_editing = False
            self.state.leaderboard_filter = ""
            self.state.leaderboard_scroll = 0
            return
        if key in (
            curses.KEY_DOWN,
            curses.KEY_UP,
            curses.KEY_NPAGE,
            curses.KEY_PPAGE,
            curses.KEY_HOME,
            curses.KEY_END,
        ):
            # Let the user scroll the filtered results without leaving the box.
            self._handle_leaderboard_key(key)
            return
        if key in (curses.KEY_BACKSPACE, 127, 8):
            self.state.leaderboard_filter = self.state.leaderboard_filter[:-1]
            self.state.leaderboard_scroll = 0
            return
        if key == 21:  # Ctrl-U clears the query
            self.state.leaderboard_filter = ""
            self.state.leaderboard_scroll = 0
            return
        if 32 <= key <= 126:
            self.state.leaderboard_filter += chr(key)
            self.state.leaderboard_scroll = 0

    def _cycle_leaderboard_sort(self) -> None:
        try:
            index = LEADERBOARD_SORTS.index(self.state.leaderboard_sort)
        except ValueError:
            index = -1
        self.state.leaderboard_sort = LEADERBOARD_SORTS[
            (index + 1) % len(LEADERBOARD_SORTS)
        ]
        self.state.leaderboard_scroll = 0
        label = LEADERBOARD_SORT_LABELS.get(
            self.state.leaderboard_sort, self.state.leaderboard_sort
        )
        self.state.message = f"usage sorted by {label}"

    def _cycle_jobs_sort(self) -> None:
        try:
            index = JOB_SORTS.index(self.state.jobs_sort)
        except ValueError:
            index = -1
        self.state.jobs_sort = JOB_SORTS[(index + 1) % len(JOB_SORTS)]
        self.state.selected = 0
        self.state.scroll = 0
        label = JOB_SORT_LABELS.get(self.state.jobs_sort, self.state.jobs_sort)
        self.state.message = f"jobs sorted by {label}"

    def _set_history_window(self, window: str) -> None:
        if window not in HISTORY_WINDOWS:
            return
        self.state.history_window = window
        self.state.selected = 0
        self.state.scroll = 0
        self._refresh_history()
        self._clamp_selection()

    def _cycle_history_window(self) -> None:
        windows = [window for window, _label in HISTORY_FILTER_OPTIONS]
        try:
            index = windows.index(self.state.history_window)
        except ValueError:
            index = -1
        self._set_history_window(windows[(index + 1) % len(windows)])
