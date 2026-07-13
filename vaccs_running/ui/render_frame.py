from __future__ import annotations

import curses
import time

from .constants import (
    ACTIVE_TAB_PAIR,
    HISTORY_FILTER_OPTIONS,
    INFO_TOP,
    LEADERBOARD_SORT_DISPLAY,
    LEADERBOARD_SORT_SHORT,
    MIN_TERMINAL_HEIGHT,
    MIN_TERMINAL_WIDTH,
    MUTED_PAIR,
    SPINNER_FRAMES,
    TITLE_PAIR,
    USER_INFO_BOLD_STYLES,
    USER_INFO_STYLE_PAIRS,
)
from .summaries import (
    job_state_filter_label,
    leaderboard_too_small,
    terminal_too_small,
)
from .info_panel import build_user_info_lines


class RenderFrameMixin:
    def _draw(self, stdscr: curses.window) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if self.state.view == "leaderboard":
            if leaderboard_too_small(width, height):
                self._draw_leaderboard_too_small(stdscr, width, height)
                stdscr.refresh()
                return
            self._draw_header(stdscr, width)
            self._draw_leaderboard(stdscr, height, width)
            stdscr.refresh()
            return
        if terminal_too_small(width, height):
            self._draw_terminal_too_small(stdscr, width, height)
            stdscr.refresh()
            return
        self._draw_header(stdscr, width)
        if self.state.view == "info":
            self._draw_info(stdscr, height, width)
        elif self.state.view == "nodes":
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
            ("leaderboard", " u Usage "),
            ("info", " i Info "),
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
            self._addstr(stdscr, 3, x, " a activity ", self._pair(MUTED_PAIR))
            x += len(" a activity ") + 1
        elif self.state.view == "history":
            x = 1
            # " f filter: 1h / 3h / 24h / 3d / 7d " with the active window highlighted.
            x = self._draw_header_choice(
                stdscr,
                x,
                " f filter: ",
                [(window, window) for window, _label in HISTORY_FILTER_OPTIONS],
                self.state.history_window,
            )
            self._addstr(stdscr, 3, x, " e efficiency ", self._pair(MUTED_PAIR))
            x += len(" e efficiency ") + 1
        elif self.state.view == "leaderboard":
            x = 1
            # Each control lists every option with the active one highlighted.
            # (Pressing 'r' still refreshes; it is intentionally not advertised.)
            x = self._draw_header_choice(
                stdscr,
                x,
                " m mode: ",
                [("user", "user"), ("group", "group")],
                "group" if self.state.leaderboard_group_mode else "user",
            )
            if self.state.leaderboard_filter_editing:
                find_text = f" f find: {self.state.leaderboard_filter}_ "
                find_attr = ACTIVE_TAB_PAIR
            elif self.state.leaderboard_filter:
                find_text = f" f find: {self.state.leaderboard_filter} "
                find_attr = ACTIVE_TAB_PAIR
            else:
                find_text = " f find "
                find_attr = MUTED_PAIR
            self._addstr(stdscr, 3, x, find_text, self._pair(find_attr))
            x += len(find_text) + 1
            x = self._draw_header_choice(
                stdscr,
                x,
                " s sort: ",
                [
                    (key, LEADERBOARD_SORT_SHORT.get(key, key))
                    for key in LEADERBOARD_SORT_DISPLAY
                ],
                self.state.leaderboard_sort,
            )
            x = self._draw_header_choice(
                stdscr,
                x,
                " o order: ",
                [("ascending", "ascending"), ("descending", "descending")],
                "ascending" if self.state.leaderboard_ascending else "descending",
            )
        elif self.state.view == "info":
            x = 1
            refresh_text = " r refresh "
            self._addstr(stdscr, 3, x, refresh_text, self._pair(MUTED_PAIR))
            x += len(refresh_text) + 1
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

        # Every view carries a right-aligned quit hint on the controls row.
        quit_label = "q quit"
        self._addstr(
            stdscr,
            3,
            max(x, width - len(quit_label) - 2),
            quit_label,
            self._pair(MUTED_PAIR),
        )

    def _draw_header_choice(
        self,
        stdscr: curses.window,
        x: int,
        prefix: str,
        options: list[tuple[str, str]],
        active: str,
    ) -> int:
        """Draw ``prefix`` then ``a/b/c`` options, highlighting the active one.

        Returns the new x cursor (with a trailing gap) for the next segment.
        """
        self._addstr(stdscr, 3, x, prefix, self._pair(MUTED_PAIR))
        x += len(prefix)
        for index, (value, label) in enumerate(options):
            if index:
                self._addstr(stdscr, 3, x, " / ", self._pair(MUTED_PAIR))
                x += 3
            attr = (
                self._pair(ACTIVE_TAB_PAIR) | curses.A_BOLD
                if value == active
                else self._pair(MUTED_PAIR)
            )
            self._addstr(stdscr, 3, x, label, attr)
            x += len(label)
        return x + 2  # trailing space + gap before the next segment

    def _draw_info(self, stdscr: curses.window, height: int, width: int) -> None:
        """Full-screen Info tab: the current user's account, usage & storage."""
        user = getattr(self.client, "user", "") or "me"
        snapshot = self._info_snapshot()
        spinner = SPINNER_FRAMES[int(time.monotonic() * 6) % len(SPINNER_FRAMES)]
        rows = build_user_info_lines(user, snapshot, spinner)

        body_height = max(1, height - INFO_TOP)
        max_scroll = max(0, len(rows) - body_height)
        self.state.info_scroll = max(0, min(self.state.info_scroll, max_scroll))
        scroll = self.state.info_scroll

        for offset, row in enumerate(rows[scroll : scroll + body_height]):
            self._draw_info_row(stdscr, INFO_TOP + offset, row, width)
        if scroll < max_scroll:
            hint = "↓ more "
            self._addstr(
                stdscr, height - 1, max(2, width - len(hint) - 2), hint,
                self._pair(MUTED_PAIR),
            )

    def _draw_info_row(
        self,
        win: curses.window,
        y: int,
        row: list[tuple[str, str]],
        width: int,
    ) -> None:
        x = 2
        for text, style in row:
            if x >= width - 1:
                break
            attr = self._pair(USER_INFO_STYLE_PAIRS.get(style, MUTED_PAIR))
            if style in USER_INFO_BOLD_STYLES:
                attr |= curses.A_BOLD
            self._addstr(win, y, x, text, attr)
            x += len(text)
