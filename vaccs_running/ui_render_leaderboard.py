from __future__ import annotations

import curses

from .ui_constants import (
    LEADERBOARD_GRID_TOP,
    LEADERBOARD_MIN_HEIGHT,
    LEADERBOARD_MIN_WIDTH,
    MUTED_PAIR,
)
from .table_layout import leaderboard_columns
from .slurm import (
    LEADERBOARD_WINDOWS,
    LeaderboardRow,
    format_fairshare,
    human_hours,
)


class RenderLeaderboardMixin:
    def _draw_leaderboard_too_small(
        self,
        stdscr: curses.window,
        width: int,
        height: int,
    ) -> None:
        lines = [
            ("The Usage view needs a bigger screen.", curses.A_BOLD),
            ("", 0),
            ("It is a desktop / laptop view -- phone", 0),
            ("terminals are too narrow to show it.", 0),
            ("", 0),
            (f"Current size:  {width} x {height}", curses.A_BOLD),
            (
                f"Needed:        {LEADERBOARD_MIN_WIDTH} x {LEADERBOARD_MIN_HEIGHT}",
                curses.A_BOLD,
            ),
            ("", 0),
            ("j n h  other views      q  quit", 0),
        ]
        top = max(0, (height - len(lines)) // 2)
        for offset, (line, attr) in enumerate(lines):
            if not line:
                continue
            x = max(0, (width - len(line)) // 2)
            self._addstr(stdscr, top + offset, x, line, self._pair(MUTED_PAIR) | attr)

    def _draw_leaderboard(
        self,
        stdscr: curses.window,
        height: int,
        width: int,
    ) -> None:
        snapshot = self._leaderboard_snapshot()
        grid_top = LEADERBOARD_GRID_TOP
        grid_height = max(0, height - grid_top)
        # One row of side-by-side panes, one column per window.
        rows, cols = 1, max(1, len(LEADERBOARD_WINDOWS))
        pane_height = grid_height // rows
        pane_width = width // cols

        # Clamp the shared scroll so the deepest pane's last row stays reachable.
        body_capacity = max(1, pane_height - 3)
        longest = max((len(info["rows"]) for info in snapshot.values()), default=0)
        # Size the rank column for the largest rank actually shown (ranks are
        # preserved through filtering, so they can exceed the visible row count).
        ranks = [rank for info in snapshot.values() for rank, _ in info["rows"]]
        max_rank = max(ranks) if ranks else 1
        max_scroll = max(0, longest - body_capacity)
        self.state.leaderboard_scroll = max(
            0, min(self.state.leaderboard_scroll, max_scroll)
        )
        scroll = self.state.leaderboard_scroll

        for index, (window, label) in enumerate(LEADERBOARD_WINDOWS):
            grid_row = index // cols
            grid_col = index % cols
            pane_top = grid_top + grid_row * pane_height
            pane_left = grid_col * pane_width
            # Right column and bottom row stretch to fill any integer remainder.
            this_width = width - pane_left if grid_col == cols - 1 else pane_width
            this_height = (
                height - pane_top if grid_row == rows - 1 else pane_height
            )
            self._draw_leaderboard_pane(
                stdscr,
                pane_top,
                pane_left,
                this_height,
                this_width,
                label,
                snapshot[window],
                scroll,
                max_rank,
            )

    def _draw_leaderboard_pane(
        self,
        stdscr: curses.window,
        top: int,
        left: int,
        height: int,
        width: int,
        label: str,
        info: dict[str, object],
        scroll: int,
        max_rank: int = 1,
    ) -> None:
        status = str(info["status"])
        rows: list[tuple[int, LeaderboardRow]] = info["rows"]  # type: ignore[assignment]
        suffix = {
            "ready": f"{len(rows)}",
            "loading": "loading...",
            "error": "error",
            "idle": "idle",
        }.get(status, status)
        self._draw_box(stdscr, top, left, height, width, f" {label} - {suffix} ")
        inner_left = left + 2
        inner_width = max(0, width - 4)
        if inner_width <= 0 or height < 4:
            return

        if status == "loading":
            self._addstr(
                stdscr, top + 2, inner_left, "running slurm query..."[:inner_width],
                self._pair(2),
            )
            return
        if status == "error":
            message = str(info["error"]) or "query failed"
            self._addstr(stdscr, top + 2, inner_left, message[:inner_width], self._pair(4))
            return
        if status == "idle":
            self._addstr(
                stdscr, top + 2, inner_left, "press r to load"[:inner_width],
                self._pair(MUTED_PAIR),
            )
            return
        if not rows:
            if self.state.leaderboard_filter:
                empty = f'no match for "{self.state.leaderboard_filter}"'
            else:
                empty = "no usage in this window"
            self._addstr(
                stdscr, top + 2, inner_left, empty[:inner_width],
                self._pair(MUTED_PAIR),
            )
            return

        user_mode = not self.state.leaderboard_group_mode
        entity = "USER" if user_mode else "GROUP"
        # Show each user's PI group only in user mode (group rows are groups).
        columns = leaderboard_columns(inner_width, entity, max_rank, group_col=user_mode)
        self._draw_leaderboard_row(
            stdscr,
            top + 1,
            inner_left,
            columns,
            [label for _key, label, _w, _a in columns],
            self._pair(MUTED_PAIR) | curses.A_BOLD,
        )
        body_capacity = max(0, height - 3)
        for offset, (rank, row) in enumerate(rows[scroll : scroll + body_capacity]):
            cells = {
                "rank": str(rank),
                "name": row.name,
                "group": row.group or "-",
                "cpu": human_hours(row.cpu_hours),
                "gpu": human_hours(row.gpu_hours),
                "fs": format_fairshare(row.fairshare),
            }
            self._draw_leaderboard_row(
                stdscr,
                top + 2 + offset,
                inner_left,
                columns,
                [cells[key] for key, _label, _w, _a in columns],
                self._pair(MUTED_PAIR),
            )

    def _draw_leaderboard_row(
        self,
        stdscr: curses.window,
        y: int,
        x: int,
        columns: list[tuple[str, str, int, str]],
        values: list[str],
        attr: int,
    ) -> None:
        cursor = x
        for (_key, _label, col_width, align), value in zip(columns, values):
            if col_width <= 0:
                continue
            text = value[:col_width]
            text = text.rjust(col_width) if align == "r" else text.ljust(col_width)
            self._addstr(stdscr, y, cursor, text, attr)
            cursor += col_width + 1
