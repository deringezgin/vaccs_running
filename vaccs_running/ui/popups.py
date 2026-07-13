from __future__ import annotations

import curses
import time

from collections.abc import Callable

from .text_layout import (
    popup_geometry,
    wrap_lines,
)
from ..slurm import (
    SlurmError,
    format_job_efficiency,
)


def command_text(fn, job_id: str) -> str:
    try:
        return fn(job_id).strip() or "No output."
    except SlurmError as exc:
        return str(exc)

class PopupMixin:
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
            "running activity by user",
            command_text(lambda _: self.client.cluster_usage(), ""),
            close_keys=(ord("a"),),
        )

    def _show_job_efficiency(self, stdscr: curses.window) -> None:
        group = self._selected_history_group()
        if not group:
            return
        job_id = group.array_parent
        name = group.name
        self._popup(
            stdscr,
            f"efficiency · {job_id}",
            lambda: self._job_efficiency_text(job_id, name),
            close_keys=(ord("e"),),
            refresh_while_open=False,
        )

    def _job_efficiency_text(self, job_id: str, name: str) -> str:
        try:
            summary = self.client.fetch_job_efficiency_for(job_id)
        except SlurmError as exc:
            return str(exc)
        return format_job_efficiency(summary, job_id, name)

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
