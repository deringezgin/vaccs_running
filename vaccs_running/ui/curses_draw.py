from __future__ import annotations

import curses

from .constants import (
    BORDER_PAIR,
    NODE_COLORS,
    STATE_COLORS,
)
from ..slurm import Node


class CursesMixin:
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
        vertical_length = bottom - top - 1
        vertical = getattr(curses, "ACS_VLINE", "│")
        try:
            win.vline(top + 1, left, vertical, vertical_length, attr)
            win.vline(top + 1, right, vertical, vertical_length, attr)
        except (AttributeError, curses.error):
            for y in range(top + 1, bottom):
                self._addstr(win, y, left, "│", attr)
                self._addstr(win, y, right, "│", attr)
        if title:
            self._addstr(win, top, left + 2, title[: max(0, width - 4)], self._pair(5) | curses.A_BOLD)
