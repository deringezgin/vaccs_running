from __future__ import annotations

import curses

from .ui_constants import (
    ACTIVE_TAB_PAIR,
    BORDER_PAIR,
    MUTED_PAIR,
    SURFACE_PAIR,
    TEXT_PAIR,
    TITLE_PAIR,
)


class ColorMixin:
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
