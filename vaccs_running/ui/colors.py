from __future__ import annotations

import curses

from .constants import (
    ACTIVE_TAB_PAIR,
    BORDER_PAIR,
    MUTED_PAIR,
    SURFACE_PAIR,
    TEXT_PAIR,
    TITLE_PAIR,
)


# Fixed xterm-256 palette entries. Unlike the configurable ANSI colors 0-15,
# entries 16-255 have stable RGB definitions across compatible terminals.
# This is the same deterministic low-color strategy used by btop when truecolor
# is unavailable.
THEME_256 = {
    "background": 16,    # #000000
    "foreground": 255,  # #eeeeee
    "muted": 250,       # #bcbcbc
    "green": 77,        # #5fd75f
    "yellow": 221,      # #ffd75f
    "cyan": 80,         # #5fd7d7
    "red": 203,         # #ff5f5f
    "orange": 173,      # #d7875f
}


# Deliberate fallback for terminals that advertise fewer than 256 colors.
# Exact RGB consistency is impossible there, but every role still follows one
# stable mapping and the background remains explicitly black.
THEME_16 = {
    "background": curses.COLOR_BLACK,
    "foreground": curses.COLOR_WHITE,
    "muted": curses.COLOR_WHITE,
    "green": curses.COLOR_GREEN,
    "yellow": curses.COLOR_YELLOW,
    "cyan": curses.COLOR_CYAN,
    "red": curses.COLOR_RED,
    "orange": curses.COLOR_YELLOW,
}


THEME_PAIR_ROLES = {
    1: ("green", "background"),
    2: ("yellow", "background"),
    3: ("cyan", "background"),
    4: ("red", "background"),
    5: ("orange", "background"),
    6: ("orange", "background"),
    7: ("background", "orange"),
    8: ("foreground", "red"),
    BORDER_PAIR: ("orange", "background"),
    TEXT_PAIR: ("foreground", "background"),
    ACTIVE_TAB_PAIR: ("background", "orange"),
    TITLE_PAIR: ("foreground", "background"),
    MUTED_PAIR: ("muted", "background"),
    SURFACE_PAIR: ("foreground", "background"),
}


class ColorMixin:
    def _init_colors(self) -> None:
        self.colors_enabled = False
        try:
            if not curses.has_colors():
                return
            curses.start_color()
            palette = THEME_256 if curses.COLORS >= 256 else THEME_16
            for pair_id, (foreground, background) in THEME_PAIR_ROLES.items():
                curses.init_pair(
                    pair_id,
                    palette[foreground],
                    palette[background],
                )
            self.colors_enabled = True
        except (AttributeError, curses.error):
            self.colors_enabled = False

    def _apply_theme_background(self, win: curses.window) -> None:
        """Make blank cells and unstyled text use the themed black surface."""
        try:
            win.bkgd(" ", self._pair(SURFACE_PAIR))
        except (AttributeError, curses.error):
            pass
