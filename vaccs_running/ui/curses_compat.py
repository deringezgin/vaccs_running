from __future__ import annotations

import curses


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
