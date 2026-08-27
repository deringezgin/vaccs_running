from __future__ import annotations

import textwrap
from collections.abc import Sequence


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


def wrap_detail_blocks(
    rows: Sequence[Sequence[str]],
    width: int,
) -> list[str]:
    """Wrap detail rows without splitting fields that fit on their own."""
    available = max(1, width)
    wrapped: list[str] = []
    for blocks in rows:
        current = ""
        continuation = False
        for block in blocks:
            separator = "  " if current else ("  " if continuation else "")
            candidate = f"{current}{separator}{block}"
            if len(candidate) <= available:
                current = candidate
                continue

            if current:
                wrapped.append(current)
                current = ""
                continuation = True

            prefix = "  " if continuation else ""
            if len(prefix) + len(block) <= available:
                current = f"{prefix}{block}"
                continue

            pieces = textwrap.wrap(
                block,
                width=max(1, available - len(prefix)),
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            wrapped.extend(f"{prefix}{piece}" for piece in pieces[:-1])
            current = f"{prefix}{pieces[-1]}"
            continuation = True

        if current:
            wrapped.append(current)
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
