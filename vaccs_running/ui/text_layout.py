from __future__ import annotations

import textwrap


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
