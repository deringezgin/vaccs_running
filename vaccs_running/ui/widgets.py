from __future__ import annotations


def meter(percent: float, width: int) -> str:
    bounded = max(0.0, min(100.0, percent))
    inner = max(1, width)
    filled = round(inner * bounded / 100.0)
    return "[" + "|" * filled + "." * (inner - filled) + "]"


def resource_count_width(pairs: list[tuple[int, int]]) -> int:
    return max((len(f"{used}/{total}") for used, total in pairs), default=0)


def resource_text_width(values: list[str]) -> int:
    return max((len(value) for value in values), default=0)


def resource_meter(
    used: int,
    total: int,
    percent: float,
    *,
    meter_width: int,
    count_width: int,
) -> str:
    count = f"{used}/{total}".rjust(count_width)
    return f"{count} {meter(percent, meter_width)}"


def resource_text_meter(
    text: str,
    percent: float,
    *,
    meter_width: int,
    count_width: int,
) -> str:
    return f"{text.rjust(count_width)} {meter(percent, meter_width)}"


def pct(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * value / total
