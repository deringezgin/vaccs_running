from __future__ import annotations


def human_bytes(value: float) -> str:
    """Compact size like '96G', '7.3G', '512M' from a byte count."""
    size = float(value)
    for unit in ("B", "K", "M", "G", "T", "P"):
        if size < 1024 or unit == "P":
            if unit in {"B", "K", "M"}:
                return f"{size:.0f}{unit}"
            text = f"{size:.1f}"
            if text.endswith(".0"):
                text = text[:-2]
            return f"{text}{unit}"
        size /= 1024
    return f"{size:.0f}P"


def human_duration(seconds: float) -> str:
    """Compact duration like '36h', '1h 3m', '2d 4h', '45m' from seconds."""
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h" if hours else f"{days}d"
    if hours:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    return f"{minutes}m"


def _efficiency_report_line(label: str, percent: float | None, detail: str) -> str:
    percent_text = f"{percent:.0f}%" if percent is not None else "n/a"
    line = f"{label:<9} {percent_text:>4}"
    return f"{line}   {detail}" if detail else line


def _cores_detail(used: float | None, alloc: float | None) -> str:
    if used is None or alloc is None:
        return ""
    return f"used {used:.1f} of {alloc:.1f} cores"


def _bytes_detail(used: float | None, requested: float | None) -> str:
    if used is None or requested is None:
        return ""
    return f"used {human_bytes(used)} of {human_bytes(requested)}"


def _time_detail(used: float | None, limit: float | None) -> str:
    if used is None or limit is None:
        return ""
    return f"ran {human_duration(used)} of {human_duration(limit)}"


def format_node_jobs(jobs: list[dict[str, str]]) -> str:
    columns = [
        ("job_id", "JOBID", 12),
        ("user", "USER", 10),
        ("state", "STATE", 8),
        ("elapsed", "ELAPSED", 8),
        ("limit", "LIMIT", 8),
        ("cpus", "CPUS", 4),
        ("gres", "GRES", 12),
        ("name", "JOB", 18),
    ]
    widths = []
    for key, label, minimum in columns:
        widths.append(max(minimum, len(label), *(len(job.get(key, "")) for job in jobs)))

    header = "  ".join(label.ljust(width) for (_, label, _), width in zip(columns, widths))
    divider = "-" * len(header)
    rows = [
        "  ".join(job.get(key, "").ljust(width) for (key, _, _), width in zip(columns, widths))
        for job in jobs
    ]
    return "\n".join([header, divider, *rows])


def human_mb(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f}T"
    if value >= 1024:
        return f"{value / 1024:.0f}G"
    return f"{value}M"


def human_hours(value: int) -> str:
    """Compact hour count for narrow columns: 123, 1.3k, 76k, 1.8M."""
    # 999_500+ would round up to "1000k", so promote it into the M scale.
    if value >= 999_500:
        return f"{value / 1_000_000:.1f}M"
    if value >= 10_000:
        return f"{value / 1000:.0f}k"
    if value >= 1_000:
        return f"{value / 1000:.1f}k"
    return str(value)


def format_fairshare(value: float | None) -> str:
    if value is None:
        return "-"
    if value == float("inf"):
        return "∞"
    return f"{value:.5f}".rstrip("0").rstrip(".")
