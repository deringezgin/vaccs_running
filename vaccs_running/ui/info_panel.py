from __future__ import annotations

from ..slurm import (
    JOB_EFFICIENCY_WINDOWS,
    USER_INFO_WINDOWS,
    format_fairshare,
    human_bytes,
    human_duration,
    parse_storage_size,
    storage_percent,
)


def fairshare_style(value: float | None) -> tuple[str, str]:
    """Color style + label for a fairshare score.

    Fairshare runs 0..1 where 1.0 means untouched / top scheduling priority and
    values near 0 mean heavy recent use / low priority.
    """
    if value is None:
        return "muted", ""
    if value >= 0.5:
        return "good", "high priority"
    if value >= 0.125:
        return "warn", "normal"
    return "bad", "low priority"


def build_user_info_lines(
    user: str,
    snapshot: dict,
    spinner: str = "",
) -> list[list[tuple[str, str]]]:
    """Build the colored, segmented rows for the Info tab.

    Pure by design: it takes a plain snapshot dict (as produced by
    ``_info_snapshot``) and returns rows of ``(text, style)`` tuples so it can be
    unit-tested without curses. ``spinner`` is the animation frame drawn while a
    section is still loading.
    """
    rows: list[list[tuple[str, str]]] = []

    def blank() -> None:
        rows.append([("", "muted")])

    status = snapshot.get("status", "idle")
    default = snapshot.get("default", "")

    # Header ------------------------------------------------------------
    rows.append([("USER   ", "muted"), (user, "title")])
    if default:
        rows.append([("GROUP  ", "muted"), (default, "muted"), ("  primary", "muted")])
    blank()

    if status in {"idle", "loading"}:
        rows.append([(f"{spinner} loading your VACC info".strip(), "muted")])
        return rows

    # Fairshare ---------------------------------------------------------
    rows.append([("fairshare", "heading")])
    fairshare = snapshot.get("fairshare", {}) or {}
    if snapshot.get("accounts_error") and not fairshare:
        rows.append([("  ", "muted"), ("unavailable", "bad")])
    elif not fairshare:
        rows.append([("  ", "muted"), ("no account associations found", "muted")])
    else:
        for account in sorted(fairshare, key=lambda name: (name != default, name)):
            score = fairshare[account]
            style, label = fairshare_style(score)
            rows.append(
                [
                    ("  ", "muted"),
                    (f"{account:<16}", "muted"),
                    (format_fairshare(score), style),
                    ("   ", "muted"),
                    (label, style),
                ]
            )
    blank()

    rows.extend(_fairshare_forecast_lines(snapshot, spinner))
    blank()

    # Compute usage (exact hours, no bars) ------------------------------
    rows.append([("compute usage", "heading")])
    rows.append(
        [
            ("  ", "muted"),
            (f"{'':<15}", "muted"),
            (f"{'CPU-hours':>12}", "cpu"),
            ("   ", "muted"),
            (f"{'GPU-hours':>12}", "gpu"),
        ]
    )
    windows = snapshot.get("windows", {})
    for window, label in USER_INFO_WINDOWS:
        value = windows.get(window)
        base = [("  ", "muted"), (f"{label:<15}", "muted")]
        if value is None:
            rows.append(base + [(f"{spinner} …".strip(), "muted")])
        elif value == "error":
            rows.append(base + [("unavailable", "bad")])
        else:
            cpu, gpu = value
            rows.append(
                base
                + [
                    (f"{cpu:>12,}", "muted"),
                    ("   ", "muted"),
                    (f"{gpu:>12,}", "muted"),
                ]
            )
    blank()

    # Storage (GPFS) ----------------------------------------------------
    rows.extend(_gpfs_lines(snapshot, default, spinner, user))
    blank()

    # Job efficiency (last; each window streams in as it loads) ---------
    rows.extend(_efficiency_lines(snapshot, spinner))
    return rows


def _fairshare_forecast_lines(
    snapshot: dict,
    spinner: str,
) -> list[list[tuple[str, str]]]:
    """Render an explicitly conditional Fair Tree outlook."""
    forecast = snapshot.get("forecast")
    account = getattr(forecast, "account", "") or snapshot.get("default", "")
    suffix = f"  {account} estimate" if account else "  estimate"
    rows: list[list[tuple[str, str]]] = [
        [("fairshare outlook", "heading"), (suffix, "muted")]
    ]
    if forecast is None:
        rows.append([("  ", "muted"), (f"{spinner} loading".strip(), "muted")])
        return rows
    if forecast == "error":
        rows.append([("  ", "muted"), ("unavailable", "bad")])
        return rows

    rows.append(
        [
            ("  ", "muted"),
            (f"{'':<13}", "muted"),
            (f"{'idle':>9}", "muted"),
            ("   ", "muted"),
            (f"{'recent pace':>11}", "muted"),
        ]
    )
    current = forecast.current
    current_style, _label = fairshare_style(current)
    rows.append(
        [
            ("  ", "muted"),
            (f"{'now':<13}", "muted"),
            (f"{format_fairshare(current):>9}", current_style),
            ("   ", "muted"),
            (f"{format_fairshare(current):>11}", current_style),
        ]
    )
    for point in forecast.points:
        idle_style, _label = fairshare_style(point.idle)
        pace_style, _label = fairshare_style(point.recent_pace)
        rows.append(
            [
                ("  ", "muted"),
                (f"{f'in {point.days} days':<13}", "muted"),
                (f"{format_fairshare(point.idle):>9}", idle_style),
                ("   ", "muted"),
                (f"{format_fairshare(point.recent_pace):>11}", pace_style),
            ]
        )
    lookback = f"{forecast.lookback_days:g}"
    half_life = f"{forecast.half_life_days:g}"
    rows.append(
        [
            ("  idle = you run nothing; others repeat ", "muted"),
            (f"the last {lookback}d", "muted"),
        ]
    )
    rows.append(
        [
            ("  recent pace = everyone repeats ", "muted"),
            (f"the last {lookback}d", "muted"),
        ]
    )
    rows.append(
        [
            (f"  Fair Tree rank model; {half_life}d usage half-life; ", "muted"),
            ("not a queue-time guarantee", "muted"),
        ]
    )
    return rows


def _efficiency_percent_cell(percent: float | None, width: int) -> str:
    if percent is None:
        return f"{'-':>{width}}"
    return f"{percent:>{width - 1}.0f}%"


def _efficiency_lines(snapshot: dict, spinner: str) -> list[list[tuple[str, str]]]:
    """Job-efficiency table: one row per window, raw percentages + job count.

    Each window (7d/30d/1y) loads independently, so a row shows a spinner until
    its own sacct query returns. Values are used-vs-allocated percentages —
    ``CPU`` = CPU-time used / allocated, ``memory`` = peak RSS / requested,
    ``walltime`` = elapsed / time limit.
    """
    rows: list[list[tuple[str, str]]] = []
    rows.append([("job efficiency", "heading"), ("  used / allocated", "muted")])
    rows.append(
        [
            ("  ", "muted"),
            (f"{'':<14}", "muted"),
            (f"{'CPU':>7}", "muted"),
            (f"{'memory':>9}", "muted"),
            (f"{'walltime':>10}", "muted"),
            (f"{'jobs':>9}", "muted"),
        ]
    )
    efficiency = snapshot.get("efficiency", {}) or {}
    for key, _window, label in JOB_EFFICIENCY_WINDOWS:
        summary = efficiency.get(key)
        base = [("  ", "muted"), (f"{label:<14}", "muted")]
        if summary is None:
            rows.append(base + [(f"{spinner} loading".strip(), "muted")])
        elif summary == "error":
            rows.append(base + [("unavailable", "bad")])
        elif getattr(summary, "job_count", 0) == 0:
            rows.append(base + [("no finished jobs", "muted")])
        else:
            rows.append(
                base
                + [
                    (_efficiency_percent_cell(summary.cpu_percent, 7), "muted"),
                    (_efficiency_percent_cell(summary.mem_percent, 9), "muted"),
                    (_efficiency_percent_cell(summary.walltime_percent, 10), "muted"),
                    (f"{summary.job_count:>9,}", "muted"),
                ]
            )

    detail = _efficiency_detail_lines(efficiency)
    if detail:
        rows.append([("", "muted")])
        rows.extend(detail)
    return rows


def _efficiency_detail_lines(efficiency: dict) -> list[list[tuple[str, str]]]:
    """Raw 'requested X but used Y' averages for the first window with data."""
    summary = None
    label = ""
    for key, _window, window_label in JOB_EFFICIENCY_WINDOWS:
        candidate = efficiency.get(key)
        if getattr(candidate, "job_count", 0):
            summary, label = candidate, window_label
            break
    if summary is None:
        return []

    def sentence(text: str) -> list[tuple[str, str]]:
        return [("    ", "muted"), (text, "muted")]

    rows: list[list[tuple[str, str]]] = [
        [("  ", "muted"), (f"on average, per job over the {label}:", "muted")]
    ]
    if summary.cpu_alloc is not None and summary.cpu_used is not None:
        rows.append(
            sentence(
                f"requested {summary.cpu_alloc:.1f} CPU cores but used "
                f"{summary.cpu_used:.1f}"
            )
        )
    if summary.mem_req_bytes is not None and summary.mem_used_bytes is not None:
        rows.append(
            sentence(
                f"requested {human_bytes(summary.mem_req_bytes)} of memory but "
                f"used {human_bytes(summary.mem_used_bytes)}"
            )
        )
    if summary.walltime_limit_sec is not None and summary.walltime_used_sec is not None:
        rows.append(
            sentence(
                f"requested {human_duration(summary.walltime_limit_sec)} of "
                f"walltime but used {human_duration(summary.walltime_used_sec)}"
            )
        )
    return rows


def _gpfs_lines(
    snapshot: dict,
    default: str,
    spinner: str,
    user: str,
) -> list[list[tuple[str, str]]]:
    rows: list[list[tuple[str, str]]] = []
    gpfs = snapshot.get("gpfs")
    group = getattr(gpfs, "primary_group", "") or default
    rows.append(
        [("storage", "heading")]
        + ([("  ", "muted"), (f"{group} group", "muted")] if group else [])
    )
    if gpfs is None:
        if snapshot.get("gpfs_error"):
            rows.append([("  ", "muted"), ("unavailable", "bad")])
        else:
            rows.append([("  ", "muted"), (f"{spinner} …".strip(), "muted")])
        return rows

    space_grace = dict(gpfs.group_space_grace)
    for filesystem, used, quota, _limit in gpfs.group_space:
        percent = storage_percent(used, quota)
        if percent is None:
            percent_style, percent_text = "muted", ""
        else:
            percent_text = f"({percent:.0f}%)"
            if percent >= 90:
                percent_style = "bad"
            elif percent >= 75:
                percent_style = "warn"
            else:
                percent_style = "good"
        rows.append(
            [
                ("  ", "muted"),
                (f"{filesystem:<10}", "muted"),
                (f"{used:>9} / {quota:<8}", "muted"),
                ("  ", "muted"),
                (percent_text, percent_style),
            ]
        )
        grace_text, grace_style = _gpfs_grace_status(
            space_grace.get(filesystem, ""), "storage"
        )
        if grace_text:
            rows.append([(" " * 21, "muted"), (grace_text, grace_style)])

    files_grace = dict(gpfs.group_files_grace)
    for filesystem, used, quota, limit in gpfs.group_files:
        percent, percent_style = _file_quota_percent(used, quota)
        soft_status, soft_style = _file_quota_remaining(used, quota, "soft")
        hard_status, hard_style = _file_quota_remaining(used, limit, "hard")
        used_text = _format_file_count(used)
        quota_text = _format_file_count(quota)
        rows.append(
            [
                ("  ", "muted"),
                (f"{filesystem:<10}", "muted"),
                ("files  ", "muted"),
                (f"{used_text:>13} / {quota_text:<13}", "muted"),
                (" soft", "muted"),
                ("  ", "muted"),
                (percent, percent_style),
            ]
        )
        if soft_status or hard_status:
            rows.append(
                [
                    (" " * 21, "muted"),
                    (soft_status, soft_style),
                    ("  ·  " if soft_status and hard_status else "", "muted"),
                    (hard_status, hard_style),
                ]
            )
        grace_text, grace_style = _gpfs_grace_status(
            files_grace.get(filesystem, ""), "file"
        )
        if grace_text:
            rows.append([(" " * 21, "muted"), (grace_text, grace_style)])

    merged: dict[str, dict[str, str]] = {}
    for filesystem, used in gpfs.personal_space:
        merged.setdefault(filesystem, {})["space"] = used
    for filesystem, files in gpfs.personal_files:
        merged.setdefault(filesystem, {})["files"] = files
    if merged:
        rows.append([("", "muted")])
        rows.append([("  ", "muted"), ("your usage", "muted")])
        for filesystem in sorted(merged):
            space = merged[filesystem].get("space", "-")
            files = merged[filesystem].get("files", "-")
            files_text = f"{int(files):,}" if files.isdigit() else files
            rows.append(
                [
                    ("  ", "muted"),
                    (f"{filesystem:<10}", "muted"),
                    (f"{space:>9}", "muted"),
                    ("   ", "muted"),
                    (f"{files_text:>13} files", "muted"),
                ]
            )
    rows.extend(_gpfs_group_top_lines(snapshot, user))
    return rows


def _gpfs_group_top_lines(
    snapshot: dict,
    user: str,
) -> list[list[tuple[str, str]]]:
    usage = snapshot.get("gpfs_group_usage")
    rows: list[list[tuple[str, str]]] = [[("", "muted")]]
    if usage is None:
        rows.append(
            [
                ("  ", "muted"),
                ("group top 5", "muted"),
                ("  unavailable", "bad"),
            ]
        )
        return rows

    rows.append([("  ", "muted"), ("group top 5", "muted")])
    for filesystem in ("gpfs1", "gpfs2"):
        entries = [entry for entry in usage if entry.filesystem == filesystem]
        by_space = sorted(
            entries,
            key=lambda entry: (-(parse_storage_size(entry.space) or 0), entry.user),
        )[:5]
        by_files = sorted(entries, key=lambda entry: (-entry.files, entry.user))[:5]
        if not by_space and not by_files:
            continue

        rows.append([("", "muted")])
        rows.append([("  ", "muted"), (filesystem, "heading")])
        rows.append(
            [
                ("    ", "muted"),
                (f"{'storage':<22}", "muted"),
                ("files", "muted"),
            ]
        )
        for index in range(max(len(by_space), len(by_files))):
            row: list[tuple[str, str]] = [("  ", "muted")]
            if index < len(by_space):
                entry = by_space[index]
                style = "title" if entry.user == user else "muted"
                row.extend(
                    [
                        (f"{index + 1:>2} ", style),
                        (f"{entry.user:<11}", style),
                        (f"{entry.space:>8}", style),
                    ]
                )
            else:
                row.append((" " * 22, "muted"))
            row.append(("  ", "muted"))
            if index < len(by_files):
                entry = by_files[index]
                style = "title" if entry.user == user else "muted"
                row.extend(
                    [
                        (f"{index + 1:>2} ", style),
                        (f"{entry.user:<11}", style),
                        (f"{entry.files:>11,}", style),
                    ]
                )
            rows.append(row)
    return rows


def _parse_file_count(value: str) -> int | None:
    try:
        return int(value.replace(",", ""))
    except (AttributeError, ValueError):
        return None


def _gpfs_grace_status(value: str, quota_type: str) -> tuple[str, str]:
    value = value.strip()
    if not value or value.lower() in {"none", "n/a"}:
        return "", "muted"
    if value.lower() == "expired":
        return f"{quota_type} grace expired", "bad"
    return f"{quota_type} grace: {value} left", "warn"


def _format_file_count(value: str) -> str:
    parsed = _parse_file_count(value)
    return f"{parsed:,}" if parsed is not None else value


def _file_quota_percent(used: str, quota: str) -> tuple[str, str]:
    used_count = _parse_file_count(used)
    quota_count = _parse_file_count(quota)
    if used_count is None or not quota_count:
        return "", "muted"
    percent = 100.0 * used_count / quota_count
    if percent >= 90:
        style = "bad"
    elif percent >= 75:
        style = "warn"
    else:
        style = "good"
    return f"({percent:.0f}%)", style


def _file_quota_remaining(used: str, limit: str, label: str) -> tuple[str, str]:
    used_count = _parse_file_count(used)
    limit_count = _parse_file_count(limit)
    if used_count is None or not limit_count:
        return "", "muted"
    remaining = limit_count - used_count
    if remaining >= 0:
        return f"{remaining:,} {label} left", "muted"
    return f"{-remaining:,} over {label}", "bad"
