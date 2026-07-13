from __future__ import annotations

from ..slurm import Job
from .constants import (
    JOB_STATE_CODES,
    JOB_STATE_FILTER_OPTIONS,
    LEADERBOARD_MIN_HEIGHT,
    LEADERBOARD_MIN_WIDTH,
    MIN_TERMINAL_HEIGHT,
    MIN_TERMINAL_WIDTH,
)


def status_title(label: str, summary: dict[str, int], preferred: list[str]) -> str:
    suffix = state_summary_text(summary, preferred)
    return f" {label}: {suffix} "


def summary_title(summary: dict[str, int], preferred: list[str]) -> str:
    return f" {state_summary_text(summary, preferred)} "


def state_summary_text(summary: dict[str, int], preferred: list[str]) -> str:
    bits: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if summary.get(key):
            bits.append(f"{key}:{summary[key]}")
            seen.add(key)
    for key, value in sorted(summary.items()):
        if key not in seen and value:
            bits.append(f"{key}:{value}")
    return " ".join(bits) if bits else "none"


def page_status(selected: int, total_items: int, page_size: int) -> str:
    if total_items <= 0 or page_size <= 0:
        return "0/0"
    page_count = (total_items + page_size - 1) // page_size
    current_page = min(page_count, max(0, selected) // page_size + 1)
    return f"{current_page}/{page_count}"


def terminal_too_small(width: int, height: int) -> bool:
    return width < MIN_TERMINAL_WIDTH or height < MIN_TERMINAL_HEIGHT


def leaderboard_too_small(width: int, height: int) -> bool:
    return width < LEADERBOARD_MIN_WIDTH or height < LEADERBOARD_MIN_HEIGHT


def job_state_filter_label(states: str) -> str:
    if states.lower() == "all":
        return "all"
    selected = {
        state.strip().upper()
        for state in states.split(",")
        if state.strip()
    }
    if selected == set(JOB_STATE_CODES):
        return "all selected"
    if len(selected) > 4:
        return f"{len(selected)} states"
    for value, label in JOB_STATE_FILTER_OPTIONS:
        if states == value:
            return f"{label} ({value})"
    return states


def filter_running_jobs(jobs: list[Job]) -> list[Job]:
    return [
        job
        for job in jobs
        if job.state.upper() in {"RUNNING", "PENDING"}
    ]


def job_group_key(job: Job) -> tuple[str, str]:
    return (job.array_parent, job.name)
