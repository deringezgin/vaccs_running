from __future__ import annotations


STATE_COLORS = {
    "RUNNING": 1,
    "PENDING": 2,
    "COMPLETED": 3,
    "FAILED": 4,
    "CANCELLED": 4,
    "NODE_FAIL": 4,
    "OUT_OF_MEMORY": 4,
    "PREEMPTED": 4,
    "TIMEOUT": 4,
}


NODE_COLORS = {
    "IDLE": 1,
    "MIXED": 2,
    "ALLOCATED": 3,
    "DOWN": 4,
    "DRAIN": 4,
    "DRAINED": 4,
}


BORDER_PAIR = 9


TEXT_PAIR = 10


ACTIVE_TAB_PAIR = 11


TITLE_PAIR = 12


MUTED_PAIR = 13


SURFACE_PAIR = 14


MIN_TERMINAL_WIDTH = 70


MIN_TERMINAL_HEIGHT = 16


LEADERBOARD_MIN_WIDTH = 120


LEADERBOARD_MIN_HEIGHT = 20


LEADERBOARD_PAGE = 10


LEADERBOARD_GRID_TOP = 5


LEADERBOARD_SORT_DISPLAY = ["gpu", "cpu", "fairshare"]


LEADERBOARD_SORT_SHORT = {"cpu": "CPU", "gpu": "GPU", "fairshare": "fairshare"}


JOB_SORT_OPTIONS = [
    ("job_id", "job-id"),
    ("state", "state"),
    ("elapsed", "elapsed"),
]


JOB_SORTS = [key for key, _label in JOB_SORT_OPTIONS]


JOB_SORT_LABELS = dict(JOB_SORT_OPTIONS)


HISTORY_REFRESH_SECONDS = 10.0


BUSY_JOBS_REFRESH_SECONDS = 5.0


BUSY_JOBS_REFRESH_THRESHOLD = 50


# The Priority view includes a sprio RPC in addition to its queue snapshot.
# Slurm explicitly discourages calling sprio in tight loops, so keep this view
# substantially slower than the lightweight per-user Jobs refresh.
PRIORITY_REFRESH_SECONDS = 30.0


# Priority's one-key GPU queue shortcut intentionally excludes gpu-debug.
PRIORITY_GPU_PARTITIONS = ("nvgpu", "gpu-preempt")


HISTORY_FILTER_OPTIONS = [
    ("1h", "last 1 hour"),
    ("3h", "last 3 hours"),
    ("24h", "last 24 hours"),
    ("3d", "last 3 days"),
    ("7d", "last 7 days"),
]


JOB_STATE_FILTER_OPTIONS = [
    ("BF", "BOOT_FAIL"),
    ("CA", "CANCELLED"),
    ("CD", "COMPLETED"),
    ("CF", "CONFIGURING"),
    ("CG", "COMPLETING"),
    ("DL", "DEADLINE"),
    ("F", "FAILED"),
    ("NF", "NODE_FAIL"),
    ("OOM", "OUT_OF_MEMORY"),
    ("PD", "PENDING"),
    ("PR", "PREEMPTED"),
    ("R", "RUNNING"),
    ("RD", "RESV_DEL_HOLD"),
    ("RF", "REQUEUE_FED"),
    ("RH", "REQUEUE_HOLD"),
    ("RQ", "REQUEUED"),
    ("RS", "RESIZING"),
    ("RV", "REVOKED"),
    ("SI", "SIGNALING"),
    ("SE", "SPECIAL_EXIT"),
    ("SO", "STAGE_OUT"),
    ("ST", "STOPPED"),
    ("S", "SUSPENDED"),
    ("TO", "TIMEOUT"),
]


JOB_STATE_CODES = [state for state, _ in JOB_STATE_FILTER_OPTIONS]


USER_INFO_STYLE_PAIRS = {
    "title": TITLE_PAIR,
    "heading": 6,
    "muted": MUTED_PAIR,
    "cpu": 3,
    "gpu": 6,
    "good": 1,
    "warn": 2,
    "bad": 4,
    "accent": ACTIVE_TAB_PAIR,
}


USER_INFO_BOLD_STYLES = {"title", "heading"}


INFO_TOP = 5


SPINNER_FRAMES = "|/-\\"


# One source of truth for the visible top-level tabs and their global keys.
# Keep the established tabs in place and append Priority so existing muscle
# memory and narrow-terminal layouts do not shift unexpectedly.
TOP_TABS = (
    ("jobs", "j", "Jobs"),
    ("nodes", "n", "Nodes"),
    ("history", "h", "History"),
    ("leaderboard", "u", "Usage"),
    ("info", "i", "Info"),
    ("priority", "w", "Priority"),
)
