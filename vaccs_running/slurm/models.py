from __future__ import annotations

from dataclasses import dataclass, field
import re

from .constants import FAILED_STATES
from .primitives import (
    parse_gpu_count,
    state_base,
)
from .format import human_mb


@dataclass(frozen=True)
class Job:
    job_id: str
    name: str
    state: str
    partition: str
    nodes: str
    reason: str
    elapsed: str
    limit: str
    node_count: str
    cpus: str
    gres: str
    submit_time: str
    start_time: str
    user: str = ""
    group: str = ""

    @property
    def array_parent(self) -> str:
        return self.job_id.split("_", 1)[0]

    @property
    def is_running(self) -> bool:
        return self.state.upper() == "RUNNING"

    @property
    def location(self) -> str:
        if self.nodes and self.nodes not in {"(null)", "N/A"}:
            return self.nodes
        if self.reason and self.reason not in {"None", "N/A"}:
            return f"pending: {self.reason}"
        return "-"


@dataclass(frozen=True)
class JobGroup:
    array_parent: str
    name: str
    total: int
    completed: int
    running: int
    pending: int
    failed: int
    other: int
    longest_running_elapsed: str
    limit: str

    @property
    def done_text(self) -> str:
        return f"{self.completed}/{self.total}"

    @property
    def dominant_state(self) -> str:
        if self.running:
            return "RUNNING"
        if self.pending:
            return "PENDING"
        if self.failed:
            return "FAILED"
        if self.completed == self.total and self.total:
            return "COMPLETED"
        return "UNKNOWN"


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    raw_job_id: str
    name: str
    state: str
    partition: str
    nodes: str
    elapsed: str
    limit: str
    node_count: str
    cpus: str
    tres: str
    submit_time: str
    start_time: str
    end_time: str
    exit_code: str
    reason: str = ""
    source: str = "sacct"
    user: str = ""
    group: str = ""

    @property
    def array_parent(self) -> str:
        return self.job_id.split("_", 1)[0]

    @property
    def base_state(self) -> str:
        return state_base(self.state)

    @property
    def is_active(self) -> bool:
        return self.base_state in {"RUNNING", "PENDING"}

    @property
    def is_running(self) -> bool:
        return self.base_state == "RUNNING"

    @property
    def is_pending(self) -> bool:
        return self.base_state == "PENDING"

    @property
    def is_failed(self) -> bool:
        return self.base_state in FAILED_STATES

    @property
    def end_text(self) -> str:
        if self.end_time and self.end_time not in {"Unknown", "None", "N/A"}:
            return self.end_time
        if self.is_running:
            return "running"
        if self.is_pending:
            return "pending"
        return "-"

    @property
    def location(self) -> str:
        if self.nodes and self.nodes not in {"(null)", "N/A", "None", "Unknown"}:
            return self.nodes
        if self.reason and self.reason not in {"None", "N/A", "(null)"}:
            return f"pending: {self.reason}"
        return "-"

    @property
    def gpu_count(self) -> int:
        return parse_gpu_count(self.tres)


@dataclass(frozen=True)
class JobRecordGroup:
    array_parent: str
    name: str
    total: int
    completed: int
    running: int
    pending: int
    failed: int
    other: int
    longest_running_elapsed: str
    limit: str
    submit_time: str
    end_time: str
    cpus: int
    gpus: int
    user: str = ""
    group: str = ""

    @property
    def done_text(self) -> str:
        return f"{self.completed}/{self.total}"

    @property
    def dominant_state(self) -> str:
        if self.running:
            return "RUNNING"
        if self.pending:
            return "PENDING"
        if self.failed:
            return "FAILED"
        if self.completed == self.total and self.total:
            return "COMPLETED"
        return "UNKNOWN"


@dataclass(frozen=True)
class Node:
    name: str
    state: str
    partitions: str
    cpu_alloc: int
    cpu_total: int
    cpu_load: float
    real_memory_mb: int
    alloc_memory_mb: int
    free_memory_mb: int
    gres: str
    alloc_tres: str
    features: str

    @property
    def base_state(self) -> str:
        return re.split(r"[+~-]", self.state, maxsplit=1)[0].upper() or "UNKNOWN"

    @property
    def free_cpus(self) -> int:
        return max(0, self.cpu_total - self.cpu_alloc)

    @property
    def cpu_percent(self) -> float:
        if not self.cpu_total:
            return 0.0
        return 100.0 * self.cpu_alloc / self.cpu_total

    @property
    def memory_percent(self) -> float:
        if not self.real_memory_mb:
            return 0.0
        return 100.0 * self.alloc_memory_mb / self.real_memory_mb

    @property
    def gpu_total(self) -> int:
        return sum(int(match) for match in re.findall(r"gpu(?::[^:,]+)*:(\d+)", self.gres))

    @property
    def has_gpus(self) -> bool:
        return self.gpu_total > 0

    @property
    def is_debug_gpu_node(self) -> bool:
        return self.has_gpus and any(
            "debug" in partition.lower()
            for partition in re.split(r"[,\s]+", self.partitions.strip())
            if partition
        )

    @property
    def gpu_alloc(self) -> int:
        match = re.search(r"(?:^|,)gres/gpu=(\d+)", self.alloc_tres)
        if not match:
            return 0
        return int(match.group(1))

    @property
    def gpu_free(self) -> int:
        # A GPU is only schedulable when the node still has a free CPU core, so
        # idle GPUs on a fully CPU-allocated node are effectively unavailable.
        if self.free_cpus == 0:
            return 0
        return max(0, self.gpu_total - self.gpu_alloc)

    @property
    def gpu_text(self) -> str:
        if self.gpu_total == 0:
            return "-"
        return f"{self.gpu_alloc}/{self.gpu_total}"

    @property
    def memory_text(self) -> str:
        return f"{human_mb(self.alloc_memory_mb)}/{human_mb(self.real_memory_mb)}"


@dataclass(frozen=True)
class UserUsage:
    user: str
    tasks: int
    cpus: int
    gpus: int
    memory_mb: int | None = None


@dataclass(frozen=True)
class UsageEntry:
    """One (login, account) usage record parsed from sreport.

    ``login`` is empty for the account-total row that sreport emits per account.
    ``cpu_hours`` and ``gpu_hours`` are CPU-hours and GPU-hours over the window.
    """

    login: str
    account: str
    cpu_hours: int
    gpu_hours: int


@dataclass(frozen=True)
class LeaderboardRow:
    """A single ranked entry (a user or a group/account) for one window."""

    name: str
    cpu_hours: int
    gpu_hours: int
    fairshare: float | None = None
    # For user rows: the account/PI group they used most this window. Empty for
    # group rows (there the name already is the group).
    group: str = ""


@dataclass(frozen=True)
class JobFilterChoices:
    users: list[str]
    groups: list[str]
    partitions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EfficiencySummary:
    """Average resource efficiency across the user's recent finished jobs.

    Percentages are 0..100 means over the sampled jobs, or None when no job in
    the window carried that metric. ``cpu_percent`` and ``mem_percent`` are the
    same figures ``seff`` reports (used vs allocated); ``walltime_percent`` is
    elapsed vs the requested time limit.
    """

    job_count: int
    cpu_percent: float | None
    mem_percent: float | None
    walltime_percent: float | None
    window_label: str = ""
    # Raw per-job averages behind the percentages (None when unavailable).
    cpu_alloc: float | None = None       # allocated cores
    cpu_used: float | None = None        # utilized cores (TotalCPU / Elapsed)
    mem_req_bytes: float | None = None   # requested memory
    mem_used_bytes: float | None = None  # peak RSS
    walltime_limit_sec: float | None = None  # requested time limit
    walltime_used_sec: float | None = None   # elapsed


@dataclass(frozen=True)
class GpfsQuota:
    """Parsed ``my_gpfs_quota`` output for the info screen.

    ``group_space`` rows are (filesystem, used, quota, limit); the personal rows
    are (filesystem, value). All values keep their human units (e.g. '17.58T').
    """

    primary_group: str
    group_space: list[tuple[str, str, str, str]] = field(default_factory=list)
    personal_space: list[tuple[str, str]] = field(default_factory=list)
    personal_files: list[tuple[str, str]] = field(default_factory=list)
