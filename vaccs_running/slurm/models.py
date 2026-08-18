from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import re

from .constants import FAILED_STATES, PRIORITY_RANKABLE_REASONS
from .primitives import (
    explain_pending_reason,
    parse_elapsed_seconds,
    parse_gpu_count,
    parse_memory_mb,
    parse_optional_int,
    parse_tres_value,
    pending_reason_code,
    state_base,
)
from .format import human_duration, human_mb


def _compact_wait(seconds: int) -> str:
    days, remainder = divmod(max(0, seconds), 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d{hours}h{minutes}m{seconds}s"


@dataclass(frozen=True)
class PriorityFactors:
    """Weighted ``sprio`` components behind one job's composite priority."""

    priority: int | None = None
    site: int | None = None
    age: int | None = None
    association: int | None = None
    fairshare: int | None = None
    job_size: int | None = None
    partition: int | None = None
    qos: int | None = None
    tres: str = ""
    nice: int | None = None


@dataclass(frozen=True)
class PriorityQueueJob:
    """One pending job from the scheduler-order ``squeue`` snapshot."""

    job_id: str
    user: str
    name: str
    partition: str
    state: str
    reason: str
    priority: int | None
    account: str
    qos: str
    submit_time: str
    start_time: str
    reservation: str
    node_count: str
    cpus: str
    gres: str
    requested_tres: str
    limit: str
    factors: PriorityFactors | None = None

    @property
    def array_parent(self) -> str:
        return self.job_id.split("_", 1)[0]

    @property
    def normalized_reservation(self) -> str:
        value = self.reservation.strip()
        if value.upper() in {"", "N/A", "NONE", "(NULL)"}:
            return ""
        return value

    @property
    def queue_label(self) -> str:
        if not self.normalized_reservation:
            return self.partition
        return f"{self.partition}/{self.normalized_reservation}"

    @property
    def reason_code(self) -> str:
        return pending_reason_code(self.reason)

    @property
    def reason_explanation(self) -> str:
        return explain_pending_reason(self.reason)

    @property
    def is_rankable(self) -> bool:
        """Whether a conservative priority-only rank is meaningful."""
        return self.priority is not None and self.reason_code in PRIORITY_RANKABLE_REASONS

    @property
    def estimated_start(self) -> str:
        value = self.start_time.strip()
        if not value or value.upper() in {
            "N/A",
            "NONE",
            "UNKNOWN",
            "(NULL)",
        }:
            return "-"
        return value

    @property
    def requested_resources(self) -> str:
        if self.requested_tres and self.requested_tres.upper() not in {
            "N/A",
            "NONE",
            "(NULL)",
        }:
            return self.requested_tres
        resources: list[str] = []
        if self.node_count and self.node_count not in {"0", "N/A", "(null)"}:
            suffix = "node" if self.node_count == "1" else "nodes"
            resources.append(f"{self.node_count} {suffix}")
        if self.cpus and self.cpus not in {"0", "N/A", "(null)"}:
            suffix = "CPU" if self.cpus == "1" else "CPUs"
            resources.append(f"{self.cpus} {suffix}")
        if self.gres and self.gres.upper() not in {"N/A", "NONE", "(NULL)"}:
            resources.append(self.gres)
        return ", ".join(resources) if resources else "not reported"

    @property
    def gpu_count(self) -> int:
        return parse_gpu_count(self.requested_tres or self.gres)

    @property
    def requested_gpu_count(self) -> int | None:
        """Requested GPUs, retaining unknown data instead of calling it zero."""
        if self.requested_tres and self.requested_tres.upper() not in {
            "N/A",
            "NONE",
            "(NULL)",
        }:
            return parse_gpu_count(self.requested_tres)
        if self.gres and self.gres.upper() not in {"N/A", "NONE", "(NULL)"}:
            return parse_gpu_count(self.gres)
        return None

    @property
    def requested_cpu_count(self) -> int | None:
        tres_cpus = parse_optional_int(parse_tres_value(self.requested_tres, "cpu"))
        if tres_cpus is not None:
            return tres_cpus
        return parse_optional_int(self.cpus)

    @property
    def requested_memory_mb(self) -> int | None:
        return parse_memory_mb(parse_tres_value(self.requested_tres, "mem"))

    @property
    def requested_walltime_seconds(self) -> int | None:
        seconds = parse_elapsed_seconds(self.limit)
        return seconds if seconds >= 0 else None


@dataclass(frozen=True)
class PriorityQueueEntry:
    """A pending queue row plus its conservative priority-rank context."""

    job: PriorityQueueJob
    priority_rank: int | None
    rank_total: int | None
    rank_end: int | None = None
    task_count: int = 1
    task_job_ids: tuple[str, ...] = ()
    group_job_ids: tuple[str, ...] = ()
    group_names: tuple[str, ...] = ()
    group_priorities: tuple[int, ...] = ()
    group_reason_codes: tuple[str, ...] = ()
    group_submit_times: tuple[str, ...] = ()
    source_group_count: int = 1
    ahead_count: int | None = None
    ahead_user_count: int | None = None
    ahead: tuple[PriorityQueueJob, ...] = ()
    rank_note: str = ""
    requested_gpus: int | None = None
    requested_cpus: int | None = None
    requested_memory_mb: int | None = None
    walltime_min_seconds: int | None = None
    walltime_max_seconds: int | None = None

    @property
    def ahead_users(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(job.user for job in self.ahead if job.user))

    @property
    def earlier_count(self) -> int:
        if self.ahead_count is not None:
            return self.ahead_count
        return len(self.ahead)

    @property
    def earlier_user_count(self) -> int:
        if self.ahead_user_count is not None:
            return self.ahead_user_count
        return len(self.ahead_users)

    @property
    def priority_rank_text(self) -> str:
        if self.priority_rank is None or self.rank_total is None:
            return "not ranked"
        if self.rank_end is not None and self.rank_end != self.priority_rank:
            return f"{self.priority_rank}-{self.rank_end} of {self.rank_total}"
        return f"{self.priority_rank} of {self.rank_total}"

    @property
    def job_count(self) -> int:
        if self.group_job_ids:
            return len(self.group_job_ids)
        parents = tuple(
            dict.fromkeys(job_id.split("_", 1)[0] for job_id in self.task_job_ids)
        )
        return len(parents) or 1

    @property
    def is_multi_job_group(self) -> bool:
        return self.job_count > 1

    @property
    def is_consecutive_user_group(self) -> bool:
        return self.source_group_count > 1

    @property
    def display_name(self) -> str:
        names = self.group_names or (self.job.name,)
        return names[0] if len(names) == 1 else "mixed jobs"

    @property
    def display_priority(self) -> str:
        priorities = self.group_priorities
        if not priorities and self.job.priority is not None:
            priorities = (self.job.priority,)
        if not priorities:
            return "-"
        if len(priorities) == 1:
            return str(priorities[0])
        return f"{max(priorities)}-{min(priorities)}"

    @property
    def display_reason(self) -> str:
        reasons = self.group_reason_codes or (self.job.reason_code or "Unknown",)
        if len(reasons) == 1:
            return reasons[0]
        combined = "/".join(reasons)
        return combined if len(combined) <= 24 else f"mixed ({len(reasons)})"

    @property
    def display_estimated_start(self) -> str:
        if self.is_consecutive_user_group:
            return "mixed"
        return self.job.estimated_start

    @property
    def display_submitted_on(self) -> str:
        values = self.group_submit_times or (self.job.submit_time,)
        known = tuple(
            value
            for value in values
            if value and value.upper() not in {"N/A", "NONE", "UNKNOWN", "(NULL)"}
        )
        if not known:
            return "-"
        if len(known) != len(values):
            return "mixed"
        earliest = min(known)
        latest = max(known)
        return earliest if earliest == latest else f"{earliest}–{latest}"

    def wait_text(self, now: datetime.datetime | None = None) -> str:
        values = self.group_submit_times or (self.job.submit_time,)
        current = now or datetime.datetime.now()
        waits: list[int] = []
        for value in values:
            try:
                submitted = datetime.datetime.fromisoformat(value)
            except (TypeError, ValueError):
                continue
            comparison_now = current
            if submitted.tzinfo is None and comparison_now.tzinfo is not None:
                comparison_now = comparison_now.replace(tzinfo=None)
            elif submitted.tzinfo is not None and comparison_now.tzinfo is None:
                comparison_now = comparison_now.replace(tzinfo=submitted.tzinfo)
            elif submitted.tzinfo is not None:
                comparison_now = comparison_now.astimezone(submitted.tzinfo)
            waits.append(max(0, int((comparison_now - submitted).total_seconds())))
        if not waits:
            return "-"
        if len(waits) != len(values):
            return "mixed"
        shortest = min(waits)
        longest = max(waits)
        if shortest == longest:
            return _compact_wait(shortest)
        return f"{_compact_wait(shortest)}–{_compact_wait(longest)}"

    @property
    def display_wait(self) -> str:
        return self.wait_text()

    @property
    def display_job_id(self) -> str:
        if self.is_multi_job_group:
            if self.task_count != self.job_count:
                return f"{self.job_count} jobs / {self.task_count} tasks"
            return f"{self.job_count} jobs"
        if self.task_count <= 1:
            return self.job.job_id
        parent = self.job.array_parent
        task_ids: list[int] = []
        for job_id in self.task_job_ids:
            job_parent, separator, task = job_id.partition("_")
            if job_parent != parent or not separator or not task.isdigit():
                return f"{parent} [{self.task_count} tasks]"
            task_ids.append(int(task))
        if not task_ids:
            return f"{parent} [{self.task_count} tasks]"

        task_ids.sort()
        ranges: list[str] = []
        start = previous = task_ids[0]
        for task in task_ids[1:]:
            if task == previous + 1:
                previous = task
                continue
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = task
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        return f"{parent}_[{','.join(ranges)}]"

    @property
    def display_gpus(self) -> str:
        return "-" if self.requested_gpus is None else str(self.requested_gpus)

    @property
    def display_cpus(self) -> str:
        return "-" if self.requested_cpus is None else str(self.requested_cpus)

    @property
    def display_memory(self) -> str:
        if self.requested_memory_mb is None:
            return "-"
        return human_mb(self.requested_memory_mb)

    @property
    def display_walltime(self) -> str:
        if self.walltime_min_seconds is None or self.walltime_max_seconds is None:
            return "-"
        minimum = human_duration(self.walltime_min_seconds)
        maximum = human_duration(self.walltime_max_seconds)
        return minimum if minimum == maximum else f"{minimum}–{maximum}"


@dataclass(frozen=True)
class PriorityQueueSnapshot:
    """Read-only pending-queue snapshot and the current user's queue context."""

    user: str
    pending_jobs: tuple[PriorityQueueJob, ...]
    my_jobs: tuple[PriorityQueueEntry, ...]
    factors_available: bool
    all_entries: tuple[PriorityQueueEntry, ...] = ()
    factors_error: str = ""
    grouped_entries: tuple[PriorityQueueEntry, ...] = ()


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

    ``group_space`` and ``group_files`` rows are (filesystem, used, soft quota,
    hard limit); the personal rows are (filesystem, value). Active grace-period
    countdowns are retained separately by filesystem. Space values keep their
    human units (e.g. '17.58T') and file counts remain decimal strings.
    """

    primary_group: str
    group_space: list[tuple[str, str, str, str]] = field(default_factory=list)
    group_files: list[tuple[str, str, str, str]] = field(default_factory=list)
    personal_space: list[tuple[str, str]] = field(default_factory=list)
    personal_files: list[tuple[str, str]] = field(default_factory=list)
    group_space_grace: list[tuple[str, str]] = field(default_factory=list)
    group_files_grace: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class GpfsMemberUsage:
    """One group member's GPFS usage on one filesystem."""

    user: str
    filesystem: str
    space: str
    files: int
