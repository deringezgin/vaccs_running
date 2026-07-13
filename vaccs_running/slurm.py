from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import getpass
import os
import re
import shlex
import subprocess
from typing import Iterable


HISTORY_WINDOWS = {
    "1h": "now-1hours",
    "3h": "now-3hours",
    "24h": "now-24hours",
    "3d": "now-3days",
    "7d": "now-7days",
}

SQUEUE_FIELDS = [
    "job_id",
    "name",
    "state",
    "partition",
    "nodes",
    "reason",
    "elapsed",
    "limit",
    "node_count",
    "cpus",
    "gres",
    "submit_time",
    "start_time",
    "user",
    "group",
]

SQUEUE_FORMAT = "%i|%j|%T|%P|%N|%R|%M|%l|%D|%C|%b|%V|%S|%u|%g"
FILTER_CHOICES_FORMAT = "%u|%g|%P"
VACC_PARTITIONS = [
    "general",
    "short",
    "week",
    "nvgpu",
    "gpu-debug",
    "gpu-preempt",
    "hgnodes",
    "goldenmaple",
]
SACCT_FIELDS = [
    "job_id",
    "raw_job_id",
    "name",
    "state",
    "partition",
    "nodes",
    "elapsed",
    "limit",
    "node_count",
    "cpus",
    "tres",
    "submit_time",
    "start_time",
    "end_time",
    "exit_code",
]
SACCT_FORMAT = (
    "JobID,JobIDRaw,JobName,State,Partition,NodeList,Elapsed,Timelimit,"
    "NNodes,NCPUS,ReqTRES,Submit,Start,End,ExitCode"
)
NODE_JOBS_FIELDS = [
    "job_id",
    "user",
    "state",
    "elapsed",
    "cpus",
    "gres",
    "name",
]
NODE_JOBS_FORMAT = "%i|%u|%T|%M|%C|%b|%j"
DEFAULT_SQUEUE_STATES = "all"
SQUEUE_STATE_RE = re.compile(r"[A-Z_]+")

# Leaderboard (usage-per-user) view. sreport AccountUtilizationByUser returns
# both account-total rows (empty login) and per-user rows in one call, so a
# single fetch per window feeds both the "by user" and "by group" tables.
SREPORT_USAGE_FORMAT = "Login,Account,TresName,Used"
SREPORT_USAGE_REPORT = "AccountUtilizationByUser"
SSHARE_FAIRSHARE_FORMAT = "User,Account,FairShare"
# sreport totals TRES named "cpu" and "gres/gpu"; Used is expressed in Hours.
USAGE_TRES = "cpu,gres/gpu"
GPU_TRES_NAMES = {"gres/gpu", "gpu"}
# The root account row is the cluster grand total; never a leaderboard entry.
ROOT_ACCOUNT = "root"
# Windows shown as the three usage panes, ordered left -> right.
# Capped at 30 days because longer sreport scans grow expensive on VACC
# (a one-year scan takes about a minute).
LEADERBOARD_WINDOWS = [
    ("24h", "last 24 hours"),
    ("7d", "last 7 days"),
    ("30d", "last 30 days"),
]
LEADERBOARD_WINDOW_DELTAS = {
    "24h": datetime.timedelta(days=1),
    "7d": datetime.timedelta(days=7),
    "30d": datetime.timedelta(days=30),
    "1y": datetime.timedelta(days=365),
}
# The per-user "my info" panel (key `i`) reports compute usage over these four
# windows. It reuses the sreport machinery but filters to a single user, so even
# the year-long scan stays tolerable; each window is still fetched off-thread.
USER_INFO_WINDOWS = [
    ("24h", "last 24 hours"),
    ("7d", "last 7 days"),
    ("30d", "last 30 days"),
    ("1y", "last year"),
]
# Job-efficiency summary on the Info tab. sacct over a week for one user is fast
# even with thousands of array tasks. MaxRSS only appears on step (.batch) rows,
# and TotalCPU only aggregates when steps are included, so we do NOT pass -X.
JOB_EFFICIENCY_WINDOW = "now-7days"
JOB_EFFICIENCY_WINDOW_LABEL = "last 7 days"
JOB_EFFICIENCY_FORMAT = (
    "JobID,State,AllocCPUS,TotalCPU,CPUTimeRAW,ElapsedRaw,"
    "TimelimitRaw,ReqMem,MaxRSS,NNodes"
)
# The Info tab reports efficiency over these three windows, each fetched in its
# own thread so it pops in as it returns (1y sacct is ~2.4s). Entries are
# (key, sacct -S value, label).
JOB_EFFICIENCY_WINDOWS = [
    ("7d", "now-7days", "last 7 days"),
    ("30d", "now-30days", "last 30 days"),
    ("1y", "now-365days", "last year"),
]
LEADERBOARD_SORTS = ["gpu", "cpu", "fairshare"]
LEADERBOARD_SORT_LABELS = {
    "gpu": "GPU hours",
    "cpu": "CPU hours",
    "fairshare": "fairshare",
}

class SlurmError(RuntimeError):
    pass


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


class CommandRunner:
    def run(self, args: Iterable[str], timeout: float = 12.0) -> str:
        argv = list(args)

        try:
            proc = subprocess.run(
                argv,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except OSError as exc:
            command = " ".join(shlex.quote(part) for part in argv)
            raise SlurmError(f"could not run {command}: {exc}") from exc
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            command = " ".join(shlex.quote(part) for part in argv)
            raise SlurmError(stderr or f"command failed: {command}")
        return proc.stdout


class SlurmClient:
    def __init__(self, user: str | None = None, states: str | None = None):
        self.user = user or os.environ.get("USER") or getpass.getuser()
        self.job_users: set[str] = {self.user}
        self.job_groups: set[str] = set()
        self.job_partitions: set[str] = set()
        self.job_all_principals = False
        self.squeue_states = normalize_squeue_states(states)
        self.runner = CommandRunner()

    @property
    def state_filter_active(self) -> bool:
        return self.squeue_states.lower() != DEFAULT_SQUEUE_STATES

    @property
    def job_user_filter_active(self) -> bool:
        return (
            self.job_all_principals
            or self.job_groups
            or self.job_users != {self.user}
        )

    @property
    def job_partition_filter_active(self) -> bool:
        return bool(self.job_partitions)

    @property
    def job_user_label(self) -> str:
        if self.job_all_principals:
            return "all users"
        bits: list[str] = []
        if self.job_users:
            bits.append(plural_label(len(self.job_users), "user"))
        if self.job_groups:
            bits.append(plural_label(len(self.job_groups), "group"))
        return " ".join(bits) if bits else "me"

    @property
    def job_filter_active(self) -> bool:
        return (
            self.state_filter_active
            or self.job_user_filter_active
            or self.job_partition_filter_active
        )

    def set_job_state_filter(self, states: str | None) -> None:
        self.squeue_states = normalize_squeue_states(states)

    def set_job_user_filter(self, user: str | None) -> None:
        if user is None:
            self.set_job_principal_filters(all_principals=True)
            return
        stripped = user.strip()
        if not stripped or stripped.lower() in {"all", "*"}:
            self.set_job_principal_filters(all_principals=True)
            return
        if stripped in {"@", "me"}:
            self.set_job_principal_filters(users={self.user})
            return
        self.set_job_principal_filters(users={stripped})

    def set_job_principal_filters(
        self,
        users: Iterable[str] | None = None,
        groups: Iterable[str] | None = None,
        *,
        all_principals: bool = False,
    ) -> None:
        self.job_all_principals = all_principals
        self.job_users = {
            user.strip()
            for user in (users or [])
            if user and user.strip()
        }
        self.job_groups = {
            group.strip()
            for group in (groups or [])
            if group and group.strip()
        }
        if not self.job_all_principals and not self.job_users and not self.job_groups:
            self.job_users = {self.user}

    def set_job_partition_filters(self, partitions: Iterable[str] | None) -> None:
        self.job_partitions = {
            partition.strip()
            for partition in (partitions or [])
            if partition and partition.strip()
        }

    def clear_job_filters(self) -> None:
        self.squeue_states = DEFAULT_SQUEUE_STATES
        self.set_job_principal_filters(users={self.user})
        self.job_partitions = set()

    def fetch_jobs(self, states: str | None = None) -> list[Job]:
        squeue_states = (
            self.squeue_states
            if states is None
            else normalize_squeue_states(states)
        )
        output = self._fetch_jobs(
            squeue_states,
            self._query_users(),
            self.job_partitions,
        )
        jobs: list[Job] = []
        for line in output.splitlines():
            if not line.strip():
                continue
            jobs.append(parse_squeue_line(line))
        return self._filter_jobs_by_partitions(self._filter_jobs_by_principals(jobs))

    def _query_users(self) -> set[str] | None:
        if self.job_all_principals or self.job_groups:
            return None
        return set(self.job_users or {self.user})

    def _filter_jobs_by_principals(self, jobs: list[Job]) -> list[Job]:
        if self.job_groups:
            groups = self.job_groups
            return [job for job in jobs if job.group in groups]
        if self.job_all_principals:
            return jobs
        return jobs

    def _filter_jobs_by_partitions(self, jobs: list[Job]) -> list[Job]:
        if not self.job_partitions:
            return jobs
        partitions = self.job_partitions
        return [job for job in jobs if job.partition in partitions]

    def _fetch_jobs(
        self,
        states: str,
        users: Iterable[str] | None,
        partitions: Iterable[str] | None = None,
    ) -> str:
        args = ["squeue", "--array", "-h"]
        user_list = sorted(user for user in (users or []) if user)
        if user_list:
            args.extend(["-u", ",".join(user_list)])
        partition_list = sorted(partition for partition in (partitions or []) if partition)
        if partition_list:
            args.extend(["-p", ",".join(partition_list)])
        args.extend(["-t", states, "-o", SQUEUE_FORMAT])
        return self.runner.run(args, timeout=15.0)

    def fetch_active_job_records(self) -> tuple[list[Job], list[JobRecord]]:
        jobs = self.fetch_jobs()
        if self.job_filter_active:
            return jobs, [record_from_job(job) for job in jobs]
        if not active_job_keys(jobs):
            return jobs, []
        return (
            jobs,
            records_for_active_jobs(
                jobs,
                self._fetch_sacct_records(active_jobs_start(jobs)),
            ),
        )

    def fetch_job_history(self, window: str) -> list[JobRecord]:
        jobs = [
            parse_squeue_line(line)
            for line in self._fetch_jobs(
                DEFAULT_SQUEUE_STATES,
                {self.user},
            ).splitlines()
            if line.strip()
        ]
        live_records = [record_from_job(job) for job in jobs]
        records_by_id = {
            record.job_id: record
            for record in self._fetch_sacct_records(history_start(window))
            if record.job_id
        }
        for record in live_records:
            if record.is_active or record.job_id not in records_by_id:
                records_by_id[record.job_id] = record
        return sorted(records_by_id.values(), key=job_record_sort_key)

    def _fetch_sacct_records(self, start: str) -> list[JobRecord]:
        output = self.runner.run(
            [
                "sacct",
                "-n",
                "-P",
                "-X",
                "--array",
                "-u",
                self.user,
                "-S",
                start,
                "-o",
                SACCT_FORMAT,
            ],
            timeout=25.0,
        )
        return parse_sacct_records(output)

    def fetch_running_filter_choices(self) -> JobFilterChoices:
        output = self.runner.run(
            [
                "squeue",
                "--array",
                "-h",
                "-t",
                "R",
                "-o",
                FILTER_CHOICES_FORMAT,
            ],
            timeout=15.0,
        )
        users: set[str] = set()
        groups: set[str] = set()
        partitions: set[str] = set(VACC_PARTITIONS)
        for line in output.splitlines():
            parts = line.rstrip("\n").split("|")
            parts.extend([""] * max(0, 3 - len(parts)))
            user, group, partition = (part.strip() for part in parts[:3])
            user = user.strip()
            group = group.strip()
            partition = partition.strip()
            if user:
                users.add(user)
            if group:
                groups.add(group)
            if partition:
                partitions.add(partition)
        return JobFilterChoices(
            users=sorted(users),
            groups=sorted(groups),
            partitions=sorted(partitions),
        )

    def fetch_nodes(self) -> list[Node]:
        output = self.runner.run(["scontrol", "show", "node"], timeout=20.0)
        return parse_scontrol_nodes(output)

    def show_job(self, job_id: str) -> str:
        return self.runner.run(["scontrol", "show", "job", job_id], timeout=12.0)

    def show_node(self, node_name: str) -> str:
        return self.runner.run(["scontrol", "show", "node", node_name], timeout=12.0)

    def node_jobs(self, node_name: str) -> str:
        output = self.runner.run(
            ["squeue", "-a", "-h", "-w", node_name, "-o", NODE_JOBS_FORMAT],
            timeout=12.0,
        )
        body = output.strip()
        if not body:
            return f"No jobs found on {node_name}."
        jobs = [parse_node_job_line(line) for line in body.splitlines() if line.strip()]
        return format_node_jobs(jobs)

    def cluster_usage(self) -> str:
        job_output = self.runner.run(["scontrol", "show", "job"], timeout=20.0)
        node_output = self.runner.run(["scontrol", "show", "node"], timeout=20.0)
        usage = parse_scontrol_job_usage(job_output)
        nodes = parse_scontrol_nodes(node_output)
        free_gpus = free_gpu_count(nodes)
        allocated_gpus = stranded_gpu_count(nodes)
        return format_user_usage(
            aggregate_user_usage(usage),
            free_gpus=free_gpus,
            allocated_gpus=allocated_gpus,
        )

    def fetch_usage_window(
        self,
        window: str,
        now: datetime.datetime | None = None,
    ) -> list[UsageEntry]:
        """Per (user, account) CPU-hour and GPU-hour usage over ``window``.

        The 30-day window can take several seconds on VACC, so callers run this
        off the UI thread.
        """
        start = usage_window_start(window, now=now)
        output = self.runner.run(
            [
                "sreport",
                "-n",
                "-P",
                "-t",
                "Hours",
                "cluster",
                SREPORT_USAGE_REPORT,
                f"Start={start}",
                "End=now",
                "-T",
                USAGE_TRES,
                f"format={SREPORT_USAGE_FORMAT}",
            ],
            timeout=180.0,
        )
        return parse_sreport_usage(output)

    def fetch_fairshare(self) -> dict[tuple[str, str], float]:
        """Current fairshare score keyed by (user, account) association."""
        output = self.runner.run(
            [
                "sshare",
                "-a",
                "-h",
                "-P",
                "-o",
                SSHARE_FAIRSHARE_FORMAT,
            ],
            timeout=30.0,
        )
        return parse_sshare_fairshare(output)

    def fetch_user_compute_usage(
        self,
        window: str,
        now: datetime.datetime | None = None,
    ) -> tuple[int, int]:
        """(cpu_hours, gpu_hours) for the current user over ``window``.

        Restricting sreport with ``Users=`` keeps even the year window
        tolerable, but callers still run it off the UI thread.
        """
        start = usage_window_start(window, now=now)
        output = self.runner.run(
            [
                "sreport",
                "-n",
                "-P",
                "-t",
                "Hours",
                "cluster",
                SREPORT_USAGE_REPORT,
                f"Start={start}",
                "End=now",
                f"Users={self.user}",
                "-T",
                USAGE_TRES,
                f"format={SREPORT_USAGE_FORMAT}",
            ],
            timeout=300.0,
        )
        cpu = 0
        gpu = 0
        for entry in parse_sreport_usage(output):
            # Sum only this user's own rows (one per account), never the
            # account-total row sreport also emits with an empty login.
            if entry.login != self.user:
                continue
            cpu += entry.cpu_hours
            gpu += entry.gpu_hours
        return cpu, gpu

    def fetch_user_fairshare(self) -> dict[str, float]:
        """Current fairshare per account for the logged-in user only."""
        return {
            account: score
            for (user, account), score in self.fetch_fairshare().items()
            if user == self.user
        }

    def fetch_user_default_account(self) -> str:
        """The user's primary Slurm account (best effort; '' if unavailable)."""
        try:
            output = self.runner.run(
                [
                    "sacctmgr",
                    "-n",
                    "-P",
                    "show",
                    "user",
                    self.user,
                    "format=DefaultAccount",
                ],
                timeout=15.0,
            )
        except SlurmError:
            return ""
        for line in output.splitlines():
            account = line.strip()
            if account:
                return account
        return ""

    def fetch_gpfs_quota(self) -> GpfsQuota:
        """Group and personal GPFS storage usage from ``my_gpfs_quota``."""
        output = self.runner.run(["my_gpfs_quota"], timeout=30.0)
        return parse_gpfs_quota(output, self.user)

    def fetch_job_efficiency(
        self,
        window: str = JOB_EFFICIENCY_WINDOW,
        window_label: str = JOB_EFFICIENCY_WINDOW_LABEL,
    ) -> EfficiencySummary:
        """Average CPU/memory/walltime efficiency of the user's recent jobs."""
        output = self.runner.run(
            [
                "sacct",
                "-n",
                "-P",
                "-u",
                self.user,
                "-S",
                window,
                "-E",
                "now",
                "-o",
                JOB_EFFICIENCY_FORMAT,
            ],
            timeout=60.0,
        )
        return summarize_job_efficiency(output, window_label)

    def fetch_job_efficiency_for(self, job_id: str) -> EfficiencySummary:
        """Efficiency for one job (or all tasks of one array), via ``sacct -j``."""
        output = self.runner.run(
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                str(job_id),
                "-o",
                JOB_EFFICIENCY_FORMAT,
            ],
            timeout=30.0,
        )
        return summarize_job_efficiency(output, str(job_id))


def parse_gpfs_quota(output: str, user: str = "") -> GpfsQuota:
    """Parse ``my_gpfs_quota`` into a GpfsQuota.

    The tool prints four blocks: group space limits, group file limits, then the
    user's personal space and file usage. Rows are whitespace-separated and
    every block is bracketed by dashed dividers, so we track the current section
    from its heading and read the filesystem rows beneath it.
    """
    primary = ""
    group_space: list[tuple[str, str, str, str]] = []
    personal_space: list[tuple[str, str]] = []
    personal_files: list[tuple[str, str]] = []
    section: str | None = None
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("group quota for your primary group"):
            primary = line.split(":", 1)[1].strip() if ":" in line else ""
            continue
        if low.startswith("space limits"):
            section = "group_space"
            continue
        if low.startswith("file limits"):
            section = "group_files"
            continue
        if low.startswith("space occupied by"):
            section = "personal_space"
            continue
        if low.startswith("files created by"):
            section = "personal_files"
            continue
        if low.startswith("note"):
            # The closing NOTE wraps onto a second line; stop capturing so its
            # continuation isn't mistaken for a filesystem row.
            section = None
            continue
        if line.startswith("-") or low.startswith("filesystem"):
            continue
        tokens = line.split()
        if section == "group_space" and len(tokens) >= 5 and tokens[1].upper() == "GRP":
            group_space.append((tokens[0], tokens[2], tokens[3], tokens[4]))
        elif section == "personal_space" and len(tokens) >= 2:
            personal_space.append((tokens[0], tokens[1]))
        elif section == "personal_files" and len(tokens) >= 2:
            personal_files.append((tokens[0], tokens[1]))
    return GpfsQuota(
        primary_group=primary,
        group_space=group_space,
        personal_space=personal_space,
        personal_files=personal_files,
    )


_STORAGE_UNITS = {
    "": 1.0,
    "K": 1024.0,
    "M": 1024.0 ** 2,
    "G": 1024.0 ** 3,
    "T": 1024.0 ** 4,
    "P": 1024.0 ** 5,
}


def parse_storage_size(text: str) -> float | None:
    """Bytes for a human size like '17.58T', '32K', '0'; None if unparseable."""
    match = re.fullmatch(
        r"([0-9]*\.?[0-9]+)\s*([KMGTP]?)B?",
        text.strip().upper(),
    )
    if not match:
        return None
    return float(match.group(1)) * _STORAGE_UNITS[match.group(2)]


def storage_percent(used: str, quota: str) -> float | None:
    """Percent of quota used, or None when either size is missing/zero."""
    used_bytes = parse_storage_size(used)
    quota_bytes = parse_storage_size(quota)
    if not used_bytes or not quota_bytes:
        return None
    return 100.0 * used_bytes / quota_bytes


def parse_duration_seconds(text: str) -> float | None:
    """Seconds for a sacct duration like '01:11:01', '12:34.567', '1-02:03:04'."""
    stripped = text.strip()
    if not stripped or stripped.upper() in {"N/A", "UNLIMITED", "INVALID"}:
        return None
    days = 0
    if "-" in stripped:
        day_part, stripped = stripped.split("-", 1)
        days = parse_int(day_part)
    pieces = stripped.split(":")
    try:
        if len(pieces) == 3:
            hours, minutes, seconds = pieces
        elif len(pieces) == 2:
            hours, (minutes, seconds) = "0", pieces
        elif len(pieces) == 1:
            hours, minutes, seconds = "0", "0", pieces[0]
        else:
            return None
        return days * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def parse_reqmem_bytes(text: str, cpus: int, nodes: int) -> float | None:
    """Total requested memory in bytes.

    Modern Slurm reports ReqMem as a plain total (e.g. '96G'). Older output used
    a per-CPU ('c') or per-node ('n') suffix, so those are scaled accordingly.
    """
    stripped = text.strip()
    if not stripped:
        return None
    suffix = ""
    if stripped[-1:].lower() in {"c", "n"}:
        suffix = stripped[-1:].lower()
        stripped = stripped[:-1]
    base = parse_storage_size(stripped)
    if base is None:
        return None
    if suffix == "c":
        return base * max(1, cpus)
    if suffix == "n":
        return base * max(1, nodes)
    return base


def summarize_job_efficiency(
    output: str,
    window_label: str = "",
) -> EfficiencySummary:
    """Average CPU/memory/walltime efficiency over the sacct rows.

    sacct emits a main row per job plus one row per step; MaxRSS lives on the
    step rows, so rows are grouped by base JobID and the peak RSS is taken across
    a job's steps. Jobs that never ran (zero elapsed) are ignored.
    """
    jobs: dict[str, dict[str, object]] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 10:
            parts.extend([""] * (10 - len(parts)))
        job_id = parts[0].strip()
        base = job_id.split(".", 1)[0]
        entry = jobs.setdefault(base, {"main": None, "maxrss": 0.0})
        rss = parse_storage_size(parts[8].strip())
        if rss is not None:
            entry["maxrss"] = max(float(entry["maxrss"]), rss)
        if "." not in job_id:
            entry["main"] = parts

    cpu_values: list[float] = []
    mem_values: list[float] = []
    wall_values: list[float] = []
    cpu_alloc_values: list[float] = []
    cpu_used_values: list[float] = []
    mem_req_values: list[float] = []
    mem_used_values: list[float] = []
    wall_limit_values: list[float] = []
    wall_used_values: list[float] = []
    count = 0
    for entry in jobs.values():
        main = entry["main"]
        if not main:
            continue
        elapsed = parse_int(main[5].strip())
        cpu_time_alloc = parse_int(main[4].strip())
        if elapsed <= 0 or cpu_time_alloc <= 0:
            continue
        count += 1
        alloc_cpus = parse_int(main[2].strip())
        total_cpu = parse_duration_seconds(main[3].strip())
        if total_cpu is not None:
            cpu_values.append(_clamp01(total_cpu / cpu_time_alloc))
            cpu_alloc_values.append(alloc_cpus)
            cpu_used_values.append(total_cpu / elapsed)
        timelimit_minutes = parse_int(main[6].strip())
        if timelimit_minutes > 0:
            limit_seconds = timelimit_minutes * 60
            wall_values.append(_clamp01(elapsed / limit_seconds))
            wall_limit_values.append(limit_seconds)
            wall_used_values.append(elapsed)
        req_mem = parse_reqmem_bytes(
            main[7].strip(), alloc_cpus, parse_int(main[9].strip())
        )
        max_rss = float(entry["maxrss"])
        if req_mem and max_rss > 0:
            mem_values.append(_clamp01(max_rss / req_mem))
            mem_req_values.append(req_mem)
            mem_used_values.append(max_rss)

    return EfficiencySummary(
        job_count=count,
        cpu_percent=_average_percent(cpu_values),
        mem_percent=_average_percent(mem_values),
        walltime_percent=_average_percent(wall_values),
        window_label=window_label,
        cpu_alloc=_average(cpu_alloc_values),
        cpu_used=_average(cpu_used_values),
        mem_req_bytes=_average(mem_req_values),
        mem_used_bytes=_average(mem_used_values),
        walltime_limit_sec=_average(wall_limit_values),
        walltime_used_sec=_average(wall_used_values),
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _average_percent(values: list[float]) -> float | None:
    if not values:
        return None
    return 100.0 * sum(values) / len(values)


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


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


def format_job_efficiency(
    summary: EfficiencySummary,
    job_id: str,
    name: str = "",
) -> str:
    """A seff-style plain-text efficiency report for one job/array selection."""
    header = str(job_id)
    if name and name not in {"-", ""}:
        header += f"  ({name})"
    if summary.job_count == 0:
        return f"{header}\n\nNo completed job data available yet."

    lines = [header, ""]
    if summary.job_count > 1:
        lines.append(f"averaged over {summary.job_count} array tasks")
        lines.append("")
    lines.append(
        _efficiency_report_line(
            "CPU",
            summary.cpu_percent,
            _cores_detail(summary.cpu_used, summary.cpu_alloc),
        )
    )
    lines.append(
        _efficiency_report_line(
            "memory",
            summary.mem_percent,
            _bytes_detail(summary.mem_used_bytes, summary.mem_req_bytes),
        )
    )
    lines.append(
        _efficiency_report_line(
            "walltime",
            summary.walltime_percent,
            _time_detail(summary.walltime_used_sec, summary.walltime_limit_sec),
        )
    )
    return "\n".join(lines)


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


def parse_squeue_line(line: str) -> Job:
    parts = line.rstrip("\n").split("|")
    if len(parts) < len(SQUEUE_FIELDS):
        parts.extend([""] * (len(SQUEUE_FIELDS) - len(parts)))
    elif len(parts) > len(SQUEUE_FIELDS):
        head = parts[: len(SQUEUE_FIELDS) - 1]
        tail = "|".join(parts[len(SQUEUE_FIELDS) - 1 :])
        parts = [*head, tail]

    values = {field: value.strip() for field, value in zip(SQUEUE_FIELDS, parts)}
    return Job(**values)


def parse_sacct_records(output: str) -> list[JobRecord]:
    records: list[JobRecord] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        records.append(parse_sacct_line(line))
    return records


def parse_sacct_line(line: str) -> JobRecord:
    parts = line.rstrip("\n").split("|")
    if len(parts) < len(SACCT_FIELDS):
        parts.extend([""] * (len(SACCT_FIELDS) - len(parts)))
    elif len(parts) > len(SACCT_FIELDS):
        head = parts[: len(SACCT_FIELDS) - 1]
        tail = "|".join(parts[len(SACCT_FIELDS) - 1 :])
        parts = [*head, tail]

    values = {field: value.strip() for field, value in zip(SACCT_FIELDS, parts)}
    return JobRecord(**values)


def record_from_job(job: Job) -> JobRecord:
    return JobRecord(
        job_id=job.job_id,
        raw_job_id=job.job_id,
        name=job.name,
        state=job.state,
        partition=job.partition,
        nodes=job.nodes,
        elapsed=job.elapsed,
        limit=job.limit,
        node_count=job.node_count,
        cpus=job.cpus,
        tres=job.gres,
        submit_time=job.submit_time,
        start_time=job.start_time,
        end_time="",
        exit_code="",
        reason=job.reason,
        source="squeue",
        user=job.user,
        group=job.group,
    )


def job_from_record(record: JobRecord) -> Job:
    return Job(
        job_id=record.job_id,
        name=record.name,
        state=record.state,
        partition=record.partition,
        nodes=record.nodes,
        reason=record.reason,
        elapsed=record.elapsed,
        limit=record.limit,
        node_count=record.node_count,
        cpus=record.cpus,
        gres=record.tres,
        submit_time=record.submit_time,
        start_time=record.start_time,
        user=record.user,
        group=record.group,
    )


def parse_node_job_line(line: str) -> dict[str, str]:
    parts = line.rstrip("\n").split("|")
    if len(parts) < len(NODE_JOBS_FIELDS):
        parts.extend([""] * (len(NODE_JOBS_FIELDS) - len(parts)))
    elif len(parts) > len(NODE_JOBS_FIELDS):
        head = parts[: len(NODE_JOBS_FIELDS) - 1]
        tail = "|".join(parts[len(NODE_JOBS_FIELDS) - 1 :])
        parts = [*head, tail]
    return {
        field: value.strip()
        for field, value in zip(NODE_JOBS_FIELDS, parts)
    }


def format_node_jobs(jobs: list[dict[str, str]]) -> str:
    columns = [
        ("job_id", "JOBID", 12),
        ("user", "USER", 10),
        ("state", "STATE", 8),
        ("elapsed", "ELAPSED", 8),
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


def group_jobs(jobs: Iterable[Job]) -> list[JobGroup]:
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for job in jobs:
        key = (job.array_parent, job.name)
        group = groups.setdefault(
            key,
            {
                "array_parent": job.array_parent,
                "name": job.name,
                "total": 0,
                "completed": 0,
                "running": 0,
                "pending": 0,
                "failed": 0,
                "other": 0,
                "longest_running_elapsed": "-",
                "longest_running_seconds": -1,
                "limit": job.limit or "-",
            },
        )
        group["total"] = int(group["total"]) + 1
        state = job.state.upper()
        if state == "COMPLETED":
            group["completed"] = int(group["completed"]) + 1
        elif state == "RUNNING":
            group["running"] = int(group["running"]) + 1
            seconds = parse_elapsed_seconds(job.elapsed)
            if seconds > int(group["longest_running_seconds"]):
                group["longest_running_seconds"] = seconds
                group["longest_running_elapsed"] = job.elapsed or "-"
        elif state == "PENDING":
            group["pending"] = int(group["pending"]) + 1
        elif state in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}:
            group["failed"] = int(group["failed"]) + 1
        else:
            group["other"] = int(group["other"]) + 1

    return [
        JobGroup(
            array_parent=str(group["array_parent"]),
            name=str(group["name"]),
            total=int(group["total"]),
            completed=int(group["completed"]),
            running=int(group["running"]),
            pending=int(group["pending"]),
            failed=int(group["failed"]),
            other=int(group["other"]),
            longest_running_elapsed=str(group["longest_running_elapsed"]),
            limit=str(group["limit"]),
        )
        for group in groups.values()
    ]


FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


def state_base(state: str) -> str:
    return state.upper().split(maxsplit=1)[0] or "UNKNOWN"


def normalize_squeue_states(states: str | None) -> str:
    if states is None:
        return DEFAULT_SQUEUE_STATES

    tokens: list[str] = []
    for token in re.split(r"[,\s]+", states.strip().strip("'\"")):
        token = token.strip().strip("'\"")
        if not token:
            continue
        if token.lower() == DEFAULT_SQUEUE_STATES:
            return DEFAULT_SQUEUE_STATES
        normalized = token.upper()
        if not SQUEUE_STATE_RE.fullmatch(normalized):
            raise ValueError(f"invalid Slurm state: {token!r}")
        tokens.append(normalized)

    return ",".join(tokens) if tokens else DEFAULT_SQUEUE_STATES


def active_job_keys(jobs: Iterable[Job]) -> set[tuple[str, str]]:
    return {
        job_key(job)
        for job in jobs
        if state_base(job.state) in {"RUNNING", "PENDING"}
    }


def job_key(job: Job) -> tuple[str, str]:
    return (job.array_parent, job.name)


def job_record_key(record: JobRecord) -> tuple[str, str]:
    return (record.array_parent, record.name)


def active_jobs_start(jobs: Iterable[Job]) -> str:
    submit_times = [
        job.submit_time
        for job in jobs
        if state_base(job.state) in {"RUNNING", "PENDING"}
        and is_slurm_timestamp(job.submit_time)
    ]
    if not submit_times:
        return HISTORY_WINDOWS["3d"]
    return min(submit_times)


def is_slurm_timestamp(value: str) -> bool:
    if not value or value in {"N/A", "None", "Unknown", "(null)"}:
        return False
    return bool(re.search(r"\d", value))


def records_for_active_jobs(
    jobs: Iterable[Job],
    accounting_records: Iterable[JobRecord],
) -> list[JobRecord]:
    job_list = list(jobs)
    active_keys = active_job_keys(job_list)
    if not active_keys:
        return []

    records_by_id = {
        record.job_id: record
        for record in accounting_records
        if record.job_id and job_record_key(record) in active_keys
    }
    for job in job_list:
        record = record_from_job(job)
        if job_record_key(record) not in active_keys:
            continue
        if record.is_active or record.job_id not in records_by_id:
            records_by_id[record.job_id] = record
    return sorted(records_by_id.values(), key=job_record_sort_key)


def group_job_records(records: Iterable[JobRecord]) -> list[JobRecordGroup]:
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        key = (record.array_parent, record.name)
        group = groups.setdefault(
            key,
            {
                "array_parent": record.array_parent,
                "name": record.name,
                "total": 0,
                "completed": 0,
                "running": 0,
                "pending": 0,
                "failed": 0,
                "other": 0,
                "longest_running_elapsed": "-",
                "longest_running_seconds": -1,
                "limit": record.limit or "-",
                "submit_time": record.submit_time,
                "end_time": record.end_text,
                "cpus": 0,
                "gpus": 0,
                "users": set(),
                "groups": set(),
            },
        )
        group["total"] = int(group["total"]) + 1
        group["cpus"] = int(group["cpus"]) + parse_int(record.cpus)
        group["gpus"] = int(group["gpus"]) + record.gpu_count
        if record.user:
            group["users"].add(record.user)
        if record.group:
            group["groups"].add(record.group)
        if record.submit_time and (
            not group["submit_time"] or record.submit_time < str(group["submit_time"])
        ):
            group["submit_time"] = record.submit_time
        if record.end_text not in {"-", "running", "pending"} and (
            not group["end_time"] or record.end_text > str(group["end_time"])
        ):
            group["end_time"] = record.end_text

        state = record.base_state
        if state == "COMPLETED":
            group["completed"] = int(group["completed"]) + 1
        elif state == "RUNNING":
            group["running"] = int(group["running"]) + 1
            seconds = parse_elapsed_seconds(record.elapsed)
            if seconds > int(group["longest_running_seconds"]):
                group["longest_running_seconds"] = seconds
                group["longest_running_elapsed"] = record.elapsed or "-"
        elif state == "PENDING":
            group["pending"] = int(group["pending"]) + 1
        elif state in FAILED_STATES:
            group["failed"] = int(group["failed"]) + 1
        else:
            group["other"] = int(group["other"]) + 1

    summaries = [
        JobRecordGroup(
            array_parent=str(group["array_parent"]),
            name=str(group["name"]),
            total=int(group["total"]),
            completed=int(group["completed"]),
            running=int(group["running"]),
            pending=int(group["pending"]),
            failed=int(group["failed"]),
            other=int(group["other"]),
            longest_running_elapsed=str(group["longest_running_elapsed"]),
            limit=str(group["limit"]),
            submit_time=str(group["submit_time"]),
            end_time=str(group["end_time"]),
            cpus=int(group["cpus"]),
            gpus=int(group["gpus"]),
            user=single_or_mixed_label(group["users"]),
            group=single_or_mixed_label(group["groups"]),
        )
        for group in groups.values()
    ]
    return sorted(summaries, key=job_record_group_sort_key)


def aggregate_user_usage(tasks: Iterable[dict[str, str]]) -> list[UserUsage]:
    usage: dict[str, dict[str, int | bool]] = {}
    for task in tasks:
        user = task.get("user") or "unknown"
        row = usage.setdefault(
            user,
            {"tasks": 0, "cpus": 0, "gpus": 0, "memory_mb": 0, "has_memory": False},
        )
        row["tasks"] = int(row["tasks"]) + 1
        row["cpus"] = int(row["cpus"]) + parse_int(task.get("cpus", ""))
        row["gpus"] = int(row["gpus"]) + parse_gpu_count(task.get("tres", ""))
        memory_mb = parse_memory_mb(task.get("memory", ""))
        if memory_mb is not None:
            row["memory_mb"] = int(row["memory_mb"]) + memory_mb
            row["has_memory"] = True

    summaries = [
        UserUsage(
            user=user,
            tasks=int(row["tasks"]),
            cpus=int(row["cpus"]),
            gpus=int(row["gpus"]),
            memory_mb=int(row["memory_mb"]) if row["has_memory"] else None,
        )
        for user, row in usage.items()
    ]
    return sorted(
        summaries,
        key=lambda row: (-row.gpus, -row.cpus, -row.tasks, row.user),
    )


def free_gpu_count(nodes: Iterable[Node]) -> int:
    return sum(
        node.gpu_free
        for node in nodes
        if node.has_gpus and not node.is_debug_gpu_node
    )


def stranded_gpu_count(nodes: Iterable[Node]) -> int:
    """Count idle GPUs that cannot be scheduled because their node has no free CPU core."""
    return sum(
        max(0, node.gpu_total - node.gpu_alloc)
        for node in nodes
        if node.has_gpus and not node.is_debug_gpu_node and node.free_cpus == 0
    )


def format_user_usage(
    usage: list[UserUsage],
    free_gpus: int | None = None,
    allocated_gpus: int | None = None,
) -> str:
    if not usage and free_gpus is None:
        return "No running tasks found."

    total_tasks = sum(row.tasks for row in usage)
    total_cpus = sum(row.cpus for row in usage)
    total_gpus = sum(row.gpus for row in usage)
    show_memory = any(row.memory_mb is not None for row in usage)
    total_memory = sum(row.memory_mb or 0 for row in usage)
    columns = [
        ("user", "USER"),
        ("tasks", "TASKS"),
        ("cpus", "CPUS"),
        ("gpus", "GPUS"),
    ]
    if show_memory:
        columns.append(("memory", "RAM_ALLOC"))

    rows: list[dict[str, str]] = []
    for row in usage:
        values = {
            "user": row.user,
            "tasks": str(row.tasks),
            "cpus": str(row.cpus),
            "gpus": str(row.gpus),
        }
        if show_memory:
            values["memory"] = (
                human_mb(row.memory_mb) if row.memory_mb is not None else "-"
            )
        rows.append(values)

    total = {
        "user": "TOTAL",
        "tasks": str(total_tasks),
        "cpus": str(total_cpus),
        "gpus": str(total_gpus),
    }
    if show_memory:
        total["memory"] = human_mb(total_memory)
    rows.append(total)

    def summary_row(label: str, gpus: int) -> dict[str, str]:
        row = {"user": label, "tasks": "-", "cpus": "-", "gpus": str(gpus)}
        if show_memory:
            row["memory"] = "-"
        return row

    if allocated_gpus is not None:
        rows.append(summary_row("ALLOCATED", allocated_gpus))
    if free_gpus is not None:
        rows.append(summary_row("FREE", free_gpus))

    widths = [
        max(len(label), *(len(row[key]) for row in rows))
        for key, label in columns
    ]
    header = "  ".join(label.ljust(width) for (_, label), width in zip(columns, widths))
    divider = "-" * len(header)
    body = [
        "  ".join(row[key].ljust(width) for (key, _), width in zip(columns, widths))
        for row in rows
    ]
    people = "person" if len(usage) == 1 else "people"
    tasks_word = "task" if total_tasks == 1 else "tasks"
    title = f"{len(usage)} {people} running {total_tasks} {tasks_word}"
    return "\n".join([title, "", header, divider, *body])


def parse_scontrol_job_usage(output: str) -> list[dict[str, str]]:
    usage: list[dict[str, str]] = []
    current: dict[str, str] = {}

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("JobId="):
            append_job_usage(usage, current)
            current = {}
        current.update(parse_key_values(stripped))

    append_job_usage(usage, current)
    return usage


def append_job_usage(
    usage: list[dict[str, str]],
    fields: dict[str, str],
) -> None:
    if not fields or fields.get("JobState", "").upper() != "RUNNING":
        return
    tres = fields.get("AllocTRES") or fields.get("ReqTRES", "")
    usage.append(
        {
            "job_id": fields.get("JobId", ""),
            "user": parse_user_id(fields.get("UserId", "")),
            "cpus": parse_tres_value(tres, "cpu") or fields.get("NumCPUs", ""),
            "tres": tres,
            "memory": parse_tres_value(tres, "mem") or fields.get("MinMemoryNode", ""),
        }
    )


def parse_user_id(value: str) -> str:
    if not value:
        return ""
    return value.split("(", 1)[0]


def parse_tres_value(tres: str, key: str) -> str:
    for part in tres.split(","):
        name, separator, value = part.partition("=")
        if separator and name == key:
            return value
    return ""


def history_start(window: str) -> str:
    return HISTORY_WINDOWS.get(window, HISTORY_WINDOWS["24h"])


def summarize_job_records(records: Iterable[JobRecord]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for record in records:
        key = record.base_state
        summary[key] = summary.get(key, 0) + 1
    return summary


def summarize_jobs(jobs: Iterable[Job]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for job in jobs:
        key = job.state.upper() or "UNKNOWN"
        summary[key] = summary.get(key, 0) + 1
    return summary


def parse_scontrol_nodes(output: str) -> list[Node]:
    nodes: list[Node] = []
    current: dict[str, str] = {}

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("NodeName=") and current:
            nodes.append(node_from_fields(current))
            current = {}
        current.update(parse_key_values(stripped))

    if current:
        nodes.append(node_from_fields(current))
    return nodes


def parse_key_values(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def node_from_fields(fields: dict[str, str]) -> Node:
    return Node(
        name=fields.get("NodeName", ""),
        state=fields.get("State", ""),
        partitions=fields.get("Partitions", ""),
        cpu_alloc=parse_int(fields.get("CPUAlloc", "")),
        cpu_total=parse_int(fields.get("CPUTot", "")),
        cpu_load=parse_float(fields.get("CPULoad", "")),
        real_memory_mb=parse_int(fields.get("RealMemory", "")),
        alloc_memory_mb=parse_int(fields.get("AllocMem", "")),
        free_memory_mb=parse_int(fields.get("FreeMem", "")),
        gres=fields.get("Gres", ""),
        alloc_tres=fields.get("AllocTRES", ""),
        features=fields.get("ActiveFeatures") or fields.get("AvailableFeatures", ""),
    )


def parse_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def plural_label(count: int, singular: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {singular}{suffix}"


def single_or_mixed_label(values: set[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def parse_gpu_count(value: str) -> int:
    return sum(
        int(match)
        for match in re.findall(r"(?:gres/)?gpu(?::[^,;:=()]+)*[:=](\d+)", value)
    )


def parse_memory_mb(value: str) -> int | None:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "NONE", "(NULL)"}:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTkmgt]?)([cnCN]?)", stripped)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).upper()
    multiplier = {
        "": 1,
        "K": 1 / 1024,
        "M": 1,
        "G": 1024,
        "T": 1024 * 1024,
    }[unit]
    memory_mb = int(amount * multiplier)
    return memory_mb if memory_mb > 0 else None


def parse_elapsed_seconds(value: str) -> int:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "UNLIMITED"}:
        return -1
    days = 0
    time_part = stripped
    if "-" in stripped:
        day_part, time_part = stripped.split("-", 1)
        days = parse_int(day_part)
    pieces = time_part.split(":")
    if len(pieces) == 2:
        hours = 0
        minutes, seconds = pieces
    elif len(pieces) == 3:
        hours, minutes, seconds = pieces
    else:
        return -1
    return (
        days * 24 * 60 * 60
        + parse_int(hours) * 60 * 60
        + parse_int(minutes) * 60
        + parse_int(seconds)
    )


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def human_mb(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f}T"
    if value >= 1024:
        return f"{value / 1024:.0f}G"
    return f"{value}M"


def usage_window_start(
    window: str,
    now: datetime.datetime | None = None,
) -> str:
    """sreport ``Start=`` value for a leaderboard window, e.g. '2026-07-05T09:00:00'."""
    reference = now or datetime.datetime.now()
    delta = LEADERBOARD_WINDOW_DELTAS.get(
        window,
        LEADERBOARD_WINDOW_DELTAS["24h"],
    )
    return (reference - delta).strftime("%Y-%m-%dT%H:%M:%S")


def parse_sreport_usage(output: str) -> list[UsageEntry]:
    """Fold sreport's per-TRES rows into one UsageEntry per (login, account)."""
    totals: dict[tuple[str, str], dict[str, int]] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        login = parts[0].strip()
        account = parts[1].strip()
        tres = parts[2].strip().lower()
        used = parse_int(parts[3].strip())
        row = totals.setdefault((login, account), {"cpu": 0, "gpu": 0})
        if tres == "cpu":
            row["cpu"] += used
        elif tres in GPU_TRES_NAMES:
            row["gpu"] += used
    return [
        UsageEntry(
            login=login,
            account=account,
            cpu_hours=row["cpu"],
            gpu_hours=row["gpu"],
        )
        for (login, account), row in totals.items()
    ]


def parse_fairshare_value(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def parse_sshare_fairshare(output: str) -> dict[tuple[str, str], float]:
    """Map (user, account) -> fairshare from parseable sshare output.

    Account-level rows (empty user) are skipped; sshare indents its columns so
    every field is stripped before use.
    """
    scores: dict[tuple[str, str], float] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        user = parts[0].strip()
        account = parts[1].strip()
        if not user:
            continue
        fairshare = parse_fairshare_value(parts[2])
        if fairshare is not None:
            scores[(user, account)] = fairshare
    return scores


def _user_fairshare(fairshare: dict[tuple[str, str], float]) -> dict[str, float]:
    """Best (highest) fairshare across each user's account associations."""
    best: dict[str, float] = {}
    for (user, _account), score in fairshare.items():
        if user not in best or score > best[user]:
            best[user] = score
    return best


def _group_fairshare(fairshare: dict[tuple[str, str], float]) -> dict[str, float]:
    """Mean fairshare of each account's member users."""
    members: dict[str, list[float]] = {}
    for (_user, account), score in fairshare.items():
        members.setdefault(account, []).append(score)
    return {
        account: sum(scores) / len(scores)
        for account, scores in members.items()
        if scores
    }


def build_user_leaderboard(
    usage: Iterable[UsageEntry],
    fairshare: dict[tuple[str, str], float] | None = None,
) -> list[LeaderboardRow]:
    """Per-user rows, summing a user's usage across all their accounts.

    Each row also records the account/PI group the user drew the most on this
    window (their dominant group), so the UI can show it alongside the user.
    """
    per_user: dict[str, dict[str, object]] = {}
    for entry in usage:
        if not entry.login:
            continue
        row = per_user.setdefault(
            entry.login, {"cpu": 0, "gpu": 0, "accounts": {}}
        )
        row["cpu"] = int(row["cpu"]) + entry.cpu_hours
        row["gpu"] = int(row["gpu"]) + entry.gpu_hours
        if entry.account:
            accounts: dict[str, int] = row["accounts"]  # type: ignore[assignment]
            accounts[entry.account] = (
                accounts.get(entry.account, 0)
                + entry.cpu_hours
                + entry.gpu_hours
            )
    fairshare_by_user = _user_fairshare(fairshare or {})
    return [
        LeaderboardRow(
            name=user,
            cpu_hours=int(row["cpu"]),
            gpu_hours=int(row["gpu"]),
            fairshare=fairshare_by_user.get(user),
            group=dominant_account(row["accounts"]),  # type: ignore[arg-type]
        )
        for user, row in per_user.items()
    ]


def dominant_account(accounts: dict[str, int]) -> str:
    """The account with the most usage; ties broken alphabetically."""
    if not accounts:
        return ""
    return min(accounts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def build_group_leaderboard(
    usage: Iterable[UsageEntry],
    fairshare: dict[tuple[str, str], float] | None = None,
) -> list[LeaderboardRow]:
    """Per-account rows, using sreport's account-total rows for usage."""
    per_group: dict[str, dict[str, int]] = {}
    for entry in usage:
        if entry.login:
            continue
        if entry.account.lower() == ROOT_ACCOUNT:
            continue
        row = per_group.setdefault(entry.account, {"cpu": 0, "gpu": 0})
        row["cpu"] += entry.cpu_hours
        row["gpu"] += entry.gpu_hours
    fairshare_by_group = _group_fairshare(fairshare or {})
    return [
        LeaderboardRow(
            name=account,
            cpu_hours=row["cpu"],
            gpu_hours=row["gpu"],
            fairshare=fairshare_by_group.get(account),
        )
        for account, row in per_group.items()
    ]


def sort_leaderboard(
    rows: Iterable[LeaderboardRow],
    sort_key: str,
    descending: bool = True,
) -> list[LeaderboardRow]:
    """Rank rows by the chosen metric, name as tie-breaker.

    ``descending`` (the default) puts the biggest values first. Rows without a
    fairshare score always sort to the bottom, regardless of direction.
    """
    sign = -1 if descending else 1
    if sort_key == "cpu":
        key = lambda row: (sign * row.cpu_hours, sign * row.gpu_hours, row.name)
    elif sort_key == "fairshare":
        key = lambda row: (
            row.fairshare is None,
            sign * (row.fairshare if row.fairshare is not None else 0.0),
            row.name,
        )
    else:  # "gpu" and any unknown key
        key = lambda row: (sign * row.gpu_hours, sign * row.cpu_hours, row.name)
    return sorted(rows, key=key)


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
    return f"{value:.3f}"


def summarize_nodes(nodes: Iterable[Node]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for node in nodes:
        key = node.base_state
        summary[key] = summary.get(key, 0) + 1
    return summary


def job_record_sort_key(record: JobRecord) -> tuple[int, str, str]:
    state_rank = {
        "RUNNING": 0,
        "PENDING": 1,
        "COMPLETED": 2,
    }.get(record.base_state, 3 if record.is_failed else 4)
    timestamp = record.start_time
    if record.base_state == "PENDING":
        timestamp = record.submit_time
    elif record.end_text not in {"-", "running", "pending"}:
        timestamp = record.end_text
    return (state_rank, reverse_lex(timestamp), record.job_id)


def job_record_group_sort_key(group: JobRecordGroup) -> tuple[int, str, str]:
    state_rank = {
        "RUNNING": 0,
        "PENDING": 1,
        "COMPLETED": 2,
    }.get(group.dominant_state, 3 if group.failed else 4)
    timestamp = group.submit_time
    if group.end_time not in {"", "-", "running", "pending"}:
        timestamp = group.end_time
    return (state_rank, reverse_lex(timestamp), group.array_parent)


def reverse_lex(value: str) -> str:
    return "".join(chr(255 - ord(char)) for char in value)
