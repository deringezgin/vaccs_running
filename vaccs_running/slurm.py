from __future__ import annotations

from dataclasses import dataclass, field
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
class JobFilterChoices:
    users: list[str]
    groups: list[str]
    partitions: list[str] = field(default_factory=list)


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
