from __future__ import annotations

from .constants import (
    GPU_TRES_NAMES,
    NODE_JOBS_FIELDS,
    SACCT_FIELDS,
    SQUEUE_FIELDS,
)
from .primitives import (
    _average,
    _average_percent,
    _clamp01,
    parse_duration_seconds,
    parse_fairshare_value,
    parse_float,
    parse_int,
    parse_key_values,
    parse_level_fairshare_value,
    parse_reqmem_bytes,
    parse_storage_size,
    parse_tres_value,
    parse_user_id,
)
from .models import (
    EfficiencySummary,
    GpfsQuota,
    Job,
    JobRecord,
    Node,
    UsageEntry,
)


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


def parse_sshare_fairshare(output: str) -> dict[tuple[str, str], float]:
    """Map (user, account) -> FairShare from parseable sshare output."""
    return parse_sshare_scores(output)[0]


def parse_sshare_scores(
    output: str,
) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    """Parse native user FairShare and account LevelFS from one sshare result."""
    fairshare: dict[tuple[str, str], float] = {}
    level_fairshare: dict[str, float] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        user = parts[0].strip()
        account = parts[1].strip()
        if user:
            score = parse_fairshare_value(parts[2])
            if score is not None:
                fairshare[(user, account)] = score
        elif account:
            score = parse_level_fairshare_value(parts[3])
            if score is not None:
                level_fairshare[account] = score
    return fairshare, level_fairshare
