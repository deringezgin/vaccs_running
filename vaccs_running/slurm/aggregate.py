from __future__ import annotations

from typing import Iterable

from .constants import (
    FAILED_STATES,
    HISTORY_WINDOWS,
    ROOT_ACCOUNT,
)
from .primitives import (
    _user_fairshare,
    dominant_account,
    is_slurm_timestamp,
    parse_elapsed_seconds,
    parse_gpu_count,
    parse_int,
    parse_memory_mb,
    reverse_lex,
    single_or_mixed_label,
    state_base,
)
from .format import (
    _bytes_detail,
    _cores_detail,
    _efficiency_report_line,
    _time_detail,
    format_fairshare,
    human_mb,
)
from .models import (
    EfficiencySummary,
    Job,
    JobGroup,
    JobRecord,
    JobRecordGroup,
    LeaderboardRow,
    Node,
    UsageEntry,
    UserUsage,
)
from .parsers import record_from_job


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
    fairshare_by_user: dict[str, float] | None = None,
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
    if fairshare_by_user is not None:
        columns.append(("fairshare", "FS"))

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
        if fairshare_by_user is not None:
            values["fairshare"] = format_fairshare(fairshare_by_user.get(row.user))
        rows.append(values)

    total = {
        "user": "TOTAL",
        "tasks": str(total_tasks),
        "cpus": str(total_cpus),
        "gpus": str(total_gpus),
    }
    if show_memory:
        total["memory"] = human_mb(total_memory)
    if fairshare_by_user is not None:
        total["fairshare"] = "-"
    rows.append(total)

    def summary_row(label: str, gpus: int) -> dict[str, str]:
        row = {"user": label, "tasks": "-", "cpus": "-", "gpus": str(gpus)}
        if show_memory:
            row["memory"] = "-"
        if fairshare_by_user is not None:
            row["fairshare"] = "-"
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


def build_user_leaderboard(
    usage: Iterable[UsageEntry],
    fairshare: dict[tuple[str, str], float] | None = None,
    default_accounts: dict[str, str] | None = None,
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
    fairshare_by_user = _user_fairshare(
        fairshare or {},
        default_accounts or {},
    )
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


def build_group_leaderboard(
    usage: Iterable[UsageEntry],
    level_fairshare: dict[str, float] | None = None,
) -> list[LeaderboardRow]:
    """Per-account usage rows with Slurm-native account LevelFS."""
    per_group: dict[str, dict[str, int]] = {}
    for entry in usage:
        if entry.login:
            continue
        if entry.account.lower() == ROOT_ACCOUNT:
            continue
        row = per_group.setdefault(entry.account, {"cpu": 0, "gpu": 0})
        row["cpu"] += entry.cpu_hours
        row["gpu"] += entry.gpu_hours
    level_fairshare = level_fairshare or {}
    return [
        LeaderboardRow(
            name=account,
            cpu_hours=row["cpu"],
            gpu_hours=row["gpu"],
            fairshare=level_fairshare.get(account),
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
