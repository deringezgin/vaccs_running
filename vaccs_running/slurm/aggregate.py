from __future__ import annotations

from dataclasses import replace
import re
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
    PriorityFactors,
    PriorityQueueEntry,
    PriorityQueueJob,
    PriorityQueueSnapshot,
    UsageEntry,
    UserUsage,
)
from .parsers import record_from_job


def _array_pattern_matches(pattern: str, job_id: str) -> bool:
    """Whether a possibly-compressed sprio array ID names ``job_id``."""
    if pattern == job_id:
        return True
    pattern_parent, separator, pattern_task = pattern.partition("_")
    job_parent, job_separator, job_task = job_id.partition("_")
    if pattern_parent != job_parent:
        return False
    if not separator:
        # Some sprio versions report one factor row for the array parent.
        return True
    if not job_separator or not job_task.isdigit():
        return False
    if not (pattern_task.startswith("[") and "]" in pattern_task):
        return False

    task = int(job_task)
    specification = pattern_task[1 : pattern_task.index("]")]
    # Ignore an array concurrency throttle such as ``[1-20%4]``.
    specification = specification.split("%", 1)[0]
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+)(?::(\d+))?)?", token)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2) or start)
        step = max(1, int(match.group(3) or 1))
        if start <= task <= end and (task - start) % step == 0:
            return True
    return False


def attach_priority_factors(
    jobs: Iterable[PriorityQueueJob],
    factor_records: Iterable[tuple[str, str, PriorityFactors]],
) -> list[PriorityQueueJob]:
    """Merge best-effort sprio factors into scheduler-order queue rows."""
    records = list(factor_records)
    with_factors: list[PriorityQueueJob] = []
    for job in jobs:
        factor: PriorityFactors | None = None
        # Prefer the exact job/partition row. A fallback handles compressed
        # array IDs and sprio versions which leave the partition column blank.
        for factor_id, partition, candidate in records:
            if factor_id == job.job_id and partition == job.partition:
                factor = candidate
                break
        if factor is None:
            for factor_id, partition, candidate in records:
                if partition and partition != job.partition:
                    continue
                if _array_pattern_matches(factor_id, job.job_id):
                    factor = candidate
                    break
        if factor is None:
            with_factors.append(job)
            continue
        with_factors.append(
            replace(
                job,
                priority=job.priority if job.priority is not None else factor.priority,
                factors=factor,
            )
        )
    return with_factors


def attach_priority_tres(
    jobs: Iterable[PriorityQueueJob],
    requested_tres: dict[str, str],
) -> list[PriorityQueueJob]:
    """Attach requested TRES from a separate long-format squeue query."""
    return [
        replace(job, requested_tres=requested_tres.get(job.job_id, ""))
        if job.job_id in requested_tres
        else job
        for job in jobs
    ]


def _normalize_reservation(reservation: str) -> str:
    if reservation.upper() in {"", "N/A", "NONE", "(NULL)"}:
        return ""
    return reservation


def _priority_scope(job: PriorityQueueJob) -> tuple[str, str]:
    return job.partition, _normalize_reservation(job.reservation)


def _natural_job_id_sort_key(job_id: str) -> tuple:
    """Match Slurm's numeric JobID ordering, including array task IDs."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", job_id)
    )


def _priority_queue_sort_key(job: PriorityQueueJob) -> tuple:
    """Documented squeue order: descending PriorityLong, then JobID."""
    priority = job.priority if job.priority is not None else -1
    return (-priority, _natural_job_id_sort_key(job.job_id))


def _complete_sum(
    entries: list[PriorityQueueEntry],
    attribute: str,
) -> int | None:
    values = [getattr(entry, attribute) for entry in entries]
    if any(value is None for value in values):
        return None
    return sum(values)


def _merge_priority_entries(
    entries: list[PriorityQueueEntry],
    *,
    ranked: bool,
) -> PriorityQueueEntry:
    """Combine same-user ranked slots or unranked entries into one row."""
    first = entries[0]
    if len(entries) == 1:
        return first

    task_job_ids = tuple(
        job_id
        for entry in entries
        for job_id in (entry.task_job_ids or (entry.job.job_id,))
    )
    group_job_ids = tuple(
        dict.fromkeys(
            job_id
            for entry in entries
            for job_id in (entry.group_job_ids or (entry.job.array_parent,))
        )
    )
    group_names = tuple(
        dict.fromkeys(
            name
            for entry in entries
            for name in (entry.group_names or (entry.job.name,))
        )
    )
    group_priorities = tuple(
        dict.fromkeys(
            priority
            for entry in entries
            for priority in (
                entry.group_priorities
                or (() if entry.job.priority is None else (entry.job.priority,))
            )
        )
    )
    group_reason_codes = tuple(
        dict.fromkeys(
            reason
            for entry in entries
            for reason in (
                entry.group_reason_codes
                or (entry.job.reason_code or "Unknown",)
            )
        )
    )
    last = entries[-1]
    rank_end = (last.rank_end or last.priority_rank) if ranked else None
    task_count = sum(entry.task_count for entry in entries)
    job_count = len(group_job_ids)
    source_group_count = sum(entry.source_group_count for entry in entries)
    scope = f"partition {first.job.partition}"
    reservation = _normalize_reservation(first.job.reservation)
    if reservation:
        scope += f", reservation {reservation}"
    walltimes = [entry.walltime_min_seconds for entry in entries]
    complete_walltimes = all(value is not None for value in walltimes)
    if ranked:
        rank_note = (
            f"Priority-rank snapshot among schedulable pending jobs in {scope}; "
            f"this row covers literal rank slots {first.priority_rank}-{rank_end}, "
            f"all owned by {first.job.user}, across {job_count} job"
            f"{'s' if job_count != 1 else ''}. "
            "Press e to inspect them individually; resource fit, reservations, "
            "preemption, and backfill can change actual start order."
        )
    else:
        reasons = ", ".join(group_reason_codes)
        rank_note = (
            f"No priority rank; this packed row combines {task_count} unranked "
            f"pending entries owned by {first.job.user} in {scope}. Slurm reports "
            f"{reasons}; these entries are outside the normal Priority/Resources "
            "queue. Press e to inspect them individually."
        )
    return replace(
        first,
        rank_end=rank_end,
        task_count=task_count,
        task_job_ids=task_job_ids,
        group_job_ids=group_job_ids,
        group_names=group_names,
        group_priorities=group_priorities,
        group_reason_codes=group_reason_codes,
        source_group_count=source_group_count,
        requested_gpus=_complete_sum(entries, "requested_gpus"),
        requested_cpus=_complete_sum(entries, "requested_cpus"),
        requested_memory_mb=_complete_sum(entries, "requested_memory_mb"),
        walltime_min_seconds=(min(walltimes) if complete_walltimes else None),
        walltime_max_seconds=(max(walltimes) if complete_walltimes else None),
        rank_note=rank_note,
    )


def _pack_priority_rank_runs(
    entries: list[PriorityQueueEntry],
) -> list[PriorityQueueEntry]:
    """Pack one already-ranked scope without losing or overlapping a slot."""
    packed: list[PriorityQueueEntry] = []
    run: list[PriorityQueueEntry] = []
    for entry in entries:
        previous_rank = run[-1].priority_rank if run else None
        if (
            run
            and entry.job.user
            and run[-1].job.user == entry.job.user
            and previous_rank is not None
            and entry.priority_rank == previous_rank + 1
        ):
            run.append(entry)
            continue
        if run:
            packed.append(_merge_priority_entries(run, ranked=True))
        run = [entry]
    if run:
        packed.append(_merge_priority_entries(run, ranked=True))
    return packed


def _pack_unranked_user_groups(
    entries: list[PriorityQueueEntry],
) -> list[PriorityQueueEntry]:
    """Pack each owner's unranked entries once without implying rank order."""
    groups: list[list[PriorityQueueEntry]] = []
    group_index_by_user: dict[str, int] = {}
    for entry in entries:
        user = entry.job.user
        if user and user in group_index_by_user:
            groups[group_index_by_user[user]].append(entry)
            continue
        if user:
            group_index_by_user[user] = len(groups)
        groups.append([entry])
    return [
        _merge_priority_entries(group, ranked=False)
        for group in groups
    ]


def build_priority_queue_snapshot(
    user: str,
    jobs: Iterable[PriorityQueueJob],
    factor_records: Iterable[tuple[str, str, PriorityFactors]] = (),
    *,
    factors_available: bool = False,
    factors_error: str = "",
) -> PriorityQueueSnapshot:
    """Build cluster-wide packed/raw ranks from pending scheduler rows.

    Ranks deliberately include only ``Priority`` and ``Resources`` rows in the
    same partition/reservation. Holds, dependencies, invalid requests, and
    policy limits can carry a high composite score (especially with
    ``ACCRUE_ALWAYS``) but are not blockers in the normal schedulable queue.
    Expanded ``squeue --array`` output is sorted again here because some Slurm
    versions do not keep expanded tasks monotonic by their displayed priority.
    """
    pending_jobs = tuple(attach_priority_factors(jobs, factor_records))
    jobs_by_scope: dict[tuple[str, str], list[PriorityQueueJob]] = {}
    for job in pending_jobs:
        jobs_by_scope.setdefault(_priority_scope(job), []).append(job)

    rankable_by_scope = {
        scope: sorted(
            (job for job in scoped_jobs if job.is_rankable),
            key=_priority_queue_sort_key,
        )
        for scope, scoped_jobs in jobs_by_scope.items()
    }

    # Build lightweight entries for the extended all-users view. Counts are
    # accumulated once per partition/reservation; unlike the packed current-user
    # rows, these do not copy every preceding job into every row.
    all_entries_by_job: dict[int, PriorityQueueEntry] = {}
    for (partition, reservation), ranked in rankable_by_scope.items():
        earlier_users: set[str] = set()
        scope = f"partition {partition}"
        if reservation:
            scope += f", reservation {reservation}"
        for index, job in enumerate(ranked):
            all_entries_by_job[id(job)] = PriorityQueueEntry(
                job=job,
                priority_rank=index + 1,
                rank_total=len(ranked),
                task_job_ids=(job.job_id,),
                group_job_ids=(job.array_parent,),
                group_names=(job.name,),
                group_priorities=(job.priority,) if job.priority is not None else (),
                group_reason_codes=(job.reason_code or "Unknown",),
                ahead_count=index,
                ahead_user_count=len(earlier_users),
                requested_gpus=job.requested_gpu_count,
                requested_cpus=job.requested_cpu_count,
                requested_memory_mb=job.requested_memory_mb,
                walltime_min_seconds=job.requested_walltime_seconds,
                walltime_max_seconds=job.requested_walltime_seconds,
                rank_note=(
                    f"Priority-rank snapshot among schedulable pending jobs in {scope}; "
                    "resource fit, reservations, preemption, and backfill can change "
                    "actual start order."
                ),
            )
            if job.user:
                earlier_users.add(job.user)

    unranked_entries_by_job: dict[int, PriorityQueueEntry] = {}
    for job in pending_jobs:
        if id(job) not in all_entries_by_job:
            unranked_entries_by_job[id(job)] = PriorityQueueEntry(
                job=job,
                priority_rank=None,
                rank_total=None,
                task_job_ids=(job.job_id,),
                group_job_ids=(job.array_parent,),
                group_names=(job.name,),
                group_priorities=(job.priority,) if job.priority is not None else (),
                group_reason_codes=(job.reason_code or "Unknown",),
                ahead_count=0,
                ahead_user_count=0,
                requested_gpus=job.requested_gpu_count,
                requested_cpus=job.requested_cpu_count,
                requested_memory_mb=job.requested_memory_mb,
                walltime_min_seconds=job.requested_walltime_seconds,
                walltime_max_seconds=job.requested_walltime_seconds,
                rank_note=(
                    f"No priority rank while Slurm reports {job.reason_code}; "
                    "this job is not in the normal Priority/Resources queue."
                ),
            )

    # Canonical queue order is a block per partition/reservation, in first-seen
    # scope order. Ranked rows always cover 1..N exactly; unranked holds and
    # dependencies remain visible after that scope's literal priority queue.
    all_entries: list[PriorityQueueEntry] = []
    grouped_entries: list[PriorityQueueEntry] = []
    for scope, scoped_jobs in jobs_by_scope.items():
        ranked_entries = [
            all_entries_by_job[id(job)] for job in rankable_by_scope[scope]
        ]
        unranked_entries = [
            unranked_entries_by_job[id(job)]
            for job in scoped_jobs
            if not job.is_rankable
        ]
        all_entries.extend(ranked_entries)
        all_entries.extend(unranked_entries)
        grouped_entries.extend(_pack_priority_rank_runs(ranked_entries))
        grouped_entries.extend(_pack_unranked_user_groups(unranked_entries))

    my_jobs = tuple(
        entry for entry in grouped_entries if entry.job.user == user
    )
    return PriorityQueueSnapshot(
        user=user,
        pending_jobs=pending_jobs,
        my_jobs=my_jobs,
        grouped_entries=tuple(grouped_entries),
        all_entries=tuple(all_entries),
        factors_available=factors_available,
        factors_error=factors_error,
    )


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
