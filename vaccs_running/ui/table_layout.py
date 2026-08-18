from __future__ import annotations

from collections.abc import Callable

from ..slurm import Job, JobRecordGroup, PriorityQueueEntry


def priority_rank_text(entry: PriorityQueueEntry) -> str:
    if entry.priority_rank is None or entry.rank_total is None:
        return "—"
    if entry.rank_end is not None and entry.rank_end != entry.priority_rank:
        return f"{entry.priority_rank}-{entry.rank_end}/{entry.rank_total}"
    return f"{entry.priority_rank}/{entry.rank_total}"


def responsive_priority_specs(
    available_width: int,
    *,
    extended: bool = False,
    current_user: str = "",
) -> list[tuple[str, int, int, Callable[[PriorityQueueEntry], str]]]:
    """Priority columns, retaining rank and requested resources when narrow."""
    if extended:
        specs: list[
            tuple[str, int, int, Callable[[PriorityQueueEntry], str]]
        ] = [
            (
                "YOU",
                3,
                3,
                lambda entry: "YOU" if entry.job.user == current_user else "",
            ),
            ("JOBID", 10, 24, lambda entry: entry.display_job_id),
            ("USER", 6, 16, lambda entry: entry.job.user or "-"),
            ("ACCOUNT", 6, 20, lambda entry: entry.job.account or "-"),
            ("JOB", 10, 28, lambda entry: entry.display_name),
            ("PARTITION", 8, 24, lambda entry: entry.job.queue_label),
            ("RANK", 7, 13, priority_rank_text),
            ("GPUS", 4, 7, lambda entry: entry.display_gpus),
            ("CPUS", 4, 7, lambda entry: entry.display_cpus),
            ("RAM", 5, 9, lambda entry: entry.display_memory),
            ("WALLTIME", 8, 15, lambda entry: entry.display_walltime),
            (
                "PRIORITY",
                8,
                12,
                lambda entry: entry.display_priority,
            ),
            ("WHY", 16, 38, lambda entry: entry.display_reason),
            ("SUBMITTED ON", 19, 39, lambda entry: entry.display_submitted_on),
            ("WAIT", 10, 25, lambda entry: entry.display_wait),
            ("EST START", 16, 20, lambda entry: entry.display_estimated_start),
        ]
        for removable in (
            "EST START",
            "SUBMITTED ON",
            "WAIT",
            "JOB",
            "ACCOUNT",
            "PRIORITY",
            "WHY",
        ):
            if minimum_table_width(label_widths(specs)) <= available_width:
                return specs
            specs = [spec for spec in specs if spec[0] != removable]
        return specs

    specs: list[tuple[str, int, int, Callable[[PriorityQueueEntry], str]]] = [
        (
            "YOU",
            3,
            3,
            lambda entry: "YOU" if entry.job.user == current_user else "",
        ),
        ("JOBS", 4, 8, lambda entry: str(entry.job_count)),
        ("TASKS", 5, 8, lambda entry: str(entry.task_count)),
        ("USER", 6, 16, lambda entry: entry.job.user or "-"),
        ("JOB", 10, 28, lambda entry: entry.display_name),
        ("PARTITION", 8, 24, lambda entry: entry.job.queue_label),
        ("RANK", 7, 13, priority_rank_text),
        ("GPUS", 4, 7, lambda entry: entry.display_gpus),
        ("CPUS", 4, 7, lambda entry: entry.display_cpus),
        ("RAM", 5, 9, lambda entry: entry.display_memory),
        ("WALLTIME", 8, 15, lambda entry: entry.display_walltime),
        (
            "PRIORITY",
            8,
            12,
            lambda entry: entry.display_priority,
        ),
        (
            "AHEAD",
            5,
            7,
            lambda entry: (
                "—" if entry.priority_rank is None else str(entry.earlier_count)
            ),
        ),
        (
            "USERS",
            5,
            7,
            lambda entry: (
                "—" if entry.priority_rank is None else str(entry.earlier_user_count)
            ),
        ),
        ("WHY", 16, 38, lambda entry: entry.display_reason),
        ("SUBMITTED ON", 19, 39, lambda entry: entry.display_submitted_on),
        ("WAIT", 10, 25, lambda entry: entry.display_wait),
        ("EST START", 16, 20, lambda entry: entry.display_estimated_start),
    ]
    # At the supported 70-column minimum, identity and rank are more useful
    # than duplicate detail/count columns. Wider screens retain everything.
    for removable in (
        "EST START",
        "SUBMITTED ON",
        "WAIT",
        "JOB",
        "USERS",
        "AHEAD",
        "PRIORITY",
        "WHY",
    ):
        if minimum_table_width(label_widths(specs)) <= available_width:
            return specs
        specs = [spec for spec in specs if spec[0] != removable]
    return specs


def leaderboard_columns(
    inner_width: int,
    entity_label: str,
    max_rank: int = 1,
    group_col: bool = False,
    fairshare_label: str = "FS",
) -> list[tuple[str, str, int, str]]:
    """Columns for one leaderboard pane, given its usable inner width.

    Returns ``(key, label, width, align)`` tuples. The name (and, in user mode,
    GROUP) columns flex to fill the remaining space. As the pane narrows, the
    fairshare column is dropped first, then GROUP, so rank/name/cpu/gpu always
    stay legible. The rank column widens to fit ``max_rank`` so four-plus digit
    ranks are never truncated.
    """
    rank_w = max(3, len(str(max(1, max_rank))))
    cpu_w, gpu_w, fs_w = 7, 6, 7
    name_min = 6

    # Widest layout first; drop FS, then GROUP, as the width shrinks.
    if group_col:
        candidates = [(True, True), (True, False), (False, False)]
    else:
        candidates = [(False, True), (False, False)]

    for with_group, with_fs in candidates:
        flex = 2 if with_group else 1  # name (+ group)
        fixed = rank_w + cpu_w + gpu_w + (fs_w if with_fs else 0)
        # rank, name, cpu, gpu are always present (4); add group and/or fs.
        ncols = 4 + (1 if with_group else 0) + (1 if with_fs else 0)
        remaining = inner_width - fixed - (ncols - 1)
        if remaining < name_min * flex:
            continue
        base = remaining // flex
        name_w = base
        group_w = remaining - base  # remainder goes to GROUP
        cols = [("rank", "#", rank_w, "r"), ("name", entity_label, name_w, "l")]
        if with_group:
            cols.append(("group", "GROUP", group_w, "l"))
        cols += [("cpu", "CPUh", cpu_w, "r"), ("gpu", "GPUh", gpu_w, "r")]
        if with_fs:
            cols.append(("fs", fairshare_label, fs_w, "r"))
        return cols

    # Ultra-narrow fallback: rank/name/cpu/gpu with the name squeezed down.
    name_w = max(4, inner_width - rank_w - cpu_w - gpu_w - 3)
    return [
        ("rank", "#", rank_w, "r"),
        ("name", entity_label, name_w, "l"),
        ("cpu", "CPUh", cpu_w, "r"),
        ("gpu", "GPUh", gpu_w, "r"),
    ]


def responsive_job_specs(
    available_width: int,
    *,
    show_principals: bool = False,
) -> list[tuple[str, int, int, Callable[[Job], str]]]:
    specs: list[tuple[str, int, int, Callable[[Job], str]]] = [
        ("JOBID", 10, 22, lambda job: job.job_id),
    ]
    if show_principals:
        specs.extend(
            [
                ("USER", 6, 16, lambda job: job.user or "-"),
                ("GROUP", 6, 18, lambda job: job.group or "-"),
            ]
        )
    specs.extend(
        [
        ("STATE", 8, 14, lambda job: job.state),
        ("PARTITION", 10, 22, lambda job: job.partition),
        ("ELAPSED", 8, 14, lambda job: job.elapsed),
        ("LIMIT", 8, 14, lambda job: job.limit),
        ("CPUS", 4, 8, lambda job: job.cpus),
        ("WHERE / WHY", 18, 56, lambda job: job.location),
        ]
    )
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs

    specs = [spec for spec in specs if spec[0] != "LIMIT"]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs

    specs = [spec for spec in specs if spec[0] != "CPUS"]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs

    specs = [spec for spec in specs if spec[0] != "GROUP"]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs

    return [spec for spec in specs if spec[0] != "USER"]


def responsive_job_group_specs(
    available_width: int,
    *,
    show_principals: bool = False,
) -> list[tuple[str, int, int, Callable[[JobRecordGroup], str]]]:
    specs: list[tuple[str, int, int, Callable[[JobRecordGroup], str]]] = [
        ("JOBID", 10, 16, lambda group: group.array_parent),
        ("JOB", 12, 28, lambda group: group.name),
    ]
    if show_principals:
        specs.extend(
            [
                ("USER", 6, 16, lambda group: group.user or "-"),
                ("GROUP", 6, 18, lambda group: group.group or "-"),
            ]
        )
    specs.extend(
        [
        ("REQ", 3, 5, lambda group: str(group.total)),
        ("DONE", 4, 5, lambda group: str(group.completed)),
        ("RUN", 3, 5, lambda group: str(group.running)),
        ("PEND", 4, 5, lambda group: str(group.pending)),
        ("FAIL", 4, 5, lambda group: str(group.failed)),
        ("RUN_FOR", 7, 12, lambda group: group.longest_running_elapsed),
        ("LIMIT", 8, 14, lambda group: group.limit),
        ]
    )
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs
    specs = [spec for spec in specs if spec[0] != "LIMIT"]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs
    specs = [spec for spec in specs if spec[0] != "GROUP"]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs
    return [spec for spec in specs if spec[0] != "USER"]


def responsive_history_group_specs(
    available_width: int,
) -> list[tuple[str, int, int, Callable[[JobRecordGroup], str]]]:
    specs: list[tuple[str, int, int, Callable[[JobRecordGroup], str]]] = [
        ("JOBID", 10, 16, lambda group: group.array_parent),
        ("JOB", 12, 28, lambda group: group.name),
        ("REQ", 3, 5, lambda group: str(group.total)),
        ("DONE", 4, 5, lambda group: str(group.completed)),
        ("RUN", 3, 5, lambda group: str(group.running)),
        ("PEND", 4, 5, lambda group: str(group.pending)),
        ("FAIL", 4, 5, lambda group: str(group.failed)),
        ("CPUS", 4, 6, lambda group: str(group.cpus)),
        ("GPUS", 4, 6, lambda group: str(group.gpus)),
        ("RUN_FOR", 7, 12, lambda group: group.longest_running_elapsed),
        ("LIMIT", 8, 14, lambda group: group.limit),
    ]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs

    specs = [spec for spec in specs if spec[0] != "LIMIT"]
    if minimum_table_width(label_widths(specs)) <= available_width:
        return specs

    return [spec for spec in specs if spec[0] not in {"CPUS", "GPUS"}]


def responsive_node_specs(
    show_resource_bars: bool,
    cpu_count_width: int,
    memory_count_width: int,
    gpu_count_width: int,
) -> list[tuple[str, int, int]]:
    if show_resource_bars:
        return [
            ("NODE", 10, 22),
            ("STATE", 8, 18),
            ("PARTITION", 10, 22),
            ("CPU", 24, 38),
            ("MEM", 24, 38),
            ("GPU", 18, 30),
            ("GRES", 12, 48),
        ]

    cpu_width = max(4, cpu_count_width)
    memory_width = max(4, memory_count_width)
    gpu_width = max(3, gpu_count_width)
    return [
        ("NODE", 10, 22),
        ("STATE", 8, 18),
        ("PARTITION", 10, 22),
        ("CPU", cpu_width, max(cpu_width, 8)),
        ("MEM", memory_width, max(memory_width, 12)),
        ("GPU", gpu_width, max(gpu_width, 8)),
        ("GRES", 12, 48),
    ]


def label_widths(
    specs: list[tuple[str, int, int, Callable[..., str]]],
) -> list[tuple[str, int, int]]:
    return [(label, min_width, max_width) for label, min_width, max_width, _ in specs]


def minimum_table_width(specs: list[tuple[str, int, int]]) -> int:
    if not specs:
        return 0
    return sum(min_width for _, min_width, _ in specs) + len(specs) - 1


def fit_columns(
    specs: list[tuple[str, int, int]],
    rows: list[list[str]],
    available_width: int,
) -> list[int]:
    widths: list[int] = []
    for index, (label, min_width, max_width) in enumerate(specs):
        content_width = len(label)
        for row in rows:
            if index < len(row):
                content_width = max(content_width, len(row[index]))
        widths.append(min(max(content_width, min_width), max_width))

    gaps = max(0, len(widths) - 1)
    target = max(1, available_width - gaps)
    while sum(widths) > target:
        candidates = [
            index
            for index, width in enumerate(widths)
            if width > specs[index][1]
        ]
        if not candidates:
            break
        widest = max(candidates, key=lambda index: widths[index])
        widths[widest] -= 1
    return widths
