from __future__ import annotations

import curses
from collections import Counter

from .constants import MUTED_PAIR
from .summaries import page_status
from .table_layout import (
    fit_columns,
    label_widths,
    priority_rank_text,
    responsive_priority_specs,
)
from .text_layout import wrap_detail_lines
from ..slurm import PriorityQueueEntry, PriorityQueueJob, PriorityQueueSnapshot


def _ahead_user_summary(
    entry: PriorityQueueEntry,
    current_user: str,
    all_entries: tuple[PriorityQueueEntry, ...],
    limit: int = 4,
) -> str:
    if entry.priority_rank is None or entry.priority_rank <= 1:
        return "none"
    reservation = entry.job.reservation
    if reservation.upper() in {"N/A", "NONE", "(NULL)"}:
        reservation = ""
    ahead: list[PriorityQueueJob] = []
    for candidate in all_entries:
        job = candidate.job
        if (
            candidate.priority_rank is None
            or candidate.priority_rank >= entry.priority_rank
            or job.partition != entry.job.partition
        ):
            continue
        job_reservation = job.reservation
        if job_reservation.upper() in {"N/A", "NONE", "(NULL)"}:
            job_reservation = ""
        if job_reservation != reservation:
            continue
        ahead.append(job)
    counts = Counter(
        (job.user or "unknown", job.account or "")
        for job in ahead
    )
    if not counts:
        return "none"
    labels: list[str] = []
    for (user, account), count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )[:limit]:
        who = "you" if user == current_user else user
        if account:
            who += f"/{account}"
        labels.append(f"{who}×{count}")
    hidden = len(counts) - len(labels)
    if hidden > 0:
        labels.append(f"+{hidden} more")
    return ", ".join(labels)


def _priority_factor_text(
    entry: PriorityQueueEntry,
    available: bool,
    current_user: str,
) -> str:
    priority = entry.display_priority
    if entry.is_consecutive_user_group:
        group_kind = (
            "unranked pending entries"
            if entry.priority_rank is None
            else "consecutive rank slots"
        )
        if entry.job.user != current_user:
            return (
                f"reported priority={priority}; press e for each entry; weighted "
                f"sprio components are queried only for {current_user}"
            )
        return (
            f"reported priority={priority}; this packed row combines {group_kind}; "
            "press e for each entry's weighted sprio components"
        )
    if entry.job.user != current_user:
        return (
            f"priority={priority}; weighted sprio components are queried only "
            f"for {current_user}"
        )
    if not available or entry.job.factors is None:
        return f"priority={priority}; weighted sprio breakdown unavailable"
    factors = entry.job.factors
    values = [
        ("age", factors.age),
        ("fair-share", factors.fairshare),
        ("job-size", factors.job_size),
        ("TRES", factors.tres),
        ("association", factors.association),
        ("partition", factors.partition),
        ("QOS", factors.qos),
        ("site", factors.site),
        ("nice", factors.nice),
    ]
    rendered = "  ".join(
        f"{label}={'-' if value is None else value}" for label, value in values
    )
    return f"priority={priority}; weighted components: {rendered}"


class RenderPriorityMixin:
    def _draw_priority(
        self,
        stdscr: curses.window,
        height: int,
        width: int,
    ) -> None:
        snapshot = self.state.priority_queue
        visible = self._visible_priority_entries()
        self._draw_priority_table(stdscr, visible, snapshot, height, width)
        self._draw_priority_detail(stdscr, height, width)

    def _draw_priority_table(
        self,
        stdscr: curses.window,
        visible: list[PriorityQueueEntry],
        snapshot: PriorityQueueSnapshot | None,
        height: int,
        width: int,
    ) -> None:
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        if snapshot is None:
            title = " Priority queue · loading "
        else:
            # --priority duplicates a multi-partition job once per route; count
            # distinct job/task IDs so the title does not double-count them.
            pending_tasks = len(
                {
                    entry.job.job_id
                    for entry in self._visible_priority_all_entries()
                    if entry.job.user == snapshot.user
                }
            )
            task_label = f"{pending_tasks} task{'s' if pending_tasks != 1 else ''}"
            queue_entries = len(self._visible_priority_all_entries())
            entry_label = (
                f"{queue_entries} queue "
                f"{'entries' if queue_entries != 1 else 'entry'}"
            )
            if self.state.priority_extended:
                title = (
                    f" Extended: {entry_label} · "
                    f"yours: {task_label} "
                )
            else:
                packed_entries = self._visible_priority_grouped_entries()
                queue_groups = sum(
                    entry.priority_rank is not None for entry in packed_entries
                )
                unranked = len(packed_entries) - queue_groups
                group_label = (
                    f"{queue_groups} rank "
                    f"{'runs' if queue_groups != 1 else 'run'}"
                )
                if unranked:
                    group_label += (
                        f" + {unranked} unranked "
                        f"{'groups' if unranked != 1 else 'group'}"
                    )
                title = (
                    f" Packed: {group_label} / {entry_label} · "
                    f"yours: {task_label} "
                )
        self._draw_box(stdscr, table_top, 0, table_height, width, title)

        header_y = table_top + 1
        first_row = table_top + 2
        rows = max(0, table_height - 3)
        available_width = max(1, width - 4)
        specs = responsive_priority_specs(
            available_width,
            extended=self.state.priority_extended,
            current_user=snapshot.user if snapshot is not None else "",
        )
        row_values = [
            [value_fn(entry) for _, _, _, value_fn in specs]
            for entry in visible
        ]
        columns = fit_columns(label_widths(specs), row_values, available_width)
        x = 2
        for (label, _, _, _), size in zip(specs, columns):
            self._addstr(
                stdscr,
                header_y,
                x,
                label[:size].ljust(size),
                self._pair(MUTED_PAIR) | curses.A_BOLD,
            )
            x += size + 1

        if self.state.selected < self.state.scroll:
            self.state.scroll = self.state.selected
        if self.state.selected >= self.state.scroll + rows:
            self.state.scroll = self.state.selected - rows + 1
        page_label = page_status(self.state.selected, len(visible), rows)
        if page_label:
            footer = f" {page_label} "
            self._addstr(
                stdscr,
                table_top + table_height - 1,
                max(2, width - len(footer) - 2),
                footer,
                self._pair(5) | curses.A_BOLD,
            )

        for screen_row, entry in enumerate(
            visible[self.state.scroll : self.state.scroll + rows],
            start=first_row,
        ):
            index = self.state.scroll + screen_row - first_row
            attr = self._state_attr("PENDING")
            if snapshot is not None and entry.job.user == snapshot.user:
                attr |= curses.A_BOLD
            if index == self.state.selected:
                attr |= curses.A_REVERSE
            x = 2
            for (_, _, _, value_fn), size in zip(specs, columns):
                value = value_fn(entry)
                self._addstr(
                    stdscr,
                    screen_row,
                    x,
                    value[:size].ljust(size)[: max(0, width - x - 1)],
                    attr,
                )
                x += size + 1

    def _draw_priority_detail(
        self,
        stdscr: curses.window,
        height: int,
        width: int,
    ) -> None:
        panel_height = min(8, max(4, height // 4))
        top = max(4, height - panel_height)
        snapshot = self.state.priority_queue
        entry = self._selected_priority_entry()
        if entry is not None and entry.is_consecutive_user_group:
            panel_title = (
                " selected rank run "
                if entry.priority_rank is not None
                else " selected unranked group "
            )
        else:
            panel_title = " selected pending job "
        self._draw_box(stdscr, top, 0, panel_height, width, panel_title)
        if snapshot is None:
            self._addstr(stdscr, top + 1, 2, "Loading priority queue…", self._pair(2))
            return
        if entry is None:
            if self._priority_partition_filter_active():
                summary = self._priority_partition_summary()
                empty = f"No pending queue entries for partition {summary}."
            else:
                empty = "No pending queue entries in the cluster."
            self._addstr(
                stdscr,
                top + 1,
                2,
                empty,
                self._pair(2),
            )
            return

        job = entry.job
        if entry.priority_rank is None:
            rank_line = entry.rank_note or "No priority rank: this job is not currently schedulable."
        else:
            reservation = job.reservation.strip()
            scope = job.partition
            if reservation.upper() not in {"", "N/A", "NONE", "(NULL)"}:
                scope += f" / reservation {reservation}"
            rank_line = (
                f"priority rank={priority_rank_text(entry)} in {scope}; "
                f"{entry.earlier_count} earlier rankable entries from "
                f"{entry.earlier_user_count} users; "
                "backfill/resource fit can run a lower rank first"
            )
        if self.state.priority_extended:
            context_line = (
                "extended view: surrounding rows show the other queued jobs; "
                "YOU marks your jobs"
            )
        else:
            ahead = _ahead_user_summary(entry, snapshot.user, snapshot.all_entries)
            if entry.is_consecutive_user_group:
                job_label = (
                    f"{entry.job_count} job"
                    f"{'s' if entry.job_count != 1 else ''}"
                )
                if entry.priority_rank is None:
                    context_line = (
                        f"packed same-user unranked group: {job_label} / "
                        f"{entry.task_count} pending entries; no priority rank "
                        "or ahead count"
                    )
                else:
                    context_line = (
                        f"packed same-user rank run: {job_label} / "
                        f"{entry.task_count} consecutive rank slots; "
                        f"users ahead: {ahead}"
                    )
            else:
                context_line = f"users ahead in snapshot: {ahead}"

        if entry.is_consecutive_user_group:
            group_ids = entry.group_job_ids or (job.array_parent,)
            id_summary = group_ids[0]
            if len(group_ids) > 1:
                id_summary += f"…{group_ids[-1]}"
            entry_label = "slots" if entry.priority_rank is not None else "entries"
            identity_line = (
                f"{entry.display_name}  {entry_label}={entry.display_job_id}  "
                f"ids={id_summary}  "
                f"partition={job.partition}  user={job.user or '-'}"
            )
            if len(entry.group_reason_codes) > 1:
                why_line = (
                    f"why waiting: {entry.display_reason} — reasons vary within this "
                    "group; press e to inspect each job"
                )
            else:
                why_line = (
                    f"why waiting: {entry.display_reason} — {job.reason_explanation}"
                )
            request_unit = "slots" if entry.priority_rank is not None else "entries"
            walltime_unit = "slot" if entry.priority_rank is not None else "entry"
            request_line = (
                f"requested across {entry.task_count} {request_unit}: "
                f"GPUs={entry.display_gpus}  CPUs={entry.display_cpus}  "
                f"RAM={entry.display_memory}; "
                f"walltime/{walltime_unit}={entry.display_walltime}"
            )
        else:
            identity_line = (
                f"{entry.display_name}  job={entry.display_job_id}  "
                f"partition={job.partition}  user={job.user or '-'}  "
                f"account={job.account or '-'}  qos={job.qos or '-'}"
            )
            why_line = (
                f"why waiting: {job.reason_code or 'Unknown'} — "
                f"{job.reason_explanation}"
            )
            request_line = (
                f"requested: GPUs={entry.display_gpus}  CPUs={entry.display_cpus}  "
                f"RAM={entry.display_memory}  walltime={entry.display_walltime}; "
                f"submitted={job.submit_time or '-'}; "
                f"estimated start={job.estimated_start}"
            )
        lines = [
            identity_line,
            why_line,
            rank_line,
            context_line,
            request_line,
            _priority_factor_text(
                entry,
                snapshot.factors_available,
                snapshot.user,
            ),
            (
                "rank is a snapshot, not a start-order promise; resource fit, "
                "reservations, partition tiers, and backfill can change what starts next"
            ),
        ]
        body_rows = max(0, min(height - 1, top + panel_height - 1) - top - 1)
        for offset, line in enumerate(
            wrap_detail_lines(lines, max(1, width - 4))[:body_rows]
        ):
            self._addstr(
                stdscr,
                top + 1 + offset,
                2,
                line,
                self._state_attr("PENDING"),
            )
