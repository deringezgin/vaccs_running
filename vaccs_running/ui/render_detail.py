from __future__ import annotations

import curses

from .widgets import (
    meter,
    pct,
)
from .text_layout import wrap_detail_blocks, wrap_detail_lines


class RenderDetailMixin:
    def _job_detail_lines(self, width: int) -> list[str]:
        job = self._selected_job()
        if not job:
            return ["No jobs found."]

        wait_label = "waiting-for" if job.is_pending else "waited-for"
        wait_value = job.waiting_for if job.is_pending else job.waited_for
        timing = [
            f"submitted={job.submit_time}",
            f"started={job.start_time}",
            f"{wait_label}={wait_value}",
        ]
        if not job.is_running:
            timing.append(f"reason={job.reason}")

        rows = [
            [
                job.name,
                f"job={job.job_id}",
                f"array-parent={job.array_parent}",
                f"state={job.state}",
                f"partition={job.partition}",
                f"nodes={job.nodes or '-'}",
            ],
            timing,
            [
                f"resources: nodes={job.node_count}",
                f"cpus={job.cpus}",
                f"gres={job.gres}",
            ],
        ]
        return wrap_detail_blocks(rows, max(1, width - 4))

    def _job_detail_height(self, height: int, width: int) -> int:
        desired = len(self._job_detail_lines(width)) + 2
        max_height = max(3, height - 5 - 4)
        return min(max_height, max(3, desired))

    def _draw_job_detail(self, stdscr: curses.window, height: int, width: int) -> None:
        panel_height = self._job_detail_height(height, width)
        top = max(4, height - panel_height)
        job = self._selected_job()
        self._draw_box(stdscr, top, 0, panel_height, width, " selected job ")
        if not job:
            self._addstr(stdscr, top + 1, 2, "No jobs found.", self._pair(2))
            return

        body_rows = max(0, min(height - 1, top + panel_height - 1) - top - 1)
        for offset, line in enumerate(self._job_detail_lines(width)[:body_rows]):
            self._addstr(
                stdscr,
                top + 1 + offset,
                2,
                line,
                self._state_attr(job.state),
            )

    def _draw_job_group_detail(
        self,
        stdscr: curses.window,
        height: int,
        width: int,
    ) -> None:
        panel_height = min(8, max(4, height // 4))
        top = max(4, height - panel_height)
        group = self._selected_job_group()
        self._draw_box(stdscr, top, 0, panel_height, width, " selected job group ")
        if not group:
            self._addstr(stdscr, top + 1, 2, "No job groups found.", self._pair(2))
            return

        lines = [
            f"{group.name}  array-parent={group.array_parent}",
            (
                f"requested={group.total}  done={group.completed} "
                f"running={group.running}  pending={group.pending}  failed={group.failed}"
            ),
            (
                f"longest-running={group.longest_running_elapsed}  "
                f"limit={group.limit}  other={group.other}"
            ),
        ]
        body_rows = max(0, min(height - 1, top + panel_height - 1) - top - 1)
        wrapped = wrap_detail_lines(lines, max(1, width - 4))
        for offset, line in enumerate(wrapped[:body_rows]):
            self._addstr(
                stdscr,
                top + 1 + offset,
                2,
                line,
                self._state_attr(group.dominant_state),
            )

    def _draw_history_group_detail(
        self,
        stdscr: curses.window,
        height: int,
        width: int,
    ) -> None:
        panel_height = min(8, max(4, height // 4))
        top = max(4, height - panel_height)
        group = self._selected_history_group()
        self._draw_box(stdscr, top, 0, panel_height, width, " selected history group ")
        if not group:
            self._addstr(stdscr, top + 1, 2, "No history groups found.", self._pair(2))
            return

        lines = [
            f"{group.name}  array-parent={group.array_parent}",
            (
                f"requested={group.total}  done={group.completed}  running={group.running} "
                f"pending={group.pending}  failed={group.failed}  other={group.other}"
            ),
            (
                f"resources: cpus={group.cpus}  gpus={group.gpus} "
                f"limit={group.limit}"
            ),
            f"submitted={group.submit_time}  latest-end={group.end_time or '-'}",
        ]
        body_rows = max(0, min(height - 1, top + panel_height - 1) - top - 1)
        wrapped = wrap_detail_lines(lines, max(1, width - 4))
        for offset, line in enumerate(wrapped[:body_rows]):
            self._addstr(
                stdscr,
                top + 1 + offset,
                2,
                line,
                self._state_attr(group.dominant_state),
            )

    def _node_detail_lines(self, width: int) -> list[str]:
        node = self._selected_node()
        if not node:
            return ["No nodes found."]

        gpu_percent = pct(node.gpu_alloc, node.gpu_total)
        lines = [
            f"{node.name}  state={node.state}  partition={node.partitions}",
            (
                f"cpu alloc={node.cpu_alloc}/{node.cpu_total} "
                f"free={node.free_cpus}  {meter(node.cpu_percent, 18)}  live-load={node.cpu_load:.2f}"
            ),
            (
                f"mem alloc={node.memory_text} ({node.memory_percent:.1f}%) "
                f"{meter(node.memory_percent, 18)}  free-os={node.free_memory_mb // 1024}G"
            ),
            f"gpu alloc={node.gpu_text} free={node.gpu_free}  {meter(gpu_percent, 18)}  tres={node.alloc_tres or '-'}",
            f"features={node.features}",
        ]
        return wrap_detail_lines(lines, max(1, width - 4))

    def _node_detail_height(self, height: int, width: int) -> int:
        desired = len(self._node_detail_lines(width)) + 2
        max_height = max(3, height - 5 - 4)
        return min(max_height, max(3, desired))

    def _draw_node_detail(self, stdscr: curses.window, height: int, width: int) -> None:
        panel_height = self._node_detail_height(height, width)
        top = max(4, height - panel_height)
        node = self._selected_node()
        self._draw_box(stdscr, top, 0, panel_height, width, " selected node ")
        if not node:
            self._addstr(stdscr, top + 1, 2, "No nodes found.", self._pair(2))
            return

        body_rows = max(0, min(height - 1, top + panel_height - 1) - top - 1)
        for offset, line in enumerate(self._node_detail_lines(width)[:body_rows]):
            self._addstr(
                stdscr,
                top + 1 + offset,
                2,
                line,
                self._node_attr(node),
            )
