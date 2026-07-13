from __future__ import annotations

import curses

from .ui_constants import MUTED_PAIR
from .widgets import (
    pct,
    resource_count_width,
    resource_meter,
    resource_text_meter,
    resource_text_width,
)
from .table_layout import (
    fit_columns,
    label_widths,
    minimum_table_width,
    responsive_history_group_specs,
    responsive_job_group_specs,
    responsive_job_specs,
    responsive_node_specs,
)
from .summaries import (
    page_status,
    status_title,
    summary_title,
)
from .slurm import (
    Job,
    JobRecordGroup,
    Node,
    summarize_job_records,
    summarize_jobs,
    summarize_nodes,
)


class RenderTablesMixin:
    def _draw_jobs_table(
        self,
        stdscr: curses.window,
        visible: list[Job],
        height: int,
        width: int,
    ) -> None:
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        title = summary_title(
            summarize_jobs(visible),
            ["RUNNING", "PENDING", "COMPLETED"],
        )
        self._draw_box(stdscr, table_top, 0, table_height, width, title)
        header_y = table_top + 1
        first_row = table_top + 2
        rows = max(0, table_height - 3)
        available_width = max(1, width - 4)
        job_specs = responsive_job_specs(
            available_width,
            show_principals=self._show_job_principal_columns(),
        )
        row_values = [
            [value_fn(job) for _, _, _, value_fn in job_specs]
            for job in visible
        ]
        columns = fit_columns(
            [
                (label, min_width, max_width)
                for label, min_width, max_width, _ in job_specs
            ],
            row_values,
            available_width,
        )
        headers = [
            (label, column_width)
            for (label, _, _, _), column_width in zip(job_specs, columns)
        ]
        x = 2
        for label, size in headers:
            self._addstr(stdscr, header_y, x, label[:size].ljust(size), self._pair(MUTED_PAIR) | curses.A_BOLD)
            x += size + 1

        if self.state.selected < self.state.scroll:
            self.state.scroll = self.state.selected
        if self.state.selected >= self.state.scroll + rows:
            self.state.scroll = self.state.selected - rows + 1
        page_label = page_status(self.state.selected, len(visible), rows)
        if page_label:
            footer = f" {page_label} "
            footer_x = max(2, width - len(footer) - 2)
            self._addstr(
                stdscr,
                table_top + table_height - 1,
                footer_x,
                footer,
                self._pair(5) | curses.A_BOLD,
            )

        for screen_row, job in enumerate(
            visible[self.state.scroll : self.state.scroll + rows],
            start=first_row,
        ):
            index = self.state.scroll + screen_row - first_row
            attr = self._state_attr(job.state)
            if index == self.state.selected:
                attr |= curses.A_REVERSE
            cells = [
                (value_fn(job), column_width)
                for (_, _, _, value_fn), column_width in zip(job_specs, columns)
            ]
            x = 2
            for value, size in cells:
                text = value[:size].ljust(size)
                if x < width:
                    self._addstr(stdscr, screen_row, x, text[: max(0, width - x - 1)], attr)
                x += size + 1

    def _draw_job_groups_table(
        self,
        stdscr: curses.window,
        visible: list[JobRecordGroup],
        height: int,
        width: int,
    ) -> None:
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        title = status_title(
            "Running Groups",
            summarize_jobs(self._visible_jobs()),
            ["RUNNING", "PENDING", "COMPLETED"],
        )
        self._draw_box(stdscr, table_top, 0, table_height, width, title)
        header_y = table_top + 1
        first_row = table_top + 2
        rows = max(0, table_height - 3)
        available_width = max(1, width - 4)
        group_specs = responsive_job_group_specs(
            available_width,
            show_principals=self._show_job_principal_columns(),
        )
        row_values = [
            [value_fn(group) for _, _, _, value_fn in group_specs]
            for group in visible
        ]
        columns = fit_columns(
            [
                (label, min_width, max_width)
                for label, min_width, max_width, _ in group_specs
            ],
            row_values,
            available_width,
        )
        headers = [
            (label, column_width)
            for (label, _, _, _), column_width in zip(group_specs, columns)
        ]
        x = 2
        for label, size in headers:
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
            footer_x = max(2, width - len(footer) - 2)
            self._addstr(
                stdscr,
                table_top + table_height - 1,
                footer_x,
                footer,
                self._pair(5) | curses.A_BOLD,
            )

        for screen_row, group in enumerate(
            visible[self.state.scroll : self.state.scroll + rows],
            start=first_row,
        ):
            index = self.state.scroll + screen_row - first_row
            attr = self._state_attr(group.dominant_state)
            if index == self.state.selected:
                attr |= curses.A_REVERSE
            cells = [
                (value_fn(group), column_width)
                for (_, _, _, value_fn), column_width in zip(group_specs, columns)
            ]
            x = 2
            for value, size in cells:
                text = value[:size].ljust(size)
                if x < width:
                    self._addstr(
                        stdscr,
                        screen_row,
                        x,
                        text[: max(0, width - x - 1)],
                        attr,
                    )
                x += size + 1

    def _draw_history_groups_table(
        self,
        stdscr: curses.window,
        visible: list[JobRecordGroup],
        height: int,
        width: int,
    ) -> None:
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        title = summary_title(
            summarize_job_records(self.state.history),
            ["RUNNING", "PENDING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"],
        )
        self._draw_box(stdscr, table_top, 0, table_height, width, title)
        header_y = table_top + 1
        first_row = table_top + 2
        rows = max(0, table_height - 3)
        available_width = max(1, width - 4)
        specs = responsive_history_group_specs(available_width)
        row_values = [
            [value_fn(group) for _, _, _, value_fn in specs]
            for group in visible
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
            footer_x = max(2, width - len(footer) - 2)
            self._addstr(
                stdscr,
                table_top + table_height - 1,
                footer_x,
                footer,
                self._pair(5) | curses.A_BOLD,
            )

        for screen_row, group in enumerate(
            visible[self.state.scroll : self.state.scroll + rows],
            start=first_row,
        ):
            index = self.state.scroll + screen_row - first_row
            attr = self._state_attr(group.dominant_state)
            if index == self.state.selected:
                attr |= curses.A_REVERSE
            x = 2
            for (_, _, _, value_fn), size in zip(specs, columns):
                text = value_fn(group)[:size].ljust(size)
                if x < width:
                    self._addstr(
                        stdscr,
                        screen_row,
                        x,
                        text[: max(0, width - x - 1)],
                        attr,
                    )
                x += size + 1

    def _draw_nodes_table(
        self,
        stdscr: curses.window,
        visible: list[Node],
        height: int,
        width: int,
    ) -> None:
        table_top = 5
        detail_height = min(8, max(4, height // 4))
        table_height = max(4, height - detail_height - table_top)
        title = summary_title(
            summarize_nodes(self.state.nodes),
            ["IDLE", "MIXED", "ALLOCATED", "DOWN"],
        )
        self._draw_box(stdscr, table_top, 0, table_height, width, title)
        header_y = table_top + 1
        first_row = table_top + 2
        rows = max(0, table_height - 3)
        available_width = max(1, width - 4)
        cpu_count_width = resource_count_width(
            [(node.cpu_alloc, node.cpu_total) for node in visible]
        )
        gpu_count_width = resource_text_width([node.gpu_text for node in visible])
        memory_count_width = resource_text_width(
            [node.memory_text for node in visible]
        )
        show_resource_bars = available_width >= minimum_table_width(
            [
                ("NODE", 10, 22),
                ("STATE", 8, 18),
                ("PARTITION", 10, 22),
                ("CPU", 24, 38),
                ("MEM", 24, 38),
                ("GPU", 18, 30),
                ("GRES", 12, 48),
            ]
        )
        node_specs = responsive_node_specs(
            show_resource_bars,
            cpu_count_width,
            memory_count_width,
            gpu_count_width,
        )
        row_values = []
        for node in visible:
            gpu_percent = pct(node.gpu_alloc, node.gpu_total)
            if show_resource_bars:
                row_values.append(
                    [
                        node.name,
                        node.state,
                        node.partitions,
                        resource_meter(
                            node.cpu_alloc,
                            node.cpu_total,
                            node.cpu_percent,
                            meter_width=16,
                            count_width=cpu_count_width,
                        ),
                        resource_text_meter(
                            node.memory_text,
                            node.memory_percent,
                            meter_width=14,
                            count_width=memory_count_width,
                        ),
                        resource_text_meter(
                            node.gpu_text,
                            gpu_percent,
                            meter_width=12,
                            count_width=gpu_count_width,
                        ),
                        node.gres,
                    ]
                )
            else:
                row_values.append(
                    [
                        node.name,
                        node.state,
                        node.partitions,
                        f"{node.cpu_alloc}/{node.cpu_total}",
                        node.memory_text,
                        node.gpu_text,
                        node.gres,
                    ]
                )
        columns = fit_columns(
            node_specs,
            row_values,
            available_width,
        )
        headers = [
            (label, column_width)
            for (label, _, _), column_width in zip(node_specs, columns)
        ]
        x = 2
        for label, size in headers:
            self._addstr(stdscr, header_y, x, label[:size].ljust(size), self._pair(MUTED_PAIR) | curses.A_BOLD)
            x += size + 1

        if self.state.selected < self.state.scroll:
            self.state.scroll = self.state.selected
        if self.state.selected >= self.state.scroll + rows:
            self.state.scroll = self.state.selected - rows + 1
        page_label = page_status(self.state.selected, len(visible), rows)
        if page_label:
            footer = f" {page_label} "
            footer_x = max(2, width - len(footer) - 2)
            self._addstr(
                stdscr,
                table_top + table_height - 1,
                footer_x,
                footer,
                self._pair(5) | curses.A_BOLD,
            )

        for screen_row, node in enumerate(
            visible[self.state.scroll : self.state.scroll + rows],
            start=first_row,
        ):
            index = self.state.scroll + screen_row - first_row
            attr = self._node_attr(node)
            if index == self.state.selected:
                attr |= curses.A_REVERSE
            gpu_percent = pct(node.gpu_alloc, node.gpu_total)
            if show_resource_bars:
                values = [
                    node.name,
                    node.state,
                    node.partitions,
                    resource_meter(
                        node.cpu_alloc,
                        node.cpu_total,
                        node.cpu_percent,
                        meter_width=16,
                        count_width=cpu_count_width,
                    ),
                    resource_text_meter(
                        node.memory_text,
                        node.memory_percent,
                        meter_width=14,
                        count_width=memory_count_width,
                    ),
                    resource_text_meter(
                        node.gpu_text,
                        gpu_percent,
                        meter_width=12,
                        count_width=gpu_count_width,
                    ),
                    node.gres,
                ]
            else:
                values = [
                    node.name,
                    node.state,
                    node.partitions,
                    f"{node.cpu_alloc}/{node.cpu_total}",
                    node.memory_text,
                    node.gpu_text,
                    node.gres,
                ]
            cells = list(zip(values, columns))
            x = 2
            for value, size in cells:
                text = value[:size].ljust(size)
                if x < width:
                    self._addstr(stdscr, screen_row, x, text[: max(0, width - x - 1)], attr)
                x += size + 1
