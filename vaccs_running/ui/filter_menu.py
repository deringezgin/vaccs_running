from __future__ import annotations

import curses

from .constants import (
    ACTIVE_TAB_PAIR,
    JOB_STATE_FILTER_OPTIONS,
    MUTED_PAIR,
)
from .text_layout import filter_choice_options
from .summaries import job_state_filter_label
from .curses_compat import (
    safe_curs_set,
    safe_getmouse,
    safe_mousemask,
)
from ..slurm import (
    JobFilterChoices,
    SlurmError,
    VACC_PARTITIONS,
)


class JobFilterMenuMixin:
    def _show_jobs_filter(self, stdscr: curses.window) -> None:
        choices = self._fetch_job_filter_choices()
        self._run_jobs_filter_menu(
            stdscr,
            "running filter",
            lambda: self._jobs_filter_home_items(),
            lambda win, height, width, item: self._activate_jobs_filter_home_item(
                stdscr,
                choices,
                item,
            ),
            " enter/click open  c clear  q close ",
            close_keys=(ord("f"),),
        )

    def _fetch_job_filter_choices(self) -> JobFilterChoices:
        fetch = getattr(self.client, "fetch_running_filter_choices", None)
        if not fetch:
            return JobFilterChoices(users=[], groups=[], partitions=list(VACC_PARTITIONS))
        try:
            return fetch()
        except SlurmError:
            return JobFilterChoices(users=[], groups=[], partitions=list(VACC_PARTITIONS))

    def _show_priority_filter(self, stdscr: curses.window) -> None:
        choices = self._priority_filter_choices()
        self._run_jobs_filter_menu(
            stdscr,
            "priority filter",
            self._priority_filter_home_items,
            lambda win, height, width, item: self._activate_priority_filter_home_item(
                stdscr,
                choices,
                item,
            ),
            " enter/click open  c clear  q close ",
            close_keys=(ord("f"),),
            clear_fn=self._clear_priority_filters,
        )

    def _priority_filter_choices(self) -> JobFilterChoices:
        partitions = list(VACC_PARTITIONS)
        snapshot = self.state.priority_queue
        if snapshot is not None:
            partitions.extend(job.partition for job in snapshot.pending_jobs)
        partitions.extend(self._selected_priority_partitions())
        return JobFilterChoices(
            users=[],
            groups=[],
            partitions=list(
                dict.fromkeys(partition for partition in partitions if partition)
            ),
        )

    def _priority_filter_home_items(self) -> list[dict[str, object]]:
        return [
            {
                "kind": "submenu",
                "action": "partition",
                "label": (
                    "Filter by partition: "
                    f"{self._priority_partition_summary()}"
                ),
            }
        ]

    def _activate_priority_filter_home_item(
        self,
        stdscr: curses.window,
        choices: JobFilterChoices,
        item: dict[str, object],
    ) -> None:
        if item.get("action") == "partition":
            self._show_priority_partition_filter(stdscr, choices)

    def _show_priority_partition_filter(
        self,
        stdscr: curses.window,
        choices: JobFilterChoices,
    ) -> None:
        self._run_jobs_filter_menu(
            stdscr,
            "filter by partition",
            lambda: self._priority_partition_filter_items(choices),
            lambda win, height, width, item: self._activate_priority_partition_filter_item(
                win,
                height,
                width,
                choices,
                item,
            ),
            " enter/click select  c clear  q back ",
            close_keys=(ord("f"),),
            clear_fn=self._clear_priority_filters,
        )

    def _run_jobs_filter_menu(
        self,
        stdscr: curses.window,
        title: str,
        items_fn,
        activate_fn,
        footer: str,
        close_keys: tuple[int, ...] = (),
        clear_fn=None,
    ) -> None:
        selected = 0
        scroll = 0
        height, width = stdscr.getmaxyx()
        content_width = max(len(title) + 4, len(footer), 58)
        box_width = min(max(44, content_width + 4), max(20, width - 8))
        box_height = min(max(8, height - 4), 28)
        body_height = max(1, box_height - 4)
        top = max(1, (height - box_height) // 2)
        left = max(1, (width - box_width) // 2)
        win = curses.newwin(box_height, box_width, top, left)
        self._apply_theme_background(win)
        win.keypad(True)
        win.nodelay(False)
        safe_mousemask()

        while True:
            items = items_fn()
            selectable_indexes = [
                index for index, item in enumerate(items) if item["kind"] != "separator"
            ]
            if not selectable_indexes:
                return
            if selected not in selectable_indexes:
                selected = selectable_indexes[0]
            selected_position = selectable_indexes.index(selected)
            selected_position = max(0, min(selected_position, len(selectable_indexes) - 1))
            selected = selectable_indexes[selected_position]
            if selected < scroll:
                scroll = selected
            if selected >= scroll + body_height:
                scroll = selected - body_height + 1

            win.erase()
            win.border()
            self._addstr(win, 0, 2, f" {title} ", self._pair(6) | curses.A_BOLD)
            hitboxes = []
            for offset, item in enumerate(items[scroll : scroll + body_height], start=1):
                item_index = scroll + offset - 1
                if item["kind"] == "separator":
                    self._addstr(
                        win,
                        offset,
                        2,
                        str(item["label"])[: box_width - 4],
                        self._pair(MUTED_PAIR) | curses.A_BOLD,
                    )
                    continue
                marker = ">" if item_index == selected else " "
                prefix = ""
                if "checked" in item:
                    prefix = "[x] " if item.get("checked") else "[ ] "
                text = f"{marker} {prefix}{item['label']}"
                attr = (
                    self._pair(ACTIVE_TAB_PAIR) | curses.A_BOLD
                    if item_index == selected
                    else self._pair(MUTED_PAIR)
                )
                self._addstr(win, offset, 2, text[: box_width - 4], attr)
                hitboxes.append((top + offset, left + 1, left + box_width - 2, item_index))
            self._addstr(win, box_height - 1, 2, footer[: box_width - 4], self._pair(5))
            win.refresh()

            key = win.getch()
            if key in (ord("q"), 27, *close_keys):
                return
            if key in (curses.KEY_DOWN, ord("j")):
                selected_position = min(len(selectable_indexes) - 1, selected_position + 1)
                selected = selectable_indexes[selected_position]
            elif key in (curses.KEY_UP, ord("k")):
                selected_position = max(0, selected_position - 1)
                selected = selectable_indexes[selected_position]
            elif key == curses.KEY_NPAGE:
                selected_position = min(
                    len(selectable_indexes) - 1,
                    selected_position + body_height,
                )
                selected = selectable_indexes[selected_position]
            elif key == curses.KEY_PPAGE:
                selected_position = max(0, selected_position - body_height)
                selected = selectable_indexes[selected_position]
            elif key in (ord("\n"), curses.KEY_ENTER, ord(" ")):
                activate_fn(win, box_height, box_width, items[selected])
            elif key == ord("c"):
                (clear_fn or self._clear_job_filters)()
            elif key in (ord("s"), ord("u"), ord("g"), ord("p")):
                shortcut_actions = {
                    ord("s"): ("status",),
                    ord("u"): ("custom_user", "user"),
                    ord("g"): ("custom_group", "group"),
                    ord("p"): ("custom_partition", "partition"),
                }
                for action in shortcut_actions[key]:
                    match = next(
                        (
                            index
                            for index in selectable_indexes
                            if items[index].get("action") == action
                        ),
                        None,
                    )
                    if match is not None:
                        selected = match
                        activate_fn(win, box_height, box_width, items[match])
                        break
            elif key == curses.KEY_MOUSE:
                mouse = safe_getmouse()
                if mouse:
                    _, mouse_x, mouse_y, _, button_state = mouse
                    if button_state & (curses.BUTTON1_CLICKED | curses.BUTTON1_PRESSED):
                        for y, x_min, x_max, item_index in hitboxes:
                            if mouse_y == y and x_min <= mouse_x <= x_max:
                                selected = item_index
                                activate_fn(win, box_height, box_width, items[item_index])
                                break

    def _jobs_filter_home_items(self) -> list[dict[str, object]]:
        return [
            {
                "kind": "submenu",
                "action": "status",
                "label": f"Filter by status: {job_state_filter_label(self._squeue_state_filter())}",
            },
            {
                "kind": "submenu",
                "action": "user",
                "label": f"Filter by user: {self._job_user_summary()}",
            },
            {
                "kind": "submenu",
                "action": "group",
                "label": f"Filter by group: {self._job_group_summary()}",
            },
            {
                "kind": "submenu",
                "action": "partition",
                "label": f"Filter by partition: {self._job_partition_summary()}",
            },
        ]

    def _activate_jobs_filter_home_item(
        self,
        stdscr: curses.window,
        choices: JobFilterChoices,
        item: dict[str, object],
    ) -> None:
        action = item.get("action")
        if action == "status":
            self._show_jobs_status_filter(stdscr)
        elif action == "user":
            self._show_jobs_user_filter(stdscr, choices)
        elif action == "group":
            self._show_jobs_group_filter(stdscr, choices)
        elif action == "partition":
            self._show_jobs_partition_filter(stdscr, choices)

    def _show_jobs_status_filter(self, stdscr: curses.window) -> None:
        self._run_jobs_filter_menu(
            stdscr,
            "filter by status",
            self._jobs_status_filter_items,
            self._activate_jobs_status_filter_item,
            " enter/space toggle  c clear  q back ",
            close_keys=(ord("f"),),
        )

    def _show_jobs_user_filter(
        self,
        stdscr: curses.window,
        choices: JobFilterChoices,
    ) -> None:
        self._run_jobs_filter_menu(
            stdscr,
            "filter by user",
            lambda: self._jobs_user_filter_items(choices),
            lambda win, height, width, item: self._activate_jobs_user_filter_item(
                win,
                height,
                width,
                choices,
                item,
            ),
            " enter/click toggle  c clear  q back ",
            close_keys=(ord("f"),),
        )

    def _show_jobs_group_filter(
        self,
        stdscr: curses.window,
        choices: JobFilterChoices,
    ) -> None:
        self._run_jobs_filter_menu(
            stdscr,
            "filter by group",
            lambda: self._jobs_group_filter_items(choices),
            lambda win, height, width, item: self._activate_jobs_group_filter_item(
                win,
                height,
                width,
                choices,
                item,
            ),
            " enter/click select  c clear  q back ",
            close_keys=(ord("f"),),
        )

    def _show_jobs_partition_filter(
        self,
        stdscr: curses.window,
        choices: JobFilterChoices,
    ) -> None:
        self._run_jobs_filter_menu(
            stdscr,
            "filter by partition",
            lambda: self._jobs_partition_filter_items(choices),
            lambda win, height, width, item: self._activate_jobs_partition_filter_item(
                win,
                height,
                width,
                choices,
                item,
            ),
            " enter/click select  c clear  q back ",
            close_keys=(ord("f"),),
        )

    def _jobs_status_filter_items(self) -> list[dict[str, object]]:
        selected_states = self._selected_job_state_codes()
        items: list[dict[str, object]] = []
        for code, label in JOB_STATE_FILTER_OPTIONS:
            items.append(
                {
                    "kind": "state",
                    "value": code,
                    "label": f"{code:<3} {label}",
                    "checked": code in selected_states,
                }
            )
        return items

    def _jobs_user_filter_items(self, choices: JobFilterChoices) -> list[dict[str, object]]:
        selected_users = self._selected_job_users()
        all_principals = self._job_all_principals()
        default_user = str(getattr(self.client, "user", "me") or "me")
        items: list[dict[str, object]] = [
            {
                "kind": "action",
                "action": "users_all",
                "label": "Select all",
                "checked": bool(choices.users) and selected_users == set(choices.users),
            },
            {
                "kind": "action",
                "action": "users_clear",
                "label": f"Clear all (only {default_user})",
                "checked": not all_principals and not selected_users,
            },
            {
                "kind": "action",
                "action": "custom_user",
                "label": "Enter user name...",
            },
        ]
        for user in choices.users:
            items.append(
                {
                    "kind": "user",
                    "value": user,
                    "label": user,
                    "checked": not all_principals and user in selected_users,
                }
            )
        return items

    def _jobs_group_filter_items(self, choices: JobFilterChoices) -> list[dict[str, object]]:
        selected_groups = self._selected_job_groups()
        items: list[dict[str, object]] = [
            {
                "kind": "action",
                "action": "groups_all",
                "label": "Select all",
                "checked": bool(choices.groups) and selected_groups == set(choices.groups),
            },
            {
                "kind": "action",
                "action": "groups_clear",
                "label": "Clear all",
                "checked": not selected_groups,
            },
            {
                "kind": "action",
                "action": "custom_group",
                "label": "Enter group name...",
            },
        ]
        for group in choices.groups:
            items.append(
                {
                    "kind": "group",
                    "value": group,
                    "label": group,
                    "checked": group in selected_groups,
                }
            )
        return items

    def _jobs_partition_filter_items(
        self,
        choices: JobFilterChoices,
    ) -> list[dict[str, object]]:
        return self._partition_filter_items(
            choices,
            self._selected_job_partitions(),
        )

    def _priority_partition_filter_items(
        self,
        choices: JobFilterChoices,
    ) -> list[dict[str, object]]:
        return self._partition_filter_items(
            choices,
            self._selected_priority_partitions(),
        )

    def _partition_filter_items(
        self,
        choices: JobFilterChoices,
        selected_partitions: set[str],
    ) -> list[dict[str, object]]:
        items: list[dict[str, object]] = [
            {
                "kind": "action",
                "action": "partitions_all",
                "label": "Select all",
                "checked": (
                    bool(choices.partitions)
                    and selected_partitions == set(choices.partitions)
                ),
            },
            {
                "kind": "action",
                "action": "partitions_clear",
                "label": "Clear all",
                "checked": not selected_partitions,
            },
            {
                "kind": "action",
                "action": "custom_partition",
                "label": "Enter partition name...",
            },
        ]
        for partition in choices.partitions:
            items.append(
                {
                    "kind": "partition",
                    "value": partition,
                    "label": partition,
                    "checked": partition in selected_partitions,
                }
            )
        return items

    def _activate_jobs_status_filter_item(
        self,
        win: curses.window,
        box_height: int,
        box_width: int,
        item: dict[str, object],
    ) -> None:
        kind = item["kind"]
        if kind == "state":
            states = self._selected_job_state_codes()
            value = str(item["value"])
            if value in states:
                states.remove(value)
            else:
                states.add(value)
            self._set_job_state_codes(states)
        self._refresh_jobs_after_filter_change()

    def _activate_jobs_user_filter_item(
        self,
        win: curses.window,
        box_height: int,
        box_width: int,
        choices: JobFilterChoices,
        item: dict[str, object],
    ) -> None:
        kind = item["kind"]
        action = item.get("action")
        if kind == "user":
            users = self._selected_job_users()
            value = str(item["value"])
            if value in users:
                users.remove(value)
            else:
                users.add(value)
            self._set_job_principal_filters(users, self._selected_job_groups())
        elif action == "users_all":
            self._set_job_principal_filters(
                set(choices.users),
                self._selected_job_groups(),
            )
        elif action == "users_clear":
            self._set_job_principal_filters(set(), self._selected_job_groups())
        elif action == "custom_user":
            value = self._read_jobs_filter_choice(
                win,
                box_height,
                box_width,
                "user",
                choices.users,
            )
            if value:
                if value not in choices.users:
                    choices.users.append(value)
                    choices.users.sort()
                self._set_job_principal_filters({value}, set())
        self._refresh_jobs_after_filter_change()

    def _activate_jobs_group_filter_item(
        self,
        win: curses.window,
        box_height: int,
        box_width: int,
        choices: JobFilterChoices,
        item: dict[str, object],
    ) -> None:
        kind = item["kind"]
        action = item.get("action")
        if kind == "group":
            groups = self._selected_job_groups()
            value = str(item["value"])
            if value in groups:
                groups.remove(value)
            else:
                groups.add(value)
            self._set_job_principal_filters(self._selected_job_users(), groups)
        elif action == "groups_all":
            self._set_job_principal_filters(
                self._selected_job_users(),
                set(choices.groups),
            )
        elif action == "groups_clear":
            self._set_job_principal_filters(self._selected_job_users(), set())
        elif action == "custom_group":
            value = self._read_jobs_filter_choice(
                win,
                box_height,
                box_width,
                "group",
                choices.groups,
            )
            if value:
                if value not in choices.groups:
                    choices.groups.append(value)
                    choices.groups.sort()
                self._set_job_principal_filters(set(), {value})
        self._refresh_jobs_after_filter_change()

    def _activate_jobs_partition_filter_item(
        self,
        win: curses.window,
        box_height: int,
        box_width: int,
        choices: JobFilterChoices,
        item: dict[str, object],
    ) -> None:
        self._activate_partition_filter_item(
            win,
            box_height,
            box_width,
            choices,
            item,
            selected_fn=self._selected_job_partitions,
            setter=self._set_job_partition_filters,
            after_change=self._refresh_jobs_after_filter_change,
        )

    def _activate_priority_partition_filter_item(
        self,
        win: curses.window,
        box_height: int,
        box_width: int,
        choices: JobFilterChoices,
        item: dict[str, object],
    ) -> None:
        self._activate_partition_filter_item(
            win,
            box_height,
            box_width,
            choices,
            item,
            selected_fn=self._selected_priority_partitions,
            setter=self._set_priority_partition_filters,
            after_change=self._reset_priority_after_filter_change,
        )

    def _activate_partition_filter_item(
        self,
        win: curses.window,
        box_height: int,
        box_width: int,
        choices: JobFilterChoices,
        item: dict[str, object],
        *,
        selected_fn,
        setter,
        after_change,
    ) -> None:
        kind = item["kind"]
        action = item.get("action")
        if kind == "partition":
            partitions = selected_fn()
            value = str(item["value"])
            if value in partitions:
                partitions.remove(value)
            else:
                partitions.add(value)
            setter(partitions)
        elif action == "partitions_all":
            setter(set(choices.partitions))
        elif action == "partitions_clear":
            setter(set())
        elif action == "custom_partition":
            value = self._read_jobs_filter_choice(
                win,
                box_height,
                box_width,
                "partition",
                choices.partitions,
            )
            if value:
                if value not in choices.partitions:
                    choices.partitions.append(value)
                    choices.partitions.sort()
                setter({value})
        after_change()

    def _read_jobs_filter_choice(
        self,
        win: curses.window,
        box_height: int,
        box_width: int,
        label: str,
        options: list[str],
    ) -> str | None:
        query = ""
        selected = 0
        footer = " type to filter  enter select/add  esc cancel "
        safe_curs_set(1)
        try:
            while True:
                matches = filter_choice_options(options, query)
                if matches:
                    selected = max(0, min(selected, len(matches) - 1))
                else:
                    selected = 0

                win.erase()
                win.border()
                self._addstr(
                    win,
                    0,
                    2,
                    f" enter {label} ",
                    self._pair(6) | curses.A_BOLD,
                )
                prompt = f"{label}: {query}"
                self._addstr(win, 1, 2, prompt[: box_width - 4], self._pair(MUTED_PAIR))
                body_height = max(1, box_height - 4)
                if matches:
                    first_match = max(0, selected - body_height + 1)
                    visible_matches = matches[first_match : first_match + body_height]
                    for row, value in enumerate(visible_matches, start=2):
                        match_index = first_match + row - 2
                        marker = ">" if match_index == selected else " "
                        attr = (
                            self._pair(ACTIVE_TAB_PAIR) | curses.A_BOLD
                            if match_index == selected
                            else self._pair(MUTED_PAIR)
                        )
                        self._addstr(
                            win,
                            row,
                            2,
                            f"{marker} {value}"[: box_width - 4],
                            attr,
                        )
                elif query.strip():
                    self._addstr(
                        win,
                        2,
                        2,
                        f'Add "{query.strip()}"'[: box_width - 4],
                        self._pair(ACTIVE_TAB_PAIR) | curses.A_BOLD,
                    )
                else:
                    self._addstr(
                        win,
                        2,
                        2,
                        "No choices.",
                        self._pair(MUTED_PAIR),
                    )
                self._addstr(win, box_height - 1, 2, footer[: box_width - 4], self._pair(5))
                win.refresh()

                key = win.getch()
                if key == 27:
                    return None
                if key in (ord("\n"), curses.KEY_ENTER):
                    if matches:
                        return matches[selected]
                    value = query.strip()
                    return value or None
                if key in (curses.KEY_DOWN, ord("j")) and matches:
                    selected = min(len(matches) - 1, selected + 1)
                elif key in (curses.KEY_UP, ord("k")) and matches:
                    selected = max(0, selected - 1)
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    query = query[:-1]
                    selected = 0
                elif key in (21,):
                    query = ""
                    selected = 0
                elif 32 <= key <= 126:
                    query += chr(key)
                    selected = 0
        finally:
            safe_curs_set(0)
