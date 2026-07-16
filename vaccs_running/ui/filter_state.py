from __future__ import annotations

from .constants import JOB_STATE_CODES, PRIORITY_GPU_PARTITIONS
from ..slurm import plural_label


class JobFilterStateMixin:
    def _squeue_state_filter(self) -> str:
        return str(getattr(self.client, "squeue_states", "all") or "all")

    def _squeue_state_filter_active(self) -> bool:
        active = getattr(self.client, "state_filter_active", None)
        if active is not None:
            return bool(active)
        return self._squeue_state_filter().lower() != "all"

    def _job_user_filter_active(self) -> bool:
        active = getattr(self.client, "job_user_filter_active", None)
        return bool(active) if active is not None else False

    def _job_partition_filter_active(self) -> bool:
        active = getattr(self.client, "job_partition_filter_active", None)
        if active is not None:
            return bool(active)
        return bool(self._selected_job_partitions())

    def _jobs_filter_active(self) -> bool:
        return (
            self._squeue_state_filter_active()
            or self._job_user_filter_active()
            or self._job_partition_filter_active()
        )

    def _show_job_principal_columns(self) -> bool:
        return (
            self._job_all_principals()
            or len(self._selected_job_users()) > 1
            or bool(self._selected_job_groups())
        )

    def _clear_job_filters(self) -> None:
        clear = getattr(self.client, "clear_job_filters", None)
        if clear:
            clear()
        else:
            self.client.squeue_states = "all"
        self._refresh_jobs_after_filter_change()

    def _refresh_jobs_after_filter_change(self) -> None:
        self._refresh_jobs()
        self.state.selected = 0
        self.state.scroll = 0
        self._clamp_selection()

    def _set_job_state_filter(self, states: str) -> None:
        setter = getattr(self.client, "set_job_state_filter", None)
        if setter:
            setter(states)
        else:
            self.client.squeue_states = states

    def _set_job_state_codes(self, states: set[str]) -> None:
        ordered = [state for state in JOB_STATE_CODES if state in states]
        self._set_job_state_filter(",".join(ordered) if ordered else "all")

    def _set_job_principal_filters(
        self,
        users: set[str],
        groups: set[str],
        *,
        all_principals: bool = False,
    ) -> None:
        setter = getattr(self.client, "set_job_principal_filters", None)
        if setter:
            setter(users, groups, all_principals=all_principals)

    def _set_job_partition_filters(self, partitions: set[str]) -> None:
        setter = getattr(self.client, "set_job_partition_filters", None)
        if setter:
            setter(partitions)
        else:
            setattr(self.client, "job_partitions", set(partitions))

    def _selected_job_state_codes(self) -> set[str]:
        states = self._squeue_state_filter()
        if states.lower() == "all":
            return set()
        return {
            state.strip().upper()
            for state in states.split(",")
            if state.strip()
        }

    def _selected_job_users(self) -> set[str]:
        users = set(getattr(self.client, "job_users", set()) or set())
        groups = self._selected_job_groups()
        all_principals = self._job_all_principals()
        default_user = str(getattr(self.client, "user", ""))
        if not all_principals and not groups and users == {default_user}:
            return set()
        return users

    def _selected_job_groups(self) -> set[str]:
        return set(getattr(self.client, "job_groups", set()) or set())

    def _selected_job_partitions(self) -> set[str]:
        return set(getattr(self.client, "job_partitions", set()) or set())

    def _job_all_principals(self) -> bool:
        return bool(getattr(self.client, "job_all_principals", False))

    def _job_user_summary(self) -> str:
        if self._job_all_principals():
            return "all users"
        users = self._selected_job_users()
        if not users:
            return "me"
        return plural_label(len(users), "user")

    def _job_group_summary(self) -> str:
        groups = self._selected_job_groups()
        if not groups:
            return "none"
        return plural_label(len(groups), "group")

    def _job_partition_summary(self) -> str:
        partitions = self._selected_job_partitions()
        if not partitions:
            return "all"
        if len(partitions) == 1:
            return next(iter(partitions))
        return plural_label(len(partitions), "partition")

    def _priority_partition_filter_active(self) -> bool:
        return bool(self.state.priority_partitions)

    def _selected_priority_partitions(self) -> set[str]:
        return set(self.state.priority_partitions)

    def _set_priority_partition_filters(self, partitions: set[str]) -> None:
        self.state.priority_partitions = set(partitions)

    def _priority_gpu_filter_active(self) -> bool:
        return self._selected_priority_partitions() == set(PRIORITY_GPU_PARTITIONS)

    def _toggle_priority_gpu_filter(self) -> None:
        enabled = not self._priority_gpu_filter_active()
        self._set_priority_partition_filters(
            set(PRIORITY_GPU_PARTITIONS) if enabled else set()
        )
        self._reset_priority_after_filter_change()
        state = "on" if enabled else "off"
        self.state.message = f"GPU partition filter {state}"

    def _priority_partition_summary(self) -> str:
        partitions = self._selected_priority_partitions()
        if not partitions:
            return "all"
        if len(partitions) == 1:
            return next(iter(partitions))
        return plural_label(len(partitions), "partition")

    def _clear_priority_filters(self) -> None:
        self._set_priority_partition_filters(set())
        self._reset_priority_after_filter_change()

    def _reset_priority_after_filter_change(self) -> None:
        self.state.selected = 0
        self.state.scroll = 0
        self._clamp_selection()
