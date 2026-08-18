from __future__ import annotations

import datetime
import getpass
import os
import shlex
import subprocess
import time
from typing import Iterable

from .constants import (
    DEFAULT_SQUEUE_STATES,
    FILTER_CHOICES_FORMAT,
    JOB_EFFICIENCY_FORMAT,
    JOB_EFFICIENCY_WINDOW,
    JOB_EFFICIENCY_WINDOW_LABEL,
    NODE_JOBS_FORMAT,
    PRIORITY_QUEUE_LONG_FORMAT,
    SACCT_FORMAT,
    SQUEUE_FORMAT,
    SREPORT_USAGE_FORMAT,
    SREPORT_USAGE_REPORT,
    SSHARE_FAIRSHARE_FORMAT,
    SPRIO_FORMAT,
    SlurmError,
    USAGE_TRES,
    VACC_PARTITIONS,
)
from .primitives import (
    _user_fairshare,
    history_start,
    normalize_squeue_states,
    plural_label,
    usage_window_start,
)
from .format import format_node_jobs
from .models import (
    EfficiencySummary,
    GpfsMemberUsage,
    GpfsQuota,
    Job,
    JobFilterChoices,
    JobRecord,
    Node,
    PriorityFactors,
    PriorityQueueSnapshot,
    UsageEntry,
)
from .parsers import (
    parse_gpfs_quota,
    parse_gpfs_group_usage,
    parse_node_job_line,
    parse_priority_queue_long_line,
    parse_sacct_records,
    parse_scontrol_job_usage,
    parse_scontrol_nodes,
    parse_squeue_line,
    parse_sreport_usage,
    parse_sprio_line,
    parse_sshare_scores,
    record_from_job,
    summarize_job_efficiency,
)
from .aggregate import (
    active_job_keys,
    active_jobs_start,
    aggregate_user_usage,
    build_priority_queue_snapshot,
    format_user_usage,
    free_gpu_count,
    job_record_sort_key,
    records_for_active_jobs,
    stranded_gpu_count,
)


PRIORITY_FACTORS_CACHE_SECONDS = 60.0


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
        self._priority_factor_records: list[
            tuple[str, str, PriorityFactors]
        ] = []
        self._priority_factors_checked_at: float | None = None
        self._priority_factors_error = ""

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

    def fetch_priority_queue(self) -> PriorityQueueSnapshot:
        """Cluster-wide pending jobs plus requested resources and user factors.

        ``squeue --priority`` supplies the order Slurm considers for scheduling.
        Priority/Resources rows and transient ReqNodeNotAvail rows receive a
        rank; jobs held by dependencies, user/admin holds, or policy/configuration
        limits are kept visible but excluded from the list of blockers.
        DependencyNeverSatisfied rows are omitted. Multifactor details are
        queried only for this user and cached to avoid repeatedly sending the
        comparatively expensive sprio RPC to slurmctld.
        """
        output = self.runner.run(
            [
                "squeue",
                "--array",
                "--all",
                "-h",
                "-t",
                "PD",
                "--priority",
                "--sort=-p,i",
                "-O",
                PRIORITY_QUEUE_LONG_FORMAT,
            ],
            timeout=20.0,
        )
        jobs = [
            parse_priority_queue_long_line(line)
            for line in output.splitlines()
            if line.strip()
        ]
        if not any(job.user == self.user for job in jobs):
            return build_priority_queue_snapshot(
                self.user,
                jobs,
                factors_available=True,
            )

        factor_records, factors_available, factors_error = (
            self._fetch_priority_factor_records()
        )
        return build_priority_queue_snapshot(
            self.user,
            jobs,
            factor_records,
            factors_available=factors_available,
            factors_error=factors_error,
        )

    def _fetch_priority_factor_records(
        self,
    ) -> tuple[list[tuple[str, str, PriorityFactors]], bool, str]:
        now = time.monotonic()
        if (
            self._priority_factors_checked_at is not None
            and now - self._priority_factors_checked_at
            < PRIORITY_FACTORS_CACHE_SECONDS
        ):
            return (
                list(self._priority_factor_records),
                bool(self._priority_factor_records),
                self._priority_factors_error,
            )

        records: list[tuple[str, str, PriorityFactors]] = []
        error = ""
        try:
            output = self.runner.run(
                [
                    "sprio",
                    "-h",
                    "-u",
                    self.user,
                    "-o",
                    SPRIO_FORMAT,
                ],
                timeout=20.0,
            )
            records = [
                parse_sprio_line(line)
                for line in output.splitlines()
                if line.strip()
            ]
            if not records:
                error = "sprio returned no multifactor priority data"
        except SlurmError as exc:
            error = str(exc)

        self._priority_factor_records = records
        self._priority_factors_checked_at = now
        self._priority_factors_error = error
        return list(records), bool(records), error

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
        fairshare_by_user: dict[str, float] = {}
        try:
            fairshare, _level_fairshare = self.fetch_fairshare_data()
            fairshare_by_user = _user_fairshare(
                fairshare,
                self.fetch_default_accounts(),
            )
        except SlurmError:
            # Live allocations are still useful if accounting is unavailable.
            pass
        return format_user_usage(
            aggregate_user_usage(usage),
            free_gpus=free_gpus,
            allocated_gpus=allocated_gpus,
            fairshare_by_user=fairshare_by_user,
        )

    def fetch_usage_window(
        self,
        window: str,
        now: datetime.datetime | None = None,
    ) -> list[UsageEntry]:
        """Per (user, account) CPU-hour and GPU-hour usage over ``window``.

        The 30-day window can take several seconds on VACC, so callers run this
        off the UI thread.
        """
        start = usage_window_start(window, now=now)
        output = self.runner.run(
            [
                "sreport",
                "-n",
                "-P",
                "-t",
                "Hours",
                "cluster",
                SREPORT_USAGE_REPORT,
                f"Start={start}",
                "End=now",
                "-T",
                USAGE_TRES,
                f"format={SREPORT_USAGE_FORMAT}",
            ],
            timeout=180.0,
        )
        return parse_sreport_usage(output)

    def fetch_fairshare(self) -> dict[tuple[str, str], float]:
        """Current fairshare score keyed by (user, account) association."""
        return self.fetch_fairshare_data()[0]

    def fetch_fairshare_data(
        self,
    ) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
        """Native user FairShare and account LevelFS from one sshare query."""
        output = self.runner.run(
            [
                "sshare",
                "-a",
                "-h",
                "-P",
                "-l",
                "-o",
                SSHARE_FAIRSHARE_FORMAT,
            ],
            timeout=30.0,
        )
        return parse_sshare_scores(output)

    def fetch_default_accounts(self) -> dict[str, str]:
        """Default Slurm account keyed by user (best effort)."""
        try:
            output = self.runner.run(
                [
                    "sacctmgr",
                    "-n",
                    "-P",
                    "show",
                    "user",
                    "format=User,DefaultAccount",
                ],
                timeout=30.0,
            )
        except SlurmError:
            return {}
        accounts: dict[str, str] = {}
        for line in output.splitlines():
            parts = line.split("|", 1)
            if len(parts) < 2:
                continue
            user, account = (part.strip() for part in parts)
            if user and account:
                accounts[user] = account
        return accounts

    def fetch_user_compute_usage(
        self,
        window: str,
        now: datetime.datetime | None = None,
    ) -> tuple[int, int]:
        """(cpu_hours, gpu_hours) for the current user over ``window``.

        Restricting sreport with ``Users=`` keeps even the year window
        tolerable, but callers still run it off the UI thread.
        """
        start = usage_window_start(window, now=now)
        output = self.runner.run(
            [
                "sreport",
                "-n",
                "-P",
                "-t",
                "Hours",
                "cluster",
                SREPORT_USAGE_REPORT,
                f"Start={start}",
                "End=now",
                f"Users={self.user}",
                "-T",
                USAGE_TRES,
                f"format={SREPORT_USAGE_FORMAT}",
            ],
            timeout=300.0,
        )
        cpu = 0
        gpu = 0
        for entry in parse_sreport_usage(output):
            # Sum only this user's own rows (one per account), never the
            # account-total row sreport also emits with an empty login.
            if entry.login != self.user:
                continue
            cpu += entry.cpu_hours
            gpu += entry.gpu_hours
        return cpu, gpu

    def fetch_user_fairshare(self) -> dict[str, float]:
        """Current fairshare per account for the logged-in user only."""
        return {
            account: score
            for (user, account), score in self.fetch_fairshare().items()
            if user == self.user
        }

    def fetch_user_default_account(self) -> str:
        """The user's primary Slurm account (best effort; '' if unavailable)."""
        try:
            output = self.runner.run(
                [
                    "sacctmgr",
                    "-n",
                    "-P",
                    "show",
                    "user",
                    self.user,
                    "format=DefaultAccount",
                ],
                timeout=15.0,
            )
        except SlurmError:
            return ""
        for line in output.splitlines():
            account = line.strip()
            if account:
                return account
        return ""

    def fetch_gpfs_quota(self) -> GpfsQuota:
        """Group and personal GPFS storage usage from ``my_gpfs_quota``."""
        output = self.runner.run(["my_gpfs_quota"], timeout=30.0)
        return parse_gpfs_quota(output, self.user)

    def fetch_gpfs_group_usage(self) -> list[GpfsMemberUsage]:
        """Per-member GPFS usage for the user's primary group."""
        output = self.runner.run(["groupquota"], timeout=30.0)
        return parse_gpfs_group_usage(output)

    def fetch_job_efficiency(
        self,
        window: str = JOB_EFFICIENCY_WINDOW,
        window_label: str = JOB_EFFICIENCY_WINDOW_LABEL,
    ) -> EfficiencySummary:
        """Average CPU/memory/walltime efficiency of the user's recent jobs."""
        output = self.runner.run(
            [
                "sacct",
                "-n",
                "-P",
                "-u",
                self.user,
                "-S",
                window,
                "-E",
                "now",
                "-o",
                JOB_EFFICIENCY_FORMAT,
            ],
            timeout=60.0,
        )
        return summarize_job_efficiency(output, window_label)

    def fetch_job_efficiency_for(self, job_id: str) -> EfficiencySummary:
        """Efficiency for one job (or all tasks of one array), via ``sacct -j``."""
        output = self.runner.run(
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                str(job_id),
                "-o",
                JOB_EFFICIENCY_FORMAT,
            ],
            timeout=30.0,
        )
        return summarize_job_efficiency(output, str(job_id))
