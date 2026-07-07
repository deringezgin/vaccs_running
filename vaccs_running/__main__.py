from __future__ import annotations

import argparse
import sys

from . import __version__
from .slurm import SlurmClient, SlurmError, normalize_squeue_states
from .ui import VaccsRunningApp


def is_all_user_selector(value: str | None) -> bool:
    return bool(value and value.strip().lower() == "all")


def slurm_state_arg(value: str) -> str:
    try:
        return normalize_squeue_states(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaccs-running",
        description="Terminal UI for viewing VACC Slurm jobs and cluster status.",
    )
    parser.add_argument(
        "-u", "--user",
        help=(
            "Slurm username to inspect, or 'all' to inspect all users. "
            "Defaults to the current VACC shell user."
        ),
    )
    parser.add_argument(
        "-g", "--group",
        help="Slurm group/account to inspect in the jobs view.",
    )
    parser.add_argument(
        "-r", "--refresh",
        type=float,
        default=2.0,
        help="Auto-refresh interval in seconds. Use 0 to disable. Default: 2.0.",
    )
    parser.add_argument(
        "-s",
        "--state",
        "--states",
        dest="states",
        type=slurm_state_arg,
        default="all",
        help=(
            "Comma-separated Slurm states passed to squeue -t for the jobs view, "
            "for example PD or RUNNING,PENDING. Default: all."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def make_client(user: str | None, states: str, group: str | None = None) -> SlurmClient:
    client = SlurmClient(
        user=None if is_all_user_selector(user) else user,
        states=states,
    )
    group = group.strip() if group else None
    if group:
        users = set()
        if user and not is_all_user_selector(user):
            users.add(user.strip())
        client.set_job_principal_filters(users=users, groups={group})
    elif is_all_user_selector(user):
        client.set_job_user_filter(user)
    return client


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = make_client(args.user, args.states, args.group)

    try:
        VaccsRunningApp(
            client=client,
            refresh_seconds=max(0.0, args.refresh),
        ).run()
        return 0
    except KeyboardInterrupt:
        return 130
    except SlurmError as exc:
        print(f"vaccs-running: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
