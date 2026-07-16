from __future__ import annotations

import datetime
import re

from .constants import (
    DEFAULT_SQUEUE_STATES,
    HISTORY_WINDOWS,
    LEADERBOARD_WINDOW_DELTAS,
    SQUEUE_STATE_RE,
    _STORAGE_UNITS,
)


def parse_storage_size(text: str) -> float | None:
    """Bytes for a human size like '17.58T', '32K', '0'; None if unparseable."""
    match = re.fullmatch(
        r"([0-9]*\.?[0-9]+)\s*([KMGTP]?)B?",
        text.strip().upper(),
    )
    if not match:
        return None
    return float(match.group(1)) * _STORAGE_UNITS[match.group(2)]


def storage_percent(used: str, quota: str) -> float | None:
    """Percent of quota used, or None when either size is missing/zero."""
    used_bytes = parse_storage_size(used)
    quota_bytes = parse_storage_size(quota)
    if not used_bytes or not quota_bytes:
        return None
    return 100.0 * used_bytes / quota_bytes


def parse_duration_seconds(text: str) -> float | None:
    """Seconds for a sacct duration like '01:11:01', '12:34.567', '1-02:03:04'."""
    stripped = text.strip()
    if not stripped or stripped.upper() in {"N/A", "UNLIMITED", "INVALID"}:
        return None
    days = 0
    if "-" in stripped:
        day_part, stripped = stripped.split("-", 1)
        days = parse_int(day_part)
    pieces = stripped.split(":")
    try:
        if len(pieces) == 3:
            hours, minutes, seconds = pieces
        elif len(pieces) == 2:
            hours, (minutes, seconds) = "0", pieces
        elif len(pieces) == 1:
            hours, minutes, seconds = "0", "0", pieces[0]
        else:
            return None
        return days * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def parse_reqmem_bytes(text: str, cpus: int, nodes: int) -> float | None:
    """Total requested memory in bytes.

    Modern Slurm reports ReqMem as a plain total (e.g. '96G'). Older output used
    a per-CPU ('c') or per-node ('n') suffix, so those are scaled accordingly.
    """
    stripped = text.strip()
    if not stripped:
        return None
    suffix = ""
    if stripped[-1:].lower() in {"c", "n"}:
        suffix = stripped[-1:].lower()
        stripped = stripped[:-1]
    base = parse_storage_size(stripped)
    if base is None:
        return None
    if suffix == "c":
        return base * max(1, cpus)
    if suffix == "n":
        return base * max(1, nodes)
    return base


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _average_percent(values: list[float]) -> float | None:
    if not values:
        return None
    return 100.0 * sum(values) / len(values)


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def state_base(state: str) -> str:
    return state.upper().split(maxsplit=1)[0] or "UNKNOWN"


def normalize_squeue_states(states: str | None) -> str:
    if states is None:
        return DEFAULT_SQUEUE_STATES

    tokens: list[str] = []
    for token in re.split(r"[,\s]+", states.strip().strip("'\"")):
        token = token.strip().strip("'\"")
        if not token:
            continue
        if token.lower() == DEFAULT_SQUEUE_STATES:
            return DEFAULT_SQUEUE_STATES
        normalized = token.upper()
        if not SQUEUE_STATE_RE.fullmatch(normalized):
            raise ValueError(f"invalid Slurm state: {token!r}")
        tokens.append(normalized)

    return ",".join(tokens) if tokens else DEFAULT_SQUEUE_STATES


def is_slurm_timestamp(value: str) -> bool:
    if not value or value in {"N/A", "None", "Unknown", "(null)"}:
        return False
    return bool(re.search(r"\d", value))


def parse_user_id(value: str) -> str:
    if not value:
        return ""
    return value.split("(", 1)[0]


def parse_tres_value(tres: str, key: str) -> str:
    for part in tres.split(","):
        name, separator, value = part.partition("=")
        if separator and name == key:
            return value
    return ""


def history_start(window: str) -> str:
    return HISTORY_WINDOWS.get(window, HISTORY_WINDOWS["24h"])


def parse_key_values(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def parse_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def parse_optional_int(value: str) -> int | None:
    """Parse an integer field without turning missing Slurm data into zero."""
    stripped = value.strip()
    if not stripped or stripped.upper() in {
        "N/A",
        "NA",
        "NONE",
        "UNKNOWN",
        "(NULL)",
    }:
        return None
    try:
        return int(stripped)
    except ValueError:
        # Some Slurm versions/plugins render integral weighted factors with a
        # decimal suffix. Accept only values which are still exactly integral.
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        if not parsed.is_integer():
            return None
        return int(parsed)


def pending_reason_code(reason: str) -> str:
    """The stable Slurm reason token, without display punctuation/details."""
    stripped = reason.strip().strip("()")
    if not stripped:
        return "Unknown"
    return stripped.split(",", 1)[0].split(maxsplit=1)[0]


_PENDING_REASON_EXPLANATIONS = {
    "Priority": "Higher-priority jobs are ahead in this partition or reservation.",
    "Resources": "The requested resources are currently in use or unavailable.",
    "Dependency": "A job dependency has not completed yet.",
    "DependencyNeverSatisfied": "A dependency can no longer be satisfied.",
    "BeginTime": "The job's requested earliest start time has not arrived yet.",
    "JobHeldUser": "The job is held by its user or account coordinator.",
    "JobHeldAdmin": "The job is held by a cluster administrator.",
    "JobHoldMaxRequeue": "The job reached the cluster's maximum requeue count.",
    "JobArrayTaskLimit": "The job array's simultaneous-task limit is currently full.",
    "JobLaunchFailure": "Slurm could not launch the job on its assigned resources.",
    "InactiveLimit": "The job exceeded the cluster's allowed inactive time.",
    "InvalidAccount": "The requested Slurm account is not valid for this job.",
    "InvalidQOS": "The requested quality of service is not valid for this job.",
    "BadConstraints": "No available node can satisfy the requested constraints.",
    "PartitionDown": "The requested partition is down.",
    "PartitionInactive": "The requested partition is inactive.",
    "PartitionNodeLimit": "The node request is outside the partition's limits.",
    "PartitionTimeLimit": "The time request exceeds the partition's limit.",
    "NodeDown": "A node required by the job is down.",
    "ReqNodeNotAvail": "A specifically required node is unavailable.",
    "Reservation": "The job is waiting for its reservation to become active.",
    "ReservationDeleted": "The job's requested reservation was deleted.",
    "Licenses": "A requested software license is unavailable.",
    "Cleaning": "The job is being requeued while its previous run is cleaned up.",
    "Prolog": "The job's cluster prolog is still running.",
    "WaitingForScheduling": "The scheduler has not assigned a final reason yet.",
    "SchedDefer": "The scheduler deferred evaluation of this job.",
    "SystemFailure": "A cluster, network, or filesystem failure is blocking the job.",
}


def explain_pending_reason(reason: str) -> str:
    """Plain-language explanation for common Slurm pending reason codes."""
    code = pending_reason_code(reason)
    explanation = _PENDING_REASON_EXPLANATIONS.get(code)
    if explanation:
        return explanation
    if code.startswith("QOSGrp"):
        return "The job's QOS has reached an aggregate usage limit."
    if code.startswith("QOSMax"):
        return "The request exceeds a per-job or per-node QOS limit."
    if code.startswith("QOS"):
        return "A quality-of-service policy or usage limit is blocking the job."
    if code.startswith("AssocGrp"):
        return "The Slurm account association has reached an aggregate limit."
    if code.startswith("AssocMax"):
        return "The request exceeds a Slurm account association limit."
    if code.startswith("Max"):
        return "The request exceeds an account or QOS resource limit."
    if code == "Unknown":
        return "Slurm has not reported why this job is pending."
    return f"Slurm reports the pending reason {code}."


def plural_label(count: int, singular: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {singular}{suffix}"


def single_or_mixed_label(values: set[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def parse_gpu_count(value: str) -> int:
    # ReqTRES can contain both a canonical total (``gres/gpu=2``) and a
    # type-specific spelling (``gres/gpu:h200=2``). Prefer the canonical total
    # so the same GPUs are not counted twice. GRES strings with only typed
    # resources still need their types summed.
    generic = re.findall(
        r"(?:^|[,;])(?:gres/)?gpu[:=](\d+)(?=$|[,;])",
        value,
    )
    if generic:
        return sum(int(match) for match in generic)
    return sum(
        int(match)
        for match in re.findall(
            r"(?:^|[,;])(?:gres/)?gpu(?::[^,;:=()]+)+[:=](\d+)",
            value,
        )
    )


def parse_memory_mb(value: str) -> int | None:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "NONE", "(NULL)"}:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTkmgt]?)([cnCN]?)", stripped)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).upper()
    multiplier = {
        "": 1,
        "K": 1 / 1024,
        "M": 1,
        "G": 1024,
        "T": 1024 * 1024,
    }[unit]
    memory_mb = int(amount * multiplier)
    return memory_mb if memory_mb > 0 else None


def parse_elapsed_seconds(value: str) -> int:
    stripped = value.strip()
    if not stripped or stripped.upper() in {"N/A", "UNLIMITED"}:
        return -1
    days = 0
    time_part = stripped
    if "-" in stripped:
        day_part, time_part = stripped.split("-", 1)
        days = parse_int(day_part)
    pieces = time_part.split(":")
    if len(pieces) == 2:
        hours = 0
        minutes, seconds = pieces
    elif len(pieces) == 3:
        hours, minutes, seconds = pieces
    else:
        return -1
    return (
        days * 24 * 60 * 60
        + parse_int(hours) * 60 * 60
        + parse_int(minutes) * 60
        + parse_int(seconds)
    )


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def usage_window_start(
    window: str,
    now: datetime.datetime | None = None,
) -> str:
    """sreport ``Start=`` value for a leaderboard window, e.g. '2026-07-05T09:00:00'."""
    reference = now or datetime.datetime.now()
    delta = LEADERBOARD_WINDOW_DELTAS.get(
        window,
        LEADERBOARD_WINDOW_DELTAS["24h"],
    )
    return (reference - delta).strftime("%Y-%m-%dT%H:%M:%S")


def parse_fairshare_value(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def parse_level_fairshare_value(value: str) -> float | None:
    """Parse Slurm LevelFS, preserving its native infinity value."""
    stripped = value.strip()
    if stripped.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    return parse_fairshare_value(stripped)


def _user_fairshare(
    fairshare: dict[tuple[str, str], float],
    default_accounts: dict[str, str],
) -> dict[str, float]:
    """Fairshare for each user's default Slurm account association."""
    return {
        user: fairshare[(user, account)]
        for user, account in default_accounts.items()
        if account and (user, account) in fairshare
    }


def dominant_account(accounts: dict[str, int]) -> str:
    """The account with the most usage; ties broken alphabetically."""
    if not accounts:
        return ""
    return min(accounts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def reverse_lex(value: str) -> str:
    return "".join(chr(255 - ord(char)) for char in value)
