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
    return sum(
        int(match)
        for match in re.findall(r"(?:gres/)?gpu(?::[^,;:=()]+)*[:=](\d+)", value)
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


def _user_fairshare(fairshare: dict[tuple[str, str], float]) -> dict[str, float]:
    """Best (highest) fairshare across each user's account associations."""
    best: dict[str, float] = {}
    for (user, _account), score in fairshare.items():
        if user not in best or score > best[user]:
            best[user] = score
    return best


def _group_fairshare(fairshare: dict[tuple[str, str], float]) -> dict[str, float]:
    """Mean fairshare of each account's member users."""
    members: dict[str, list[float]] = {}
    for (_user, account), score in fairshare.items():
        members.setdefault(account, []).append(score)
    return {
        account: sum(scores) / len(scores)
        for account, scores in members.items()
        if scores
    }


def dominant_account(accounts: dict[str, int]) -> str:
    """The account with the most usage; ties broken alphabetically."""
    if not accounts:
        return ""
    return min(accounts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def reverse_lex(value: str) -> str:
    return "".join(chr(255 - ord(char)) for char in value)
