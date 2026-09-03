from __future__ import annotations

import math
from collections import defaultdict

from .models import (
    FairshareAssociation,
    FairshareForecast,
    FairshareForecastPoint,
)


AssociationKey = tuple[str, str]


def _project_usage(
    current: float,
    rate: float,
    horizon_seconds: float,
    half_life_seconds: float,
) -> float:
    """Continuously approximate Slurm's periodic usage decay and accrual."""
    decay_rate = math.log(2.0) / half_life_seconds
    decay = math.exp(-decay_rate * horizon_seconds)
    return current * decay + rate * (1.0 - decay) / decay_rate


def _account_children(
    associations: list[FairshareAssociation],
) -> dict[str, list[FairshareAssociation]]:
    children: dict[str, list[FairshareAssociation]] = defaultdict(list)
    for association in associations:
        if association.is_user:
            children[association.account].append(association)
        elif association.parent:
            children[association.parent].append(association)
    return dict(children)


def _projected_raw_usage(
    associations: list[FairshareAssociation],
    recent_usage: dict[AssociationKey, float],
    target_user: str,
    horizon_seconds: float,
    half_life_seconds: float,
    lookback_seconds: float,
    target_idle: bool,
) -> dict[AssociationKey, float]:
    children = _account_children(associations)
    account_nodes = {
        association.account: association
        for association in associations
        if not association.is_user
    }
    projected: dict[AssociationKey, float] = {}
    decay = math.exp(-math.log(2.0) * horizon_seconds / half_life_seconds)

    for association in associations:
        if not association.is_user:
            continue
        rate = recent_usage.get(association.key, 0.0) / lookback_seconds
        if target_idle and association.user == target_user:
            rate = 0.0
        projected[association.key] = _project_usage(
            association.raw_usage,
            rate,
            horizon_seconds,
            half_life_seconds,
        )

    visiting: set[str] = set()

    def project_account(account: str) -> float:
        key = ("", account)
        if key in projected:
            return projected[key]
        if account in visiting:
            return 0.0
        visiting.add(account)
        node = account_nodes.get(account)
        if node is None:
            visiting.remove(account)
            return 0.0
        current_children = 0.0
        future_children = 0.0
        for child in children.get(account, []):
            current_children += child.raw_usage
            if child.is_user:
                future_children += projected.get(child.key, 0.0)
            else:
                future_children += project_account(child.account)
        # Preserve unattributed/deleted-association usage so the projection is
        # calibrated to the account's live RawUsage at horizon zero.
        residual = max(0.0, node.raw_usage - current_children)
        value = future_children + residual * decay
        projected[key] = value
        visiting.remove(account)
        return value

    for account in account_nodes:
        project_account(account)
    return projected


def _fair_tree_scores(
    associations: list[FairshareAssociation],
    projected_usage: dict[AssociationKey, float],
) -> dict[AssociationKey, float]:
    """Reproduce Fair Tree's LevelFS sort, depth-first rank, and tie handling."""
    children = _account_children(associations)
    roots = [
        association.account
        for association in associations
        if not association.is_user and not association.parent
    ]
    user_count = sum(association.is_user for association in associations)
    if not roots or not user_count:
        return {}

    def level_fs(child: FairshareAssociation) -> float:
        siblings = children.get(child.parent if not child.is_user else child.account, [])
        total_shares = sum(sibling.shares for sibling in siblings)
        parent_name = child.account if child.is_user else child.parent
        parent_usage = projected_usage.get(("", parent_name), 0.0)
        usage = projected_usage.get(child.key, 0.0)
        if child.shares <= 0.0 or total_shares <= 0.0:
            return 0.0
        if usage <= 0.0:
            return math.inf
        if parent_usage <= 0.0:
            return math.inf
        return (child.shares / total_shares) / (usage / parent_usage)

    def rank_children(parent_accounts: list[str]) -> list[list[AssociationKey]]:
        merged: list[FairshareAssociation] = []
        for account in parent_accounts:
            merged.extend(children.get(account, []))
        ordered = sorted(merged, key=level_fs, reverse=True)
        buckets: list[list[AssociationKey]] = []
        index = 0
        while index < len(ordered):
            value = level_fs(ordered[index])
            tied: list[FairshareAssociation] = []
            while index < len(ordered) and level_fs(ordered[index]) == value:
                tied.append(ordered[index])
                index += 1
            users = [association.key for association in tied if association.is_user]
            accounts = [
                association.account for association in tied if not association.is_user
            ]
            nested = rank_children(accounts) if accounts else []
            if users and nested:
                buckets.append(users + nested[0])
                buckets.extend(nested[1:])
            elif users:
                buckets.append(users)
            else:
                buckets.extend(nested)
        return buckets

    # Multiple roots are treated as a merged top-level list; VACC normally has
    # exactly one (``root``), but this keeps malformed snapshots non-fatal.
    buckets = rank_children(roots)
    scores: dict[AssociationKey, float] = {}
    processed = 0
    for bucket in buckets:
        score = (user_count - processed) / user_count
        for key in bucket:
            scores[key] = score
        processed += len(bucket)
    return scores


def build_fairshare_forecast(
    associations: list[FairshareAssociation],
    recent_usage: dict[AssociationKey, float],
    target_user: str,
    target_account: str,
    half_life_seconds: float,
    lookback_seconds: float,
    horizons: tuple[int, ...],
) -> FairshareForecast:
    """Project Fair Tree rank under idle and recent-rate continuation scenarios."""
    if half_life_seconds <= 0.0 or lookback_seconds <= 0.0:
        raise ValueError("fairshare forecast requires positive time windows")
    target_key = (target_user, target_account)
    target = next(
        (association for association in associations if association.key == target_key),
        None,
    )
    if target is None or target.fairshare is None:
        raise ValueError("current user fairshare association was not found")

    baseline_usage = _projected_raw_usage(
        associations,
        recent_usage,
        target_user,
        0.0,
        half_life_seconds,
        lookback_seconds,
        target_idle=False,
    )
    baseline = _fair_tree_scores(associations, baseline_usage).get(target_key)
    user_count = sum(association.is_user for association in associations)
    tolerance = 1e-6 + (0.49 / user_count if user_count else 0.0)
    if baseline is None or abs(baseline - target.fairshare) > tolerance:
        raise ValueError("live Fair Tree rank could not be reproduced")

    points: list[FairshareForecastPoint] = []
    for days in horizons:
        horizon_seconds = float(days * 86400)
        idle_usage = _projected_raw_usage(
            associations,
            recent_usage,
            target_user,
            horizon_seconds,
            half_life_seconds,
            lookback_seconds,
            target_idle=True,
        )
        pace_usage = _projected_raw_usage(
            associations,
            recent_usage,
            target_user,
            horizon_seconds,
            half_life_seconds,
            lookback_seconds,
            target_idle=False,
        )
        idle_score = _fair_tree_scores(associations, idle_usage).get(target_key)
        pace_score = _fair_tree_scores(associations, pace_usage).get(target_key)
        if idle_score is None or pace_score is None:
            raise ValueError("projected Fair Tree rank was unavailable")
        points.append(
            FairshareForecastPoint(
                days=days,
                idle=idle_score,
                recent_pace=pace_score,
            )
        )

    return FairshareForecast(
        account=target_account,
        current=target.fairshare,
        half_life_days=half_life_seconds / 86400.0,
        lookback_days=lookback_seconds / 86400.0,
        points=tuple(points),
    )
