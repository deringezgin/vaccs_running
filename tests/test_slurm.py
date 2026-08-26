import unittest

import datetime

from vaccs_running.slurm import (
    FILTER_CHOICES_FORMAT,
    SACCT_FORMAT,
    NODE_JOBS_FORMAT,
    PRIORITY_QUEUE_LONG_FORMAT,
    SQUEUE_FORMAT,
    SREPORT_USAGE_FORMAT,
    SPRIO_FORMAT,
    SSHARE_FAIRSHARE_FORMAT,
    USAGE_TRES,
    USER_INFO_WINDOWS,
    EfficiencySummary,
    GpfsMemberUsage,
    GpfsQuota,
    JOB_EFFICIENCY_FORMAT,
    LeaderboardRow,
    PriorityFactors,
    SlurmClient,
    SlurmError,
    UsageEntry,
    format_job_efficiency,
    human_bytes,
    human_duration,
    parse_duration_seconds,
    parse_gpfs_quota,
    parse_gpfs_group_usage,
    parse_reqmem_bytes,
    parse_storage_size,
    storage_percent,
    summarize_job_efficiency,
    VACC_PARTITIONS,
    aggregate_user_usage,
    build_group_leaderboard,
    build_priority_queue_snapshot,
    build_user_leaderboard,
    format_fairshare,
    format_node_jobs,
    format_user_usage,
    free_gpu_count,
    human_hours,
    stranded_gpu_count,
    group_job_records,
    group_jobs,
    history_start,
    parse_sacct_line,
    parse_fairshare_value,
    parse_level_fairshare_value,
    parse_node_job_line,
    parse_priority_queue_line,
    parse_priority_queue_long_line,
    parse_priority_tres,
    parse_scontrol_nodes,
    parse_scontrol_job_usage,
    parse_sreport_usage,
    parse_sprio_line,
    parse_sshare_fairshare,
    parse_sshare_scores,
    parse_squeue_line,
    parse_elapsed_seconds,
    parse_gpu_count,
    parse_memory_mb,
    parse_tres_value,
    normalize_squeue_states,
    explain_pending_reason,
    sort_leaderboard,
    summarize_jobs,
    usage_window_start,
)


class FakeRunner:
    def __init__(self, output=""):
        self.calls = []
        if isinstance(output, (list, tuple)):
            self.outputs = list(output)
            self.output = ""
        else:
            self.outputs = None
            self.output = output

    def run(self, args, timeout=12.0):
        self.calls.append((args, timeout))
        if self.outputs is not None:
            return self.outputs.pop(0) if self.outputs else ""
        return self.output


def priority_queue_line(
    job_id,
    user,
    partition,
    reason,
    priority,
    name="job",
    reservation="N/A",
):
    return (
        f"{job_id}|{user}|{partition}|PENDING|{reason}|{priority}|pi-{user}|"
        f"normal|2026-07-15T09:00:00|N/A|{reservation}|1|4|N/A|02:00:00|{name}"
    )


def priority_queue_long_line(
    job_id,
    user,
    partition,
    reason,
    priority,
    name="job",
    *,
    array_task="N/A",
    requested_tres="cpu=4,mem=16G,node=1",
    cpus="4",
    limit="02:00:00",
    reservation="N/A",
):
    return (
        f"{job_id}|{array_task}|{user}|{partition}|PENDING|{reason}|{priority}|"
        f"pi-{user}|normal|2026-07-15T09:00:00|N/A|{reservation}|1|{cpus}|"
        f"{requested_tres}|{limit}|{name}|"
    )


class PriorityQueueTests(unittest.TestCase):
    def test_priority_parsers_preserve_names_missing_factors_and_array_tres(self):
        job = parse_priority_queue_line(
            priority_queue_line("100_2", "me", "nvgpu", "Priority", "700", "a|b")
        )
        self.assertEqual(job.name, "a|b")
        self.assertEqual(job.priority, 700)
        self.assertEqual(job.reason_code, "Priority")
        self.assertTrue(job.is_rankable)

        factor_id, partition, factors = parse_sprio_line(
            "100_[1-3]|me|pi-me|nvgpu|700||100||500"
        )
        self.assertEqual((factor_id, partition), ("100_[1-3]", "nvgpu"))
        self.assertEqual(factors.age, 100)
        self.assertEqual(factors.fairshare, 500)
        self.assertIsNone(factors.site)
        self.assertIsNone(factors.job_size)

        self.assertEqual(
            parse_priority_tres(
                "100|2|cpu=4,mem=16G,node=1,gres/gpu=1|\n"
                "200|N/A|cpu=1,mem=2G,node=1|\n"
            ),
            {
                "100_2": "cpu=4,mem=16G,node=1,gres/gpu=1",
                "200": "cpu=1,mem=2G,node=1",
            },
        )

    def test_priority_long_parser_reconstructs_array_id_and_preserves_name_pipes(self):
        job = parse_priority_queue_long_line(
            priority_queue_long_line(
                "100",
                "me",
                "nvgpu",
                "Resources",
                700,
                "training|restart|3",
                array_task="2",
                requested_tres=(
                    "cpu=8,mem=64G,node=1,gres/gpu=2,gres/gpu:h100=2"
                ),
                cpus="8",
                limit="1-12:00:00",
            )
        )

        self.assertEqual(job.job_id, "100_2")
        self.assertEqual(job.array_parent, "100")
        self.assertEqual(job.name, "training|restart|3")
        self.assertEqual(job.requested_cpu_count, 8)
        self.assertEqual(job.requested_memory_mb, 64 * 1024)
        self.assertEqual(job.requested_gpu_count, 2)
        self.assertEqual(job.requested_walltime_seconds, 36 * 60 * 60)

    def test_snapshot_ranks_transient_node_unavailability_and_omits_dead_dependencies(self):
        snapshot = build_priority_queue_snapshot(
            "me",
            [
                parse_priority_queue_line(
                    priority_queue_line(
                        "900",
                        "blocked",
                        "nvgpu",
                        "DependencyNeverSatisfied",
                        999,
                    )
                ),
                parse_priority_queue_line(
                    priority_queue_line(
                        "800",
                        "me",
                        "nvgpu",
                        "ReqNodeNotAvail, Reserved for maintenance",
                        800,
                    )
                ),
                parse_priority_queue_line(
                    priority_queue_line("700", "alice", "nvgpu", "Resources", 700)
                ),
            ],
        )

        self.assertEqual(
            [job.job_id for job in snapshot.pending_jobs],
            ["800", "700"],
        )
        self.assertEqual(
            [entry.priority_rank_text for entry in snapshot.all_entries],
            ["1 of 2", "2 of 2"],
        )
        self.assertEqual(snapshot.my_jobs[0].job.reason_code, "ReqNodeNotAvail")

    def test_snapshot_ranks_only_schedulable_jobs_in_same_partition(self):
        jobs = [
            parse_priority_queue_line(
                priority_queue_line("500", "dave", "nvgpu", "JobHeldUser", 1200)
            ),
            parse_priority_queue_line(
                priority_queue_line("200", "bob", "nvgpu", "Priority", 900)
            ),
            parse_priority_queue_line(
                priority_queue_line("101", "me", "nvgpu", "Dependency", 850)
            ),
            parse_priority_queue_line(
                priority_queue_line("300", "alice", "nvgpu", "Resources", 800)
            ),
            parse_priority_queue_line(
                priority_queue_line("400", "carol", "general", "Priority", 950)
            ),
            parse_priority_queue_line(
                priority_queue_line("100", "me", "nvgpu", "Priority", 700)
            ),
        ]
        snapshot = build_priority_queue_snapshot(
            "me",
            jobs,
            [("100", "nvgpu", PriorityFactors(priority=700, age=100, fairshare=500))],
            factors_available=True,
        )

        ranked, held = snapshot.my_jobs
        self.assertEqual(ranked.priority_rank_text, "3 of 3")
        self.assertEqual(ranked.earlier_count, 2)
        self.assertEqual(ranked.earlier_user_count, 2)
        self.assertEqual(ranked.ahead, ())
        self.assertEqual(ranked.job.factors.fairshare, 500)
        self.assertIsNone(held.priority_rank)
        self.assertIn("Dependency", held.rank_note)
        self.assertEqual(len(snapshot.grouped_entries), 6)
        self.assertEqual(
            [entry.job.user for entry in snapshot.grouped_entries],
            ["bob", "alice", "me", "dave", "me", "carol"],
        )

    def test_snapshot_compacts_array_tasks_into_partition_rank_bands(self):
        jobs = [
            parse_priority_queue_line(
                priority_queue_line("900", "alice", "nvgpu", "Priority", 900)
            ),
            *[
                parse_priority_queue_line(
                    priority_queue_line(
                        f"100_{task}", "me", "nvgpu", "Resources", 700
                    )
                )
                for task in (2, 3, 4)
            ],
            *[
                parse_priority_queue_line(
                    priority_queue_line(
                        f"100_{task}", "me", "gpu-preempt", "Resources", 600
                    )
                )
                for task in (2, 3, 4)
            ],
        ]

        snapshot = build_priority_queue_snapshot("me", jobs)

        self.assertEqual(len(snapshot.pending_jobs), 7)
        self.assertEqual(len(snapshot.all_entries), 7)
        self.assertEqual(len(snapshot.grouped_entries), 3)
        self.assertEqual(len(snapshot.my_jobs), 2)
        nvgpu, preempt = snapshot.my_jobs
        self.assertEqual(nvgpu.display_job_id, "100_[2-4]")
        self.assertEqual(nvgpu.task_count, 3)
        self.assertEqual(nvgpu.priority_rank_text, "2-4 of 4")
        self.assertEqual(nvgpu.earlier_count, 1)
        self.assertEqual(nvgpu.earlier_user_count, 1)
        self.assertEqual(nvgpu.ahead, ())
        self.assertEqual(preempt.priority_rank_text, "1-3 of 3")
        self.assertEqual(snapshot.all_entries[1].earlier_count, 1)
        self.assertEqual(snapshot.all_entries[2].earlier_count, 2)
        self.assertEqual(snapshot.all_entries[2].earlier_user_count, 2)
        self.assertEqual(snapshot.all_entries[2].ahead, ())
        self.assertTrue(all(not entry.ahead for entry in snapshot.grouped_entries))

    def test_snapshot_packs_every_users_arrays_but_keeps_raw_extended_rows(self):
        jobs = [
            *[
                parse_priority_queue_line(
                    priority_queue_line(
                        f"900_{task}", "alice", "nvgpu", "Priority", 900
                    )
                )
                for task in (1, 2)
            ],
            parse_priority_queue_line(
                priority_queue_line("800", "bob", "nvgpu", "Priority", 800)
            ),
            *[
                parse_priority_queue_line(
                    priority_queue_line(
                        f"700_{task}", "me", "nvgpu", "Resources", 700
                    )
                )
                for task in (1, 2)
            ],
        ]

        snapshot = build_priority_queue_snapshot("me", jobs)

        self.assertEqual(len(snapshot.all_entries), 5)
        self.assertEqual(len(snapshot.grouped_entries), 3)
        alice, bob, mine = snapshot.grouped_entries
        self.assertEqual(alice.display_job_id, "900_[1-2]")
        self.assertEqual(alice.priority_rank_text, "1-2 of 5")
        self.assertEqual((alice.earlier_count, alice.earlier_user_count), (0, 0))
        self.assertEqual(bob.priority_rank_text, "3 of 5")
        self.assertEqual((bob.earlier_count, bob.earlier_user_count), (2, 1))
        self.assertEqual(mine.display_job_id, "700_[1-2]")
        self.assertEqual(mine.priority_rank_text, "4-5 of 5")
        self.assertEqual((mine.earlier_count, mine.earlier_user_count), (3, 2))
        self.assertEqual(snapshot.my_jobs, (mine,))
        self.assertTrue(all(not entry.ahead for entry in snapshot.grouped_entries))

    def test_interleaved_tasks_from_one_array_do_not_create_overlapping_rank_runs(self):
        snapshot = build_priority_queue_snapshot(
            "me",
            [
                parse_priority_queue_line(
                    priority_queue_line("100_1", "me", "nvgpu", "Priority", 900)
                ),
                parse_priority_queue_line(
                    priority_queue_line("200", "bob", "nvgpu", "Priority", 850)
                ),
                parse_priority_queue_line(
                    priority_queue_line("100_2", "me", "nvgpu", "Resources", 800)
                ),
            ],
        )

        self.assertEqual(len(snapshot.grouped_entries), 3)
        first_task, bob, second_task = snapshot.grouped_entries
        self.assertEqual(first_task.task_job_ids, ("100_1",))
        self.assertEqual(first_task.priority_rank_text, "1 of 3")
        self.assertEqual(bob.task_job_ids, ("200",))
        self.assertEqual(bob.priority_rank_text, "2 of 3")
        self.assertEqual(second_task.task_job_ids, ("100_2",))
        self.assertEqual(second_task.priority_rank_text, "3 of 3")
        self.assertEqual(
            set(first_task.task_job_ids)
            & set(bob.task_job_ids + second_task.task_job_ids),
            set(),
        )

    def test_snapshot_resorts_expanded_rows_by_priority_then_numeric_job_id(self):
        snapshot = build_priority_queue_snapshot(
            "me",
            [
                parse_priority_queue_line(
                    priority_queue_line("100_10", "me", "nvgpu", "Priority", 501)
                ),
                parse_priority_queue_line(
                    priority_queue_line("100_20", "me", "nvgpu", "Priority", 444)
                ),
                parse_priority_queue_line(
                    priority_queue_line("90_2", "ccoil", "nvgpu", "Priority", 501)
                ),
                parse_priority_queue_line(
                    priority_queue_line("100_2", "me", "nvgpu", "Priority", 501)
                ),
                parse_priority_queue_line(
                    priority_queue_line("100_3", "me", "nvgpu", "Priority", 503)
                ),
                parse_priority_queue_line(
                    priority_queue_line("90_1", "ccoil", "nvgpu", "Priority", 503)
                ),
                parse_priority_queue_line(
                    priority_queue_line("100_1", "me", "nvgpu", "Resources", 517)
                ),
            ],
        )

        self.assertEqual(
            [
                (
                    entry.job.user,
                    entry.job.job_id,
                    entry.job.priority,
                    entry.priority_rank,
                    entry.earlier_count,
                )
                for entry in snapshot.all_entries
            ],
            [
                ("me", "100_1", 517, 1, 0),
                ("ccoil", "90_1", 503, 2, 1),
                ("me", "100_3", 503, 3, 2),
                ("ccoil", "90_2", 501, 4, 3),
                ("me", "100_2", 501, 5, 4),
                ("me", "100_10", 501, 6, 5),
                ("me", "100_20", 444, 7, 6),
            ],
        )
        priorities = [entry.job.priority for entry in snapshot.all_entries]
        self.assertEqual(priorities, sorted(priorities, reverse=True))
        self.assertEqual(
            [entry.job.user for entry in snapshot.grouped_entries],
            ["me", "ccoil", "me", "ccoil", "me"],
        )
        final_run = snapshot.grouped_entries[-1]
        self.assertEqual(final_run.priority_rank_text, "5-7 of 7")
        self.assertEqual(final_run.display_job_id, "100_[2,10,20]")
        self.assertEqual(final_run.display_priority, "501-444")

    def test_snapshot_splits_one_array_when_pending_reasons_differ(self):
        snapshot = build_priority_queue_snapshot(
            "me",
            [
                parse_priority_queue_line(
                    priority_queue_line("100_1", "me", "nvgpu", "Priority", 700)
                ),
                parse_priority_queue_line(
                    priority_queue_line("100_2", "me", "nvgpu", "Dependency", 700)
                ),
            ],
        )

        self.assertEqual(len(snapshot.grouped_entries), 2)
        ranked, dependency = snapshot.grouped_entries
        self.assertEqual(ranked.task_job_ids, ("100_1",))
        self.assertEqual(ranked.priority_rank_text, "1 of 1")
        self.assertEqual(dependency.task_job_ids, ("100_2",))
        self.assertIsNone(dependency.priority_rank)

    def test_snapshot_packs_each_users_unranked_entries_without_assigning_ranks(self):
        snapshot = build_priority_queue_snapshot(
            "me",
            [
                parse_priority_queue_line(
                    priority_queue_line("900", "alice", "nvgpu", "Priority", 900)
                ),
                parse_priority_queue_line(
                    priority_queue_line(
                        "4877755_0",
                        "achawla1",
                        "nvgpu",
                        "Dependency",
                        2243,
                    )
                ),
                parse_priority_queue_line(
                    priority_queue_line(
                        "800", "bob", "nvgpu", "PartitionTimeLimit", 1200
                    )
                ),
                parse_priority_queue_line(
                    priority_queue_line(
                        "4877755_1",
                        "achawla1",
                        "nvgpu",
                        "Dependency",
                        2243,
                    )
                ),
                parse_priority_queue_line(
                    priority_queue_line(
                        "4880000",
                        "achawla1",
                        "nvgpu",
                        "DependencyNeverSatisfied",
                        2200,
                    )
                ),
            ],
        )

        self.assertEqual(len(snapshot.pending_jobs), 4)
        self.assertEqual(len(snapshot.all_entries), 4)
        self.assertEqual(len(snapshot.grouped_entries), 3)
        ranked, achawla, bob = snapshot.grouped_entries
        self.assertEqual(ranked.job.user, "alice")
        self.assertEqual(achawla.job.user, "achawla1")
        self.assertEqual(achawla.task_count, 2)
        self.assertEqual(achawla.job_count, 1)
        self.assertEqual(achawla.display_job_id, "4877755_[0-1]")
        self.assertEqual(
            achawla.task_job_ids,
            ("4877755_0", "4877755_1"),
        )
        self.assertEqual(
            achawla.group_reason_codes,
            ("Dependency",),
        )
        self.assertEqual(achawla.display_reason, "Dependency")
        self.assertIsNone(achawla.priority_rank)
        self.assertIsNone(achawla.rank_end)
        self.assertIn("unranked pending entries", achawla.rank_note)
        self.assertNotIn("rank slots", achawla.rank_note)
        self.assertEqual(bob.job.user, "bob")
        self.assertNotIn(
            "4880000",
            {job.job_id for job in snapshot.pending_jobs},
        )

    def test_snapshot_packs_consecutive_ranked_jobs_from_the_same_user(self):
        jobs = [
            parse_priority_queue_line(
                priority_queue_line("1000", "alice", "nvgpu", "Priority", 1000)
            ),
            *[
                parse_priority_queue_line(
                    priority_queue_line(
                        str(900 + index),
                        "fsabokro",
                        "nvgpu",
                        "Priority",
                        900 - index,
                        f"job-{index}",
                    )
                )
                for index in range(10)
            ],
            parse_priority_queue_line(
                priority_queue_line("800", "bob", "nvgpu", "Priority", 700)
            ),
        ]

        snapshot = build_priority_queue_snapshot("me", jobs)

        self.assertEqual(len(snapshot.all_entries), 12)
        self.assertEqual(len(snapshot.grouped_entries), 3)
        alice, fsabokro, bob = snapshot.grouped_entries
        self.assertEqual(alice.job.user, "alice")
        self.assertEqual(fsabokro.job.user, "fsabokro")
        self.assertEqual(fsabokro.display_job_id, "10 jobs")
        self.assertEqual(fsabokro.job_count, 10)
        self.assertEqual(fsabokro.source_group_count, 10)
        self.assertEqual(fsabokro.task_count, 10)
        self.assertEqual(fsabokro.priority_rank_text, "2-11 of 12")
        self.assertEqual(fsabokro.display_priority, "900-891")
        self.assertEqual(fsabokro.display_name, "mixed jobs")
        self.assertEqual((fsabokro.earlier_count, fsabokro.earlier_user_count), (1, 1))
        self.assertIn("Press e", fsabokro.rank_note)
        self.assertEqual(bob.job.user, "bob")
        self.assertTrue(all(not entry.ahead for entry in snapshot.grouped_entries))

    def test_same_user_jobs_separated_by_another_user_form_separate_runs(self):
        jobs = [
            *[
                parse_priority_queue_line(
                    priority_queue_line(
                        str(900 + index), "fsabokro", "nvgpu", "Priority", 900 - index
                    )
                )
                for index in range(3)
            ],
            parse_priority_queue_line(
                priority_queue_line("800", "bob", "nvgpu", "Priority", 800)
            ),
            *[
                parse_priority_queue_line(
                    priority_queue_line(
                        str(700 + index), "fsabokro", "nvgpu", "Resources", 700 - index
                    )
                )
                for index in range(2)
            ],
        ]

        snapshot = build_priority_queue_snapshot("me", jobs)

        self.assertEqual(len(snapshot.grouped_entries), 3)
        first_run, bob, second_run = snapshot.grouped_entries
        self.assertEqual((first_run.job.user, first_run.job_count), ("fsabokro", 3))
        self.assertEqual((bob.job.user, bob.job_count), ("bob", 1))
        self.assertEqual((second_run.job.user, second_run.job_count), ("fsabokro", 2))
        self.assertEqual(second_run.display_reason, "Resources")

    def test_same_user_consecutive_partition_ranks_merge_despite_other_scope_rows(self):
        snapshot = build_priority_queue_snapshot(
            "me",
            [
                parse_priority_queue_line(
                    priority_queue_line("900", "fsabokro", "nvgpu", "Priority", 900)
                ),
                parse_priority_queue_line(
                    priority_queue_line(
                        "850", "alice", "gpu-preempt", "Priority", 850
                    )
                ),
                parse_priority_queue_line(
                    priority_queue_line("800", "fsabokro", "nvgpu", "Resources", 800)
                ),
            ],
        )

        self.assertEqual(len(snapshot.grouped_entries), 2)
        nvgpu_run, preempt = snapshot.grouped_entries
        self.assertEqual(nvgpu_run.job_count, 2)
        self.assertEqual(nvgpu_run.priority_rank_text, "1-2 of 2")
        self.assertEqual(nvgpu_run.display_reason, "Priority/Resources")
        self.assertEqual(preempt.job.partition, "gpu-preempt")

    def test_same_user_jobs_do_not_merge_across_reservations(self):
        snapshot = build_priority_queue_snapshot(
            "me",
            [
                parse_priority_queue_line(
                    priority_queue_line(
                        "900", "fsabokro", "nvgpu", "Priority", 900, reservation="a"
                    )
                ),
                parse_priority_queue_line(
                    priority_queue_line(
                        "800", "fsabokro", "nvgpu", "Priority", 800, reservation="b"
                    )
                ),
            ],
        )

        self.assertEqual(len(snapshot.grouped_entries), 2)
        self.assertTrue(all(entry.job_count == 1 for entry in snapshot.grouped_entries))

    def test_packed_runs_cover_exactly_one_through_n_in_each_rank_scope(self):
        snapshot = build_priority_queue_snapshot(
            "me",
            [
                parse_priority_queue_line(
                    priority_queue_line("900", "alice", "nvgpu", "Priority", 900)
                ),
                parse_priority_queue_line(
                    priority_queue_line("850", "carol", "general", "Priority", 850)
                ),
                parse_priority_queue_line(
                    priority_queue_line("800", "alice", "nvgpu", "Resources", 800)
                ),
                parse_priority_queue_line(
                    priority_queue_line(
                        "750", "me", "nvgpu", "Priority", 750, reservation="lab"
                    )
                ),
                parse_priority_queue_line(
                    priority_queue_line("700", "bob", "nvgpu", "Priority", 700)
                ),
                parse_priority_queue_line(
                    priority_queue_line(
                        "650", "me", "nvgpu", "Resources", 650, reservation="lab"
                    )
                ),
                parse_priority_queue_line(
                    priority_queue_line("600", "dave", "general", "Resources", 600)
                ),
            ],
        )

        coverage = {}
        totals = {}
        for entry in snapshot.grouped_entries:
            if entry.priority_rank is None:
                continue
            reservation = entry.job.reservation
            if reservation.upper() in {"N/A", "NONE", "(NULL)"}:
                reservation = ""
            scope = (entry.job.partition, reservation)
            coverage.setdefault(scope, []).extend(
                range(entry.priority_rank, (entry.rank_end or entry.priority_rank) + 1)
            )
            totals.setdefault(scope, entry.rank_total)
            self.assertEqual(totals[scope], entry.rank_total)

        self.assertEqual(
            coverage,
            {
                ("nvgpu", ""): [1, 2, 3],
                ("general", ""): [1, 2],
                ("nvgpu", "lab"): [1, 2],
            },
        )
        for scope, ranks in coverage.items():
            self.assertEqual(ranks, list(range(1, totals[scope] + 1)))

    def test_unranked_holds_and_dependencies_follow_ranked_scope_rows(self):
        snapshot = build_priority_queue_snapshot(
            "me",
            [
                parse_priority_queue_line(
                    priority_queue_line("400", "held", "nvgpu", "JobHeldUser", 999)
                ),
                parse_priority_queue_line(
                    priority_queue_line("300", "alice", "nvgpu", "Priority", 900)
                ),
                parse_priority_queue_line(
                    priority_queue_line("200", "blocked", "nvgpu", "Dependency", 850)
                ),
                parse_priority_queue_line(
                    priority_queue_line("100", "bob", "nvgpu", "Resources", 800)
                ),
            ],
        )

        self.assertEqual(
            [entry.job.reason_code for entry in snapshot.all_entries],
            ["Priority", "Resources", "JobHeldUser", "Dependency"],
        )
        self.assertEqual(
            [entry.priority_rank for entry in snapshot.all_entries],
            [1, 2, None, None],
        )
        self.assertEqual(
            [entry.job.job_id for entry in snapshot.grouped_entries],
            ["300", "100", "400", "200"],
        )

    def test_rank_run_sums_requested_resources_and_shows_walltime_range(self):
        snapshot = build_priority_queue_snapshot(
            "me",
            [
                parse_priority_queue_long_line(
                    priority_queue_long_line(
                        "200",
                        "alice",
                        "nvgpu",
                        "Priority",
                        900,
                        requested_tres=(
                            "cpu=4,mem=16G,node=1,gres/gpu=2,gres/gpu:h100=2"
                        ),
                        cpus="4",
                        limit="02:00:00",
                    )
                ),
                parse_priority_queue_long_line(
                    priority_queue_long_line(
                        "100",
                        "alice",
                        "nvgpu",
                        "Resources",
                        800,
                        requested_tres="cpu=8,mem=32G,node=1,gres/gpu:a100=1",
                        cpus="8",
                        limit="04:00:00",
                    )
                ),
            ],
        )

        self.assertEqual(len(snapshot.grouped_entries), 1)
        run = snapshot.grouped_entries[0]
        self.assertEqual(run.priority_rank_text, "1-2 of 2")
        self.assertEqual(run.requested_gpus, 3)
        self.assertEqual(run.requested_cpus, 12)
        self.assertEqual(run.requested_memory_mb, 48 * 1024)
        self.assertEqual(run.display_memory, "48G")
        self.assertEqual(run.walltime_min_seconds, 2 * 60 * 60)
        self.assertEqual(run.walltime_max_seconds, 4 * 60 * 60)
        self.assertEqual(run.display_walltime, "2h–4h")

    def test_rank_run_marks_incomplete_resource_and_walltime_totals_unknown(self):
        snapshot = build_priority_queue_snapshot(
            "me",
            [
                parse_priority_queue_long_line(
                    priority_queue_long_line(
                        "200",
                        "alice",
                        "nvgpu",
                        "Priority",
                        900,
                        requested_tres="cpu=4,mem=16G,node=1,gres/gpu=1",
                        cpus="4",
                        limit="02:00:00",
                    )
                ),
                parse_priority_queue_long_line(
                    priority_queue_long_line(
                        "100",
                        "alice",
                        "nvgpu",
                        "Resources",
                        800,
                        requested_tres="",
                        cpus="2",
                        limit="N/A",
                    )
                ),
            ],
        )

        run = snapshot.grouped_entries[0]
        self.assertIsNone(run.requested_gpus)
        self.assertEqual(run.requested_cpus, 6)
        self.assertIsNone(run.requested_memory_mb)
        self.assertIsNone(run.walltime_min_seconds)
        self.assertIsNone(run.walltime_max_seconds)
        self.assertEqual(run.display_gpus, "-")
        self.assertEqual(run.display_cpus, "6")
        self.assertEqual(run.display_memory, "-")
        self.assertEqual(run.display_walltime, "-")

    def test_gpu_parser_prefers_generic_total_over_type_breakdown(self):
        self.assertEqual(
            parse_gpu_count("gres/gpu=2,gres/gpu:h100=2"),
            2,
        )
        self.assertEqual(
            parse_gpu_count("gres/gpu:a100=1,gres/gpu:h100=2"),
            3,
        )

    def test_client_uses_scheduler_order_user_sprio_and_cached_factors(self):
        queue = "\n".join(
            [
                priority_queue_long_line(
                    "200", "bob", "nvgpu", "Priority", 900
                ),
                priority_queue_long_line(
                    "100",
                    "me",
                    "nvgpu",
                    "Resources",
                    700,
                    array_task="2",
                    requested_tres="cpu=4,mem=16G,node=1,billing=4,gres/gpu=1",
                ),
            ]
        )
        sprio = "100_[1-3]|me|pi-me|nvgpu|700|0|100|0|500|10|0|0|gpu=80|0\n"
        client = SlurmClient(user="me")
        fake_runner = FakeRunner([queue, sprio, queue])
        client.runner = fake_runner

        first = client.fetch_priority_queue()
        first_queue_calls = [
            call
            for call in fake_runner.calls
            if call[0][0] == "squeue" and "-O" in call[0]
        ]
        self.assertEqual(len(first_queue_calls), 1)
        second = client.fetch_priority_queue()

        self.assertEqual(first.my_jobs[0].priority_rank_text, "2 of 2")
        self.assertEqual(first.my_jobs[0].job.gpu_count, 1)
        self.assertEqual(first.my_jobs[0].requested_gpus, 1)
        self.assertEqual(first.my_jobs[0].requested_cpus, 4)
        self.assertEqual(first.my_jobs[0].requested_memory_mb, 16 * 1024)
        self.assertEqual(first.my_jobs[0].job.factors.age, 100)
        self.assertTrue(first.factors_available)
        self.assertEqual(second.my_jobs[0].job.factors.fairshare, 500)
        self.assertEqual(
            fake_runner.calls[0][0],
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
        )
        sprio_calls = [call for call in fake_runner.calls if call[0][0] == "sprio"]
        queue_calls = [
            call
            for call in fake_runner.calls
            if call[0][0] == "squeue" and "-O" in call[0]
        ]
        self.assertEqual(len(sprio_calls), 1)
        self.assertEqual(len(queue_calls), 2)
        self.assertTrue(all("-u" not in call[0] for call in queue_calls))
        self.assertEqual(
            sprio_calls[0][0],
            ["sprio", "-h", "-u", "me", "-o", SPRIO_FORMAT],
        )

    def test_client_without_current_user_still_reports_other_users_resources(self):
        queue = priority_queue_long_line(
            "200",
            "bob",
            "nvgpu",
            "Priority",
            900,
            requested_tres=(
                "cpu=12,mem=96G,node=1,gres/gpu=2,gres/gpu:h100=2"
            ),
            cpus="12",
            limit="1-00:00:00",
        )
        client = SlurmClient(user="me")
        fake_runner = FakeRunner(queue)
        client.runner = fake_runner

        snapshot = client.fetch_priority_queue()

        self.assertEqual(snapshot.my_jobs, ())
        self.assertEqual(len(snapshot.grouped_entries), 1)
        entry = snapshot.grouped_entries[0]
        self.assertEqual(entry.job.user, "bob")
        self.assertEqual(entry.requested_gpus, 2)
        self.assertEqual(entry.requested_cpus, 12)
        self.assertEqual(entry.requested_memory_mb, 96 * 1024)
        self.assertEqual(entry.walltime_min_seconds, 24 * 60 * 60)
        self.assertEqual(len(fake_runner.calls), 1)
        self.assertEqual(fake_runner.calls[0][0][0], "squeue")
        self.assertIn("-O", fake_runner.calls[0][0])
        self.assertNotIn("-u", fake_runner.calls[0][0])

    def test_sprio_failure_keeps_queue_rank_and_reason(self):
        class FailingSprioRunner(FakeRunner):
            def run(self, args, timeout=12.0):
                self.calls.append((args, timeout))
                if args[0] == "sprio":
                    raise SlurmError("priority/basic does not support sprio")
                return priority_queue_long_line(
                    "100", "me", "general", "Priority", 20
                )

        client = SlurmClient(user="me")
        client.runner = FailingSprioRunner()
        snapshot = client.fetch_priority_queue()

        self.assertFalse(snapshot.factors_available)
        self.assertIn("does not support sprio", snapshot.factors_error)
        self.assertEqual(snapshot.my_jobs[0].priority_rank_text, "1 of 1")
        self.assertEqual(
            snapshot.my_jobs[0].job.reason_explanation,
            "Higher-priority jobs are ahead in this partition or reservation.",
        )

    def test_reason_explanations_cover_holds_and_policy_families(self):
        self.assertIn("held by its user", explain_pending_reason("(JobHeldUser)"))
        self.assertIn("aggregate usage limit", explain_pending_reason("QOSGrpCpuLimit"))
        self.assertIn("association", explain_pending_reason("AssocGrpMemLimit"))


class SlurmParsingTests(unittest.TestCase):
    def test_fetch_jobs_expands_array_tasks(self):
        client = SlurmClient(user="testuser")
        fake_runner = FakeRunner()
        client.runner = fake_runner

        self.assertEqual(client.fetch_jobs(), [])
        self.assertEqual(
            fake_runner.calls[0][0],
            [
                "squeue",
                "--array",
                "-h",
                "-u",
                "testuser",
                "-t",
                "all",
                "-o",
                SQUEUE_FORMAT,
            ],
        )

    def test_fetch_jobs_passes_requested_state_filter_to_squeue(self):
        client = SlurmClient(user="testuser", states="pd, running")
        fake_runner = FakeRunner()
        client.runner = fake_runner

        self.assertEqual(client.fetch_jobs(), [])

        self.assertEqual(client.squeue_states, "PD,RUNNING")
        self.assertEqual(
            fake_runner.calls[0][0],
            [
                "squeue",
                "--array",
                "-h",
                "-u",
                "testuser",
                "-t",
                "PD,RUNNING",
                "-o",
                SQUEUE_FORMAT,
            ],
        )

    def test_fetch_jobs_can_query_all_users(self):
        client = SlurmClient(user="testuser")
        client.set_job_user_filter("all")
        fake_runner = FakeRunner()
        client.runner = fake_runner

        self.assertEqual(client.fetch_jobs(), [])

        self.assertEqual(
            fake_runner.calls[0][0],
            [
                "squeue",
                "--array",
                "-h",
                "-t",
                "all",
                "-o",
                SQUEUE_FORMAT,
            ],
        )

    def test_normalize_squeue_states_accepts_all_or_comma_list(self):
        self.assertEqual(normalize_squeue_states(None), "all")
        self.assertEqual(normalize_squeue_states("all"), "all")
        self.assertEqual(normalize_squeue_states("'PD', R"), "PD,R")
        with self.assertRaises(ValueError):
            normalize_squeue_states("PD;RUNNING")

    def test_state_prefiltered_jobs_skip_accounting_expansion(self):
        client = SlurmClient(user="testuser", states="PD")
        fake_runner = FakeRunner(
            "4492653_42|direct-xcon-nsga2|PENDING|nvgpu||"
            "(Resources)|0:00|2-00:00:00|1|4|N/A|"
            "2026-06-28T08:37:36|2026-06-28T16:53:08\n"
        )
        client.runner = fake_runner

        jobs, records = client.fetch_active_job_records()

        self.assertEqual(len(fake_runner.calls), 1)
        self.assertEqual([job.job_id for job in jobs], ["4492653_42"])
        self.assertEqual([record.job_id for record in records], ["4492653_42"])
        self.assertEqual(records[0].source, "squeue")

    def test_user_prefiltered_jobs_skip_accounting_expansion(self):
        client = SlurmClient(user="testuser")
        client.set_job_user_filter("other")
        fake_runner = FakeRunner(
            "4492653_42|direct-xcon-nsga2|PENDING|nvgpu||"
            "(Resources)|0:00|2-00:00:00|1|4|N/A|"
            "2026-06-28T08:37:36|2026-06-28T16:53:08\n"
        )
        client.runner = fake_runner

        jobs, records = client.fetch_active_job_records()

        self.assertEqual(len(fake_runner.calls), 1)
        self.assertIn("-u", fake_runner.calls[0][0])
        self.assertEqual(
            fake_runner.calls[0][0][fake_runner.calls[0][0].index("-u") + 1],
            "other",
        )
        self.assertEqual([job.job_id for job in jobs], ["4492653_42"])
        self.assertEqual([record.job_id for record in records], ["4492653_42"])
        self.assertEqual(records[0].source, "squeue")

    def test_group_prefilter_fetches_broadly_and_filters_locally(self):
        client = SlurmClient(user="testuser")
        client.set_job_principal_filters(groups={"pi-example"})
        fake_runner = FakeRunner(
            "1|keep|RUNNING|nvgpu|node01|None|1:00|2:00|1|4|N/A|"
            "2026-06-28T08:37:36|2026-06-28T08:41:26|alice|pi-example\n"
            "2|drop|RUNNING|nvgpu|node02|None|1:00|2:00|1|4|N/A|"
            "2026-06-28T08:37:36|2026-06-28T08:41:26|bob|pi-other\n"
        )
        client.runner = fake_runner

        jobs = client.fetch_jobs()

        self.assertNotIn("-u", fake_runner.calls[0][0])
        self.assertEqual([job.job_id for job in jobs], ["1"])
        self.assertEqual(jobs[0].group, "pi-example")

    def test_group_prefilter_still_passes_state_filter(self):
        client = SlurmClient(user="testuser", states="PD")
        client.set_job_principal_filters(groups={"pi-example"})
        fake_runner = FakeRunner()
        client.runner = fake_runner

        self.assertEqual(client.fetch_jobs(), [])

        self.assertNotIn("-u", fake_runner.calls[0][0])
        self.assertEqual(
            fake_runner.calls[0][0],
            [
                "squeue",
                "--array",
                "-h",
                "-t",
                "PD",
                "-o",
                SQUEUE_FORMAT,
            ],
        )

    def test_partition_prefilter_passes_partition_filter_to_squeue(self):
        client = SlurmClient(user="testuser")
        client.set_job_partition_filters({"nvgpu", "gpu-preempt"})
        fake_runner = FakeRunner()
        client.runner = fake_runner

        self.assertEqual(client.fetch_jobs(), [])

        self.assertEqual(
            fake_runner.calls[0][0],
            [
                "squeue",
                "--array",
                "-h",
                "-u",
                "testuser",
                "-p",
                "gpu-preempt,nvgpu",
                "-t",
                "all",
                "-o",
                SQUEUE_FORMAT,
            ],
        )

    def test_group_filter_takes_priority_over_selected_users(self):
        client = SlurmClient(user="testuser")
        client.set_job_principal_filters(
            users={"alice", "bob", "carol"},
            groups={"pi-example"},
        )
        fake_runner = FakeRunner(
            "1|drop-alice|RUNNING|nvgpu|node01|None|1:00|2:00|1|4|N/A|"
            "2026-06-28T08:37:36|2026-06-28T08:41:26|alice|pi-other\n"
            "2|drop-bob|RUNNING|nvgpu|node02|None|1:00|2:00|1|4|N/A|"
            "2026-06-28T08:37:36|2026-06-28T08:41:26|bob|pi-other\n"
            "3|keep-carol|RUNNING|nvgpu|node03|None|1:00|2:00|1|4|N/A|"
            "2026-06-28T08:37:36|2026-06-28T08:41:26|carol|pi-example\n"
        )
        client.runner = fake_runner

        jobs = client.fetch_jobs()

        self.assertNotIn("-u", fake_runner.calls[0][0])
        self.assertEqual([job.job_id for job in jobs], ["3"])
        self.assertEqual(jobs[0].user, "carol")
        self.assertEqual(jobs[0].group, "pi-example")

    def test_empty_principal_selection_defaults_to_configured_user(self):
        client = SlurmClient(user="testuser")
        client.set_job_principal_filters(users=set(), groups=set())
        fake_runner = FakeRunner()
        client.runner = fake_runner

        self.assertEqual(client.fetch_jobs(), [])

        self.assertIn("-u", fake_runner.calls[0][0])
        self.assertEqual(
            fake_runner.calls[0][0][fake_runner.calls[0][0].index("-u") + 1],
            "testuser",
        )

    def test_fetch_running_filter_choices_lists_users_groups_and_partitions(self):
        client = SlurmClient(user="testuser")
        fake_runner = FakeRunner(
            "alice|pi-example|nvgpu\n"
            "alice|pi-example|nvgpu\n"
            "bob|pi-other|gpu-preempt\n"
            "zoe|pi-custom|custom-partition\n"
        )
        client.runner = fake_runner

        choices = client.fetch_running_filter_choices()

        self.assertEqual(
            fake_runner.calls[0][0],
            [
                "squeue",
                "--array",
                "-h",
                "-t",
                "R",
                "-o",
                FILTER_CHOICES_FORMAT,
            ],
        )
        self.assertEqual(choices.users, ["alice", "bob", "zoe"])
        self.assertEqual(choices.groups, ["pi-custom", "pi-example", "pi-other"])
        self.assertIn("custom-partition", choices.partitions)
        self.assertEqual(set(VACC_PARTITIONS).difference(choices.partitions), set())

    def test_history_uses_unfiltered_squeue_snapshot(self):
        client = SlurmClient(user="testuser", states="PD")
        fake_runner = FakeRunner(
            [
                (
                    "4492653_3|direct-xcon-nsga2|RUNNING|nvgpu|h2node05|"
                    "h2node05|00:10:00|2-00:00:00|1|4|gpu:h200:1|"
                    "2026-06-28T08:37:36|2026-06-28T08:41:26\n"
                ),
                "",
            ]
        )
        client.runner = fake_runner

        records = client.fetch_job_history("3h")

        self.assertEqual(
            fake_runner.calls[0][0],
            [
                "squeue",
                "--array",
                "-h",
                "-u",
                "testuser",
                "-t",
                "all",
                "-o",
                SQUEUE_FORMAT,
            ],
        )
        self.assertEqual([record.job_id for record in records], ["4492653_3"])

    def test_history_uses_default_user_when_jobs_filter_is_all_users(self):
        client = SlurmClient(user="testuser")
        client.set_job_user_filter("all")
        fake_runner = FakeRunner(["", ""])
        client.runner = fake_runner

        self.assertEqual(client.fetch_job_history("3h"), [])

        self.assertEqual(
            fake_runner.calls[0][0],
            [
                "squeue",
                "--array",
                "-h",
                "-u",
                "testuser",
                "-t",
                "all",
                "-o",
                SQUEUE_FORMAT,
            ],
        )

    def test_fetch_job_history_merges_sacct_with_current_squeue_rows(self):
        client = SlurmClient(user="testuser")
        fake_runner = FakeRunner(
            [
                (
                    "4492653_1|direct-xcon-nsga2|COMPLETED|nvgpu|h2node03|"
                    "h2node03|45:07|2-00:00:00|1|4|N/A|"
                    "2026-06-28T08:37:36|2026-06-28T08:41:26\n"
                    "4492653_42|direct-xcon-nsga2|PENDING|nvgpu||"
                    "(Resources)|0:00|2-00:00:00|1|4|N/A|"
                    "2026-06-28T08:37:36|2026-06-28T16:53:08\n"
                ),
                (
                    "4492653_1|4492655|direct-xcon-nsga2|COMPLETED|nvgpu|"
                    "h2node03|00:45:07|2-00:00:00|1|4|"
                    "billing=4,cpu=4,gres/gpu=1,mem=96G,node=1|"
                    "2026-06-28T08:37:36|2026-06-28T08:41:26|"
                    "2026-06-28T09:26:33|0:0\n"
                ),
            ]
        )
        client.runner = fake_runner

        records = client.fetch_job_history("3h")

        self.assertEqual(
            fake_runner.calls[1][0],
            [
                "sacct",
                "-n",
                "-P",
                "-X",
                "--array",
                "-u",
                "testuser",
                "-S",
                "now-3hours",
                "-o",
                SACCT_FORMAT,
            ],
        )
        self.assertEqual({record.job_id for record in records}, {"4492653_1", "4492653_42"})
        completed = next(record for record in records if record.job_id == "4492653_1")
        self.assertEqual(completed.source, "sacct")
        self.assertEqual(completed.end_text, "2026-06-28T09:26:33")
        pending = next(record for record in records if record.job_id == "4492653_42")
        self.assertEqual(pending.state, "PENDING")
        self.assertEqual(pending.location, "pending: (Resources)")

    def test_fetch_active_job_records_counts_completed_accounting_siblings(self):
        client = SlurmClient(user="testuser")
        fake_runner = FakeRunner(
            [
                (
                    "4492653_3|direct-xcon-nsga2|RUNNING|nvgpu|h2node05|"
                    "h2node05|00:10:00|2-00:00:00|1|4|gpu:h200:1|"
                    "2026-06-28T08:37:36|2026-06-28T08:41:26\n"
                    "4492653_4|direct-xcon-nsga2|PENDING|nvgpu||"
                    "(Resources)|0:00|2-00:00:00|1|4|gpu:h200:1|"
                    "2026-06-28T08:37:36|2026-06-28T16:53:08\n"
                    "9999999_1|finished-array|COMPLETED|nvgpu|h2node01|"
                    "None|00:10:00|2-00:00:00|1|4|gpu:h200:1|"
                    "2026-06-28T07:00:00|2026-06-28T07:10:00\n"
                ),
                (
                    "4492653_1|4492655|direct-xcon-nsga2|COMPLETED|nvgpu|"
                    "h2node03|00:45:07|2-00:00:00|1|4|"
                    "billing=4,cpu=4,gres/gpu=1,mem=96G,node=1|"
                    "2026-06-28T08:37:36|2026-06-28T08:41:26|"
                    "2026-06-28T09:26:33|0:0\n"
                    "4492653_2|4492656|direct-xcon-nsga2|COMPLETED|nvgpu|"
                    "h2node04|00:12:07|2-00:00:00|1|4|"
                    "billing=4,cpu=4,gres/gpu=1,mem=96G,node=1|"
                    "2026-06-28T08:37:36|2026-06-28T08:41:26|"
                    "2026-06-28T09:00:33|0:0\n"
                    "9999999_1|9999999|finished-array|COMPLETED|nvgpu|"
                    "h2node01|00:10:00|2-00:00:00|1|4|"
                    "billing=4,cpu=4,gres/gpu=1,mem=96G,node=1|"
                    "2026-06-28T07:00:00|2026-06-28T07:10:00|"
                    "2026-06-28T07:20:00|0:0\n"
                ),
            ]
        )
        client.runner = fake_runner

        jobs, records = client.fetch_active_job_records()

        self.assertEqual(
            fake_runner.calls[1][0],
            [
                "sacct",
                "-n",
                "-P",
                "-X",
                "--array",
                "-u",
                "testuser",
                "-S",
                "2026-06-28T08:37:36",
                "-o",
                SACCT_FORMAT,
            ],
        )
        self.assertEqual(
            [job.job_id for job in jobs],
            ["4492653_3", "4492653_4", "9999999_1"],
        )
        self.assertEqual(
            {record.job_id for record in records},
            {"4492653_1", "4492653_2", "4492653_3", "4492653_4"},
        )
        group = group_job_records(records)[0]
        self.assertEqual(group.done_text, "2/4")
        self.assertEqual(group.running, 1)
        self.assertEqual(group.pending, 1)

    def test_parse_squeue_line_running_job(self):
        job = parse_squeue_line(
            "4340534_1|lcb-w2d-lr|RUNNING|nvgpu|h2xnode05|h2xnode05|19:21|"
            "6:00:00|1|4|N/A|2026-05-30T12:00:26|2026-05-30T12:00:27"
        )

        self.assertEqual(job.job_id, "4340534_1")
        self.assertEqual(job.array_parent, "4340534")
        self.assertEqual(job.location, "h2xnode05")
        self.assertTrue(job.is_running)

    def test_parse_squeue_line_pending_job(self):
        job = parse_squeue_line(
            "4340534|lcb-w2d-lr|PENDING|nvgpu,gpu-preempt||Resources|0:00|"
            "6:00:00|1|4|N/A|2026-05-30T12:00:26|2026-05-30T14:31:45"
        )

        self.assertEqual(job.location, "pending: Resources")
        self.assertEqual(summarize_jobs([job]), {"PENDING": 1})

    def test_parse_sacct_line_completed_array_task(self):
        record = parse_sacct_line(
            "4492653_1|4492655|direct-xcon-nsga2|COMPLETED|nvgpu|h2node03|"
            "00:45:07|2-00:00:00|1|4|"
            "billing=4,cpu=4,gres/gpu=1,mem=96G,node=1|"
            "2026-06-28T08:37:36|2026-06-28T08:41:26|"
            "2026-06-28T09:26:33|0:0"
        )

        self.assertEqual(record.job_id, "4492653_1")
        self.assertEqual(record.array_parent, "4492653")
        self.assertEqual(record.end_text, "2026-06-28T09:26:33")
        self.assertEqual(record.gpu_count, 1)
        self.assertFalse(record.is_active)

    def test_history_start_defaults_to_24h_for_unknown_window(self):
        self.assertEqual(history_start("1h"), "now-1hours")
        self.assertEqual(history_start("7d"), "now-7days")
        self.assertEqual(history_start("bogus"), "now-24hours")

    def test_group_jobs_counts_array_progress_and_longest_running_task(self):
        jobs = [
            parse_squeue_line(
                "4413548_1|ae-pert-cand|COMPLETED|gpu-preempt|h2node01|None|"
                "1:00:00|4:00:00|1|4|N/A|2026-06-11T16:55:01|2026-06-11T16:55:02"
            ),
            parse_squeue_line(
                "4413548_2|ae-pert-cand|RUNNING|gpu-preempt|h2node02|None|"
                "2:11:04|4:00:00|1|4|N/A|2026-06-11T16:55:01|2026-06-11T16:55:02"
            ),
            parse_squeue_line(
                "4413548_3|ae-pert-cand|RUNNING|gpu-preempt|h2node03|None|"
                "45:19|4:00:00|1|4|N/A|2026-06-11T16:55:01|2026-06-11T18:20:47"
            ),
            parse_squeue_line(
                "4413548_4|ae-pert-cand|PENDING|gpu-preempt||Resources|"
                "0:00|4:00:00|1|4|N/A|2026-06-11T16:55:01|2026-06-11T18:20:47"
            ),
            parse_squeue_line(
                "4413548_5|ae-pert-cand|FAILED|gpu-preempt|h2node04|None|"
                "10:43|4:00:00|1|4|N/A|2026-06-11T16:55:01|2026-06-11T18:20:47"
            ),
        ]

        groups = group_jobs(jobs)

        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group.array_parent, "4413548")
        self.assertEqual(group.name, "ae-pert-cand")
        self.assertEqual(group.done_text, "1/5")
        self.assertEqual(group.completed, 1)
        self.assertEqual(group.running, 2)
        self.assertEqual(group.pending, 1)
        self.assertEqual(group.failed, 1)
        self.assertEqual(group.longest_running_elapsed, "2:11:04")
        self.assertEqual(group.limit, "4:00:00")
        self.assertEqual(group.dominant_state, "RUNNING")

    def test_group_job_records_groups_recent_array_tasks_by_parent(self):
        records = [
            parse_sacct_line(
                "4492653_1|4492655|direct-xcon-nsga2|COMPLETED|nvgpu|h2node03|"
                "00:45:07|2-00:00:00|1|4|"
                "billing=4,cpu=4,gres/gpu=1,mem=96G,node=1|"
                "2026-06-28T08:37:36|2026-06-28T08:41:26|"
                "2026-06-28T09:26:33|0:0"
            ),
            parse_sacct_line(
                "4492653_2|4492656|direct-xcon-nsga2|FAILED|nvgpu|h2node04|"
                "00:01:07|2-00:00:00|1|4|"
                "billing=4,cpu=4,gres/gpu=1,mem=96G,node=1|"
                "2026-06-28T08:37:36|2026-06-28T08:41:26|"
                "2026-06-28T08:42:33|1:0"
            ),
            parse_sacct_line(
                "4492653_3|4492657|direct-xcon-nsga2|RUNNING|nvgpu|h2node05|"
                "00:10:00|2-00:00:00|1|4|"
                "billing=4,cpu=4,gres/gpu=1,mem=96G,node=1|"
                "2026-06-28T08:37:36|2026-06-28T08:41:26|Unknown|0:0"
            ),
        ]

        groups = group_job_records(records)

        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group.array_parent, "4492653")
        self.assertEqual(group.done_text, "1/3")
        self.assertEqual(group.running, 1)
        self.assertEqual(group.failed, 1)
        self.assertEqual(group.cpus, 12)
        self.assertEqual(group.gpus, 3)

    def test_parse_elapsed_seconds_handles_slurm_elapsed_formats(self):
        self.assertEqual(parse_elapsed_seconds("45:19"), 2719)
        self.assertEqual(parse_elapsed_seconds("2:11:04"), 7864)
        self.assertEqual(parse_elapsed_seconds("1-02:20:01"), 94801)
        self.assertEqual(parse_elapsed_seconds("N/A"), -1)

    def test_parse_scontrol_node_load(self):
        nodes = parse_scontrol_nodes(
            """NodeName=h2node01 Arch=x86_64 CoresPerSocket=96
   CPUAlloc=13 CPUEfctv=192 CPUTot=192 CPULoad=4.74
   AvailableFeatures=GPU_SKU:H200,GPU_FP:FP64,GPU_ANY,h200
   ActiveFeatures=GPU_SKU:H200,GPU_FP:FP64,GPU_ANY,h200
   Gres=gpu:h200:4
   RealMemory=1000000 AllocMem=198656 FreeMem=942714
   State=MIXED+PLANNED ThreadsPerCore=1 TmpDisk=0 Weight=1
   Partitions=nvgpu
   AllocTRES=cpu=13,mem=194G,gres/gpu=4
"""
        )

        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.name, "h2node01")
        self.assertEqual(node.base_state, "MIXED")
        self.assertEqual(node.free_cpus, 179)
        self.assertEqual(node.cpu_load, 4.74)
        self.assertTrue(node.has_gpus)
        self.assertFalse(node.is_debug_gpu_node)
        self.assertEqual(node.gpu_text, "4/4")
        self.assertEqual(node.gpu_free, 0)

    def test_free_gpu_count_excludes_debug_gpu_partitions(self):
        nodes = parse_scontrol_nodes(
            """NodeName=gpunode001 Arch=x86_64 CoresPerSocket=64
   CPUAlloc=0 CPUTot=128 CPULoad=0.18
   Gres=gpu:a100:2
   RealMemory=1000000 AllocMem=0 FreeMem=966060
   State=IDLE ThreadsPerCore=1
   Partitions=gpu-debug
   AllocTRES=
NodeName=h2node01 Arch=x86_64 CoresPerSocket=96
   CPUAlloc=13 CPUTot=192 CPULoad=4.74
   Gres=gpu:h200:4
   RealMemory=1000000 AllocMem=198656 FreeMem=942714
   State=MIXED ThreadsPerCore=1
   Partitions=nvgpu
   AllocTRES=cpu=13,mem=194G,gres/gpu=1
NodeName=cpu01 Arch=x86_64 CoresPerSocket=32
   CPUAlloc=0 CPUTot=64 CPULoad=0.00
   Gres=(null)
   RealMemory=100000 AllocMem=0 FreeMem=90000
   State=IDLE ThreadsPerCore=1
   Partitions=general
   AllocTRES=
"""
        )

        self.assertTrue(nodes[0].is_debug_gpu_node)
        self.assertEqual(free_gpu_count(nodes), 3)

    def test_idle_gpus_on_full_cpu_node_are_not_free(self):
        nodes = parse_scontrol_nodes(
            """NodeName=r6node01 Arch=x86_64 CoresPerSocket=96
   CPUAlloc=192 CPUTot=192 CPULoad=190.0
   Gres=gpu:rtx6000:8
   RealMemory=1000000 AllocMem=65536 FreeMem=900000
   State=ALLOCATED ThreadsPerCore=1
   Partitions=nvgpu
   AllocTRES=cpu=192,mem=64G,gres/gpu=2
"""
        )

        node = nodes[0]
        self.assertEqual(node.free_cpus, 0)
        self.assertEqual(node.gpu_text, "2/8")
        self.assertEqual(node.gpu_free, 0)
        self.assertEqual(free_gpu_count(nodes), 0)
        self.assertEqual(stranded_gpu_count(nodes), 6)

    def test_stranded_gpu_count_only_counts_full_cpu_gpu_nodes(self):
        nodes = parse_scontrol_nodes(
            """NodeName=r6node01 Arch=x86_64 CoresPerSocket=96
   CPUAlloc=192 CPUTot=192 CPULoad=190.0
   Gres=gpu:rtx6000:8
   RealMemory=1000000 AllocMem=65536 FreeMem=900000
   State=ALLOCATED ThreadsPerCore=1
   Partitions=nvgpu
   AllocTRES=cpu=192,mem=64G,gres/gpu=2
NodeName=h2node01 Arch=x86_64 CoresPerSocket=96
   CPUAlloc=13 CPUTot=192 CPULoad=4.74
   Gres=gpu:h200:4
   RealMemory=1000000 AllocMem=198656 FreeMem=942714
   State=MIXED ThreadsPerCore=1
   Partitions=nvgpu
   AllocTRES=cpu=13,mem=194G,gres/gpu=1
NodeName=gpudebug01 Arch=x86_64 CoresPerSocket=64
   CPUAlloc=128 CPUTot=128 CPULoad=100.0
   Gres=gpu:a100:2
   RealMemory=1000000 AllocMem=0 FreeMem=966060
   State=ALLOCATED ThreadsPerCore=1
   Partitions=gpu-debug
   AllocTRES=cpu=128
"""
        )

        # Only the fully CPU-allocated non-debug GPU node contributes: 8 - 2 = 6.
        self.assertEqual(stranded_gpu_count(nodes), 6)
        self.assertEqual(free_gpu_count(nodes), 3)

    def test_node_jobs_queries_selected_node(self):
        client = SlurmClient(user="testuser")
        fake_runner = FakeRunner(
            "4341591_1|testuser|RUNNING|12:34|4:00:00|4|gpu:h200:1|train\n"
        )
        client.runner = fake_runner

        output = client.node_jobs("h2node01")

        self.assertEqual(
            fake_runner.calls[0][0],
            ["squeue", "-a", "-h", "-w", "h2node01", "-o", NODE_JOBS_FORMAT],
        )
        self.assertIn("JOBID", output)
        self.assertIn("USER", output)
        self.assertIn("LIMIT", output)
        self.assertIn("4:00:00", output)
        self.assertIn("testuser", output)
        self.assertIn("train", output)

    def test_node_jobs_reports_empty_node(self):
        client = SlurmClient(user="testuser")
        client.runner = FakeRunner("")

        self.assertEqual(client.node_jobs("h2node01"), "No jobs found on h2node01.")

    def test_cluster_usage_queries_running_tasks_across_all_nodes(self):
        client = SlurmClient(user="testuser")
        fake_runner = FakeRunner(
            [
                """JobId=4341591 ArrayJobId=4341591 ArrayTaskId=1 JobName=train
   UserId=testuser(512550) GroupId=pi-ncheney(170095)
   JobState=RUNNING Reason=None Dependency=(null)
   NumCPUs=4 ReqTRES=cpu=4,mem=16G,node=1,billing=4,gres/gpu=1
   AllocTRES=cpu=4,mem=16G,node=1,billing=4,gres/gpu=1
JobId=4341592 ArrayJobId=4341592 ArrayTaskId=7 JobName=train
   UserId=other(1234) GroupId=pi-example(5678)
   JobState=RUNNING Reason=None Dependency=(null)
   NumCPUs=8 ReqTRES=cpu=8,mem=32G,node=1,billing=8
   AllocTRES=cpu=8,mem=32G,node=1,billing=8
""",
                """NodeName=gpunode001 Arch=x86_64 CoresPerSocket=64
   CPUAlloc=0 CPUTot=128 CPULoad=0.18
   Gres=gpu:a100:2
   RealMemory=1000000 AllocMem=0 FreeMem=966060
   State=IDLE ThreadsPerCore=1
   Partitions=gpu-debug
   AllocTRES=
NodeName=h2node01 Arch=x86_64 CoresPerSocket=96
   CPUAlloc=13 CPUTot=192 CPULoad=4.74
   Gres=gpu:h200:4
   RealMemory=1000000 AllocMem=198656 FreeMem=942714
   State=MIXED ThreadsPerCore=1
   Partitions=nvgpu
   AllocTRES=cpu=13,mem=194G,gres/gpu=1
""",
                """testuser| pi-test|0.680600|0.750000
other| pi-other|0.125000|0.500000
""",
                "testuser|pi-test\nother|pi-other\n",
            ]
        )
        client.runner = fake_runner

        output = client.cluster_usage()

        self.assertEqual(
            fake_runner.calls[0][0],
            ["scontrol", "show", "job"],
        )
        self.assertEqual(
            fake_runner.calls[1][0],
            ["scontrol", "show", "node"],
        )
        self.assertEqual(fake_runner.calls[2][0][0], "sshare")
        self.assertEqual(fake_runner.calls[3][0][0], "sacctmgr")
        self.assertIn("2 people running 2 tasks", output)
        self.assertIn("FS", output)
        self.assertIn("testuser", output)
        self.assertRegex(output, r"(?m)^testuser\s+1\s+4\s+1\s+16G\s+0\.68060$")
        self.assertIn("other", output)
        stripped_output = "\n".join(line.rstrip() for line in output.splitlines())
        self.assertRegex(
            stripped_output,
            r"(?m)^other\s+1\s+8\s+0\s+32G\s+0\.12500$",
        )
        self.assertIn("TOTAL", output)
        self.assertRegex(stripped_output, r"(?m)^FREE\s+-\s+-\s+3\s+-\s+-$")

    def test_parse_node_job_line_strips_fields(self):
        job = parse_node_job_line(
            " 4341679_19 | testuser | RUNNING | 44:18 | 2:00:00 | 4 | N/A | lcb-ant-omni-lr "
        )

        self.assertEqual(job["job_id"], "4341679_19")
        self.assertEqual(job["user"], "testuser")
        self.assertEqual(job["state"], "RUNNING")
        self.assertEqual(job["limit"], "2:00:00")
        self.assertEqual(job["name"], "lcb-ant-omni-lr")

    def test_format_node_jobs_aligns_rows(self):
        text = format_node_jobs(
            [
                parse_node_job_line(
                    "4341591_66|testuser|RUNNING|58:18|2:00:00|4|N/A|lcb-ant-lr"
                ),
                parse_node_job_line(
                    "4341679_19|testuser|RUNNING|44:18|4:00:00|4|N/A|lcb-ant-omni-lr"
                ),
            ]
        )
        lines = text.splitlines()

        self.assertEqual(lines[0].index("USER"), lines[2].index("testuser"))
        self.assertEqual(lines[0].index("STATE"), lines[2].index("RUNNING"))
        self.assertEqual(lines[0].index("LIMIT"), lines[2].index("2:00:00"))
        self.assertEqual(lines[0].rindex("JOB"), lines[2].index("lcb-ant-lr"))

    def test_parse_gpu_count_handles_slurm_gres_shapes(self):
        self.assertEqual(parse_gpu_count("gpu:h200:1"), 1)
        self.assertEqual(parse_gpu_count("gpu:2"), 2)
        self.assertEqual(parse_gpu_count("gres/gpu=4"), 4)
        self.assertEqual(parse_gpu_count("cpu=4,mem=64G,gres/gpu=1"), 1)
        self.assertEqual(parse_gpu_count("gpu:h200:1,gpu:a100:2"), 3)
        self.assertEqual(parse_gpu_count("N/A"), 0)

    def test_parse_memory_mb_handles_slurm_units_and_unavailable_values(self):
        self.assertEqual(parse_memory_mb("4096M"), 4096)
        self.assertEqual(parse_memory_mb("16G"), 16384)
        self.assertEqual(parse_memory_mb("1.5T"), 1572864)
        self.assertEqual(parse_memory_mb("2Gc"), 2048)
        self.assertIsNone(parse_memory_mb("N/A"))
        self.assertIsNone(parse_memory_mb("0"))

    def test_aggregate_user_usage_sums_tasks_and_requested_resources(self):
        usage = aggregate_user_usage(
            [
                {
                    "job_id": "1",
                    "user": "alice",
                    "cpus": "4",
                    "tres": "cpu=4,mem=16G,node=1,gres/gpu=1",
                    "memory": "16G",
                },
                {
                    "job_id": "2",
                    "user": "alice",
                    "cpus": "2",
                    "tres": "cpu=2,mem=8G,node=1",
                    "memory": "8G",
                },
                {
                    "job_id": "3",
                    "user": "bob",
                    "cpus": "16",
                    "tres": "cpu=16,node=1,gres/gpu=2",
                    "memory": "N/A",
                },
            ]
        )

        self.assertEqual([row.user for row in usage], ["bob", "alice"])
        self.assertEqual(usage[0].tasks, 1)
        self.assertEqual(usage[0].cpus, 16)
        self.assertEqual(usage[0].gpus, 2)
        self.assertIsNone(usage[0].memory_mb)
        self.assertEqual(usage[1].tasks, 2)
        self.assertEqual(usage[1].cpus, 6)
        self.assertEqual(usage[1].gpus, 1)
        self.assertEqual(usage[1].memory_mb, 24576)

    def test_format_user_usage_omits_ram_when_unavailable(self):
        usage = aggregate_user_usage(
            [
                {
                    "job_id": "1",
                    "user": "alice",
                    "cpus": "4",
                    "tres": "cpu=4,gres/gpu=1",
                    "memory": "N/A",
                },
                {
                    "job_id": "2",
                    "user": "bob",
                    "cpus": "8",
                    "tres": "cpu=8",
                    "memory": "N/A",
                },
            ]
        )

        text = format_user_usage(usage)

        self.assertIn("2 people running 2 tasks", text)
        self.assertIn("USER", text)
        self.assertIn("TASKS", text)
        self.assertIn("CPUS", text)
        self.assertIn("GPUS", text)
        self.assertNotIn("RAM", text)

    def test_format_user_usage_adds_free_row_after_total(self):
        usage = aggregate_user_usage(
            [
                {
                    "job_id": "1",
                    "user": "alice",
                    "cpus": "4",
                    "tres": "cpu=4,gres/gpu=1",
                    "memory": "N/A",
                }
            ]
        )

        text = format_user_usage(usage, free_gpus=7)
        lines = text.splitlines()
        total_index = next(
            index for index, line in enumerate(lines) if line.startswith("TOTAL")
        )
        free_index = next(
            index for index, line in enumerate(lines) if line.startswith("FREE")
        )

        self.assertEqual(free_index, total_index + 1)
        self.assertRegex(lines[free_index].rstrip(), r"^FREE\s+-\s+-\s+7$")

    def test_format_user_usage_shows_fairshare_for_each_user(self):
        usage = aggregate_user_usage(
            [
                {
                    "job_id": "1",
                    "user": "alice",
                    "cpus": "4",
                    "tres": "cpu=4,gres/gpu=1",
                    "memory": "N/A",
                },
                {
                    "job_id": "2",
                    "user": "bob",
                    "cpus": "8",
                    "tres": "cpu=8",
                    "memory": "N/A",
                },
            ]
        )

        text = format_user_usage(
            usage,
            fairshare_by_user={"alice": 0.000736},
        )
        lines = text.splitlines()

        self.assertIn("FS", lines[2])
        self.assertRegex(
            next(line for line in lines if line.startswith("alice")),
            r"0\.00074$",
        )
        self.assertRegex(
            next(line for line in lines if line.startswith("bob")).rstrip(),
            r"-$",
        )

    def test_format_user_usage_renders_allocated_row_before_free(self):
        usage = aggregate_user_usage(
            [
                {
                    "job_id": "1",
                    "user": "alice",
                    "cpus": "4",
                    "tres": "cpu=4,gres/gpu=1",
                    "memory": "N/A",
                }
            ]
        )

        text = format_user_usage(usage, free_gpus=7, allocated_gpus=6)
        lines = text.splitlines()
        total_index = next(
            index for index, line in enumerate(lines) if line.startswith("TOTAL")
        )
        allocated_index = next(
            index for index, line in enumerate(lines) if line.startswith("ALLOCATED")
        )
        free_index = next(
            index for index, line in enumerate(lines) if line.startswith("FREE")
        )

        self.assertEqual(allocated_index, total_index + 1)
        self.assertEqual(free_index, allocated_index + 1)
        self.assertRegex(lines[allocated_index].rstrip(), r"^ALLOCATED\s+-\s+-\s+6$")
        self.assertRegex(lines[free_index].rstrip(), r"^FREE\s+-\s+-\s+7$")

    def test_parse_scontrol_job_usage_counts_running_alloc_tres(self):
        usage = parse_scontrol_job_usage(
            """JobId=4414236 ArrayJobId=4413548 ArrayTaskId=235 JobName=ae-pert-cand
   UserId=testuser(512550) GroupId=pi-ncheney(170095)
   JobState=RUNNING Reason=None Dependency=(null)
   NumCPUs=4 NumTasks=1 CPUs/Task=4
   ReqTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1
   AllocTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1
JobId=4414192 ArrayJobId=4413548 ArrayTaskId=214 JobName=ae-pert-cand
   UserId=testuser(512550) GroupId=pi-ncheney(170095)
   JobState=COMPLETED Reason=None Dependency=(null)
   NumCPUs=4 NumTasks=1 CPUs/Task=4
   ReqTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1
   AllocTRES=cpu=4,mem=64G,node=1,billing=4,gres/gpu=1
"""
        )

        self.assertEqual(
            usage,
            [
                {
                    "job_id": "4414236",
                    "user": "testuser",
                    "cpus": "4",
                    "tres": "cpu=4,mem=64G,node=1,billing=4,gres/gpu=1",
                    "memory": "64G",
                }
            ],
        )

    def test_parse_tres_value_extracts_named_value(self):
        self.assertEqual(
            parse_tres_value("cpu=4,mem=64G,node=1,billing=4,gres/gpu=1", "mem"),
            "64G",
        )
        self.assertEqual(parse_tres_value("cpu=4,node=1", "mem"), "")


# Real sreport AccountUtilizationByUser output: account-total rows have an empty
# Login field, per-user rows carry a Login, and each account emits one row per
# TRES. Used values are in Hours.
SREPORT_SAMPLE = "\n".join(
    [
        "|root|cpu|1770704",
        "|root|gres/gpu|14336",
        "|pi-aelledge|cpu|75782",
        "|pi-aelledge|gres/gpu|811",
        "bhimberg|pi-aelledge|cpu|75782",
        "bhimberg|pi-aelledge|gres/gpu|811",
        "|pi-alwoodwa|cpu|1601",
        "|pi-alwoodwa|gres/gpu|0",
        "jstonge1|pi-alwoodwa|cpu|1211",
        "jstonge1|pi-alwoodwa|gres/gpu|0",
        "mvarnold|pi-alwoodwa|cpu|391",
        "mvarnold|pi-alwoodwa|gres/gpu|0",
    ]
)

# Real 4-column sshare output: account rows have LevelFS but no user FairShare,
# and sshare indents the Account column even in parseable mode.
SSHARE_SAMPLE = "\n".join(
    [
        "|root||",
        "root| root|1.000000|inf",
        "| pi-alwoodwa||0.750000",
        "jstonge1|  pi-alwoodwa|0.812345|1.500000",
        "mvarnold|  pi-alwoodwa|0.912345|0.500000",
        "| pi-aelledge||inf",
        "bhimberg|  pi-aelledge|0.123456|0.250000",
    ]
)


class LeaderboardParsingTests(unittest.TestCase):
    def test_parse_sreport_usage_folds_cpu_and_gpu_per_association(self):
        entries = parse_sreport_usage(SREPORT_SAMPLE)
        by_key = {(e.login, e.account): e for e in entries}

        self.assertEqual(
            by_key[("bhimberg", "pi-aelledge")],
            UsageEntry("bhimberg", "pi-aelledge", 75782, 811),
        )
        self.assertEqual(
            by_key[("", "pi-alwoodwa")],
            UsageEntry("", "pi-alwoodwa", 1601, 0),
        )
        self.assertEqual(
            by_key[("jstonge1", "pi-alwoodwa")],
            UsageEntry("jstonge1", "pi-alwoodwa", 1211, 0),
        )

    def test_parse_sreport_usage_ignores_blank_and_short_lines(self):
        self.assertEqual(parse_sreport_usage(""), [])
        self.assertEqual(parse_sreport_usage("\n  \nbad|line\n"), [])

    def test_parse_sshare_skips_account_rows_and_strips_indent(self):
        scores = parse_sshare_fairshare(SSHARE_SAMPLE)

        self.assertEqual(
            scores,
            {
                ("root", "root"): 1.0,
                ("jstonge1", "pi-alwoodwa"): 0.812345,
                ("mvarnold", "pi-alwoodwa"): 0.912345,
                ("bhimberg", "pi-aelledge"): 0.123456,
            },
        )

    def test_parse_sshare_scores_keeps_native_account_level_fairshare(self):
        fairshare, level_fairshare = parse_sshare_scores(SSHARE_SAMPLE)

        self.assertEqual(fairshare[("jstonge1", "pi-alwoodwa")], 0.812345)
        self.assertEqual(level_fairshare["pi-alwoodwa"], 0.75)
        self.assertEqual(level_fairshare["pi-aelledge"], float("inf"))
        self.assertNotIn("root", level_fairshare)

    def test_parse_fairshare_value_handles_blanks_and_inf(self):
        self.assertIsNone(parse_fairshare_value(""))
        self.assertIsNone(parse_fairshare_value("inf"))
        self.assertIsNone(parse_fairshare_value("nonsense"))
        self.assertEqual(parse_fairshare_value(" 0.5 "), 0.5)

    def test_parse_level_fairshare_value_preserves_positive_infinity(self):
        self.assertEqual(parse_level_fairshare_value("inf"), float("inf"))
        self.assertEqual(parse_level_fairshare_value(" 0.009805 "), 0.009805)
        self.assertIsNone(parse_level_fairshare_value("nonsense"))

    def test_build_user_leaderboard_sums_usage_across_accounts(self):
        usage = [
            UsageEntry("", "pi-a", 500, 5),  # account total, ignored for users
            UsageEntry("alice", "pi-a", 300, 4),
            UsageEntry("alice", "pi-b", 100, 1),
            UsageEntry("bob", "pi-a", 200, 0),
        ]
        fairshare = {("alice", "pi-a"): 0.4, ("alice", "pi-b"): 0.9, ("bob", "pi-a"): 0.2}
        default_accounts = {"alice": "pi-a", "bob": "pi-a"}
        rows = {
            r.name: r
            for r in build_user_leaderboard(usage, fairshare, default_accounts)
        }

        self.assertEqual(rows["alice"].cpu_hours, 400)
        self.assertEqual(rows["alice"].gpu_hours, 5)
        # Fairshare comes from the user's default account, not their highest score.
        self.assertEqual(rows["alice"].fairshare, 0.4)
        self.assertEqual(rows["bob"].cpu_hours, 200)
        self.assertNotIn("", rows)
        # The group column shows the account each user drew on most.
        self.assertEqual(rows["alice"].group, "pi-a")  # 304 combined vs pi-b's 101
        self.assertEqual(rows["bob"].group, "pi-a")

    def test_build_user_leaderboard_group_is_the_most_used_account(self):
        usage = [
            UsageEntry("dana", "pi-a", 100, 0),
            UsageEntry("dana", "pi-b", 500, 10),  # clearly her dominant account
        ]
        rows = {r.name: r for r in build_user_leaderboard(usage, {})}
        self.assertEqual(rows["dana"].group, "pi-b")

    def test_build_user_leaderboard_missing_fairshare_is_none(self):
        rows = build_user_leaderboard(
            [UsageEntry("carol", "pi-a", 10, 0)],
            {},
            {"carol": "pi-a"},
        )
        self.assertIsNone(rows[0].fairshare)

    def test_build_user_leaderboard_missing_default_account_is_none(self):
        rows = build_user_leaderboard(
            [UsageEntry("carol", "pi-a", 10, 0)],
            {("carol", "pi-a"): 0.7},
            {},
        )
        self.assertIsNone(rows[0].fairshare)

    def test_build_group_leaderboard_uses_account_rows_and_drops_root(self):
        entries = parse_sreport_usage(SREPORT_SAMPLE)
        _fairshare, level_fairshare = parse_sshare_scores(SSHARE_SAMPLE)
        rows = {
            r.name: r
            for r in build_group_leaderboard(entries, level_fairshare)
        }

        self.assertNotIn("root", rows)
        self.assertEqual(rows["pi-alwoodwa"].cpu_hours, 1601)
        self.assertEqual(rows["pi-aelledge"].gpu_hours, 811)
        # Group scores are Slurm's native account LevelFS, not a user average.
        self.assertEqual(rows["pi-alwoodwa"].fairshare, 0.75)
        self.assertEqual(rows["pi-aelledge"].fairshare, float("inf"))

    def test_sort_leaderboard_orders_by_requested_metric_descending(self):
        rows = [
            LeaderboardRow("a", cpu_hours=10, gpu_hours=1, fairshare=0.2),
            LeaderboardRow("b", cpu_hours=5, gpu_hours=9, fairshare=0.8),
            LeaderboardRow("c", cpu_hours=20, gpu_hours=0, fairshare=None),
        ]
        self.assertEqual([r.name for r in sort_leaderboard(rows, "gpu")], ["b", "a", "c"])
        self.assertEqual([r.name for r in sort_leaderboard(rows, "cpu")], ["c", "a", "b"])
        # None fairshare sorts last.
        self.assertEqual(
            [r.name for r in sort_leaderboard(rows, "fairshare")], ["b", "a", "c"]
        )

    def test_sort_leaderboard_ascending_reverses_the_metric(self):
        rows = [
            LeaderboardRow("a", cpu_hours=10, gpu_hours=1, fairshare=0.2),
            LeaderboardRow("b", cpu_hours=5, gpu_hours=9, fairshare=0.8),
            LeaderboardRow("c", cpu_hours=20, gpu_hours=0, fairshare=None),
        ]
        self.assertEqual(
            [r.name for r in sort_leaderboard(rows, "gpu", descending=False)],
            ["c", "a", "b"],
        )
        self.assertEqual(
            [r.name for r in sort_leaderboard(rows, "cpu", descending=False)],
            ["b", "a", "c"],
        )
        # Rows without a fairshare score stay at the bottom in both directions.
        self.assertEqual(
            [r.name for r in sort_leaderboard(rows, "fairshare", descending=False)],
            ["a", "b", "c"],
        )

    def test_human_hours_is_compact(self):
        self.assertEqual(human_hours(5), "5")
        self.assertEqual(human_hours(999), "999")
        self.assertEqual(human_hours(1200), "1.2k")
        self.assertEqual(human_hours(12000), "12k")
        self.assertEqual(human_hours(75782), "76k")
        self.assertEqual(human_hours(1770704), "1.8M")

    def test_human_hours_promotes_near_million_to_M_scale(self):
        # Values that would round up to "1000k" must cross into the M scale.
        self.assertEqual(human_hours(999_499), "999k")
        self.assertEqual(human_hours(999_500), "1.0M")
        self.assertEqual(human_hours(999_750), "1.0M")

    def test_format_fairshare_renders_dash_for_missing(self):
        self.assertEqual(format_fairshare(None), "-")
        self.assertEqual(format_fairshare(float("inf")), "∞")
        self.assertEqual(format_fairshare(0.6806), "0.68060")
        self.assertEqual(format_fairshare(0.000736), "0.00074")
        self.assertEqual(format_fairshare(0.5), "0.50000")
        self.assertEqual(format_fairshare(1.0), "1.00000")

    def test_usage_window_start_subtracts_window_from_now(self):
        now = datetime.datetime(2026, 7, 12, 9, 0, 0)
        self.assertEqual(usage_window_start("24h", now=now), "2026-07-11T09:00:00")
        self.assertEqual(usage_window_start("7d", now=now), "2026-07-05T09:00:00")
        self.assertEqual(usage_window_start("30d", now=now), "2026-06-12T09:00:00")
        # Unknown windows fall back to the 24h window.
        self.assertEqual(usage_window_start("bogus", now=now), "2026-07-11T09:00:00")


class LeaderboardClientTests(unittest.TestCase):
    def test_fetch_usage_window_builds_sreport_command(self):
        client = SlurmClient(user="testuser")
        fake_runner = FakeRunner(SREPORT_SAMPLE)
        client.runner = fake_runner
        now = datetime.datetime(2026, 7, 12, 9, 0, 0)

        entries = client.fetch_usage_window("7d", now=now)

        self.assertTrue(entries)
        args = fake_runner.calls[0][0]
        self.assertEqual(args[0], "sreport")
        self.assertIn("AccountUtilizationByUser", args)
        self.assertIn("Start=2026-07-05T09:00:00", args)
        self.assertIn("End=now", args)
        self.assertIn("-T", args)
        self.assertIn(USAGE_TRES, args)
        self.assertIn(f"format={SREPORT_USAGE_FORMAT}", args)
        self.assertIn("Hours", args)

    def test_fetch_fairshare_data_builds_one_native_sshare_command(self):
        client = SlurmClient(user="testuser")
        fake_runner = FakeRunner(SSHARE_SAMPLE)
        client.runner = fake_runner

        scores, level_scores = client.fetch_fairshare_data()

        self.assertEqual(scores[("jstonge1", "pi-alwoodwa")], 0.812345)
        self.assertEqual(level_scores["pi-alwoodwa"], 0.75)
        self.assertEqual(
            fake_runner.calls[0][0],
            [
                "sshare",
                "-a",
                "-h",
                "-P",
                "-l",
                "-o",
                SSHARE_FAIRSHARE_FORMAT,
            ],
        )

    def test_fetch_fairshare_returns_user_scores_from_combined_query(self):
        client = SlurmClient(user="testuser")
        client.runner = FakeRunner(SSHARE_SAMPLE)

        scores = client.fetch_fairshare()

        self.assertEqual(scores[("bhimberg", "pi-aelledge")], 0.123456)

    def test_fetch_default_accounts_reads_all_users_from_sacctmgr(self):
        client = SlurmClient(user="testuser")
        fake_runner = FakeRunner(
            "alice|pi-a\n"
            "bob|pi-b\n"
            "malformed\n"
            "|missing-user\n"
            "missing-account|\n"
        )
        client.runner = fake_runner

        self.assertEqual(
            client.fetch_default_accounts(),
            {"alice": "pi-a", "bob": "pi-b"},
        )
        self.assertEqual(
            fake_runner.calls[0][0],
            [
                "sacctmgr",
                "-n",
                "-P",
                "show",
                "user",
                "format=User,DefaultAccount",
            ],
        )

    def test_fetch_default_accounts_degrades_to_empty_on_error(self):
        client = SlurmClient(user="testuser")

        def boom(args, timeout=12.0):
            raise SlurmError("sacctmgr unavailable")

        client.runner.run = boom
        self.assertEqual(client.fetch_default_accounts(), {})


class UserInfoClientTests(unittest.TestCase):
    def test_usage_window_start_supports_one_year(self):
        now = datetime.datetime(2026, 7, 12, 9, 0, 0)
        self.assertEqual(usage_window_start("1y", now=now), "2025-07-12T09:00:00")

    def test_user_info_windows_cover_the_four_requested_spans(self):
        self.assertEqual(
            [window for window, _label in USER_INFO_WINDOWS],
            ["24h", "7d", "30d", "1y"],
        )

    def test_fetch_user_compute_usage_filters_to_user_and_sums_own_rows(self):
        client = SlurmClient(user="bhimberg")
        fake_runner = FakeRunner(SREPORT_SAMPLE)
        client.runner = fake_runner
        now = datetime.datetime(2026, 7, 12, 9, 0, 0)

        cpu, gpu = client.fetch_user_compute_usage("30d", now=now)

        # Only the per-user (non-empty login) rows are summed; account totals are
        # skipped so a filtered roll-up never double-counts.
        self.assertEqual((cpu, gpu), (75782, 811))
        args = fake_runner.calls[0][0]
        self.assertEqual(args[0], "sreport")
        self.assertIn("Users=bhimberg", args)
        self.assertIn("AccountUtilizationByUser", args)
        self.assertIn("Start=2026-06-12T09:00:00", args)

    def test_fetch_user_fairshare_keeps_only_the_current_user(self):
        client = SlurmClient(user="jstonge1")
        client.runner = FakeRunner(SSHARE_SAMPLE)

        scores = client.fetch_user_fairshare()

        self.assertEqual(scores, {"pi-alwoodwa": 0.812345})

    def test_fetch_user_default_account_reads_sacctmgr(self):
        client = SlurmClient(user="jstonge1")
        fake_runner = FakeRunner("pi-alwoodwa\n")
        client.runner = fake_runner

        self.assertEqual(client.fetch_user_default_account(), "pi-alwoodwa")
        args = fake_runner.calls[0][0]
        self.assertEqual(args[0], "sacctmgr")
        self.assertIn("jstonge1", args)
        self.assertIn("format=DefaultAccount", args)

    def test_fetch_user_default_account_degrades_to_empty_on_error(self):
        client = SlurmClient(user="jstonge1")

        def boom(args, timeout=12.0):
            raise SlurmError("sacctmgr unavailable")

        client.runner.run = boom
        self.assertEqual(client.fetch_user_default_account(), "")


GPFS_SAMPLE = "\n".join(
    [
        "",
        "Group quota for your primary group: pi-ncheney",
        "",
        "Space limits",
        "-" * 78,
        "Filesystem type          blocks      quota      limit   in_doubt    grace ",
        "gpfs1      GRP           17.58T        20T        25T     16.34G     none ",
        "gpfs2      GRP           16.22T        35T        45T     5.962G     none ",
        "gpfs3tmp   GRP           1.219T     7.812T     7.891T        80M     none ",
        "-" * 78,
        "",
        "File Limits",
        "-" * 78,
        "Filesystem type           files   quota    limit in_doubt    grace  Remarks",
        "gpfs1      GRP          6495522 6291456 12582912     3712   4 days ",
        "-" * 78,
        "",
        "SPACE occupied by dgezgin within the pi-ncheney group",
        "-" * 78,
        "Filesystem   blocks",
        "gpfs1        6.897T",
        "gpfs2        32K",
        "-" * 78,
        "",
        "FILES created by dgezgin within the pi-ncheney group",
        "-" * 78,
        "Filesystem   files",
        "gpfs1        1523523",
        "gpfs2        17",
        "-" * 78,
        "",
        "NOTE:  Quotas are based on your group, so the figures in the first block are for",
        "your group.  Your personal usage is in the second block. ",
    ]
)


class GpfsQuotaTests(unittest.TestCase):
    def test_parse_gpfs_quota_reads_group_and_personal_blocks(self):
        quota = parse_gpfs_quota(GPFS_SAMPLE, user="dgezgin")

        self.assertEqual(quota.primary_group, "pi-ncheney")
        # Both group blocks keep (filesystem, used, soft quota, hard limit).
        self.assertEqual(
            quota.group_space,
            [
                ("gpfs1", "17.58T", "20T", "25T"),
                ("gpfs2", "16.22T", "35T", "45T"),
                ("gpfs3tmp", "1.219T", "7.812T", "7.891T"),
            ],
        )
        self.assertEqual(
            quota.group_files,
            [("gpfs1", "6495522", "6291456", "12582912")],
        )
        self.assertEqual(
            quota.group_space_grace,
            [("gpfs1", "none"), ("gpfs2", "none"), ("gpfs3tmp", "none")],
        )
        self.assertEqual(quota.group_files_grace, [("gpfs1", "4 days")])
        self.assertEqual(quota.personal_space, [("gpfs1", "6.897T"), ("gpfs2", "32K")])
        self.assertEqual(quota.personal_files, [("gpfs1", "1523523"), ("gpfs2", "17")])

    def test_parse_gpfs_quota_handles_empty_output(self):
        self.assertEqual(parse_gpfs_quota(""), GpfsQuota(primary_group=""))

    def test_fetch_gpfs_quota_runs_my_gpfs_quota(self):
        client = SlurmClient(user="dgezgin")
        fake_runner = FakeRunner(GPFS_SAMPLE)
        client.runner = fake_runner

        quota = client.fetch_gpfs_quota()

        self.assertEqual(quota.primary_group, "pi-ncheney")
        self.assertEqual(fake_runner.calls[0][0], ["my_gpfs_quota"])

    def test_parse_gpfs_group_usage_handles_buffered_member_labels(self):
        output = "\n".join(
            [
                "Filesystem type blocks quota limit in_doubt grace | files quota limit",
                "gpfs1 GRP 3T 20T 25T 0 none | 300 1000 2000 0 none",
                "Filesystem type blocks quota limit in_doubt grace | files quota limit",
                "gpfs1 USR 2T 0 0 0 none | 200 0 0 0 none",
                "gpfs2 USR 20G 0 0 0 none | 20 0 0 0 none",
                "Filesystem type blocks quota limit in_doubt grace | files quota limit",
                "gpfs1 USR 1T 0 0 0 none | 100 0 0 0 none",
                "gpfs2 USR 10G 0 0 0 none | 10 0 0 0 none",
                "------Breakdown of quotas for the MEMBERS of your group.------.",
                "------Member alice------",
                "------Member bob------",
            ]
        )

        self.assertEqual(
            parse_gpfs_group_usage(output),
            [
                GpfsMemberUsage("alice", "gpfs1", "2T", 200),
                GpfsMemberUsage("alice", "gpfs2", "20G", 20),
                GpfsMemberUsage("bob", "gpfs1", "1T", 100),
                GpfsMemberUsage("bob", "gpfs2", "10G", 10),
            ],
        )

    def test_fetch_gpfs_group_usage_runs_groupquota(self):
        client = SlurmClient(user="dgezgin")
        fake_runner = FakeRunner("")
        client.runner = fake_runner

        self.assertEqual(client.fetch_gpfs_group_usage(), [])
        self.assertEqual(fake_runner.calls[0], (["groupquota"], 30.0))

    def test_parse_storage_size_converts_human_units(self):
        self.assertEqual(parse_storage_size("1T"), 1024.0 ** 4)
        self.assertEqual(parse_storage_size("32K"), 32 * 1024.0)
        self.assertEqual(parse_storage_size("0"), 0.0)
        self.assertIsNone(parse_storage_size("none"))

    def test_storage_percent_is_used_over_quota(self):
        self.assertAlmostEqual(storage_percent("10T", "20T"), 50.0)
        self.assertIsNone(storage_percent("10T", "0"))
        self.assertIsNone(storage_percent("bad", "20T"))


# Real sacct rows (no -X): a main row per job plus .batch/.extern steps. MaxRSS
# is only on the step rows; TotalCPU aggregates onto the main row.
EFFICIENCY_SAMPLE = "\n".join(
    [
        # JobID|State|AllocCPUS|TotalCPU|CPUTimeRAW|ElapsedRaw|TimelimitRaw|ReqMem|MaxRSS|NNodes
        "4566789_0|COMPLETED|4|01:11:01|15252|3813|2160|96G||1",
        "4566789_0.batch|COMPLETED|4|01:11:01|15252|3813|||7666696K|1",
        "4566789_0.extern|COMPLETED|4|00:00:00|15252|3813|||1000K|1",
        "4566789_1|COMPLETED|4|00:00:00|200|100|10|8G||1",
        "4566789_1.batch|COMPLETED|4|00:00:00|200|100|||4194304K|1",
        # A queued/never-ran task: zero elapsed, must be ignored.
        "4599999|PENDING|4|00:00:00|0|0|60|8G||1",
    ]
)


class JobEfficiencyTests(unittest.TestCase):
    def test_parse_duration_seconds_handles_slurm_formats(self):
        self.assertEqual(parse_duration_seconds("01:11:01"), 4261.0)
        self.assertEqual(parse_duration_seconds("12:34"), 754.0)
        self.assertAlmostEqual(parse_duration_seconds("12:34.500"), 754.5)
        self.assertEqual(parse_duration_seconds("1-02:00:00"), 93600.0)
        self.assertIsNone(parse_duration_seconds(""))

    def test_parse_reqmem_bytes_totals_and_suffixes(self):
        self.assertEqual(parse_reqmem_bytes("96G", 4, 1), 96 * 1024.0 ** 3)
        # Legacy per-CPU / per-node suffixes scale by cpus / nodes.
        self.assertEqual(parse_reqmem_bytes("4Gc", 8, 1), 4 * 1024.0 ** 3 * 8)
        self.assertEqual(parse_reqmem_bytes("4Gn", 8, 2), 4 * 1024.0 ** 3 * 2)
        self.assertIsNone(parse_reqmem_bytes("", 4, 1))

    def test_summarize_job_efficiency_aggregates_across_jobs(self):
        summary = summarize_job_efficiency(EFFICIENCY_SAMPLE, "last 7 days")

        # Two jobs actually ran; the pending one is dropped.
        self.assertEqual(summary.job_count, 2)
        self.assertEqual(summary.window_label, "last 7 days")
        # Job 0: TotalCPU 4261 / CPUTimeRAW 15252 = 27.9%.  Job 1: 0 / 200 = 0%.
        # Mean = ~13.97%.
        self.assertAlmostEqual(summary.cpu_percent, 13.97, places=1)
        # Job 0 memory: 7666696K / 96G = 7.62%.  Job 1: 4194304K(=4G) / 8G = 50%.
        self.assertAlmostEqual(summary.mem_percent, (7.6224 + 50.0) / 2, places=1)
        # Walltime: 3813/(2160*60)=2.94%, 100/(10*60)=16.7%.  Mean ~9.8%.
        self.assertAlmostEqual(summary.walltime_percent, (2.944 + 16.667) / 2, places=1)
        # Raw averages behind the percentages.
        self.assertEqual(summary.cpu_alloc, 4.0)  # both jobs allocated 4 cores
        # Utilized cores: job0 4261/3813=1.117, job1 0/100=0.  Mean ~0.559.
        self.assertAlmostEqual(summary.cpu_used, (4261 / 3813 + 0.0) / 2, places=2)
        # Requested memory: 96G and 8G.  Used: 7666696K and 4194304K(=4G).
        self.assertAlmostEqual(
            summary.mem_req_bytes, (96 + 8) * 1024 ** 3 / 2, places=0
        )
        # Requested walltime seconds: 2160*60 and 10*60.  Elapsed: 3813 and 100.
        self.assertEqual(summary.walltime_limit_sec, (2160 * 60 + 10 * 60) / 2)
        self.assertEqual(summary.walltime_used_sec, (3813 + 100) / 2)

    def test_summarize_job_efficiency_empty(self):
        summary = summarize_job_efficiency("", "last 7 days")
        self.assertEqual(summary.job_count, 0)
        self.assertIsNone(summary.cpu_percent)
        self.assertIsNone(summary.mem_percent)

    def test_human_bytes_formats_sizes(self):
        self.assertEqual(human_bytes(96 * 1024 ** 3), "96G")
        self.assertEqual(human_bytes(7.31 * 1024 ** 3), "7.3G")
        self.assertEqual(human_bytes(512 * 1024 ** 2), "512M")
        self.assertEqual(human_bytes(2 * 1024 ** 4), "2T")

    def test_human_duration_formats_times(self):
        self.assertEqual(human_duration(2160 * 60), "1d 12h")  # 36h
        self.assertEqual(human_duration(3813), "1h 3m")
        self.assertEqual(human_duration(45 * 60), "45m")
        self.assertEqual(human_duration(30), "30s")

    def test_fetch_job_efficiency_builds_sacct_command(self):
        client = SlurmClient(user="dgezgin")
        fake_runner = FakeRunner(EFFICIENCY_SAMPLE)
        client.runner = fake_runner

        summary = client.fetch_job_efficiency()

        self.assertEqual(summary.job_count, 2)
        args = fake_runner.calls[0][0]
        self.assertEqual(args[0], "sacct")
        self.assertIn("-u", args)
        self.assertIn("dgezgin", args)
        self.assertIn(JOB_EFFICIENCY_FORMAT, args)
        self.assertNotIn("-X", args)  # steps are needed for MaxRSS

    def test_fetch_job_efficiency_for_uses_sacct_dash_j(self):
        client = SlurmClient(user="dgezgin")
        fake_runner = FakeRunner(EFFICIENCY_SAMPLE)
        client.runner = fake_runner

        summary = client.fetch_job_efficiency_for("4566789")

        self.assertEqual(summary.job_count, 2)
        args = fake_runner.calls[0][0]
        self.assertEqual(args[0], "sacct")
        self.assertIn("-j", args)
        self.assertIn("4566789", args)
        self.assertIn(JOB_EFFICIENCY_FORMAT, args)

    def test_format_job_efficiency_single_job(self):
        sample = "\n".join(
            [
                "4607842|CANCELLED by 5|4|01:45.007|288|72|120|64G||1",
                "4607842.batch|CANCELLED|4|01:45.007|288|72|||1485656K|1",
            ]
        )
        text = format_job_efficiency(
            summarize_job_efficiency(sample, "4607842"), "4607842", "craftax"
        )
        self.assertIn("4607842  (craftax)", text)
        self.assertIn("CPU", text)
        self.assertIn("36%", text)  # matches seff for this job
        self.assertIn("used 1.4G of 64G", text)
        self.assertNotIn("array tasks", text)  # single job, no averaging note

    def test_format_job_efficiency_array_notes_task_count(self):
        arr = "\n".join(
            [
                "99_0|COMPLETED|4|01:11:01|15252|3813|2160|96G||1",
                "99_0.batch|COMPLETED|4|01:11:01|15252|3813|||7666696K|1",
                "99_1|COMPLETED|4|01:03:00|15000|3750|2160|96G||1",
                "99_1.batch|COMPLETED|4|01:03:00|15000|3750|||7000000K|1",
            ]
        )
        text = format_job_efficiency(summarize_job_efficiency(arr, "99"), "99", "arr")
        self.assertIn("averaged over 2 array tasks", text)

    def test_format_job_efficiency_no_data(self):
        text = format_job_efficiency(
            summarize_job_efficiency("", "123"), "123", "pending"
        )
        self.assertIn("No completed job data", text)


if __name__ == "__main__":
    unittest.main()
