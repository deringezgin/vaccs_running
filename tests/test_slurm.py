import unittest

from vaccs_running.slurm import (
    FILTER_CHOICES_FORMAT,
    SACCT_FORMAT,
    NODE_JOBS_FORMAT,
    SQUEUE_FORMAT,
    SlurmClient,
    VACC_PARTITIONS,
    aggregate_user_usage,
    format_node_jobs,
    format_user_usage,
    free_gpu_count,
    stranded_gpu_count,
    group_job_records,
    group_jobs,
    history_start,
    parse_sacct_line,
    parse_node_job_line,
    parse_scontrol_nodes,
    parse_scontrol_job_usage,
    parse_squeue_line,
    parse_elapsed_seconds,
    parse_gpu_count,
    parse_memory_mb,
    parse_tres_value,
    normalize_squeue_states,
    summarize_jobs,
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
            "4341591_1|testuser|RUNNING|12:34|4|gpu:h200:1|train\n"
        )
        client.runner = fake_runner

        output = client.node_jobs("h2node01")

        self.assertEqual(
            fake_runner.calls[0][0],
            ["squeue", "-a", "-h", "-w", "h2node01", "-o", NODE_JOBS_FORMAT],
        )
        self.assertIn("JOBID", output)
        self.assertIn("USER", output)
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
        self.assertIn("2 people running 2 tasks", output)
        self.assertIn("testuser", output)
        self.assertIn("other", output)
        self.assertIn("TOTAL", output)
        self.assertRegex(output, r"(?m)^FREE\s+-\s+-\s+3")

    def test_parse_node_job_line_strips_fields(self):
        job = parse_node_job_line(
            " 4341679_19 | testuser | RUNNING | 44:18 | 4 | N/A | lcb-ant-omni-lr "
        )

        self.assertEqual(job["job_id"], "4341679_19")
        self.assertEqual(job["user"], "testuser")
        self.assertEqual(job["state"], "RUNNING")
        self.assertEqual(job["name"], "lcb-ant-omni-lr")

    def test_format_node_jobs_aligns_rows(self):
        text = format_node_jobs(
            [
                parse_node_job_line("4341591_66|testuser|RUNNING|58:18|4|N/A|lcb-ant-lr"),
                parse_node_job_line("4341679_19|testuser|RUNNING|44:18|4|N/A|lcb-ant-omni-lr"),
            ]
        )
        lines = text.splitlines()

        self.assertEqual(lines[0].index("USER"), lines[2].index("testuser"))
        self.assertEqual(lines[0].index("STATE"), lines[2].index("RUNNING"))
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


if __name__ == "__main__":
    unittest.main()
