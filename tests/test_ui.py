import curses
import unittest
from unittest import mock

from vaccs_running.slurm import (
    EfficiencySummary,
    GpfsQuota,
    Job,
    JobFilterChoices,
    JobRecord,
    LEADERBOARD_WINDOWS,
    Node,
    PriorityFactors,
    PriorityQueueJob,
    PriorityQueueSnapshot,
    SlurmError,
    UsageEntry,
    build_priority_queue_snapshot,
)
from vaccs_running.ui import (
    BUSY_JOBS_REFRESH_SECONDS,
    HISTORY_REFRESH_SECONDS,
    LEADERBOARD_GRID_TOP,
    LEADERBOARD_MIN_HEIGHT,
    LEADERBOARD_MIN_WIDTH,
    LEADERBOARD_PAGE,
    PRIORITY_GPU_PARTITIONS,
    PRIORITY_REFRESH_SECONDS,
    VaccsRunningApp,
    build_user_info_lines,
    command_text,
    fairshare_style,
    filter_choice_options,
    leaderboard_columns,
    leaderboard_too_small,
    page_status,
    popup_geometry,
    responsive_priority_specs,
    resource_count_width,
    resource_meter,
    resource_text_meter,
    resource_text_width,
    status_title,
    terminal_too_small,
    wrap_detail_lines,
)


class FakeClient:
    user = "tester"

    def fetch_jobs(self):
        return []

    def fetch_active_job_records(self):
        return [], []

    def fetch_nodes(self):
        return []

    def fetch_job_history(self, window):
        return []

    def fetch_priority_queue(self):
        return PriorityQueueSnapshot(
            user=self.user,
            pending_jobs=(),
            my_jobs=(),
            factors_available=True,
        )

    def node_jobs(self, node_name):
        return f"jobs for {node_name}"

    def cluster_usage(self):
        return "usage by user"

    def fetch_usage_window(self, window):
        return []

    def fetch_fairshare(self):
        return {}

    def fetch_fairshare_data(self):
        return self.fetch_fairshare(), {}

    def fetch_default_accounts(self):
        return {}

    def fetch_user_fairshare(self):
        return {"pi-test": 0.5}

    def fetch_user_default_account(self):
        return "pi-test"

    def fetch_user_compute_usage(self, window):
        return (10, 2)

    def fetch_gpfs_quota(self):
        return GpfsQuota(
            primary_group="pi-test",
            group_space=[("gpfs1", "1T", "2T", "3T")],
            group_files=[("gpfs1", "42", "100", "150")],
            personal_space=[("gpfs1", "500G")],
            personal_files=[("gpfs1", "42")],
        )

    def fetch_job_efficiency(self, window=None, window_label=""):
        return EfficiencySummary(
            job_count=5,
            cpu_percent=82.0,
            mem_percent=60.0,
            walltime_percent=50.0,
            window_label=window_label or "last 7 days",
        )

    def fetch_job_efficiency_for(self, job_id):
        return EfficiencySummary(
            job_count=1,
            cpu_percent=50.0,
            mem_percent=25.0,
            walltime_percent=40.0,
            window_label=str(job_id),
            cpu_alloc=4.0,
            cpu_used=2.0,
            mem_req_bytes=8 * 1024 ** 3,
            mem_used_bytes=2 * 1024 ** 3,
            walltime_limit_sec=3600,
            walltime_used_sec=1440,
        )


class LeaderboardClient(FakeClient):
    """Returns fixed usage/fairshare and records how often each is fetched."""

    def __init__(self):
        self.usage_calls = []
        self.fairshare_calls = 0
        self.default_account_calls = 0

    def fetch_usage_window(self, window):
        self.usage_calls.append(window)
        return [
            UsageEntry("", "pi-x", 1000, 50),  # account total (used by groups)
            UsageEntry("alice", "pi-x", 700, 40),
            UsageEntry("bob", "pi-x", 300, 10),
            UsageEntry("", "root", 99999, 9999),  # cluster total, never shown
        ]

    def fetch_fairshare_data(self):
        self.fairshare_calls += 1
        return (
            {("alice", "pi-x"): 0.42, ("bob", "pi-x"): 0.90},
            {"pi-x": 0.125},
        )

    def fetch_default_accounts(self):
        self.default_account_calls += 1
        return {"alice": "pi-x", "bob": "pi-x"}


class FindLeaderboardClient(FakeClient):
    """Named users so the live find filter is observable."""

    def fetch_usage_window(self, window):
        return [
            UsageEntry("dgezgin", "pi-x", 300, 30),
            UsageEntry("derek", "pi-x", 200, 20),
            UsageEntry("alice", "pi-x", 100, 10),
        ]

    def fetch_fairshare(self):
        return {}


class ScrollLeaderboardClient(FakeClient):
    """Returns enough users (u00..u29) to overflow a pane so scrolling matters."""

    def __init__(self, count=30):
        self.count = count

    def fetch_usage_window(self, window):
        # Descending gpu_hours so sort-by-gpu yields u00 (rank 1) .. u29.
        return [
            UsageEntry(f"u{i:02d}", "pi-x", (self.count - i) * 10, self.count - i)
            for i in range(self.count)
        ]

    def fetch_fairshare(self):
        return {}


class StateFilteredClient(FakeClient):
    squeue_states = "PD"
    state_filter_active = True


class JobsFilterClient(FakeClient):
    def __init__(self):
        self.user = "testuser"
        self.squeue_states = "all"
        self.job_users = {"testuser"}
        self.job_groups = set()
        self.job_partitions = set()
        self.job_all_principals = False
        self.refreshes = 0

    @property
    def state_filter_active(self):
        return self.squeue_states != "all"

    @property
    def job_user_filter_active(self):
        return (
            self.job_all_principals
            or self.job_groups
            or self.job_users != {self.user}
        )

    @property
    def job_partition_filter_active(self):
        return bool(self.job_partitions)

    @property
    def job_user_label(self):
        if self.job_all_principals:
            return "all"
        bits = []
        if self.job_users:
            bits.append(",".join(sorted(self.job_users)))
        if self.job_groups:
            bits.append("groups:" + ",".join(sorted(self.job_groups)))
        return " ".join(bits) if bits else self.user

    def set_job_state_filter(self, states):
        self.squeue_states = states

    def set_job_user_filter(self, user):
        if user in {None, "", "all"}:
            self.set_job_principal_filters(all_principals=True)
        elif user in {"@", "me"}:
            self.set_job_principal_filters(users={self.user})
        else:
            self.set_job_principal_filters(users={user})

    def set_job_principal_filters(self, users=None, groups=None, *, all_principals=False):
        self.job_all_principals = all_principals
        self.job_users = set(users or [])
        self.job_groups = set(groups or [])
        if not self.job_all_principals and not self.job_users and not self.job_groups:
            self.job_users = {self.user}

    def set_job_partition_filters(self, partitions=None):
        self.job_partitions = set(partitions or [])

    def clear_job_filters(self):
        self.squeue_states = "all"
        self.set_job_principal_filters(users={self.user})
        self.job_partitions = set()

    def fetch_active_job_records(self):
        self.refreshes += 1
        return [], []

    def fetch_running_filter_choices(self):
        return JobFilterChoices(
            users=["alice", "other", "testuser"],
            groups=["pi-example", "pi-other"],
            partitions=["gpu-preempt", "nvgpu"],
        )


class FakeScreen:
    def __init__(self, height=64, width=120):
        self.height = height
        self.width = width
        self.writes = []
        self.erase_count = 0
        self.refresh_count = 0

    def getmaxyx(self):
        return self.height, self.width

    def addstr(self, y, x, text, attr=0):
        self.writes.append((y, x, text, attr))

    def erase(self):
        self.erase_count += 1

    def refresh(self):
        self.refresh_count += 1


class FakePopupWindow(FakeScreen):
    def __init__(self, keys, text_inputs=None):
        super().__init__(height=1, width=1)
        self.keys = list(keys)
        self.text_inputs = list(text_inputs or [])
        self.refresh_count = 0
        self.sizes = []
        self.positions = []

    def keypad(self, value):
        self.keypad_value = value

    def nodelay(self, value):
        self.nodelay_value = value

    def erase(self):
        pass

    def border(self):
        pass

    def refresh(self):
        self.refresh_count += 1

    def getch(self):
        return self.keys.pop(0) if self.keys else ord("q")

    def resize(self, height, width):
        self.height = height
        self.width = width
        self.sizes.append((height, width))

    def mvwin(self, top, left):
        self.positions.append((top, left))


def make_node(name, gres, alloc_tres="", state="IDLE"):
    return Node(
        name=name,
        state=state,
        partitions="nvgpu",
        cpu_alloc=0,
        cpu_total=1,
        cpu_load=0.0,
        real_memory_mb=1,
        alloc_memory_mb=0,
        free_memory_mb=1,
        gres=gres,
        alloc_tres=alloc_tres,
        features="",
    )


def make_job(
    job_id,
    state="RUNNING",
    name="job",
    elapsed="0:01",
    limit="1:00:00",
    user="",
    group="",
    partition="nvgpu",
):
    return Job(
        job_id=job_id,
        name=name,
        state=state,
        partition=partition,
        nodes="h2node01",
        reason="",
        elapsed=elapsed,
        limit=limit,
        node_count="1",
        cpus="1",
        gres="",
        submit_time="",
        start_time="",
        user=user,
        group=group,
    )


def make_priority_job(
    job_id,
    *,
    user="tester",
    name="pending-job",
    partition="nvgpu",
    reason="Priority",
    priority=100,
    account="pi-test",
    requested_tres="cpu=4,mem=16G,node=1,gres/gpu=1",
    limit="1-00:00:00",
    reservation="",
    factors=None,
):
    return PriorityQueueJob(
        job_id=job_id,
        user=user,
        name=name,
        partition=partition,
        state="PENDING",
        reason=reason,
        priority=priority,
        account=account,
        qos="normal",
        submit_time="2026-07-15T08:00:00",
        start_time="N/A",
        reservation=reservation,
        node_count="1",
        cpus="4",
        gres="N/A",
        requested_tres=requested_tres,
        limit=limit,
        factors=factors,
    )


def make_priority_snapshot(*jobs, user="tester", factors_available=True):
    return build_priority_queue_snapshot(
        user,
        jobs,
        factors_available=factors_available,
    )


def make_record(
    job_id,
    state="COMPLETED",
    name="hist-job",
    elapsed="0:10:00",
    limit="1:00:00",
    end_time="2026-06-28T09:00:00",
    tres="cpu=4,gres/gpu=1,mem=16G",
    user="",
    group="",
):
    return JobRecord(
        job_id=job_id,
        raw_job_id=job_id,
        name=name,
        state=state,
        partition="nvgpu",
        nodes="h2node01",
        elapsed=elapsed,
        limit=limit,
        node_count="1",
        cpus="4",
        tres=tres,
        submit_time="2026-06-28T08:00:00",
        start_time="2026-06-28T08:10:00",
        end_time=end_time,
        exit_code="0:0",
        user=user,
        group=group,
    )


class NodeFilterTests(unittest.TestCase):
    def test_header_draws_jobs_and_nodes_tabs_on_top_bar_left(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        screen = FakeScreen(height=12, width=100)

        app._draw_header(screen, 100)

        self.assertIn((1, 2, " j Jobs ", curses.A_BOLD), screen.writes)
        self.assertIn((1, 11, " n Nodes ", 0), screen.writes)
        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(" h History ", written)
        self.assertIn(" w Priority ", written)
        self.assertNotIn(" r Running ", written)

    def test_header_does_not_show_refresh_interval(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0.25)
        screen = FakeScreen(height=12, width=100)

        app._draw_header(screen, 100)

        written = " ".join(write[2] for write in screen.writes)
        self.assertNotIn("refresh", written)
        self.assertNotIn("0.25s", written)

    def test_history_view_uses_ten_second_refresh_interval(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0.25)
        self.assertEqual(app._active_refresh_seconds(), 0.25)

        app.state.view = "history"
        self.assertEqual(app._active_refresh_seconds(), HISTORY_REFRESH_SECONDS)

        disabled = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="history")
        self.assertEqual(disabled._active_refresh_seconds(), 0)

    def test_jobs_view_uses_five_second_refresh_interval_after_fifty_jobs(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=2)
        app.state.jobs = [make_job(str(index)) for index in range(50)]
        self.assertEqual(app._active_refresh_seconds(), 2)

        app.state.jobs.append(make_job("50"))
        self.assertEqual(app._active_refresh_seconds(), BUSY_JOBS_REFRESH_SECONDS)

        slower = VaccsRunningApp(FakeClient(), refresh_seconds=10)
        slower.state.jobs = [make_job(str(index)) for index in range(51)]
        self.assertEqual(slower._active_refresh_seconds(), 10)

        disabled = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        disabled.state.jobs = [make_job(str(index)) for index in range(51)]
        self.assertEqual(disabled._active_refresh_seconds(), 0)

    def test_jobs_header_shows_group_toggle(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        screen = FakeScreen(height=12, width=100)

        app._draw_header(screen, 100)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(" g group ", written)
        self.assertNotIn(" c show-completed ", written)
        self.assertIn(" f filter ", written)
        self.assertNotIn(" s script ", written)

    def test_jobs_header_shows_squeue_state_filter(self):
        app = VaccsRunningApp(StateFilteredClient(), refresh_seconds=0)
        screen = FakeScreen(height=12, width=100)

        app._draw_header(screen, 100)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(" state: PENDING (PD) ", written)

    def test_jobs_header_shows_user_filter(self):
        client = JobsFilterClient()
        client.set_job_user_filter("all")
        app = VaccsRunningApp(client, refresh_seconds=0)
        screen = FakeScreen(height=12, width=100)

        app._draw_header(screen, 100)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(" user: all users ", written)

    def test_jobs_header_shows_partition_filter(self):
        client = JobsFilterClient()
        client.set_job_partition_filters({"nvgpu"})
        app = VaccsRunningApp(client, refresh_seconds=0)
        screen = FakeScreen(height=12, width=120)

        app._draw_header(screen, 120)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(" partition: nvgpu ", written)
        self.assertIn("Filter by partition: nvgpu", " ".join(
            str(item["label"])
            for item in app._jobs_filter_home_items()
        ))

    def test_jobs_header_summarizes_multiple_users_and_groups(self):
        client = JobsFilterClient()
        client.set_job_principal_filters(
            users={"alice", "testuser"},
            groups={"pi-example"},
        )
        app = VaccsRunningApp(client, refresh_seconds=0)
        screen = FakeScreen(height=12, width=120)

        app._draw_header(screen, 120)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(" user: 2 users ", written)
        self.assertIn(" group: 1 group ", written)
        self.assertNotIn("alice", written)
        self.assertNotIn("testuser", written)
        self.assertNotIn("pi-example", written)

        filter_labels = " ".join(
            str(item["label"])
            for item in app._jobs_filter_home_items()
        )
        self.assertIn("Filter by user: 2 users", filter_labels)
        self.assertIn("Filter by group: 1 group", filter_labels)
        self.assertNotIn("alice", filter_labels)
        self.assertNotIn("testuser", filter_labels)
        self.assertNotIn("pi-example", filter_labels)

    def test_history_header_shows_all_filter_windows_inline(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="history")
        screen = FakeScreen(height=12, width=120)

        app._draw_header(screen, 120)

        written = " ".join(write[2] for write in screen.writes)
        # Every window is listed inline; the active one (24h) is highlighted.
        self.assertIn("f filter:", written)
        for window in ("1h", "3h", "24h", "3d", "7d"):
            self.assertIn(window, written)
        self.assertNotIn(" g group ", written)
        # 24h is the active window and carries the highlight attribute.
        active_attr = next(w[3] for w in screen.writes if w[2] == "24h")
        other_attr = next(w[3] for w in screen.writes if w[2] == "7d")
        self.assertNotEqual(active_attr, other_attr)

    def test_accent_color_uses_requested_dc582a(self):
        import vaccs_running.ui as ui

        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        calls = []
        missing = object()
        original_colors = getattr(ui.curses, "COLORS", missing)
        original_can_change = ui.curses.can_change_color
        original_init_color = ui.curses.init_color
        try:
            ui.curses.COLORS = 256
            ui.curses.can_change_color = lambda: True
            ui.curses.init_color = lambda slot, red, green, blue: calls.append(
                (slot, red, green, blue)
            )

            self.assertEqual(app._orange_color(), 16)
        finally:
            if original_colors is missing:
                delattr(ui.curses, "COLORS")
            else:
                ui.curses.COLORS = original_colors
            ui.curses.can_change_color = original_can_change
            ui.curses.init_color = original_init_color

        self.assertEqual(calls, [(16, 863, 345, 165)])

    def test_terminal_too_small_uses_minimum_size(self):
        self.assertTrue(terminal_too_small(69, 32))
        self.assertTrue(terminal_too_small(70, 15))
        self.assertFalse(terminal_too_small(70, 16))

    def test_draw_shows_terminal_too_small_message(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        screen = FakeScreen(height=32, width=69)

        app._draw(screen)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("Terminal size too small:", written)
        self.assertIn("Width = 69 Height = 32", written)
        self.assertIn("Needed for current config:", written)
        self.assertIn("Width = 70 Height = 16", written)
        self.assertNotIn("VACC's Running?", written)
        self.assertEqual(screen.refresh_count, 1)

    def test_status_title_uses_full_state_names(self):
        self.assertEqual(
            status_title(
                "Jobs",
                {"RUNNING": 2, "PENDING": 1, "FAILED": 1},
                ["RUNNING", "PENDING", "FAILED"],
            ),
            " Jobs: RUNNING:2 PENDING:1 FAILED:1 ",
        )
        self.assertEqual(status_title("Groups", {}, ["IDLE"]), " Groups: none ")

    def test_detail_lines_wrap_with_indent(self):
        wrapped = wrap_detail_lines(
            ["submitted=2026-05-31T10:04:06  started=2026-05-31T11:00:00"],
            width=36,
        )

        self.assertEqual(
            wrapped,
            ["submitted=2026-05-31T10:04:06", "  started=2026-05-31T11:00:00"],
        )

    def test_box_draws_bottom_right_corner(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        screen = FakeScreen(height=6, width=20)

        app._draw_box(screen, 2, 0, 4, screen.width, " selected job ")

        self.assertIn((5, 19, "╯", curses.A_DIM), screen.writes)

    def test_jobs_table_title_includes_visible_running_job_status(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [
            make_job("1", "RUNNING"),
            make_job("2", "RUNNING"),
            make_job("3", "PENDING"),
            make_job("4", "COMPLETED"),
        ]
        screen = FakeScreen(height=40, width=140)

        app._draw_jobs_table(screen, app._visible_jobs(), screen.height, screen.width)

        self.assertIn(
            (5, 2, " RUNNING:2 PENDING:1 ", curses.A_BOLD),
            screen.writes,
        )
        written = " ".join(write[2].strip() for write in screen.writes)
        self.assertNotIn("COMPLETED", written)

    def test_jobs_table_hides_limit_before_cpus_when_narrow(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [make_job("1")]
        screen = FakeScreen(height=40, width=70)

        app._draw_jobs_table(screen, app._visible_jobs(), screen.height, screen.width)

        written = " ".join(write[2].strip() for write in screen.writes)
        self.assertNotIn("LIMIT", written)
        self.assertIn("CPUS", written)

    def test_jobs_table_then_hides_cpus_when_narrower(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [make_job("1")]
        screen = FakeScreen(height=40, width=62)

        app._draw_jobs_table(screen, app._visible_jobs(), screen.height, screen.width)

        written = " ".join(write[2].strip() for write in screen.writes)
        self.assertNotIn("LIMIT", written)
        self.assertNotIn("CPUS", written)

    def test_jobs_table_shows_user_and_group_for_multi_user_filter(self):
        client = JobsFilterClient()
        client.set_job_principal_filters(users={"alice", "other"})
        app = VaccsRunningApp(client, refresh_seconds=0)
        app.state.jobs = [
            make_job("1", user="alice", group="pi-example"),
            make_job("2", user="other", group="pi-other"),
        ]
        screen = FakeScreen(height=40, width=150)

        app._draw_jobs_table(screen, app._visible_jobs(), screen.height, screen.width)

        written = " ".join(write[2].strip() for write in screen.writes)
        self.assertIn("USER", written)
        self.assertIn("GROUP", written)
        self.assertIn("alice", written)
        self.assertIn("pi-example", written)
        self.assertIn("other", written)
        self.assertIn("pi-other", written)

    def test_grouped_jobs_table_shows_user_and_group_for_multi_user_filter(self):
        client = JobsFilterClient()
        client.set_job_principal_filters(users={"alice", "other"})
        app = VaccsRunningApp(client, refresh_seconds=0)
        app.state.jobs_grouped = True
        app.state.job_records = [
            make_record("1_1", "RUNNING", name="train", user="alice", group="pi-example"),
            make_record("2_1", "PENDING", name="eval", user="other", group="pi-other"),
        ]
        screen = FakeScreen(height=40, width=150)

        app._draw_job_groups_table(
            screen,
            app._visible_job_groups(),
            screen.height,
            screen.width,
        )

        written = " ".join(write[2].strip() for write in screen.writes)
        self.assertIn("USER", written)
        self.assertIn("GROUP", written)
        self.assertIn("alice", written)
        self.assertIn("pi-example", written)
        self.assertIn("other", written)
        self.assertIn("pi-other", written)

    def test_grouped_jobs_table_shows_progress_counts_and_runtime(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs_grouped = True
        app.state.jobs = [
            make_job(
                "4413548_3",
                "RUNNING",
                name="ae-pert-cand",
                elapsed="2:11:04",
                limit="4:00:00",
            ),
            make_job("4413548_4", "PENDING", name="ae-pert-cand"),
        ]
        app.state.job_records = [
            make_record("4413548_1", "COMPLETED", name="ae-pert-cand", limit="4:00:00"),
            make_record("4413548_2", "COMPLETED", name="ae-pert-cand", limit="4:00:00"),
            make_record(
                "4413548_3",
                "RUNNING",
                name="ae-pert-cand",
                elapsed="2:11:04",
                limit="4:00:00",
                end_time="Unknown",
            ),
            make_record(
                "4413548_4",
                "PENDING",
                name="ae-pert-cand",
                elapsed="0:00",
                limit="4:00:00",
                end_time="Unknown",
            ),
        ]
        screen = FakeScreen(height=40, width=140)

        app._draw_job_groups_table(
            screen,
            app._visible_job_groups(),
            screen.height,
            screen.width,
        )

        header = [
            text.strip()
            for y, _, text, _ in screen.writes
            if y == 6 and text.strip() != "│"
        ]
        row = [
            text.strip()
            for y, _, text, _ in screen.writes
            if y == 7 and text.strip() != "│"
        ]
        self.assertEqual(header[:6], ["JOBID", "JOB", "REQ", "DONE", "RUN", "PEND"])
        self.assertIn("RUN_FOR", header)
        self.assertNotIn("DONE/REQ", header)
        self.assertEqual(row[:6], ["4413548", "ae-pert-cand", "4", "2", "1", "1"])
        self.assertIn("2:11:04", row)

    def test_history_view_draws_grouped_rows_by_default(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="history")
        app.state.history = [
            make_record("4492653_1", "COMPLETED", name="direct-xcon-nsga2"),
            make_record(
                "4492653_2",
                "RUNNING",
                name="direct-xcon-nsga2",
                elapsed="0:22:00",
                end_time="Unknown",
            ),
        ]
        screen = FakeScreen(height=40, width=140)

        app._draw(screen)

        written = " ".join(write[2].strip() for write in screen.writes)
        header = [
            text.strip()
            for y, _, text, _ in screen.writes
            if y == 6 and text.strip() != "│"
        ]
        row = [
            text.strip()
            for y, _, text, _ in screen.writes
            if y == 7 and text.strip() != "│"
        ]
        self.assertNotIn("History Groups", written)
        self.assertNotIn("History:", written)
        self.assertIn("RUNNING:1 COMPLETED:1", written)
        self.assertEqual(header[:6], ["JOBID", "JOB", "REQ", "DONE", "RUN", "PEND"])
        self.assertNotIn("DONE/ALL", header)
        self.assertEqual(row[:5], ["4492653", "direct-xcon-nsga2", "2", "1", "1"])
        self.assertIn("0:22:00", row)
        self.assertIn("selected history group", written)
        self.assertNotIn("selected task", written)

    def test_grouped_history_table_shows_full_job_progress(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="history")
        app.state.history = [
            make_record("4492653_1", "COMPLETED", name="direct-xcon-nsga2"),
            make_record("4492653_2", "FAILED", name="direct-xcon-nsga2"),
            make_record(
                "4492653_3",
                "RUNNING",
                name="direct-xcon-nsga2",
                elapsed="0:22:00",
                end_time="Unknown",
            ),
        ]
        screen = FakeScreen(height=40, width=140)

        app._draw_history_groups_table(
            screen,
            app._visible_history_groups(),
            screen.height,
            screen.width,
        )

        header = [
            text.strip()
            for y, _, text, _ in screen.writes
            if y == 6 and text.strip() != "│"
        ]
        row = [
            text.strip()
            for y, _, text, _ in screen.writes
            if y == 7 and text.strip() != "│"
        ]
        self.assertEqual(header[:6], ["JOBID", "JOB", "REQ", "DONE", "RUN", "PEND"])
        self.assertNotIn("DONE/ALL", header)
        self.assertEqual(row[:6], ["4492653", "direct-xcon-nsga2", "3", "1", "1", "0"])
        self.assertIn("0:22:00", row)

    def test_grouped_job_detail_shows_requested_total(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs_grouped = True
        app.state.jobs = [
            make_job("4413548_1", "COMPLETED", name="ae-pert-cand"),
            make_job("4413548_2", "RUNNING", name="ae-pert-cand"),
        ]
        app.state.job_records = [
            make_record("4413548_1", "COMPLETED", name="ae-pert-cand"),
            make_record(
                "4413548_2",
                "RUNNING",
                name="ae-pert-cand",
                end_time="Unknown",
            ),
        ]
        screen = FakeScreen(height=40, width=120)

        app._draw_job_group_detail(screen, screen.height, screen.width)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("selected job group", written)
        self.assertIn("requested=2", written)
        self.assertIn("done=1", written)

    def test_nodes_table_title_includes_overall_node_status(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.nodes = [
            make_node("node01", "(null)", state="IDLE"),
            make_node("node02", "(null)", state="MIXED"),
            make_node("node03", "(null)", state="ALLOCATED"),
        ]
        screen = FakeScreen(height=40, width=140)

        app._draw_nodes_table(screen, app._visible_nodes(), screen.height, screen.width)

        self.assertIn(
            (5, 2, " IDLE:1 MIXED:1 ALLOCATED:1 ", curses.A_BOLD),
            screen.writes,
        )

    def test_nodes_table_removes_resource_bars_when_narrow(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.nodes = [make_node("node01", "gpu:h200:4", "gres/gpu=1")]
        screen = FakeScreen(height=40, width=100)

        app._draw_nodes_table(screen, app._visible_nodes(), screen.height, screen.width)

        written = " ".join(write[2] for write in screen.writes)
        self.assertNotIn("[", written)
        self.assertNotIn("]", written)
        self.assertIn("0/1", written)
        self.assertIn("0M/1M", written)
        self.assertIn("1/4", written)

    def test_nodes_table_keeps_resource_bars_when_wide(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.nodes = [make_node("node01", "gpu:h200:4", "gres/gpu=1")]
        screen = FakeScreen(height=40, width=140)

        app._draw_nodes_table(screen, app._visible_nodes(), screen.height, screen.width)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("[", written)
        self.assertIn("]", written)

    def test_j_and_n_switch_main_views(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)

        self.assertEqual(app.state.view, "jobs")

        self.assertTrue(app._handle_key(None, ord("n")))
        self.assertEqual(app.state.view, "nodes")

        self.assertTrue(app._handle_key(None, ord("j")))
        self.assertEqual(app.state.view, "jobs")

        self.assertTrue(app._handle_key(None, ord("n")))
        self.assertEqual(app.state.view, "nodes")

        self.assertTrue(app._handle_key(None, ord("r")))
        self.assertEqual(app.state.view, "nodes")

    def test_h_switches_to_history_and_unused_key_keeps_view(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)

        self.assertTrue(app._handle_key(None, ord("h")))
        self.assertEqual(app.state.view, "history")

        self.assertTrue(app._handle_key(None, 111))
        self.assertEqual(app.state.view, "history")

    def test_history_direct_window_keys_do_not_refresh_selected_window(self):
        class HistoryClient(FakeClient):
            def __init__(self):
                self.windows = []

            def fetch_job_history(self, window):
                self.windows.append(window)
                return [make_record(f"job-{window}")]

        client = HistoryClient()
        app = VaccsRunningApp(client, refresh_seconds=0, initial_view="history")

        self.assertTrue(app._handle_key(None, ord("1")))
        self.assertEqual(app.state.history_window, "24h")
        self.assertEqual(client.windows, [])

        self.assertTrue(app._handle_key(None, ord("d")))
        self.assertEqual(app.state.history_window, "24h")
        self.assertEqual(client.windows, [])

    def test_f_cycles_history_window_and_refetches(self):
        class HistoryClient(FakeClient):
            def __init__(self):
                self.windows = []

            def fetch_job_history(self, window):
                self.windows.append(window)
                return [make_record(f"job-{window}")]

        client = HistoryClient()
        app = VaccsRunningApp(client, refresh_seconds=0, initial_view="history")
        self.assertEqual(app.state.history_window, "24h")

        # 'f' advances to the next window (24h -> 3d) and refetches that window.
        self.assertTrue(app._handle_key(None, ord("f")))
        self.assertEqual(app.state.history_window, "3d")
        self.assertEqual(client.windows[-1], "3d")
        self.assertEqual(app.state.history[0].job_id, "job-3d")

        # Continues through the list and wraps back to the start.
        for expected in ("7d", "1h", "3h", "24h"):
            app._handle_key(None, ord("f"))
            self.assertEqual(app.state.history_window, expected)

    def test_status_filter_empty_selection_means_all(self):
        import vaccs_running.ui as ui

        client = JobsFilterClient()
        client.set_job_principal_filters(
            users={"alice", "testuser"},
            groups={"pi-example"},
        )
        app = VaccsRunningApp(client, refresh_seconds=0)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(keys=[ord("\n"), ord(" "), ord(" "), ord("q")])
        original_newwin = curses.newwin
        try:
            def fake_newwin(height, width, top, left):
                popup.height = height
                popup.width = width
                popup.positions.append((top, left))
                return popup

            curses.newwin = fake_newwin

            self.assertTrue(app._handle_key(screen, ord("f")))
        finally:
            curses.newwin = original_newwin

        self.assertEqual(client.squeue_states, "all")
        self.assertEqual(client.refreshes, 2)
        written = " ".join(write[2] for write in popup.writes)
        self.assertIn("running filter", written)
        self.assertIn("Filter by status", written)
        self.assertIn("Filter by user", written)
        self.assertIn("Filter by group", written)
        self.assertIn("Filter by partition", written)
        self.assertNotIn("Clear filters", written)
        self.assertIn("filter by status", written)
        self.assertNotIn("a select-all", written)
        self.assertNotIn("Select all statuses", written)
        self.assertNotIn("Clear status selection", written)
        self.assertIn("BF  BOOT_FAIL", written)
        home_labels = [str(item["label"]) for item in app._jobs_filter_home_items()]
        self.assertTrue(any(label.startswith("Filter by status") for label in home_labels))
        self.assertTrue(any(label.startswith("Filter by user") for label in home_labels))
        self.assertTrue(any(label.startswith("Filter by group") for label in home_labels))
        self.assertTrue(any(label.startswith("Filter by partition") for label in home_labels))
        self.assertNotIn("Clear filters", home_labels)

    def test_jobs_filter_menu_accepts_typed_user(self):
        import vaccs_running.ui as ui

        client = JobsFilterClient()
        app = VaccsRunningApp(client, refresh_seconds=0)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(
            keys=[
                ord("u"),
                ord("u"),
                ord("o"),
                ord("t"),
                ord("h"),
                ord("\n"),
                ord("q"),
            ],
        )
        original_newwin = curses.newwin
        try:
            def fake_newwin(height, width, top, left):
                popup.height = height
                popup.width = width
                popup.positions.append((top, left))
                return popup

            curses.newwin = fake_newwin

            self.assertTrue(app._handle_key(screen, ord("f")))
        finally:
            curses.newwin = original_newwin

        self.assertEqual(client.job_users, {"other"})
        self.assertEqual(client.job_groups, set())
        self.assertEqual(client.refreshes, 1)

    def test_jobs_filter_menu_accepts_typed_group(self):
        import vaccs_running.ui as ui

        client = JobsFilterClient()
        client.set_job_principal_filters(
            users={"alice", "testuser"},
            groups={"pi-example"},
        )
        app = VaccsRunningApp(client, refresh_seconds=0)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(
            keys=[
                ord("g"),
                ord("g"),
                ord("p"),
                ord("i"),
                ord("-"),
                ord("c"),
                ord("u"),
                ord("s"),
                ord("t"),
                ord("o"),
                ord("m"),
                ord("\n"),
                ord("q"),
            ],
        )
        original_newwin = curses.newwin
        try:
            def fake_newwin(height, width, top, left):
                popup.height = height
                popup.width = width
                popup.positions.append((top, left))
                return popup

            curses.newwin = fake_newwin

            self.assertTrue(app._handle_key(screen, ord("f")))
        finally:
            curses.newwin = original_newwin

        self.assertEqual(client.job_users, set())
        self.assertEqual(client.job_groups, {"pi-custom"})
        self.assertEqual(client.refreshes, 1)

    def test_jobs_filter_menu_accepts_typed_partition(self):
        import vaccs_running.ui as ui

        client = JobsFilterClient()
        app = VaccsRunningApp(client, refresh_seconds=0)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(
            keys=[
                ord("p"),
                ord("p"),
                ord("g"),
                ord("p"),
                ord("u"),
                ord("-"),
                ord("c"),
                ord("u"),
                ord("s"),
                ord("t"),
                ord("o"),
                ord("m"),
                ord("\n"),
                ord("q"),
            ],
        )
        original_newwin = curses.newwin
        try:
            def fake_newwin(height, width, top, left):
                popup.height = height
                popup.width = width
                popup.positions.append((top, left))
                return popup

            curses.newwin = fake_newwin

            self.assertTrue(app._handle_key(screen, ord("f")))
        finally:
            curses.newwin = original_newwin

        self.assertEqual(client.job_partitions, {"gpu-custom"})
        self.assertEqual(client.refreshes, 1)

    def test_jobs_filter_typeahead_filters_and_adds_custom_values(self):
        self.assertEqual(
            filter_choice_options(["alice", "other", "testuser"], "oth"),
            ["other"],
        )
        self.assertEqual(
            filter_choice_options(["pi-example", "pi-other"], "custom"),
            [],
        )

        client = JobsFilterClient()
        app = VaccsRunningApp(client, refresh_seconds=0)
        choices = client.fetch_running_filter_choices()
        popup = FakePopupWindow(keys=[ord("z"), ord("o"), ord("e"), ord("\n")])
        popup.height = 10
        popup.width = 60

        app._activate_jobs_user_filter_item(
            popup,
            10,
            60,
            choices,
            {"kind": "action", "action": "custom_user"},
        )

        self.assertEqual(client.job_users, {"zoe"})
        self.assertEqual(client.job_groups, set())
        self.assertIn("zoe", choices.users)

        popup = FakePopupWindow(keys=[ord("d"), ord("e"), ord("b"), ord("u"), ord("g"), ord("\n")])
        popup.height = 10
        popup.width = 60

        app._activate_jobs_partition_filter_item(
            popup,
            10,
            60,
            choices,
            {"kind": "action", "action": "custom_partition"},
        )

        self.assertEqual(client.job_partitions, {"debug"})
        self.assertIn("debug", choices.partitions)

    def test_jobs_filter_submenus_list_current_users_groups_and_partitions(self):
        client = JobsFilterClient()
        app = VaccsRunningApp(client, refresh_seconds=0)
        choices = client.fetch_running_filter_choices()

        user_labels = [
            str(item["label"])
            for item in app._jobs_user_filter_items(choices)
        ]
        group_labels = [
            str(item["label"])
            for item in app._jobs_group_filter_items(choices)
        ]
        partition_labels = [
            str(item["label"])
            for item in app._jobs_partition_filter_items(choices)
        ]

        self.assertEqual(
            user_labels[:3],
            ["Select all", "Clear all (only testuser)", "Enter user name..."],
        )
        self.assertNotIn("Select all users", user_labels)
        self.assertNotIn("Clear all", user_labels)
        self.assertNotIn("Clear user selection (me)", user_labels)
        self.assertIn("alice", user_labels)
        self.assertIn("testuser", user_labels)
        self.assertNotIn("Custom user...", user_labels)
        self.assertEqual(
            group_labels[:3],
            ["Select all", "Clear all", "Enter group name..."],
        )
        self.assertNotIn("Select all groups", group_labels)
        self.assertNotIn("Clear group selection", group_labels)
        self.assertIn("pi-example", group_labels)
        self.assertEqual(
            partition_labels[:3],
            ["Select all", "Clear all", "Enter partition name..."],
        )
        self.assertIn("gpu-preempt", partition_labels)
        self.assertIn("nvgpu", partition_labels)

    def test_jobs_filter_menu_clears_filters(self):
        import vaccs_running.ui as ui

        client = JobsFilterClient()
        client.set_job_state_filter("PD")
        client.set_job_user_filter("all")
        client.set_job_partition_filters({"nvgpu"})
        app = VaccsRunningApp(client, refresh_seconds=0)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(keys=[ord("c"), ord("q")])
        original_newwin = curses.newwin
        try:
            def fake_newwin(height, width, top, left):
                popup.height = height
                popup.width = width
                popup.positions.append((top, left))
                return popup

            curses.newwin = fake_newwin

            self.assertTrue(app._handle_key(screen, ord("f")))
        finally:
            curses.newwin = original_newwin

        self.assertEqual(client.squeue_states, "all")
        self.assertEqual(client.job_users, {"testuser"})
        self.assertEqual(client.job_groups, set())
        self.assertEqual(client.job_partitions, set())
        self.assertEqual(client.refreshes, 1)

    def test_g_toggles_job_grouping_in_jobs_view(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.view = "jobs"
        app.state.selected = 5
        app.state.scroll = 3

        self.assertTrue(app._handle_key(None, ord("g")))

        self.assertTrue(app.state.jobs_grouped)
        self.assertEqual(app.state.selected, 0)
        self.assertEqual(app.state.scroll, 0)
        self.assertEqual(app.state.message, "job grouping on")

        self.assertTrue(app._handle_key(None, ord("g")))

        self.assertFalse(app.state.jobs_grouped)
        self.assertEqual(app.state.message, "job grouping off")

    def test_running_view_hides_completed_and_c_is_unused(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [
            make_job("4413548_1", "COMPLETED", name="active-array"),
            make_job("4413548_2", "RUNNING", name="active-array"),
            make_job("4413548_3", "PENDING", name="active-array"),
            make_job("9999999_1", "COMPLETED", name="finished-array"),
        ]

        self.assertEqual(
            [job.job_id for job in app._visible_jobs()],
            ["4413548_2", "4413548_3"],
        )

        self.assertTrue(app._handle_key(None, ord("c")))

        self.assertEqual(app.state.message, "")
        self.assertEqual(
            [job.job_id for job in app._visible_jobs()],
            ["4413548_2", "4413548_3"],
        )
        self.assertNotIn("9999999_1", [job.job_id for job in app._visible_jobs()])

    def test_jobs_are_sorted_by_job_id_ascending(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [
            make_job("100_10", "PENDING"),
            make_job("20", "RUNNING"),
            make_job("100_2", "RUNNING"),
            make_job("3", "PENDING"),
        ]

        self.assertEqual(
            [job.job_id for job in app._visible_jobs()],
            ["3", "20", "100_2", "100_10"],
        )

    def test_grouped_jobs_are_sorted_by_job_id_ascending(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.job_records = [
            make_record("100_1", "RUNNING", name="last"),
            make_record("20_1", "PENDING", name="middle"),
            make_record("3_1", "RUNNING", name="first"),
        ]

        self.assertEqual(
            [group.array_parent for group in app._visible_job_groups()],
            ["3", "20", "100"],
        )

    def test_state_prefiltered_jobs_are_not_filtered_again_by_ui(self):
        client = StateFilteredClient()
        client.squeue_states = "COMPLETED"
        app = VaccsRunningApp(client, refresh_seconds=0)
        app.state.jobs = [make_job("4413548_1", "COMPLETED", name="active-array")]

        self.assertEqual([job.job_id for job in app._visible_jobs()], ["4413548_1"])

        group = app._visible_job_groups()[0]
        self.assertEqual(group.done_text, "1/1")
        self.assertEqual(group.dominant_state, "COMPLETED")

    def test_unfiltered_running_view_ignores_completed_accounting_records(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [
            make_job("4413548_2", "RUNNING", name="active-array"),
            make_job("4413548_3", "PENDING", name="active-array"),
        ]
        app.state.job_records = [
            make_record("4413548_1", "COMPLETED", name="active-array"),
            make_record("4413548_2", "RUNNING", name="active-array", end_time="Unknown"),
            make_record("4413548_3", "PENDING", name="active-array", end_time="Unknown"),
            make_record("9999999_1", "COMPLETED", name="finished-array"),
        ]

        self.assertEqual(
            [job.job_id for job in app._visible_jobs()],
            ["4413548_2", "4413548_3"],
        )

    def test_g_is_unused_in_history_view(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="history")
        app.state.history = [
            make_record("4492653_1", "COMPLETED", name="direct-xcon-nsga2"),
            make_record("4492654_1", "COMPLETED", name="other-job"),
        ]
        app.state.selected = 1
        app.state.scroll = 0
        app.state.message = "steady"

        self.assertTrue(app._handle_key(None, ord("g")))

        self.assertEqual(app.state.view, "history")
        self.assertEqual(app.state.selected, 1)
        self.assertEqual(app.state.scroll, 0)
        self.assertEqual(app.state.message, "steady")

    def test_g_toggles_gpu_node_filter(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.view = "nodes"
        app.state.nodes = [
            make_node("cpu01", "(null)"),
            make_node("gpu-full", "gpu:h200:4", "gres/gpu=4"),
            make_node("gpu-free", "gpu:h200:4", "gres/gpu=1"),
        ]

        self.assertEqual(
            [node.name for node in app._visible_nodes()],
            ["cpu01", "gpu-full", "gpu-free"],
        )

        self.assertTrue(app._handle_key(None, ord("g")))

        self.assertTrue(app.state.gpu_nodes_only)
        self.assertEqual(
            [node.name for node in app._visible_nodes()],
            ["gpu-full", "gpu-free"],
        )
        self.assertEqual(app.state.message, "GPU node filter on")

        self.assertTrue(app._handle_key(None, ord("g")))

        self.assertFalse(app.state.gpu_nodes_only)
        self.assertEqual(
            [node.name for node in app._visible_nodes()],
            ["cpu01", "gpu-full", "gpu-free"],
        )

    def test_node_filters_are_mutually_exclusive(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.view = "nodes"
        app.state.nodes = [
            make_node("cpu01", "(null)"),
            make_node("gpu-full", "gpu:h200:4", "gres/gpu=4"),
            make_node("gpu-free", "gpu:h200:4", "gres/gpu=1"),
        ]

        self.assertTrue(app._handle_key(None, ord("g")))
        self.assertTrue(app.state.gpu_nodes_only)
        self.assertFalse(app.state.free_gpu_only)
        self.assertEqual(
            [node.name for node in app._visible_nodes()],
            ["gpu-full", "gpu-free"],
        )

        self.assertTrue(app._handle_key(None, ord("f")))
        self.assertFalse(app.state.gpu_nodes_only)
        self.assertTrue(app.state.free_gpu_only)
        self.assertEqual([node.name for node in app._visible_nodes()], ["gpu-free"])

        self.assertTrue(app._handle_key(None, ord("g")))
        self.assertTrue(app.state.gpu_nodes_only)
        self.assertFalse(app.state.free_gpu_only)
        self.assertEqual(
            [node.name for node in app._visible_nodes()],
            ["gpu-full", "gpu-free"],
        )

    def test_p_peeks_at_selected_node_jobs(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        calls = []
        app.state.view = "nodes"
        app.state.nodes = [make_node("h2node01", "gpu:h200:4")]
        app._popup_command = lambda stdscr, title, fn, arg, close_keys=(): calls.append(
            (title, fn(arg), close_keys)
        )

        self.assertTrue(app._handle_key(None, ord("p")))

        self.assertEqual(
            calls,
            [("squeue -a -w h2node01", "jobs for h2node01", (ord("p"),))],
        )

    def test_nodes_header_shows_activity_shortcut(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="nodes")
        screen = FakeScreen(height=12, width=120)

        app._draw_header(screen, 120)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(" a activity ", written)

    def test_a_opens_cluster_usage_from_nodes_view(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        calls = []
        app.state.view = "nodes"
        app.state.nodes = [make_node("h2node01", "gpu:h200:4")]
        app._popup = (
            lambda stdscr, title, text, close_keys=(), refresh_while_open=True: calls.append(
                (title, text, close_keys, refresh_while_open)
            )
        )

        self.assertTrue(app._handle_key(None, ord("a")))

        self.assertEqual(
            calls,
            [("running activity by user", "usage by user", (ord("a"),), True)],
        )

    def test_a_is_nodes_only(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        calls = []
        app.state.view = "jobs"
        app._popup = (
            lambda stdscr, title, text, close_keys=(), refresh_while_open=True: calls.append(
                (title, text, close_keys, refresh_while_open)
            )
        )

        self.assertTrue(app._handle_key(None, ord("a")))

        self.assertEqual(calls, [])

    def test_popup_command_passes_close_keys_to_popup(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        calls = []
        app._popup = lambda stdscr, title, get_text, close_keys=(): calls.append(
            (title, get_text(), close_keys)
        )

        app._popup_command(
            None,
            "title",
            lambda value: f"text for {value}",
            "node01",
            close_keys=(ord("p"),),
        )

        self.assertEqual(calls, [("title", "text for node01", (ord("p"),))])

    def test_command_text_formats_empty_and_errors(self):
        self.assertEqual(command_text(lambda value: "", "job"), "No output.")
        self.assertEqual(command_text(lambda value: " text\n", "job"), "text")

    def test_popup_refreshes_live_text(self):
        import vaccs_running.ui as ui

        app = VaccsRunningApp(FakeClient(), refresh_seconds=0.25)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(keys=[-1, ord("q")])
        calls = []
        background_calls = []
        times = [0.0, 0.0, 0.30]
        original_newwin = curses.newwin
        original_monotonic = ui.time.monotonic
        original_sleep = ui.time.sleep
        try:
            curses.newwin = lambda height, width, top, left: popup
            ui.time.monotonic = lambda: times.pop(0) if times else 0.30
            ui.time.sleep = lambda seconds: None
            app._refresh_current = lambda: background_calls.append("refresh")
            app._draw = lambda stdscr: background_calls.append("draw")
            app._popup(screen, "title", lambda: calls.append("call") or f"text {len(calls)}")
        finally:
            curses.newwin = original_newwin
            ui.time.monotonic = original_monotonic
            ui.time.sleep = original_sleep

        self.assertEqual(calls, ["call", "call"])
        self.assertEqual(background_calls, ["refresh", "draw"])
        self.assertGreaterEqual(popup.refresh_count, 2)

    def test_popup_refreshes_background_for_static_text(self):
        import vaccs_running.ui as ui

        app = VaccsRunningApp(FakeClient(), refresh_seconds=0.25)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(keys=[-1, ord("q")])
        background_calls = []
        times = [0.0, 0.0, 0.30]
        original_newwin = curses.newwin
        original_monotonic = ui.time.monotonic
        original_sleep = ui.time.sleep
        try:
            curses.newwin = lambda height, width, top, left: popup
            ui.time.monotonic = lambda: times.pop(0) if times else 0.30
            ui.time.sleep = lambda seconds: None
            app._refresh_current = lambda: background_calls.append("refresh")
            app._draw = lambda stdscr: background_calls.append("draw")
            app._popup(screen, "title", "snapshot")
        finally:
            curses.newwin = original_newwin
            ui.time.monotonic = original_monotonic
            ui.time.sleep = original_sleep

        self.assertEqual(background_calls, ["refresh", "draw"])
        self.assertGreaterEqual(popup.refresh_count, 2)

    def test_popup_can_disable_live_refresh(self):
        import vaccs_running.ui as ui

        app = VaccsRunningApp(FakeClient(), refresh_seconds=0.25)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(keys=[-1, ord("q")])
        calls = []
        background_calls = []
        times = [0.0, 0.0, 0.30]
        original_newwin = curses.newwin
        original_monotonic = ui.time.monotonic
        original_sleep = ui.time.sleep
        try:
            curses.newwin = lambda height, width, top, left: popup
            ui.time.monotonic = lambda: times.pop(0) if times else 0.30
            ui.time.sleep = lambda seconds: None
            app._refresh_current = lambda: background_calls.append("refresh")
            app._draw = lambda stdscr: background_calls.append("draw")
            app._popup(
                screen,
                "title",
                lambda: calls.append("call") or f"text {len(calls)}",
                refresh_while_open=False,
            )
        finally:
            curses.newwin = original_newwin
            ui.time.monotonic = original_monotonic
            ui.time.sleep = original_sleep

        self.assertEqual(calls, ["call"])
        self.assertEqual(background_calls, [])
        self.assertGreaterEqual(popup.refresh_count, 2)

    def test_popup_footer_draws_on_bottom_border(self):
        import vaccs_running.ui as ui

        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(keys=[ord("q")])
        original_newwin = curses.newwin
        try:
            def fake_newwin(height, width, top, left):
                popup.height = height
                popup.width = width
                return popup

            curses.newwin = fake_newwin
            app._popup(screen, "title", "body")
        finally:
            curses.newwin = original_newwin

        footer_writes = [
            write for write in popup.writes if "up/down scroll" in write[2]
        ]
        self.assertEqual(len(footer_writes), 1)
        self.assertEqual(footer_writes[0][0], popup.height - 1)

    def test_popup_geometry_shrinks_to_short_content(self):
        top, left, height, width = popup_geometry(
            screen_height=60,
            screen_width=160,
            title="peek",
            text="one short line",
        )

        self.assertEqual((height, width), (8, 40))
        self.assertEqual(top, 26)
        self.assertEqual(left, 60)

    def test_popup_geometry_caps_to_screen_for_long_content(self):
        long_line = "x" * 300

        top, left, height, width = popup_geometry(
            screen_height=30,
            screen_width=100,
            title="detail",
            text="\n".join([long_line] * 40),
        )

        self.assertEqual((height, width), (26, 92))
        self.assertEqual(top, 2)
        self.assertEqual(left, 4)

    def test_left_and_right_arrows_jump_visible_page(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [make_job(str(index)) for index in range(60)]
        screen = FakeScreen(height=64)

        self.assertEqual(app._page_size(screen), 48)

        app.state.selected = 0
        self.assertTrue(app._handle_key(screen, curses.KEY_RIGHT))
        self.assertEqual(app.state.selected, 48)
        self.assertEqual(app.state.scroll, 48)

        self.assertTrue(app._handle_key(screen, curses.KEY_LEFT))
        self.assertEqual(app.state.selected, 0)
        self.assertEqual(app.state.scroll, 0)

        app.state.selected = 20
        app.state.scroll = 0
        self.assertTrue(app._handle_key(screen, curses.KEY_LEFT))
        self.assertEqual(app.state.selected, 0)
        self.assertEqual(app.state.scroll, 0)

    def test_right_arrow_stops_at_partial_last_page_start(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        app.state.jobs = [make_job(str(index)) for index in range(101)]
        screen = FakeScreen(height=64)

        self.assertEqual(app._page_size(screen), 48)

        app.state.selected = 48
        app.state.scroll = 48
        self.assertTrue(app._handle_key(screen, curses.KEY_RIGHT))
        self.assertEqual(app.state.selected, 96)
        self.assertEqual(app.state.scroll, 96)

        self.assertTrue(app._handle_key(screen, curses.KEY_RIGHT))
        self.assertEqual(app.state.selected, 96)
        self.assertEqual(app.state.scroll, 96)

    def test_page_status_uses_selected_item_page(self):
        self.assertEqual(page_status(0, total_items=150, page_size=50), "1/3")
        self.assertEqual(page_status(49, total_items=150, page_size=50), "1/3")
        self.assertEqual(page_status(50, total_items=150, page_size=50), "2/3")
        self.assertEqual(page_status(149, total_items=150, page_size=50), "3/3")
        self.assertEqual(page_status(150, total_items=150, page_size=50), "3/3")
        self.assertEqual(page_status(0, total_items=0, page_size=50), "0/0")

    def test_resource_meter_aligns_count_prefix(self):
        count_width = resource_count_width([(1, 8), (12, 192), (192, 192)])

        rows = [
            resource_meter(1, 8, 12.5, meter_width=4, count_width=count_width),
            resource_meter(12, 192, 6.25, meter_width=4, count_width=count_width),
            resource_meter(192, 192, 100.0, meter_width=4, count_width=count_width),
        ]

        self.assertEqual(count_width, len("192/192"))
        self.assertEqual([row.index("[") for row in rows], [8, 8, 8])

    def test_resource_text_meter_aligns_text_prefix(self):
        count_width = resource_text_width(["-", "0/4", "12/16"])

        rows = [
            resource_text_meter("-", 0.0, meter_width=4, count_width=count_width),
            resource_text_meter("0/4", 0.0, meter_width=4, count_width=count_width),
            resource_text_meter("12/16", 75.0, meter_width=4, count_width=count_width),
        ]

        self.assertEqual(count_width, len("12/16"))
        self.assertEqual([row.index("[") for row in rows], [6, 6, 6])

    def test_resource_text_meter_aligns_memory_prefix(self):
        count_width = resource_text_width(["0M/8G", "120G/1000G", "1.0T/1.0T"])

        rows = [
            resource_text_meter("0M/8G", 0.0, meter_width=4, count_width=count_width),
            resource_text_meter(
                "120G/1000G",
                12.0,
                meter_width=4,
                count_width=count_width,
            ),
            resource_text_meter(
                "1.0T/1.0T",
                100.0,
                meter_width=4,
                count_width=count_width,
            ),
        ]

        self.assertEqual(count_width, len("120G/1000G"))
        self.assertEqual([row.index("[") for row in rows], [11, 11, 11])


def rendered_text(screen):
    return " ".join(write[2].strip() for write in screen.writes)


def load_leaderboard(app):
    """Kick off the background fetch and block until every thread finishes."""
    app._start_leaderboard_refresh()
    for thread in list(app._lb_threads):
        thread.join(timeout=5)


class LeaderboardTests(unittest.TestCase):
    def test_leaderboard_too_small_requires_desktop_size(self):
        self.assertTrue(leaderboard_too_small(80, 40))
        self.assertTrue(leaderboard_too_small(120, 18))
        self.assertFalse(
            leaderboard_too_small(LEADERBOARD_MIN_WIDTH, LEADERBOARD_MIN_HEIGHT)
        )

    def test_leaderboard_columns_drop_fairshare_when_narrow(self):
        def keys(width, **kw):
            return [key for key, _label, _w, _a in leaderboard_columns(width, "USER", **kw)]

        # Group mode (no GROUP column): wide keeps FS, narrow drops it.
        self.assertEqual(keys(48), ["rank", "name", "cpu", "gpu", "fs"])
        self.assertEqual(keys(20), ["rank", "name", "cpu", "gpu"])

    def test_leaderboard_columns_show_group_and_drop_it_last(self):
        def keys(width):
            return [key for key, _l, _w, _a in leaderboard_columns(width, "USER", group_col=True)]

        # Wide: rank, name, GROUP, cpu, gpu, FS all present.
        self.assertEqual(keys(60), ["rank", "name", "group", "cpu", "gpu", "fs"])
        # Narrower: FS drops first, GROUP stays (the user asked for it).
        self.assertEqual(keys(36), ["rank", "name", "group", "cpu", "gpu"])
        # Narrowest: GROUP drops too, keeping the core metrics legible.
        self.assertEqual(keys(24), ["rank", "name", "cpu", "gpu"])

    def test_header_shows_leaders_tab_and_controls(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="leaderboard")
        screen = FakeScreen(height=40, width=140)

        app._draw_header(screen, screen.width)

        written = rendered_text(screen)
        self.assertIn("u Usage", written)
        # The refresh control is intentionally not advertised in the header.
        self.assertNotIn("r refresh", written)
        # Each control lists all its options; the active one is highlighted.
        self.assertIn("m mode:", written)
        self.assertIn("user", written)
        self.assertIn("group", written)
        self.assertIn("f find", written)
        self.assertIn("s sort:", written)
        self.assertIn("CPU", written)
        self.assertIn("GPU", written)
        self.assertIn("fairshare", written)
        self.assertIn("o order:", written)
        self.assertIn("ascending", written)
        self.assertIn("descending", written)
        self.assertIn("q quit", written)

    def test_header_highlights_the_active_mode_option(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="leaderboard")

        def attr_of(word):
            screen = FakeScreen(height=40, width=140)
            app._draw_header(screen, screen.width)
            return next(w[3] for w in screen.writes if w[2] == word)

        # In user mode, "user" is highlighted (active-tab pair) and "group" is not.
        user_attr, group_attr = attr_of("user"), attr_of("group")
        self.assertNotEqual(user_attr, group_attr)
        # Switching to group mode flips which word carries the highlight.
        app.state.leaderboard_group_mode = True
        self.assertEqual(attr_of("user"), group_attr)
        self.assertEqual(attr_of("group"), user_attr)

    def test_u_key_switches_to_usage(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0)
        self.assertTrue(app._handle_key(None, ord("u")))
        self.assertEqual(app.state.view, "leaderboard")

    def test_leaderboard_never_auto_refreshes(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=2, initial_view="leaderboard")
        self.assertEqual(app._active_refresh_seconds(), 0.0)

    def test_mode_key_toggles_grouping_and_resets_scroll(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="leaderboard")
        app.state.leaderboard_scroll = 4

        self.assertTrue(app._handle_leaderboard_key(ord("m")))
        self.assertTrue(app.state.leaderboard_group_mode)
        self.assertEqual(app.state.leaderboard_scroll, 0)

        app._handle_leaderboard_key(ord("m"))
        self.assertFalse(app.state.leaderboard_group_mode)

    def test_sort_key_cycles_through_metrics(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="leaderboard")
        # Cycles left-to-right through the header display: GPU -> CPU -> fairshare.
        self.assertEqual(app.state.leaderboard_sort, "gpu")
        app._handle_leaderboard_key(ord("s"))
        self.assertEqual(app.state.leaderboard_sort, "cpu")
        app._handle_leaderboard_key(ord("s"))
        self.assertEqual(app.state.leaderboard_sort, "fairshare")
        app._handle_leaderboard_key(ord("s"))
        self.assertEqual(app.state.leaderboard_sort, "gpu")

    def test_order_key_toggles_direction_and_resets_scroll(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="leaderboard")
        app.state.leaderboard_scroll = 5
        self.assertFalse(app.state.leaderboard_ascending)  # descending by default

        self.assertTrue(app._handle_leaderboard_key(ord("o")))
        self.assertTrue(app.state.leaderboard_ascending)
        self.assertEqual(app.state.leaderboard_scroll, 0)

        app._handle_leaderboard_key(ord("o"))
        self.assertFalse(app.state.leaderboard_ascending)

    def test_header_highlights_active_sort_option(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="leaderboard")

        def attr_of(word):
            screen = FakeScreen(height=40, width=160)
            app._draw_header(screen, screen.width)
            return next(w[3] for w in screen.writes if w[2] == word)

        gpu_attr, cpu_attr = attr_of("GPU"), attr_of("CPU")  # GPU active by default
        self.assertNotEqual(gpu_attr, cpu_attr)
        app.state.leaderboard_sort = "cpu"
        self.assertEqual(attr_of("CPU"), gpu_attr)
        self.assertEqual(attr_of("GPU"), cpu_attr)

    def test_header_highlights_active_order_option(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="leaderboard")

        def attr_of(word):
            screen = FakeScreen(height=40, width=160)
            app._draw_header(screen, screen.width)
            return next(w[3] for w in screen.writes if w[2] == word)

        desc_attr, asc_attr = attr_of("descending"), attr_of("ascending")
        self.assertNotEqual(desc_attr, asc_attr)  # descending active by default
        app.state.leaderboard_ascending = True
        self.assertEqual(attr_of("ascending"), desc_attr)
        self.assertEqual(attr_of("descending"), asc_attr)

    def test_order_direction_reranks_the_panes(self):
        app = VaccsRunningApp(
            ScrollLeaderboardClient(5), refresh_seconds=0, initial_view="leaderboard"
        )
        load_leaderboard(app)
        # Descending by GPU: u00 (highest) is rank 1.
        top_desc = app._leaderboard_snapshot()["24h"]["rows"][0]
        self.assertEqual((top_desc[0], top_desc[1].name), (1, "u00"))
        # Ascending flips it: the lowest-GPU user becomes rank 1.
        app.state.leaderboard_ascending = True
        top_asc = app._leaderboard_snapshot()["24h"]["rows"][0]
        self.assertEqual((top_asc[0], top_asc[1].name), (1, "u04"))

    def test_arrow_keys_scroll_but_tab_keys_pass_through(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="leaderboard")

        self.assertTrue(app._handle_leaderboard_key(curses.KEY_DOWN))
        self.assertEqual(app.state.leaderboard_scroll, 1)
        self.assertTrue(app._handle_leaderboard_key(curses.KEY_UP))
        self.assertEqual(app.state.leaderboard_scroll, 0)
        # Never scroll above the top.
        self.assertTrue(app._handle_leaderboard_key(curses.KEY_UP))
        self.assertEqual(app.state.leaderboard_scroll, 0)

        # Tab-switch keys must fall through so the user can leave the view.
        for key in map(ord, "jnhu"):
            self.assertFalse(app._handle_leaderboard_key(key))

    def test_refresh_key_fetches_every_window_once(self):
        client = LeaderboardClient()
        app = VaccsRunningApp(client, refresh_seconds=0, initial_view="leaderboard")

        self.assertTrue(app._handle_leaderboard_key(ord("r")))
        for thread in list(app._lb_threads):
            thread.join(timeout=5)

        self.assertEqual(
            sorted(client.usage_calls),
            sorted(window for window, _ in LEADERBOARD_WINDOWS),
        )
        self.assertEqual(client.fairshare_calls, 1)
        self.assertEqual(client.default_account_calls, 1)

    def test_refresh_is_ignored_while_a_fetch_is_still_running(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="leaderboard")
        with app._lb_lock:
            app._lb_windows["24h"]["status"] = "loading"

        self.assertFalse(app._start_leaderboard_refresh())

    def test_draw_leaderboard_ranks_users_and_hides_root(self):
        client = LeaderboardClient()
        app = VaccsRunningApp(client, refresh_seconds=0, initial_view="leaderboard")
        load_leaderboard(app)

        # Account-total (empty login) and root rows never become ranked users.
        ranked = [row.name for _rank, row in app._leaderboard_snapshot()["24h"]["rows"]]
        self.assertEqual(sorted(ranked), ["alice", "bob"])

        screen = FakeScreen(height=40, width=140)
        app._draw(screen)

        written = rendered_text(screen)
        self.assertIn("alice", written)
        self.assertIn("bob", written)
        self.assertIn("USER", written)
        # Each user's PI group is shown in its own column.
        self.assertIn("GROUP", written)
        self.assertIn("pi-x", written)
        # Fairshare and compact hour counts are rendered.
        self.assertIn("0.42", written)
        self.assertIn("700", written)

    def test_usage_leaves_a_blank_row_between_the_menu_and_the_panes(self):
        client = LeaderboardClient()
        app = VaccsRunningApp(client, refresh_seconds=0, initial_view="leaderboard")
        load_leaderboard(app)
        screen = FakeScreen(height=40, width=140)

        app._draw(screen)

        rows_used = {write[0] for write in screen.writes}
        self.assertIn(3, rows_used)  # the controls menu row
        self.assertNotIn(4, rows_used)  # blank separator row above the panes
        self.assertEqual(min(r for r in rows_used if r >= 5), LEADERBOARD_GRID_TOP)

    def test_draw_leaderboard_group_mode_shows_accounts(self):
        client = LeaderboardClient()
        app = VaccsRunningApp(client, refresh_seconds=0, initial_view="leaderboard")
        app.state.leaderboard_group_mode = True
        load_leaderboard(app)
        screen = FakeScreen(height=40, width=140)

        app._draw(screen)

        written = rendered_text(screen)
        self.assertIn("GROUP", written)
        self.assertIn("pi-x", written)
        self.assertIn("LevelFS", written)
        self.assertIn("0.125", written)
        self.assertNotIn("root", written)

    def test_draw_leaderboard_shows_loading_and_error_panes(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="leaderboard")
        with app._lb_lock:
            app._lb_windows["24h"] = {"status": "loading", "usage": [], "error": ""}
            app._lb_windows["7d"] = {
                "status": "error",
                "usage": [],
                "error": "sreport exploded",
            }
            app._lb_windows["30d"] = {"status": "ready", "usage": [], "error": ""}
        screen = FakeScreen(height=40, width=160)

        app._draw(screen)

        written = rendered_text(screen)
        self.assertIn("running slurm query", written)
        self.assertIn("sreport exploded", written)
        self.assertIn("no usage in this window", written)

    def test_draw_shows_leaderboard_too_small_notice(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="leaderboard")
        screen = FakeScreen(height=20, width=80)

        app._draw(screen)

        written = rendered_text(screen)
        self.assertIn("bigger screen", written)
        self.assertIn(f"{LEADERBOARD_MIN_WIDTH} x {LEADERBOARD_MIN_HEIGHT}", written)
        self.assertNotIn("USER", written)

    def test_draw_leaderboard_scrolls_and_numbers_ranks_from_offset(self):
        app = VaccsRunningApp(
            ScrollLeaderboardClient(30), refresh_seconds=0, initial_view="leaderboard"
        )
        load_leaderboard(app)
        app.state.leaderboard_scroll = 10
        screen = FakeScreen(height=24, width=120)

        app._draw(screen)

        # 30 users overflow the pane body; scroll=10 makes rank 11 the top row.
        cells = {write[2].strip() for write in screen.writes}
        self.assertIn("u10", cells)  # first visible row (rank scroll+1 = 11)
        self.assertIn("11", cells)  # its rank, numbered from the scroll offset
        self.assertNotIn("u09", cells)  # rank 10, scrolled above the fold
        self.assertNotIn("u00", cells)  # rank 1, well above the fold

    def test_draw_leaderboard_clamps_scroll_so_last_row_is_reachable(self):
        app = VaccsRunningApp(
            ScrollLeaderboardClient(30), refresh_seconds=0, initial_view="leaderboard"
        )
        load_leaderboard(app)
        # KEY_END parks the scroll at a sentinel; the draw must clamp it so the
        # last row stays reachable.
        app._handle_leaderboard_key(curses.KEY_END)
        screen = FakeScreen(height=24, width=120)

        app._draw(screen)

        self.assertLess(app.state.leaderboard_scroll, 10 ** 9)  # sentinel clamped
        cells = {write[2].strip() for write in screen.writes}
        self.assertIn("u29", cells)  # the last user (rank 30) is on screen
        self.assertIn("30", cells)  # its rank

    def test_paging_keys_move_the_scroll_offset(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="leaderboard")

        self.assertTrue(app._handle_leaderboard_key(curses.KEY_NPAGE))
        self.assertEqual(app.state.leaderboard_scroll, LEADERBOARD_PAGE)
        self.assertTrue(app._handle_leaderboard_key(curses.KEY_PPAGE))
        self.assertEqual(app.state.leaderboard_scroll, 0)
        # PgUp never scrolls above the top.
        self.assertTrue(app._handle_leaderboard_key(curses.KEY_PPAGE))
        self.assertEqual(app.state.leaderboard_scroll, 0)

        app.state.leaderboard_scroll = 5
        self.assertTrue(app._handle_leaderboard_key(curses.KEY_HOME))
        self.assertEqual(app.state.leaderboard_scroll, 0)
        self.assertTrue(app._handle_leaderboard_key(curses.KEY_END))
        self.assertEqual(app.state.leaderboard_scroll, 10 ** 9)

    def test_wide_ranks_are_not_truncated(self):
        app = VaccsRunningApp(
            ScrollLeaderboardClient(1200), refresh_seconds=0, initial_view="leaderboard"
        )
        load_leaderboard(app)
        app._handle_leaderboard_key(curses.KEY_END)
        screen = FakeScreen(height=24, width=120)

        app._draw(screen)

        # The 1200th rank must render in full, not truncated to "120".
        cells = {write[2].strip() for write in screen.writes}
        self.assertIn("1200", cells)
        self.assertIn("u1199", cells)

    def test_stale_generation_results_are_discarded(self):
        client = LeaderboardClient()
        app = VaccsRunningApp(client, refresh_seconds=0, initial_view="leaderboard")
        load_leaderboard(app)

        # Simulate a newer 'r' press superseding the running generation.
        with app._lb_lock:
            app._lb_generation += 1
            app._lb_windows["24h"] = {"status": "loading", "usage": [], "error": ""}
            app._lb_fairshare = {}
            app._lb_level_fairshare = {}
            app._lb_default_accounts = {}
        stale = app._lb_generation - 1

        # A leftover thread from the old generation must not clobber the new one.
        app._fetch_leaderboard_window(stale, "24h")
        app._fetch_leaderboard_fairshare(stale)
        with app._lb_lock:
            self.assertEqual(app._lb_windows["24h"]["status"], "loading")
            self.assertEqual(app._lb_fairshare, {})
            self.assertEqual(app._lb_level_fairshare, {})
            self.assertEqual(app._lb_default_accounts, {})

        # A current-generation result is accepted.
        app._fetch_leaderboard_window(app._lb_generation, "24h")
        app._fetch_leaderboard_fairshare(app._lb_generation)
        with app._lb_lock:
            self.assertEqual(app._lb_windows["24h"]["status"], "ready")
            self.assertTrue(app._lb_fairshare)
            self.assertTrue(app._lb_level_fairshare)
            self.assertTrue(app._lb_default_accounts)

    def _open_find(self, client=None):
        app = VaccsRunningApp(
            client or FindLeaderboardClient(),
            refresh_seconds=0,
            initial_view="leaderboard",
        )
        load_leaderboard(app)
        self.assertTrue(app._handle_leaderboard_key(ord("f")))
        self.assertTrue(app.state.leaderboard_filter_editing)
        return app

    def test_find_filters_rows_by_name_as_you_type(self):
        app = self._open_find()
        for ch in "der":
            self.assertTrue(app._handle_key(None, ord(ch)))
        self.assertEqual(app.state.leaderboard_filter, "der")
        names = [row.name for _rank, row in app._leaderboard_snapshot()["24h"]["rows"]]
        self.assertEqual(names, ["derek"])  # "dgezgin" has no "der" substring

    def test_find_backspace_broadens_the_match(self):
        app = self._open_find()
        for ch in "der":
            app._handle_key(None, ord(ch))
        app._handle_key(None, curses.KEY_BACKSPACE)  # -> "de"
        app._handle_key(None, 127)  # -> "d"
        self.assertEqual(app.state.leaderboard_filter, "d")
        names = sorted(
            row.name for _rank, row in app._leaderboard_snapshot()["24h"]["rows"]
        )
        self.assertEqual(names, ["derek", "dgezgin"])

    def test_find_is_case_insensitive(self):
        app = self._open_find()
        for ch in "ALICE":
            app._handle_key(None, ord(ch))
        names = [row.name for _rank, row in app._leaderboard_snapshot()["24h"]["rows"]]
        self.assertEqual(names, ["alice"])

    def test_typing_in_find_captures_tab_and_quit_keys(self):
        app = self._open_find()
        # 'q' must be typed into the query, not quit the app.
        self.assertTrue(app._handle_key(None, ord("q")))
        self.assertEqual(app.state.leaderboard_filter, "q")
        # A tab key ('j') is typed, not a view switch.
        self.assertTrue(app._handle_key(None, ord("j")))
        self.assertEqual(app.state.view, "leaderboard")
        self.assertEqual(app.state.leaderboard_filter, "qj")

    def test_find_enter_keeps_filter_and_esc_clears_it(self):
        app = self._open_find()
        for ch in "al":
            app._handle_key(None, ord(ch))
        app._handle_key(None, ord("\n"))  # Enter confirms
        self.assertFalse(app.state.leaderboard_filter_editing)
        self.assertEqual(app.state.leaderboard_filter, "al")
        # With the box closed, 'q' quits again.
        self.assertFalse(app._handle_key(None, ord("q")))
        # Re-open and Esc clears the filter entirely.
        app._handle_leaderboard_key(ord("f"))
        app._handle_key(None, 27)
        self.assertFalse(app.state.leaderboard_filter_editing)
        self.assertEqual(app.state.leaderboard_filter, "")

    def test_scrolling_still_works_while_the_find_box_is_open(self):
        app = self._open_find(ScrollLeaderboardClient(30))
        app._handle_key(None, curses.KEY_NPAGE)
        self.assertEqual(app.state.leaderboard_scroll, LEADERBOARD_PAGE)
        self.assertTrue(app.state.leaderboard_filter_editing)  # still typing

    def test_draw_shows_find_query_and_no_match_message(self):
        app = VaccsRunningApp(
            FindLeaderboardClient(), refresh_seconds=0, initial_view="leaderboard"
        )
        load_leaderboard(app)
        app.state.leaderboard_filter_editing = True
        app.state.leaderboard_filter = "zzz"
        screen = FakeScreen(height=40, width=140)

        app._draw(screen)

        written = rendered_text(screen)
        self.assertIn("f find: zzz", written)  # query echoed in the header
        self.assertIn('no match for "zzz"', written)

    def test_draw_only_shows_matching_rows(self):
        app = VaccsRunningApp(
            FindLeaderboardClient(), refresh_seconds=0, initial_view="leaderboard"
        )
        load_leaderboard(app)
        app.state.leaderboard_filter = "derek"
        screen = FakeScreen(height=40, width=140)

        app._draw(screen)

        written = rendered_text(screen)
        self.assertIn("derek", written)
        self.assertNotIn("alice", written)
        self.assertNotIn("dgezgin", written)

    def test_find_preserves_the_original_rank(self):
        # u00..u29 rank 1..30 by GPU; u25 sits at rank 26 in the full list.
        app = VaccsRunningApp(
            ScrollLeaderboardClient(30), refresh_seconds=0, initial_view="leaderboard"
        )
        load_leaderboard(app)
        app.state.leaderboard_filter = "u25"

        ranked = app._leaderboard_snapshot()["24h"]["rows"]
        self.assertEqual([(rank, row.name) for rank, row in ranked], [(26, "u25")])

        # It renders with its overall rank (26), not re-numbered to 1.
        screen = FakeScreen(height=24, width=120)
        app._draw(screen)
        cells = {write[2].strip() for write in screen.writes}
        self.assertIn("u25", cells)
        self.assertIn("26", cells)

    def test_wide_preserved_rank_is_not_truncated(self):
        # Filtering to a single deep-ranked user must still size the rank column.
        app = VaccsRunningApp(
            ScrollLeaderboardClient(1200), refresh_seconds=0, initial_view="leaderboard"
        )
        load_leaderboard(app)
        app.state.leaderboard_filter = "u1150"  # rank 1151
        screen = FakeScreen(height=24, width=120)

        app._draw(screen)

        cells = {write[2].strip() for write in screen.writes}
        self.assertIn("u1150", cells)
        self.assertIn("1151", cells)

    def test_quit_hint_is_right_aligned_on_every_view(self):
        for view in ("jobs", "nodes", "history", "leaderboard"):
            app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view=view)
            screen = FakeScreen(height=40, width=140)
            app._draw_header(screen, 140)
            quit_writes = [w for w in screen.writes if w[2] == "q quit"]
            self.assertEqual(len(quit_writes), 1, f"{view}: expected exactly one quit hint")
            y, x, _text, _attr = quit_writes[0]
            self.assertEqual(y, 3, f"{view}: quit hint on the controls row")
            # Right-aligned: sits in the right half, near the edge.
            self.assertGreater(x, 140 // 2, f"{view}: quit hint should be right-aligned")
            self.assertLessEqual(x + len("q quit"), 140)


def _info_text(rows):
    """Flatten the segmented info rows into plain-text lines."""
    return ["".join(text for text, _style in row) for row in rows]


class UserInfoTests(unittest.TestCase):
    READY = {
        "status": "ready",
        "default": "pi-smith",
        "fairshare": {"pi-smith": 0.742, "pi-jones": 0.081},
        "accounts_error": "",
        "windows": {
            "24h": (0, 0),
            "7d": (90, 20),
            "30d": (35120, 8540),
            "1y": (582137, 142830),
        },
        "efficiency": {
            "7d": EfficiencySummary(
                1410, 27.94, 7.62, 2.9, "last 7 days",
                cpu_alloc=4.0, cpu_used=1.05,
                mem_req_bytes=96 * 1024 ** 3, mem_used_bytes=7.31 * 1024 ** 3,
                walltime_limit_sec=2160 * 60, walltime_used_sec=3813,
            ),
            "30d": EfficiencySummary(6222, 30.0, 9.0, 12.0, "last 30 days"),
            "1y": None,  # still streaming in
        },
        "gpfs": GpfsQuota(
            primary_group="pi-smith",
            group_space=[("gpfs1", "17.58T", "20T", "25T"), ("gpfs2", "1T", "35T", "45T")],
            group_files=[
                ("gpfs1", "6495522", "6291456", "12582912"),
                ("gpfs2", "4000000", "8000000", "16000000"),
            ],
            personal_space=[("gpfs1", "6.897T"), ("gpfs2", "32K")],
            personal_files=[("gpfs1", "1523523"), ("gpfs2", "17")],
        ),
        "gpfs_error": "",
    }

    def test_fairshare_style_maps_score_to_priority_band(self):
        self.assertEqual(fairshare_style(0.9), ("good", "high priority"))
        self.assertEqual(fairshare_style(0.3), ("warn", "normal"))
        self.assertEqual(fairshare_style(0.05), ("bad", "low priority"))
        self.assertEqual(fairshare_style(None), ("muted", ""))

    def test_build_user_info_lines_renders_the_full_screen(self):
        lines = _info_text(build_user_info_lines("dgezgin", self.READY))
        blob = "\n".join(lines)

        self.assertIn("dgezgin", blob)
        # Primary account is shown in the header and first in the fairshare list.
        self.assertIn("pi-smith", blob)
        self.assertIn("primary", blob)
        self.assertLess(blob.index("pi-smith"), blob.index("pi-jones"))
        self.assertIn("0.742", blob)
        self.assertIn("low priority", blob)
        # All four windows with EXACT, comma-grouped hour counts (no bars).
        self.assertIn("last 24 hours", blob)
        self.assertIn("last year", blob)
        self.assertIn("582,137", blob)
        self.assertIn("142,830", blob)
        self.assertNotIn("|", blob)  # no progress bars anywhere
        # No VACC command legend.
        self.assertNotIn("my_compute_usage", blob)
        self.assertNotIn("user_tools", blob)
        # Job efficiency table: raw percentages + job counts per window, and the
        # windows that are still loading show a spinner. No invented tiers/tips.
        self.assertIn("job efficiency", blob)
        self.assertIn("28%", blob)  # 7d CPU 27.94 -> 28
        self.assertIn("1,410", blob)  # 7d job count
        self.assertIn("6,222", blob)  # 30d job count
        self.assertIn("loading", blob)  # 1y window still streaming
        self.assertNotIn("wasteful", blob)
        self.assertNotIn("tip", blob)
        self.assertNotIn("--mem", blob)
        # Raw "requested X but used Y" averages for the first window with data.
        self.assertIn("requested 4.0 CPU cores but used 1.1", blob)
        self.assertIn("requested 96G of memory but used 7.3G", blob)
        self.assertIn("walltime but used", blob)
        # GPFS storage: group space and file quotas (with remaining counts),
        # plus personal usage.
        self.assertIn("storage", blob)
        self.assertIn("17.58T / 20T", blob)
        self.assertIn("(88%)", blob)
        self.assertIn("6,495,522 / 6,291,456", blob)
        self.assertIn("(103%)", blob)
        self.assertIn("204,066 over soft", blob)
        self.assertIn("6,087,390 hard left", blob)
        self.assertIn("4,000,000 soft left", blob)
        self.assertIn("12,000,000 hard left", blob)
        self.assertIn("your usage", blob)
        self.assertIn("1,523,523 files", blob)

    def test_efficiency_rows_are_raw_numbers_with_no_tiers(self):
        # Percentages are shown plainly (no verdict words), and no metric gets a
        # good/warn/bad "grade" style — only muted/heading/cpu/gpu appear.
        rows = build_user_info_lines("dgezgin", self.READY)
        eff_rows = [
            row
            for row in rows
            if any("last 7 days" in text for text, _ in row)
            and any("%" in text for text, _ in row)
        ]
        self.assertTrue(eff_rows)
        styles = {style for row in eff_rows for _text, style in row}
        self.assertFalse(styles & {"good", "warn", "bad"})

    def test_efficiency_windows_stream_in_independently(self):
        snapshot = {
            "status": "ready",
            "default": "",
            "fairshare": {},
            "windows": {},
            "efficiency": {
                "7d": EfficiencySummary(3, 50.0, 25.0, 40.0, "last 7 days"),
                "30d": "error",
                "1y": None,
            },
            "gpfs": None,
            "gpfs_error": "x",
        }
        blob = "\n".join(_info_text(build_user_info_lines("dgezgin", snapshot, "/")))
        self.assertIn("last 7 days", blob)
        self.assertIn("50%", blob)          # 7d ready
        self.assertIn("loading", blob)      # 1y still loading
        # 30d failed + storage failed -> two "unavailable".
        self.assertEqual(blob.count("unavailable"), 2)

    def test_efficiency_zero_jobs_says_so(self):
        snapshot = {
            "status": "ready",
            "default": "",
            "fairshare": {},
            "windows": {},
            "efficiency": {
                "7d": EfficiencySummary(0, None, None, None, "last 7 days"),
                "30d": EfficiencySummary(0, None, None, None, "last 30 days"),
                "1y": EfficiencySummary(0, None, None, None, "last year"),
            },
            "gpfs": None,
            "gpfs_error": "",
        }
        blob = "\n".join(_info_text(build_user_info_lines("dgezgin", snapshot)))
        self.assertIn("no finished jobs", blob)

    def test_build_user_info_lines_shows_loading_then_stops(self):
        rows = build_user_info_lines(
            "dgezgin", {"status": "loading"}, spinner="/"
        )
        blob = "\n".join(_info_text(rows))
        self.assertIn("dgezgin", blob)
        self.assertIn("loading your VACC info", blob)
        # Nothing else renders until data arrives.
        self.assertNotIn("fairshare", blob)

    def test_build_user_info_lines_shows_unavailable_sections(self):
        snapshot = {
            "status": "ready",
            "default": "",
            "fairshare": {},
            "accounts_error": "sshare boom",
            "windows": {"24h": "error", "7d": (1, 0), "30d": (2, 0), "1y": (3, 0)},
            "gpfs": None,
            "gpfs_error": "no my_gpfs_quota",
        }
        blob = "\n".join(_info_text(build_user_info_lines("dgezgin", snapshot)))
        # Fairshare, the 24h window, and storage each degrade independently.
        self.assertEqual(blob.count("unavailable"), 3)

    def test_i_switches_to_the_info_tab(self):
        for view in ("jobs", "nodes", "history", "leaderboard"):
            app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view=view)

            self.assertTrue(app._handle_key(None, ord("i")))
            self.assertEqual(app.state.view, "info")

    def test_i_edits_the_filter_while_the_find_box_is_open(self):
        app = VaccsRunningApp(
            FakeClient(), refresh_seconds=0, initial_view="leaderboard"
        )
        app.state.leaderboard_filter_editing = True

        self.assertTrue(app._handle_key(None, ord("i")))

        # Typing 'i' filters instead of switching tabs.
        self.assertEqual(app.state.view, "leaderboard")
        self.assertEqual(app.state.leaderboard_filter, "i")

    def test_switching_to_info_starts_the_loader(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="jobs")
        app._handle_key(None, ord("i"))
        self.assertTrue(app._info_started)
        self.assertIn(app._info_snapshot()["status"], {"loading", "ready"})

    def test_fetch_info_populates_a_ready_snapshot(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="jobs")
        app._info_generation = 1
        app._info_data = {"status": "loading", "efficiency": {"7d": None}}

        app._fetch_info_base(1)  # run the base fetch body synchronously
        app._fetch_info_efficiency(1, "7d", "now-7days", "last 7 days")

        snapshot = app._info_snapshot()
        self.assertEqual(snapshot["status"], "ready")
        self.assertEqual(snapshot["default"], "pi-test")
        self.assertEqual(snapshot["windows"]["1y"], (10, 2))
        self.assertEqual(snapshot["gpfs"].primary_group, "pi-test")
        self.assertEqual(snapshot["efficiency"]["7d"].job_count, 5)

    def test_r_refreshes_info_and_arrow_keys_move_the_offset(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="info")
        app.state.info_scroll = 0

        self.assertTrue(app._handle_info_key(curses.KEY_DOWN))
        self.assertEqual(app.state.info_scroll, 1)
        self.assertTrue(app._handle_info_key(curses.KEY_UP))
        self.assertEqual(app.state.info_scroll, 0)
        self.assertTrue(app._handle_info_key(ord("r")))
        # 'r' triggers a refresh; the message reflects it.
        self.assertIn("info", app.state.message)
        # Tab-switch keys are NOT consumed by the info handler.
        self.assertFalse(app._handle_info_key(ord("j")))
        self.assertFalse(app._handle_info_key(ord("n")))

    def test_j_switches_from_info_to_jobs_before_info_handler(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="info")

        with mock.patch.object(app, "_handle_info_key") as info_handler:
            self.assertTrue(app._handle_key(None, ord("j")))

        self.assertEqual(app.state.view, "jobs")
        info_handler.assert_not_called()

    def test_header_shows_info_tab(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="info")
        screen = FakeScreen(height=40, width=120)

        app._draw_header(screen, 120)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(" i Info ", written)
        self.assertIn(" r refresh ", written)

    def test_draw_info_writes_the_screen_without_error(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="info")
        app._info_data = dict(self.READY)
        screen = FakeScreen(height=40, width=120)

        app._draw_info(screen, 40, 120)

        written = "\n".join(write[2] for write in screen.writes)
        self.assertIn("tester", written)  # username comes from the client
        self.assertIn("582,137", written)
        self.assertIn("17.58T / 20T", written)


class HistoryEfficiencyTests(unittest.TestCase):
    def _app_with_selection(self, job_id="4566789", name="myjob"):
        from types import SimpleNamespace

        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="history")
        app._selected_history_group = lambda: SimpleNamespace(
            array_parent=job_id, name=name
        )
        return app

    def test_e_opens_efficiency_popup_for_selected_job(self):
        app = self._app_with_selection()
        calls = []
        app._popup = (
            lambda stdscr, title, text, close_keys=(), refresh_while_open=True: calls.append(
                (title, text() if callable(text) else text, close_keys, refresh_while_open)
            )
        )

        self.assertTrue(app._handle_key(None, ord("e")))

        self.assertEqual(len(calls), 1)
        title, text, close_keys, refresh = calls[0]
        self.assertIn("4566789", title)
        self.assertEqual(close_keys, (ord("e"),))
        self.assertFalse(refresh)  # computed once, no auto-refresh
        # The report references the job and shows the raw used/allocated figures.
        self.assertIn("4566789", text)
        self.assertIn("myjob", text)
        self.assertIn("used 2.0 of 4.0 cores", text)

    def test_e_does_nothing_without_a_selection(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="history")
        app._selected_history_group = lambda: None
        calls = []
        app._popup = lambda *a, **k: calls.append(a)

        self.assertTrue(app._handle_key(None, ord("e")))
        self.assertEqual(calls, [])

    def test_e_is_history_only(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="jobs")
        opened = []
        app._show_job_efficiency = lambda stdscr: opened.append(True)

        self.assertTrue(app._handle_key(None, ord("e")))
        self.assertEqual(opened, [])  # not triggered outside history

    def test_history_header_shows_efficiency_shortcut(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="history")
        screen = FakeScreen(height=40, width=120)

        app._draw_header(screen, 120)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(" e efficiency ", written)


class PriorityQueueViewTests(unittest.TestCase):
    def _snapshot(self):
        factors = PriorityFactors(
            priority=100,
            site=0,
            age=18,
            association=0,
            fairshare=51,
            job_size=4,
            partition=0,
            qos=0,
            tres="cpu=1,mem=1,gres/gpu=25",
            nice=0,
        )
        return make_priority_snapshot(
            make_priority_job(
                "900_1", user="alice", account="pi-a", priority=300
            ),
            make_priority_job(
                "901", user="bob", account="pi-b", priority=200
            ),
            make_priority_job(
                "902", reason="Resources", priority=100, factors=factors
            ),
        )

    def test_w_is_a_global_priority_shortcut(self):
        for initial_view in ("jobs", "nodes", "history", "info"):
            app = VaccsRunningApp(
                FakeClient(), refresh_seconds=0, initial_view=initial_view
            )

            self.assertTrue(app._handle_key(None, ord("w")))
            self.assertEqual(app.state.view, "priority")

    def test_priority_header_is_visible_and_does_not_overlap_clock_at_width_70(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        screen = FakeScreen(height=20, width=70)

        app._draw_header(screen, 70)

        self.assertTrue(any(write[2] == " w Priority " for write in screen.writes))
        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(" e extend ", written)
        self.assertIn(" f filter ", written)
        self.assertIn(" g gpu-queue ", written)
        self.assertNotIn("explain", written)
        # The optional clock normally starts at column 60; narrow layouts omit it.
        self.assertFalse(any(write[0] == 1 and write[1] == 60 for write in screen.writes))

    def test_priority_uses_a_slow_refresh_floor_and_zero_still_disables(self):
        app = VaccsRunningApp(
            FakeClient(), refresh_seconds=2, initial_view="priority"
        )
        self.assertEqual(app._active_refresh_seconds(), PRIORITY_REFRESH_SECONDS)

        slower = VaccsRunningApp(
            FakeClient(), refresh_seconds=60, initial_view="priority"
        )
        self.assertEqual(slower._active_refresh_seconds(), 60)

        disabled = VaccsRunningApp(
            FakeClient(), refresh_seconds=0, initial_view="priority"
        )
        self.assertEqual(disabled._active_refresh_seconds(), 0)

    def test_g_toggles_gpu_work_queues_in_packed_and_extended_views(self):
        snapshot = make_priority_snapshot(
            make_priority_job("900", partition="general", user="alice"),
            make_priority_job("800", partition="nvgpu", user="bob"),
            make_priority_job("700", partition="gpu-debug", user="carol"),
            make_priority_job("600", partition="gpu-preempt", user="dave"),
        )
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot
        app.state.priority_partitions = {"nvgpu"}
        app.state.selected = 2
        app.state.scroll = 1

        self.assertTrue(app._handle_key(None, ord("g")))

        self.assertEqual(
            app.state.priority_partitions,
            {"nvgpu", "gpu-preempt"},
        )
        self.assertEqual(
            set(PRIORITY_GPU_PARTITIONS),
            {"nvgpu", "gpu-preempt"},
        )
        self.assertEqual(app.state.selected, 0)
        self.assertEqual(app.state.scroll, 0)
        self.assertEqual(app.state.message, "GPU partition filter on")
        self.assertEqual(
            {entry.job.partition for entry in app._visible_priority_entries()},
            set(PRIORITY_GPU_PARTITIONS),
        )

        app.state.priority_extended = True
        self.assertEqual(
            {entry.job.partition for entry in app._visible_priority_entries()},
            set(PRIORITY_GPU_PARTITIONS),
        )

        self.assertTrue(app._handle_key(None, ord("g")))

        self.assertEqual(app.state.priority_partitions, set())
        self.assertEqual(app.state.message, "GPU partition filter off")
        self.assertEqual(app._visible_count(), 4)

    def test_priority_gpu_control_is_highlighted_and_clickable(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app._pair = lambda pair: pair
        inactive_screen = FakeScreen(height=20, width=100)
        app._draw_header(inactive_screen, inactive_screen.width)
        inactive_attr = next(
            write[3]
            for write in inactive_screen.writes
            if write[2] == " g gpu-queue "
        )

        gpu_x = (
            1
            + len(" e extend ")
            + 1
            + len(" f filter ")
            + 1
            + 2
        )
        click = (0, gpu_x, 3, 0, curses.BUTTON1_CLICKED)
        with mock.patch("vaccs_running.ui.keys.safe_getmouse", return_value=click):
            self.assertTrue(app._handle_key(inactive_screen, curses.KEY_MOUSE))

        self.assertEqual(app.state.priority_partitions, set(PRIORITY_GPU_PARTITIONS))
        active_screen = FakeScreen(height=20, width=100)
        app._draw_header(active_screen, active_screen.width)
        active_attr = next(
            write[3]
            for write in active_screen.writes
            if write[2] == " g gpu-queue "
        )
        self.assertNotEqual(inactive_attr, active_attr)

    def test_priority_table_and_detail_answer_rank_who_and_why(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = self._snapshot()
        app.state.selected = 2
        screen = FakeScreen(height=42, width=140)

        app._draw_priority(screen, screen.height, screen.width)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("Packed: 3 rank runs / 3 queue entries", written)
        self.assertIn("alice", written)
        self.assertIn("bob", written)
        self.assertIn("YOU", written)
        self.assertIn("3/3", written)
        self.assertIn("Resources", written)
        self.assertIn("users ahead in snapshot: alice/pi-a×1, bob/pi-b×1", written)
        self.assertIn("why waiting: Resources", written)
        self.assertIn("fair-share=51", written)
        self.assertIn("GPUs=1", written)
        self.assertIn("CPUs=4", written)
        self.assertIn("RAM=16G", written)
        self.assertIn("walltime=1d", written)
        self.assertIn("backfill", written)

    def test_priority_detail_uses_corrected_order_for_users_ahead(self):
        snapshot = make_priority_snapshot(
            make_priority_job(
                "100", user="low", account="pi-low", priority=100
            ),
            make_priority_job("200", account="pi-test", priority=500),
            make_priority_job(
                "300", user="high", account="pi-high", priority=900
            ),
        )
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot
        app.state.selected = 1
        screen = FakeScreen(height=42, width=140)

        app._draw_priority_detail(screen, screen.height, screen.width)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("users ahead in snapshot: high/pi-high×1", written)
        self.assertNotIn("users ahead in snapshot: low/pi-low×1", written)

    def test_narrow_priority_table_keeps_core_columns(self):
        snapshot = self._snapshot()
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot
        screen = FakeScreen(height=40, width=70)

        app._draw_priority_table(
            screen,
            app._visible_priority_entries(),
            snapshot,
            screen.height,
            screen.width,
        )

        headers = {write[2].strip() for write in screen.writes if write[0] == 6}
        self.assertTrue(
            {
                "YOU",
                "JOBS",
                "TASKS",
                "USER",
                "PARTITION",
                "RANK",
                "GPUS",
                "CPUS",
                "RAM",
                "WALLTIME",
            }
            .issubset(headers)
        )
        self.assertNotIn("JOB", headers)
        self.assertNotIn("AHEAD", headers)
        self.assertNotIn("USERS", headers)
        self.assertNotIn("EST START", headers)
        self.assertNotIn("WHY", headers)

    def test_narrow_priority_detail_still_shows_the_pending_reason(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = self._snapshot()
        app.state.selected = 2
        screen = FakeScreen(height=40, width=70)

        app._draw_priority_detail(screen, screen.height, screen.width)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("why waiting: Resources", written)
        self.assertIn("use or unavailable", written)

    def test_priority_table_shows_array_job_and_task_counts_with_a_rank_band(self):
        snapshot = make_priority_snapshot(
            make_priority_job("900", user="alice", priority=900),
            make_priority_job("100_2", priority=700),
            make_priority_job("100_3", priority=700),
            make_priority_job("100_4", priority=700),
        )
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot
        screen = FakeScreen(height=40, width=120)

        app._draw_priority_table(
            screen,
            app._visible_priority_entries(),
            snapshot,
            screen.height,
            screen.width,
        )

        group = app._visible_priority_entries()[1]
        specs = {
            label: value_fn
            for label, _minimum, _maximum, value_fn in responsive_priority_specs(
                300,
                current_user=snapshot.user,
            )
        }
        self.assertNotIn("SLOTS", specs)
        self.assertEqual(specs["JOBS"](group), "1")
        self.assertEqual(specs["TASKS"](group), "3")

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("2-4/4", written)
        self.assertIn("Packed: 2 rank runs / 4 queue entries", written)

    def test_narrow_extended_table_keeps_user_rank_and_reason(self):
        snapshot = self._snapshot()
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot
        app.state.priority_extended = True
        screen = FakeScreen(height=40, width=70)

        app._draw_priority_table(
            screen,
            app._visible_priority_entries(),
            snapshot,
            screen.height,
            screen.width,
        )

        headers = {write[2].strip() for write in screen.writes if write[0] == 6}
        self.assertTrue(
            {
                "YOU",
                "JOBID",
                "USER",
                "PARTITION",
                "RANK",
                "GPUS",
                "CPUS",
                "RAM",
                "WALLTIME",
            }
            .issubset(headers)
        )
        self.assertNotIn("ACCOUNT", headers)
        self.assertNotIn("JOB", headers)
        self.assertNotIn("EST START", headers)
        self.assertNotIn("WHY", headers)

    def test_packed_groups_every_users_arrays_and_extend_unpacks_them(self):
        snapshot = make_priority_snapshot(
            make_priority_job("900_1", user="alice", priority=900),
            make_priority_job("900_2", user="alice", priority=900),
            make_priority_job("800", user="bob", priority=800),
            make_priority_job("700_1", priority=700),
            make_priority_job("700_2", priority=700),
        )
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot

        self.assertEqual(app._visible_count(), 3)
        self.assertEqual(
            [entry.display_job_id for entry in app._visible_priority_entries()],
            ["900_[1-2]", "800", "700_[1-2]"],
        )

        app._handle_key(None, ord("e"))
        self.assertEqual(app._visible_count(), 5)
        self.assertEqual(
            [entry.job.job_id for entry in app._visible_priority_entries()[:2]],
            ["900_1", "900_2"],
        )

        app._handle_key(None, ord("e"))
        self.assertEqual(app._visible_count(), 3)
        self.assertEqual(app._selected_priority_entry().display_job_id, "900_[1-2]")

    def test_packed_groups_consecutive_same_user_jobs_and_extend_unpacks_them(self):
        snapshot = make_priority_snapshot(
            *[
                make_priority_job(
                    str(900 + index),
                    user="fsabokro",
                    name=f"job-{index}",
                    priority=900 - index,
                )
                for index in range(10)
            ],
            make_priority_job("800", user="bob", priority=700),
        )
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot

        self.assertEqual(app._visible_count(), 2)
        group = app._visible_priority_entries()[0]
        self.assertEqual(group.display_job_id, "10 jobs")
        self.assertEqual(group.priority_rank_text, "1-10 of 11")
        specs = {
            label: value_fn
            for label, _minimum, _maximum, value_fn in responsive_priority_specs(
                300,
                current_user=snapshot.user,
            )
        }
        self.assertEqual(specs["JOBS"](group), "10")
        self.assertEqual(specs["TASKS"](group), "10")

        screen = FakeScreen(height=42, width=140)
        app._draw_priority(screen, screen.height, screen.width)
        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("10 jobs", written)
        self.assertIn("fsabokro", written)
        self.assertIn("1-10/11", written)
        self.assertIn("selected rank run", written)
        self.assertIn(
            "packed same-user rank run: 10 jobs / 10 consecutive rank slots",
            written,
        )
        self.assertIn("requested across 10 slots: GPUs=10  CPUs=40  RAM=160G", written)
        self.assertIn("walltime/slot=1d", written)
        self.assertIn("press e", written)

        app._handle_key(None, ord("e"))

        self.assertEqual(app._visible_count(), 11)
        self.assertEqual(app._selected_priority_entry().job.job_id, "900")

    def test_packed_unranked_tasks_use_a_group_without_rank_or_ahead_counts(self):
        snapshot = make_priority_snapshot(
            make_priority_job("900", user="alice", priority=900),
            *[
                make_priority_job(
                    f"4877755_{task}",
                    user="achawla1",
                    name="decoder-8b-fsdp",
                    reason="DependencyNeverSatisfied",
                    priority=2243,
                )
                for task in range(3)
            ],
        )
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot
        app.state.selected = 1
        group = app._selected_priority_entry()

        self.assertIsNotNone(group)
        self.assertEqual(group.display_job_id, "4877755_[0-2]")
        specs = {
            label: value_fn
            for label, _minimum, _maximum, value_fn in responsive_priority_specs(
                300,
                current_user=snapshot.user,
            )
        }
        self.assertEqual(specs["AHEAD"](group), "—")
        self.assertEqual(specs["USERS"](group), "—")
        self.assertEqual(specs["JOBS"](group), "1")
        self.assertEqual(specs["TASKS"](group), "3")

        screen = FakeScreen(height=42, width=180)
        app._draw_priority(screen, screen.height, screen.width)
        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(
            "Packed: 1 rank run + 1 unranked group / 4 queue entries",
            written,
        )
        self.assertIn("selected unranked group", written)
        self.assertIn("entries=4877755_[0-2]", written)
        self.assertIn(
            "packed same-user unranked group: 1 job / 3 pending entries",
            written,
        )
        self.assertIn("no priority rank or ahead count", written)
        self.assertIn("requested across 3 entries", written)
        self.assertNotIn("3 consecutive rank slots", written)

    def test_packing_selects_the_exact_split_array_group(self):
        snapshot = make_priority_snapshot(
            make_priority_job("100_1", reason="Priority", priority=700),
            make_priority_job("100_2", reason="Dependency", priority=700),
        )
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot
        app.state.priority_extended = True
        app.state.selected = 1

        app._handle_key(None, ord("e"))

        self.assertFalse(app.state.priority_extended)
        self.assertEqual(app.state.selected, 1)
        self.assertEqual(app._selected_priority_entry().task_job_ids, ("100_2",))

    def test_packing_keeps_selection_in_the_same_reservation_queue(self):
        snapshot = make_priority_snapshot(
            make_priority_job("100", reservation="morning"),
            make_priority_job("100", reservation="evening"),
        )
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot
        app.state.priority_extended = True
        app.state.selected = 1

        app._handle_key(None, ord("e"))

        self.assertFalse(app.state.priority_extended)
        selected = app._selected_priority_entry()
        self.assertEqual(selected.job.normalized_reservation, "evening")

        screen = FakeScreen(height=40, width=140)
        app._draw_priority_detail(screen, screen.height, screen.width)
        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("nvgpu / reservation evening", written)

    def test_priority_partition_filter_applies_to_packed_and_extended_rows(self):
        snapshot = make_priority_snapshot(
            make_priority_job(
                "900_1", user="alice", partition="nvgpu", priority=900
            ),
            make_priority_job(
                "900_2", user="alice", partition="nvgpu", priority=900
            ),
            make_priority_job(
                "800", user="bob", partition="nvgpu", priority=800
            ),
            make_priority_job(
                "700", user="carol", partition="gpu-preempt", priority=700
            ),
        )
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot
        app.state.priority_partitions = {"nvgpu"}

        self.assertEqual(app._visible_count(), 2)
        self.assertTrue(
            all(entry.job.partition == "nvgpu" for entry in app._visible_priority_entries())
        )
        self.assertEqual(app._visible_priority_entries()[0].priority_rank_text, "1-2 of 3")

        app._handle_key(None, ord("e"))
        self.assertEqual(app._visible_count(), 3)
        self.assertTrue(
            all(entry.job.partition == "nvgpu" for entry in app._visible_priority_entries())
        )

        screen = FakeScreen(height=40, width=140)
        app._draw_priority(screen, screen.height, screen.width)
        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("Extended: 3 queue entries", written)
        self.assertNotIn("gpu-preempt", written)

    def test_priority_filter_has_partition_as_its_only_option(self):
        snapshot = make_priority_snapshot(
            make_priority_job("900", user="alice", partition="custom-gpu")
        )
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot

        items = app._priority_filter_home_items()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["action"], "partition")
        self.assertEqual(items[0]["label"], "Filter by partition: all")
        choices = app._priority_filter_choices()
        self.assertIn("nvgpu", choices.partitions)
        self.assertIn("custom-gpu", choices.partitions)

    def test_priority_partition_filter_is_independent_from_jobs_filter(self):
        client = JobsFilterClient()
        client.set_job_partition_filters({"general"})
        app = VaccsRunningApp(client, refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = self._snapshot()
        app.state.priority_partitions = {"nvgpu"}
        app.state.selected = 2
        app.state.scroll = 1
        choices = app._priority_filter_choices()

        app._activate_priority_partition_filter_item(
            FakePopupWindow(keys=[]),
            10,
            60,
            choices,
            {"kind": "partition", "value": "gpu-preempt"},
        )

        self.assertEqual(app.state.priority_partitions, {"nvgpu", "gpu-preempt"})
        self.assertEqual(client.job_partitions, {"general"})
        self.assertEqual(app.state.selected, 0)
        self.assertEqual(app.state.scroll, 0)

    def test_priority_header_shows_active_partition_filter(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_partitions = {"nvgpu"}
        screen = FakeScreen(height=20, width=120)

        app._draw_header(screen, screen.width)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn(" e extend ", written)
        self.assertIn(" f filter ", written)
        self.assertIn(" partition: nvgpu ", written)

    def test_f_opens_priority_filter_with_no_jobs_only_options(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(keys=[ord("q")])
        original_newwin = curses.newwin
        try:
            def fake_newwin(height, width, top, left):
                popup.height = height
                popup.width = width
                popup.positions.append((top, left))
                return popup

            curses.newwin = fake_newwin
            self.assertTrue(app._handle_key(screen, ord("f")))
        finally:
            curses.newwin = original_newwin

        written = " ".join(write[2] for write in popup.writes)
        self.assertIn("priority filter", written)
        self.assertIn("Filter by partition: all", written)
        self.assertNotIn("Filter by status", written)
        self.assertNotIn("Filter by user", written)
        self.assertNotIn("Filter by group", written)

    def test_empty_priority_partition_filter_names_the_partition(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = make_priority_snapshot(
            make_priority_job("900", user="alice", partition="general")
        )
        app.state.priority_partitions = {"nvgpu"}
        screen = FakeScreen(height=30, width=120)

        app._draw_priority(screen, screen.height, screen.width)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("No pending queue entries for partition nvgpu.", written)

    def test_clicking_priority_filter_opens_the_partition_only_menu(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        screen = FakeScreen(height=40, width=120)
        popup = FakePopupWindow(keys=[ord("q")])
        click = (0, 15, 3, 0, curses.BUTTON1_CLICKED)
        original_newwin = curses.newwin
        try:
            def fake_newwin(height, width, top, left):
                popup.height = height
                popup.width = width
                return popup

            curses.newwin = fake_newwin
            with mock.patch("vaccs_running.ui.keys.safe_getmouse", return_value=click):
                self.assertTrue(app._handle_key(screen, curses.KEY_MOUSE))
        finally:
            curses.newwin = original_newwin

        written = " ".join(write[2] for write in popup.writes)
        self.assertIn("priority filter", written)
        self.assertIn("Filter by partition: all", written)

    def test_unrankable_dependency_explains_that_priority_does_not_release_it(self):
        snapshot = make_priority_snapshot(
            make_priority_job("700", reason="Dependency", priority=999)
        )
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot
        screen = FakeScreen(height=40, width=120)

        app._draw_priority_detail(screen, screen.height, screen.width)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("No priority rank while Slurm reports Dependency", written)
        self.assertIn("job dependency has not completed", written)

    def test_priority_refresh_preserves_selected_job_route(self):
        first = make_priority_snapshot(
            make_priority_job("1", priority=200),
            make_priority_job("2", priority=100),
        )
        second = make_priority_snapshot(
            make_priority_job("2", priority=300),
            make_priority_job("1", priority=200),
        )

        class PriorityClient(FakeClient):
            def fetch_priority_queue(self):
                return second

        app = VaccsRunningApp(
            PriorityClient(), refresh_seconds=0, initial_view="priority"
        )
        app.state.priority_queue = first
        app.state.selected = 1

        app._refresh_priority()

        self.assertEqual(app.state.selected, 0)
        self.assertEqual(app._selected_priority_entry().job.job_id, "2")

    def test_priority_refresh_preserves_exact_extended_job(self):
        first = make_priority_snapshot(
            make_priority_job("other", user="alice", priority=300),
            make_priority_job("mine", priority=200),
        )
        second = make_priority_snapshot(
            make_priority_job("mine", priority=400),
            make_priority_job("other", user="alice", priority=300),
        )

        class PriorityClient(FakeClient):
            def fetch_priority_queue(self):
                return second

        app = VaccsRunningApp(
            PriorityClient(), refresh_seconds=0, initial_view="priority"
        )
        app.state.priority_queue = first
        app.state.priority_extended = True
        app.state.selected = 0

        app._refresh_priority()

        self.assertEqual(app.state.selected, 1)
        self.assertEqual(app._selected_priority_entry().job.job_id, "other")

    def test_priority_refresh_keeps_last_snapshot_on_scheduler_error(self):
        snapshot = self._snapshot()

        class BrokenPriorityClient(FakeClient):
            def fetch_priority_queue(self):
                raise SlurmError("controller busy")

        app = VaccsRunningApp(
            BrokenPriorityClient(), refresh_seconds=0, initial_view="priority"
        )
        app.state.priority_queue = snapshot

        message = app._refresh_priority()

        self.assertIs(app.state.priority_queue, snapshot)
        self.assertEqual(message, "priority: controller busy")

    def test_e_extends_to_other_jobs_and_packs_again(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = self._snapshot()
        app.state.selected = 2

        self.assertTrue(app._handle_key(None, ord("e")))

        self.assertTrue(app.state.priority_extended)
        self.assertEqual(app._visible_count(), 3)
        self.assertEqual(app.state.selected, 2)  # keeps the selected own job
        self.assertEqual(app.state.message, "priority queue extended")

        screen = FakeScreen(height=40, width=140)
        app._draw_priority(screen, screen.height, screen.width)
        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("Extended: 3 queue entries", written)
        self.assertIn("alice", written)
        self.assertIn("bob", written)
        self.assertIn("YOU", written)

        self.assertTrue(app._handle_key(None, ord("e")))
        self.assertFalse(app.state.priority_extended)
        self.assertEqual(app._visible_count(), 3)
        self.assertEqual(app.state.selected, 2)
        self.assertEqual(app.state.message, "priority queue packed")

    def test_empty_priority_snapshot_is_clear(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = FakeClient().fetch_priority_queue()
        screen = FakeScreen(height=30, width=100)

        app._draw_priority(screen, screen.height, screen.width)

        self.assertIn(
            "No pending queue entries in the cluster.",
            " ".join(write[2] for write in screen.writes),
        )

    def test_packed_and_extended_modes_show_cluster_jobs_when_user_has_none(self):
        snapshot = make_priority_snapshot(
            make_priority_job("other", user="alice", priority=300)
        )
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = snapshot
        self.assertEqual(app._visible_count(), 1)
        self.assertEqual(app._selected_priority_entry().job.user, "alice")

        app._handle_key(None, ord("e"))

        self.assertEqual(app._visible_count(), 1)
        self.assertEqual(app._selected_priority_entry().job.user, "alice")

    def test_other_users_do_not_claim_a_sprio_factor_breakdown(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = self._snapshot()
        app.state.priority_extended = True
        app.state.selected = 0
        screen = FakeScreen(height=40, width=140)

        app._draw_priority_detail(screen, screen.height, screen.width)

        written = " ".join(write[2] for write in screen.writes)
        self.assertIn("weighted sprio components are queried only for tester", written)
        self.assertNotIn("weighted sprio breakdown unavailable", written)

    def test_clicking_extend_toggles_the_priority_view(self):
        app = VaccsRunningApp(FakeClient(), refresh_seconds=0, initial_view="priority")
        app.state.priority_queue = self._snapshot()
        click = (0, 5, 3, 0, curses.BUTTON1_CLICKED)

        with mock.patch("vaccs_running.ui.keys.safe_getmouse", return_value=click):
            self.assertTrue(app._handle_key(None, curses.KEY_MOUSE))
            self.assertTrue(app.state.priority_extended)
            self.assertTrue(app._handle_key(None, curses.KEY_MOUSE))

        self.assertFalse(app.state.priority_extended)


if __name__ == "__main__":
    unittest.main()
