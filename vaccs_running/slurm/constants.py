from __future__ import annotations

import datetime
import re


HISTORY_WINDOWS = {
    "1h": "now-1hours",
    "3h": "now-3hours",
    "24h": "now-24hours",
    "3d": "now-3days",
    "7d": "now-7days",
}


SQUEUE_FIELDS = [
    "job_id",
    "name",
    "state",
    "partition",
    "nodes",
    "reason",
    "elapsed",
    "limit",
    "node_count",
    "cpus",
    "gres",
    "submit_time",
    "start_time",
    "user",
    "group",
]


SQUEUE_FORMAT = "%i|%j|%T|%P|%N|%R|%M|%l|%D|%C|%b|%V|%S|%u|%g"


FILTER_CHOICES_FORMAT = "%u|%g|%P"


VACC_PARTITIONS = [
    "general",
    "short",
    "week",
    "nvgpu",
    "gpu-debug",
    "gpu-preempt",
    "hgnodes",
    "goldenmaple",
]


SACCT_FIELDS = [
    "job_id",
    "raw_job_id",
    "name",
    "state",
    "partition",
    "nodes",
    "elapsed",
    "limit",
    "node_count",
    "cpus",
    "tres",
    "submit_time",
    "start_time",
    "end_time",
    "exit_code",
]


SACCT_FORMAT = (
    "JobID,JobIDRaw,JobName,State,Partition,NodeList,Elapsed,Timelimit,"
    "NNodes,NCPUS,ReqTRES,Submit,Start,End,ExitCode"
)


NODE_JOBS_FIELDS = [
    "job_id",
    "user",
    "state",
    "elapsed",
    "cpus",
    "gres",
    "name",
]


NODE_JOBS_FORMAT = "%i|%u|%T|%M|%C|%b|%j"


DEFAULT_SQUEUE_STATES = "all"


SQUEUE_STATE_RE = re.compile(r"[A-Z_]+")


SREPORT_USAGE_FORMAT = "Login,Account,TresName,Used"


SREPORT_USAGE_REPORT = "AccountUtilizationByUser"


SSHARE_FAIRSHARE_FORMAT = "User,Account,FairShare"


USAGE_TRES = "cpu,gres/gpu"


GPU_TRES_NAMES = {"gres/gpu", "gpu"}


ROOT_ACCOUNT = "root"


LEADERBOARD_WINDOWS = [
    ("24h", "last 24 hours"),
    ("7d", "last 7 days"),
    ("30d", "last 30 days"),
]


LEADERBOARD_WINDOW_DELTAS = {
    "24h": datetime.timedelta(days=1),
    "7d": datetime.timedelta(days=7),
    "30d": datetime.timedelta(days=30),
    "1y": datetime.timedelta(days=365),
}


USER_INFO_WINDOWS = [
    ("24h", "last 24 hours"),
    ("7d", "last 7 days"),
    ("30d", "last 30 days"),
    ("1y", "last year"),
]


JOB_EFFICIENCY_WINDOW = "now-7days"


JOB_EFFICIENCY_WINDOW_LABEL = "last 7 days"


JOB_EFFICIENCY_FORMAT = (
    "JobID,State,AllocCPUS,TotalCPU,CPUTimeRAW,ElapsedRaw,"
    "TimelimitRaw,ReqMem,MaxRSS,NNodes"
)


JOB_EFFICIENCY_WINDOWS = [
    ("7d", "now-7days", "last 7 days"),
    ("30d", "now-30days", "last 30 days"),
    ("1y", "now-365days", "last year"),
]


LEADERBOARD_SORTS = ["gpu", "cpu", "fairshare"]


LEADERBOARD_SORT_LABELS = {
    "gpu": "GPU hours",
    "cpu": "CPU hours",
    "fairshare": "fairshare",
}


class SlurmError(RuntimeError):
    pass


_STORAGE_UNITS = {
    "": 1.0,
    "K": 1024.0,
    "M": 1024.0 ** 2,
    "G": 1024.0 ** 3,
    "T": 1024.0 ** 4,
    "P": 1024.0 ** 5,
}


FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}
