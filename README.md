<h1 align="center">VACC's Running?</h1>

<p align="center">
  <img alt="Supported Python versions" src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue">
  <a href="https://github.com/deringezgin/vaccs_running/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/deringezgin/vaccs_running/actions/workflows/ci.yml/badge.svg">
  </a>
</p>

A colorful terminal UI for checking your jobs on the Vermont Advanced Computing Cluster, viewing node availability, and browsing per-user cluster usage.

> This project is not affiliated in any way with UVM, VACC, or the Vermont Complex Systems Institute.

## Quick Start

Clone the repository:

```bash
git clone https://github.com/deringezgin/vaccs_running.git
cd vaccs_running
```

From this directory:

```bash
./vaccs-running
```

The TUI auto-refreshes the active view every 2 seconds by default. To change
that:

```bash
./vaccs-running --refresh 1
```

To prefilter the Jobs view by Slurm state, group, or partition:

```bash
./vaccs-running --state PD
./vaccs-running --states RUNNING,PENDING
./vaccs-running --user all --state PD
./vaccs-running --group pi-example
./vaccs-running -g pi-example --state RUNNING
./vaccs-running --partition nvgpu
./vaccs-running --partitions nvgpu,gpu-preempt
./vaccs-running --admin
```

> ⚠️  As auto-refresh queries Slurm, please use an interval larger than 1 second.

## Features

The TUI has five tabs, switched with a single key at any time:

| Key | Tab | What it shows |
| --- | --- | --- |
| `j` | **Jobs** | your Slurm jobs (running, pending, …) |
| `n` | **Nodes** | every compute node and how busy it is |
| `h` | **History** | your recently finished jobs |
| `u` | **Usage** | a per-user cluster-usage leaderboard |
| `i` | **Info** | your account, compute usage, and storage |

Keys shared by the list views:

- `↑` / `↓` — move the selection
- `PgUp` / `PgDn` — jump ten rows
- `←` / `→` — jump a full page
- `Home` / `End` — first / last row
- `q` or `Esc` — quit

Job and node state is color-coded (green = running/idle, yellow = pending/mixed,
cyan = completed/allocated, red = failed/down). Jobs, Nodes, and History
auto-refresh while you are on them (roughly on the `--refresh` interval); Usage
and Info are refresh-on-demand only.

<details>
<summary><strong>Jobs</strong> (<code>j</code>)</summary>

Lists your jobs — by default the ones that are running or pending. Columns:
JOBID, STATE, PARTITION, ELAPSED, LIMIT, CPUS, and WHERE/WHY (the node it landed
on, or the reason it is pending). Narrow terminals drop the least important
columns first.

- `g` — group array tasks into one row per array, with REQ/DONE/RUN/PEND/FAIL counts
- `f` — open the filter menu: filter by **status**, **user**, **group**, or **partition**
- `d` — pop up the full `scontrol show job` for the selected job

You can also preselect these filters at launch (see [Quick Start](#quick-start)):
`--user`/`-u`, `--group`/`-g`, `--partition`/`-p`, `--state`/`-s`, and `--admin`
(all users' running and pending jobs).

</details>

<details>
<summary><strong>Nodes</strong> (<code>n</code>)</summary>

Shows every node with live CPU, memory, and GPU allocation bars, plus its state,
partitions, and GRES. A GPU counts as free only when its node still has a spare
CPU core (an idle GPU on a fully-booked node cannot actually be scheduled).

- `g` — show only nodes that have GPUs
- `f` — show only nodes with a **free** GPU
- `d` — pop up the full `scontrol show node`
- `p` — peek at the jobs currently running on the node (`squeue -w`)
- `a` — pop up a cluster-wide "activity by user" summary, with free and allocated GPU totals

</details>

<details>
<summary><strong>History</strong> (<code>h</code>)</summary>

Your recently finished jobs (plus anything still active), grouped by array, with
REQ/DONE/RUN/PEND/FAIL counts, CPU/GPU totals, longest runtime, and time limit.

- `f` — cycle the time window: `1h → 3h → 24h → 3d → 7d`. Every window is listed
  in the header with the active one highlighted.
- `e` — pop up the efficiency for the selected job (or array): CPU, memory, and
  walltime used vs allocated, the same figures `seff` reports. Arrays are
  averaged across their tasks.

</details>

<details>
<summary><strong>Usage</strong> (<code>u</code>)</summary>

A per-user cluster-usage leaderboard, opened with `u` (or launched directly with
`--usage` / `-U`). It is split into three panes — **last 24 hours**, **last 7
days**, and **last 30 days** — each ranking users by GPU-hours, CPU-hours, and
their current Slurm fairshare score, alongside the PI group each user drew on
most.

Because the underlying `sreport` queries are heavy, this view does **not**
auto-refresh: it loads once when opened, each pane filling in as its query
returns, and you refresh on demand with `r`.

- `m` — **mode**: rank by individual **user** or by **group** / account
- `f` — **find**: filter the rows live by name as you type. `Enter` keeps the
  filter, `Esc` clears it. A match keeps its overall rank (find the 32nd user and
  they still show as #32)
- `s` — **sort**: cycle the ranking column (`GPU → CPU → fairshare`)
- `o` — **order**: toggle `ascending` / `descending`
- `r` — refresh all panes
- `↑` `↓` `PgUp` `PgDn` `Home` `End` — scroll the ranking in all panes together

The active option in each menu is highlighted in the header. The Usage view is a
wide, desktop-oriented layout; on a screen that is too small (for example a phone
terminal) it shows a "needs a bigger screen" notice instead of a broken layout,
and on narrow-but-valid widths it drops the fairshare, then the group, column to
keep the core numbers legible.

</details>

<details>
<summary><strong>Info</strong> (<code>i</code>)</summary>

A one-screen card for **your** account on the cluster:

- **Account** — your username and primary PI group
- **Fairshare** — your current Slurm fairshare score per account, labelled
  high / normal / low priority
- **Compute usage** — exact CPU-hours and GPU-hours over the **last 24 hours, 7
  days, 30 days, and 1 year** (per-user `sreport`, so even the year window is
  fast)
- **Storage** — your PI group's GPFS space quota per filesystem (with % used),
  plus your own space and file counts (from `my_gpfs_quota`)
- **Job efficiency** (shown last) — your average CPU, memory, and walltime
  efficiency (used vs allocated, the same figures `seff` reports) over the last
  **7 days, 30 days, and 1 year**, with the job count for each. Each window is
  queried separately and streams in as it returns (the year is the slow one).
  Below the table, the raw per-job averages spell it out — e.g. "requested 4.2
  CPU cores but used 1.0", the same for memory and walltime

It loads in the background when you open the tab and is refresh-on-demand with
`r`.

</details>

## Install As A Command

If you want `vaccs-running` on your path:

```bash
cd vaccs_running
python3 -m pip install --user .
vaccs-running
```
