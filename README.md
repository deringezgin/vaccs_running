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

## Usage

Press `u` (or start with `--usage` / `-U`) to open the **Usage** tab: a per-user
ranking of cluster usage split into three panes — last 24 hours, last 7 days,
and last 30 days — each showing CPU-hours, GPU-hours, and the current Slurm
fairshare score.

Because these `sreport` queries are heavy, this view does **not** auto-refresh.
It loads once when you open it and each pane fills in as its query returns;
press `r` to refresh on demand.

- `m` — switch mode between ranking by user and by group/account
- `f` — find: filter the rows live by name as you type (`Enter` keeps it, `Esc` clears); a
  matched row keeps its overall rank (find the 32nd user and they still show as 32)
- `s` — cycle the sort column (CPU / GPU / fairshare)
- `o` — toggle the order (ascending / descending)
- `r` — refresh all panes (loads once on open; refresh is manual)
- `↑ ↓ PgUp PgDn Home End` — scroll the ranking in all panes

The Usage view is a wide, desktop-oriented view; on a screen that is too small
(for example a phone terminal) it shows a notice asking for a bigger screen
instead of a broken layout.

## Install As A Command

If you want `vaccs-running` on your path:

```bash
cd vaccs_running
python3 -m pip install --user .
vaccs-running
```
