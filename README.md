<h1 align="center">VACC's Running?</h1>

<p align="center">
  <img alt="Supported Python versions" src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue">
  <a href="https://github.com/deringezgin/vaccs_running/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/deringezgin/vaccs_running/actions/workflows/ci.yml/badge.svg">
  </a>
</p>

A colorful terminal UI for checking your jobs on the Vermont Advanced Computing Cluster and viewing the node availability.

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

To prefilter the Jobs view by Slurm state, pass the state list through to
`squeue`:

```bash
./vaccs-running --state PD
./vaccs-running --states RUNNING,PENDING
./vaccs-running --user all --state PD
./vaccs-running --group pi-example
./vaccs-running -g pi-example --state RUNNING
```

Inside the Running view, press `s` to see the selected job's sbatch script.
Press `f` to change the live filter. The filter popup lets you click or press
Enter/Space to select multiple states, users, and groups. It lists users and
groups from currently running jobs. Custom user/group entry works as a
typeahead: typing narrows the choices, Enter selects the highlighted match, and
Enter adds the typed value for this session when no choices match. Use `c` to
clear the active filter. With no user/group selected, the Running view falls
back to your own user. In the status filter, Space/Enter toggles individual
statuses, and selecting no statuses means all statuses. When the Running view
can show jobs from multiple users, the table includes user and group columns.
If both user and group filters are selected, the group filter takes priority.

> ⚠️  As auto-refresh queries Slurm, please use an interval larger than 1 second.

## Install As A Command

If you want `vaccs-running` on your path:

```bash
cd vaccs_running
python3 -m pip install --user .
vaccs-running
```
