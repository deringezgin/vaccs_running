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

## Install As A Command

If you want `vaccs-running` on your path:

```bash
cd vaccs_running
python3 -m pip install --user .
vaccs-running
```
