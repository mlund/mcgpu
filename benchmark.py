#!/usr/bin/env python3
"""Benchmark script for MC simulator backends."""

import subprocess
import shutil
import sys
import re
from pathlib import Path

BACKENDS = ["cpu", "gpu-uncached", "gpu"]
MOLECULES = [100, 200, 400]


def run_benchmark(backend: str, n_molecules: int, steps: int) -> float:
    """Run benchmark and return steps/s."""
    restart_eq = Path(f"restart_{n_molecules}eq.dat")
    restart = Path("restart.dat")

    # Copy equilibrated restart
    shutil.copy(restart_eq, restart)

    # Run simulation
    cmd = [
        "cargo", "run", "--release", "--",
        "-n", str(steps),
        "--n-molecules", str(n_molecules),
        "-c",
        "--backend", backend
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Extract performance
    match = re.search(r"Performance:\s+([\d.]+)\s+steps/s", output)
    if match:
        return float(match.group(1))
    return 0.0


def create_restart_if_needed(n_molecules: int):
    """Create equilibrated restart file if it doesn't exist."""
    restart_eq = Path(f"restart_{n_molecules}eq.dat")
    if not restart_eq.exists():
        print(f"Creating equilibrated restart for {n_molecules} molecules...")
        cmd = [
            "cargo", "run", "--release", "--",
            "-n", "500",
            "--n-molecules", str(n_molecules),
            "--backend", "cpu"
        ]
        subprocess.run(cmd, capture_output=True)
        shutil.copy("restart.dat", restart_eq)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    print("=" * 60)
    print("MC Simulator Benchmark")
    print(f"Steps: {steps}")
    print("=" * 60)
    print()

    # Ensure restart files exist
    for n in MOLECULES:
        create_restart_if_needed(n)

    # Run benchmarks
    results = {}
    print("Running benchmarks...")
    print()

    for backend in BACKENDS:
        for n in MOLECULES:
            perf = run_benchmark(backend, n, steps)
            results[(backend, n)] = perf
            print(f"  {backend:14s} / {n:3d} mol: {perf:.1f} steps/s")

    # Print results table
    print()
    print("=" * 60)
    print("Results (steps/s)")
    print("=" * 60)
    print()
    print(f"{'Backend':<14s}", end="")
    for n in MOLECULES:
        print(f"{n:>12d} mol", end="")
    print()
    print("-" * 60)

    for backend in BACKENDS:
        print(f"{backend:<14s}", end="")
        for n in MOLECULES:
            print(f"{results[(backend, n)]:>12.1f}", end="")
        print()

    # GPU cached vs uncached
    print()
    print("=" * 60)
    print("Speedup: GPU cached vs uncached")
    print("=" * 60)
    print()
    print(f"{'Molecules':<14s}{'Uncached':>12s}{'Cached':>12s}{'Speedup':>12s}")
    print("-" * 60)

    for n in MOLECULES:
        uncached = results[("gpu-uncached", n)]
        cached = results[("gpu", n)]
        speedup = cached / uncached if uncached > 0 else 0
        print(f"{n:<14d}{uncached:>12.1f}{cached:>12.1f}{speedup:>11.1f}x")

    # GPU vs CPU
    print()
    print("=" * 60)
    print("Speedup: GPU cached vs CPU cached")
    print("=" * 60)
    print()
    print(f"{'Molecules':<14s}{'CPU':>12s}{'GPU':>12s}{'Speedup':>12s}")
    print("-" * 60)

    for n in MOLECULES:
        cpu = results[("cpu", n)]
        gpu = results[("gpu", n)]
        speedup = gpu / cpu if cpu > 0 else 0
        print(f"{n:<14d}{cpu:>12.1f}{gpu:>12.1f}{speedup:>11.1f}x")

    print()


if __name__ == "__main__":
    main()
