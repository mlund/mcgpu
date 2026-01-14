# Backend Performance Report

## Overview

The Monte Carlo simulator supports two energy calculation backends:
- **CPU/SIMD**: Parallel computation using SIMD (ARM NEON/x86 SSE) with Rayon thread pool
- **GPU**: Compute shader via wgpu with pairwise energy caching

## Benchmark Configuration

- **Hardware**: Apple Silicon M4 (10 CPU cores, integrated GPU)
- **Molecule**: 699 sites per molecule
- **Potential**: Lennard-Jones with 10 Å cutoff
- **Simulation**: Production runs (1000 steps) from equilibrated configurations

## Performance Results

| Molecules | Atoms | CPU (steps/s) | GPU (steps/s) | GPU Speedup |
|-----------|-------|---------------|---------------|-------------|
| 50 | 34,950 | 97.0 | 209.0 | 2.2x |
| 100 | 69,900 | 47.7 | 118.7 | 2.5x |
| 200 | 139,800 | 23.6 | 82.1 | 3.5x |

The GPU advantage increases with system size due to better amortization of dispatch overhead.

## CPU Backend Implementation

The CPU backend (`src/cpu.rs`) uses:

1. **SIMD Vectorization**: 128-bit `f32x4` operations via the `wide` crate
   - Native support on both ARM NEON and x86 SSE
   - Processes 4 site pairs per SIMD instruction

2. **Parallel Iteration**: Rayon for multi-threaded computation
   - Parallelizes over sites within each molecule
   - Scales with available CPU cores

3. **On-demand Calculation**: No caching; energies computed fresh each call

```
Thread scaling (100 molecules):
- 1 thread:  9.6 steps/s
- 4 threads: 35.7 steps/s (3.7x)
- 10 threads: 51.1 steps/s (5.3x)
```

## GPU Backend Implementation

The GPU backend (`src/gpu.rs`) uses a **pairwise energy caching** strategy:

### Cache Structure
- **Pairwise matrix**: N×N array storing E_ij (energy between molecules i and j)
- **Molecule energies**: Cached row sums for O(1) lookup
- **Dirty flags**: Track which molecules need recomputation

### Algorithm

1. **Initialization**: Compute full N×N pairwise matrix (N GPU dispatches)

2. **Per MC step**:
   - Get `e_old` from cache (O(1) lookup)
   - Invalidate moved molecule's cache entry
   - Propose move, update positions
   - Get `e_new` by recomputing one matrix row (1 GPU dispatch)
   - Accept/reject via Metropolis criterion
   - If rejected: invalidate cache, revert positions

3. **Cache update on acceptance**:
   - New row values update both row i and column i (symmetry: E_ij = E_ji)
   - All affected molecule energies updated incrementally

### GPU Dispatch Efficiency

| Approach | GPU Dispatches per Step | Synchronizations |
|----------|------------------------|------------------|
| Naive (before) | 2 | 2 |
| Cached (after) | ~1 (only if dirty) | ~1 |

The caching reduced GPU synchronization overhead by computing N pairwise energies per dispatch instead of summing all interactions.

### Performance Improvement from Caching

With 100 molecules:
- **Before caching**: 35 steps/s
- **After caching**: 119 steps/s
- **Improvement**: 3.4x faster

## Memory Usage

| Component | Size (100 molecules) |
|-----------|---------------------|
| Pairwise cache | 80 KB (N² × 8 bytes) |
| Position buffer | 1.1 MB (N × 699 × 16 bytes) |
| Parameter buffer | 16.8 KB (699 × 24 bytes) |

## Recommendations

- **Small systems (< 50 molecules)**: CPU backend preferred due to lower overhead
- **Large systems (> 100 molecules)**: GPU backend provides significant speedup
- **CPU thread count**: Use all available cores (`--threads 0`) for best performance

## Usage

```bash
# CPU backend (default threads = all available)
cargo run --release -- -n 10000 --n-molecules 100 --backend cpu

# GPU backend
cargo run --release -- -n 10000 --n-molecules 100 --backend gpu

# CPU with specific thread count
cargo run --release -- -n 10000 --n-molecules 100 --backend cpu --threads 4
```
