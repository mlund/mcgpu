# Metropolis Monte Carlo Simulation of Rigid Bodies

- Number of rigid bodies: ~ 100
- Atoms in each rigid body: ~1000
- Cuboidal box with periodic boundary conditions
- Atom-atom pair potential: Lennard-Jones + Yukawa / Screened coulomb
- Propagate using random rigid body rotation or rotation with Metropolis Monte Carlo algorithm
- Units: length = Å; energy = kJ/mol
- Use `wgpu` crate for GPU acceleration
- Use `nalgebra` for geometrical transformations

## Backend Performance Report

### Overview

The Monte Carlo simulator supports three energy calculation backends:
- **GPU (cached)**: Compute shader via wgpu with pairwise energy caching (default)
- **GPU (uncached)**: Compute shader without caching, for benchmarking
- **CPU/SIMD**: Parallel computation using SIMD (ARM NEON/x86 SSE) with Rayon and caching

The cached backends use **pairwise energy caching** to avoid redundant calculations.

### Benchmark Configuration

- **Hardware**: Apple Silicon M4 (10 CPU cores, integrated GPU)
- **Molecule**: 699 sites per molecule
- **Potential**: Lennard-Jones with 10 Å cutoff
- **Simulation**: Production runs (1000 steps) from equilibrated configurations

### Performance Results

| Molecules | Atoms | CPU (steps/s) | GPU (steps/s) | GPU Speedup |
|-----------|-------|---------------|---------------|-------------|
| 50 | 34,950 | 185.8 | 209.0 | 1.1x |
| 100 | 69,900 | 88.7 | 130.6 | 1.5x |
| 200 | 139,800 | 44.5 | 85.6 | 1.9x |

The GPU advantage increases with system size due to better parallelization of pairwise computations.

### Impact of Pairwise Caching

Both backends benefit significantly from caching:

#### CPU Backend
| Molecules | Before Caching | After Caching | Improvement |
|-----------|----------------|---------------|-------------|
| 50 | 97.0 steps/s | 185.8 steps/s | 1.9x |
| 100 | 47.7 steps/s | 88.7 steps/s | 1.9x |
| 200 | 23.6 steps/s | 44.5 steps/s | 1.9x |

#### GPU Backend
| Molecules | Uncached | Cached | Improvement |
|-----------|----------|--------|-------------|
| 100 | 35.4 steps/s | 130.6 steps/s | 3.7x |
| 200 | 17.1 steps/s | 85.6 steps/s | 5.0x |

### Caching Strategy

Both backends use identical caching logic:

#### Cache Structure
- **Pairwise matrix**: N×N array storing E_ij (energy between molecules i and j)
- **Molecule energies**: Cached row sums for O(1) lookup
- **Dirty flags**: Track which molecules need recomputation

#### Algorithm

1. **Initialization**: Compute full N×N pairwise matrix

2. **Per MC step**:
   - Get `e_old` from cache: **O(1) lookup**
   - Invalidate moved molecule's cache entry
   - Propose move, update positions
   - Get `e_new` by recomputing one matrix row: **O(N) pairwise calculations**
   - Accept/reject via Metropolis criterion
   - If rejected: invalidate cache, revert positions

3. **Cache update on acceptance**:
   - New row values update both row i and column i (symmetry: E_ij = E_ji)
   - All affected molecule energies updated incrementally

#### Complexity Reduction

| Operation | Without Cache | With Cache |
|-----------|---------------|------------|
| Get e_old | O(N × sites²) | O(1) |
| Get e_new | O(N × sites²) | O(N × sites²) |
| **Per step** | **2 × O(N × sites²)** | **O(N × sites²)** |

The cache eliminates redundant computation of e_old, halving the work per step.

### CPU Backend Implementation

The CPU backend (`src/cpu.rs`) uses:

1. **SIMD Vectorization**: 128-bit `f32x4` operations via the `wide` crate
   - Native support on both ARM NEON and x86 SSE
   - Processes 4 site pairs per SIMD instruction

2. **Parallel Iteration**: Rayon for multi-threaded computation
   - Parallelizes pairwise energy calculations across molecules
   - Scales with available CPU cores

3. **Pairwise Caching**: Same strategy as GPU backend

```
Thread scaling (100 molecules, without caching):
- 1 thread:  9.6 steps/s
- 4 threads: 35.7 steps/s (3.7x)
- 10 threads: 51.1 steps/s (5.3x)
```

### GPU Backend Implementation

The GPU backend (`src/gpu.rs`) uses:

1. **Compute Shaders**: wgpu with WGSL shaders
   - One workgroup per target molecule
   - Parallel reduction within workgroup

2. **Pairwise Caching**: Reduces GPU dispatches from 2 to ~1 per step
   - Eliminates synchronization overhead for e_old lookup

### Memory Usage

| Component | Size (100 molecules) |
|-----------|---------------------|
| Pairwise cache | 80 KB (N² × 8 bytes) |
| Position buffer (GPU) | 1.1 MB (N × 699 × 16 bytes) |
| Parameter buffer | 16.8 KB (699 × 24 bytes) |

### Recommendations

- **Small systems (< 50 molecules)**: CPU and GPU perform similarly
- **Large systems (> 100 molecules)**: GPU provides increasing speedup
- **Memory-constrained**: CPU backend uses less GPU memory
- **CPU thread count**: Use all available cores (`--threads 0`) for best performance

### Usage

```bash
# GPU backend with caching (default)
cargo run --release -- -n 10000 --n-molecules 100 --backend gpu

# GPU backend without caching (for comparison)
cargo run --release -- -n 10000 --n-molecules 100 --backend gpu-uncached

# CPU backend with caching
cargo run --release -- -n 10000 --n-molecules 100 --backend cpu

# CPU with specific thread count
cargo run --release -- -n 10000 --n-molecules 100 --backend cpu --threads 4
```
