# Cassandra vs mcgpu Energy Caching Comparison

This document compares the pairwise energy caching approach used in [Cassandra](https://github.com/MaginnGroup/Cassandra) (Fortran) with the implementation in mcgpu (Rust), based on the supplementary material for "Hastier Monte Carlo with cached state energies" (Sukeník, Lund, Vácha, 2026).

## Overview

Both implementations cache an N×N pairwise energy matrix to avoid O(N) energy recalculations per Monte Carlo move. The key difference is in how they handle cache updates and energy retrieval.

| Aspect | Cassandra (Fortran) | mcgpu (Rust) |
|--------|---------------------|--------------|
| **Matrix Structure** | 2D arrays: `pair_nrg_vdw(i,j)`, `pair_nrg_qq(i,j)` | 1D linear: `pairwise_cache[i*n+j]` |
| **Energy Types** | Separate VDW and electrostatic | Combined single value |
| **Symmetry Storage** | Duplicated: stores both (i,j) and (j,i) | Single storage, updates both positions |
| **Dirty Tracking** | None (implicit via temp arrays) | Explicit `dirty[i]` flags |
| **Row Sums** | Computed on-demand by summing | Pre-cached in `mol_energies[i]` |
| **Parallelization** | OpenMP/MPI | GPU (WGPU) + Rayon (CPU) |

## Data Structures

### Cassandra

From `global_variables.f90`:
```fortran
! Pair energy arrays (N×N matrices)
REAL(DP), DIMENSION(:,:), ALLOCATABLE :: pair_nrg_vdw, pair_nrg_qq

! Temporary storage for backup during moves
REAL(DP), ALLOCATABLE :: pair_vdw_temp(:), pair_qq_temp(:)
```

### mcgpu

From `gpu.rs` and `cpu.rs`:
```rust
// Cache: pairwise[i * n_molecules + j] = E_ij
pairwise_cache: Vec<f64>,
// Cached molecule energies (row sums)
mol_energies: Vec<f64>,
// Track which molecules need recomputation
dirty: Vec<bool>,
```

## Move Workflow Comparison

### Cassandra Workflow

From `move_translate.f90`:
```fortran
! 1. BEFORE move: backup row to temp arrays
CALL Store_Molecule_Pair_Interaction_Arrays(lm,is,ibox,E_vdw,E_qq)

! 2. Propose move, compute new energy (updates pair_nrg arrays directly)
CALL Compute_Molecule_Nonbond_Inter_Energy(lm,is,E_vdw_move,E_qq_move,...)

! 3. Accept/reject
IF (accept) THEN
   DEALLOCATE(pair_vdw_temp, pair_qq_temp)  ! discard backup
ELSE
   CALL Reset_Molecule_Pair_Interaction_Arrays(lm,is,ibox)  ! restore from backup
ENDIF
```

### mcgpu Workflow

From `monte_carlo.rs`:
```rust
// 1. Get old energy from cache (O(1) lookup)
let e_old = self.energy.molecule_energy(&self.system, i).await;

// 2. Mark dirty BEFORE move
self.energy.invalidate_molecule(i);

// 3. Propose move, get new energy (recomputes row since dirty)
let e_new = self.energy.molecule_energy(&self.system, i).await;

// 4. Accept/reject
if accepted {
    // Cache already updated, nothing to do
} else {
    // Revert positions, mark dirty again (lazy recompute on next access)
    self.energy.invalidate_molecule(i);
}
```

## Symmetry Handling

### Cassandra

From `energy_routines.f90`:
```fortran
pair_nrg_vdw(locate_im, locate_jm) = vlj_pair
pair_nrg_vdw(locate_jm, locate_im) = vlj_pair   ! explicit duplicate

pair_nrg_qq(locate_im, locate_jm) = vqq_pair
pair_nrg_qq(locate_jm, locate_im) = vqq_pair    ! explicit duplicate
```

### mcgpu

From `gpu.rs`:
```rust
// Update row i
self.pairwise_cache[mol_i * n + j] = new_val;
// Update column i (symmetry)
self.pairwise_cache[j * n + mol_i] = new_val;

// ALSO update affected mol_energies incrementally
if j != mol_i {
    self.mol_energies[j] += delta;
}
```

mcgpu goes further by incrementally updating `mol_energies[j]` for all affected molecules, avoiding O(N) re-summation.

## Efficiency Analysis

### Per-Move Complexity

| Operation | Cassandra | mcgpu |
|-----------|-----------|-------|
| Get `e_old` | O(N) sum | **O(1)** lookup |
| Get `e_new` | O(N) compute + update | O(N) compute + update |
| On **accept** | O(1) deallocate | O(N) delta updates to `mol_energies` |
| On **reject** | O(N) restore | **O(1)** set dirty flag |

### Expected Cost Per MC Step

Let `α` = acceptance ratio (typically 0.3-0.5 for good sampling):

| | Cassandra | mcgpu |
|-|-----------|-------|
| **Energy retrieval** | 2 × O(N) = O(N) | 2 × O(1) = **O(1)** |
| **Accept path** | α × O(1) | α × O(N) |
| **Reject path** | (1-α) × O(N) | (1-α) × **O(1)** |
| **Total** | O(N) + (1-α)×O(N) | O(N)×α + O(1) |

**Simplified:**
- Cassandra: ~(2-α) × O(N) per step
- mcgpu: ~α × O(N) per step

With typical α ≈ 0.4:
- Cassandra: ~1.6N operations
- mcgpu: ~0.4N operations

**mcgpu is ~4× more efficient** in cache management overhead.

### Why mcgpu is More Efficient

1. **O(1) rejection handling** - In MC, rejections are frequent. Cassandra does O(N) restore; mcgpu just sets a boolean.

2. **O(1) energy retrieval** - Pre-cached `mol_energies[i]` avoids summing N values every time.

3. **Lazy evaluation** - Dirty flags defer work until actually needed. If a molecule is rejected then moved again before any other molecule queries it, no wasted recomputation.

4. **Lower memory** - Single combined energy vs separate VDW/QQ arrays (~50% reduction).

### Where Cassandra Could Win

1. **Predictability** - No lazy evaluation means more predictable memory access patterns (better for some CPU cache behaviors).

2. **Separate energy components** - If you need VDW and electrostatic energies separately for analysis, Cassandra has them ready.

3. **Simpler debugging** - Explicit backup/restore is easier to reason about than dirty flag state.

## Memory Usage

For N molecules:

| | Cassandra | mcgpu |
|-|-----------|-------|
| **Pair matrix** | 2 × N² (VDW + QQ) | N² |
| **Temp arrays** | 2 × N | 0 (uses dirty flags) |
| **Row sums** | 0 (computed on demand) | N |
| **Dirty flags** | 0 | N booleans |
| **Total** | ~2N² + 2N | ~N² + 2N |

mcgpu uses roughly **half the memory** by combining VDW/electrostatic into a single value.

## Key Files

### Cassandra
- `global_variables.f90` - Array declarations
- `pair_nrg_routines.f90` - Store/Reset routines
- `energy_routines.f90` - Compute_MoleculePair_Energy
- `move_translate.f90` - MC move logic

### mcgpu
- `src/gpu.rs` - GPU cached backend
- `src/cpu.rs` - CPU/SIMD cached backend
- `src/monte_carlo.rs` - MC move logic
- `src/energy.wgsl` - GPU compute shader

## Conclusion

The mcgpu approach is more efficient for typical Monte Carlo simulations because:

- Rejection is common → O(1) rejection handling dominates
- Energy queries are frequent → O(1) lookup beats O(N) sum
- The extra O(N) work on acceptance (updating `mol_energies[j]`) is offset by savings elsewhere

The efficiency gain scales with system size N and is most pronounced at moderate acceptance ratios (30-50%).

## References

1. Shah, J. K., et al. "Cassandra: An open source Monte Carlo package for molecular simulation." Journal of Computational Chemistry 38, 1727–1739 (2017).
2. Sukeník, L., Lund, M., Vácha, R. "Hastier Monte Carlo with cached state energies" (2026).
