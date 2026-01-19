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

## Literature Overview

### Energy Caching and Tabulation Approaches

The concept of caching or tabulating energies to accelerate Monte Carlo simulations has been explored in various forms:

#### Energy Table/Lookup Methods

- **Distance and orientation dependent energy tables**: A tabulation strategy that replaces all non-bonded energy calculations between pairs of molecules with simple lookups. The table is both distance and orientation dependent, providing substantial speedup with modest precision loss. This approach is particularly effective for rigid or semi-rigid molecules where orientational degrees of freedom are limited. ([Chaimovich et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3408236/))

#### Algorithmic Foundations

- **"Dividing and Conquering" and "Caching" in Molecular Modeling**: Reviews how caching intermediate results has been a fundamental principle in molecular modeling for over half a century. Introduces the "local free energy landscape approach" for caching distributions of local clusters of molecular degrees of freedom. ([Hajjar et al., 2021](https://www.mdpi.com/1422-0067/22/9/5053))

- **Efficient Energy Computation for Monte Carlo Simulation of Proteins**: Addresses efficient energy computation strategies specifically for MC protein simulations where the conformational space is large. ([Chikenji et al., 2003](https://link.springer.com/chapter/10.1007/978-3-540-39763-2_26))

### Related Optimizations: Neighbor and Cell Lists

While not directly caching energies, these methods reduce the computational complexity of energy calculations:

- **Cell List Algorithms**: The linked cell list algorithm reduces neighbor searching from O(N²) to O(N) by spatially partitioning particles. GPU implementations have been developed for Monte Carlo simulations. ([Schwiebert & Hailat](https://www.semanticscholar.org/paper/An-Efficient-Cell-List-Implementation-for-Monte-on-Schwiebert-Hailat/18761563c8473581cb7a2029c093975920bcd2ff))

- **Stenciled Cell Lists**: Extension where each particle type searches a different "stencil" of adjacent cells based on cutoff radius, with precomputed distances to skip unnecessary checks.

- **Linear Bounding Volume Hierarchies (LBVH)**: GPU algorithm for computing neighbor lists, particularly effective for colloidal systems with large size disparities. ([Howard et al., 2016](https://www.sciencedirect.com/science/article/abs/pii/S0010465516300182))

### Monte Carlo Software Implementations

| Software | Caching Strategy | Language | Notes |
|----------|------------------|----------|-------|
| [Cassandra](https://github.com/MaginnGroup/Cassandra) | Pair energy arrays | Fortran | `pair_nrg_vdw`, `pair_nrg_qq` matrices |
| [Faunus](https://github.com/mlund/faunus) | Parallel evaluation, splines | C++ | Object-oriented framework ([Lund et al., 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2266748/)) |
| [GOMC](https://github.com/GOMC-WSU/GOMC) | Cell lists + GPU | C++/CUDA | GPU-optimized for phase equilibria ([Nejahi et al., 2019](https://www.sciencedirect.com/science/article/pii/S2352711018301171)) |
| [MCCCS Towhee](https://towhee.sourceforge.net/) | Force field tables | Fortran | Focus on Gibbs ensemble ([Martin, 2013](https://www.tandfonline.com/doi/abs/10.1080/08927022.2013.828208)) |
| [RASPA](https://github.com/iRASPA/RASPA2) | Energy grids | C | Pre-computed grids for rigid frameworks |
| [DL_MONTE](https://www.ccp5.ac.uk/DL_MONTE/) | Various | Fortran | General purpose MC |
| **mcgpu** | Pair matrix + row sums | Rust | Dirty flags, GPU/CPU backends |

### Key Observations

1. **Pair energy matrix caching** (as in Cassandra/mcgpu) is relatively uncommon in published literature. Most papers focus on neighbor lists or energy tabulation by distance/orientation.

2. **Energy grid pre-computation** is standard for adsorption simulations (RASPA, GCMC codes) where the adsorbent framework is rigid and grid points can be pre-calculated.

3. **The O(N²) memory cost** of full pair energy matrices limits applicability to systems with moderate N (hundreds to low thousands of molecules). Larger systems typically use cell/neighbor lists for O(N) scaling.

4. **Hybrid approaches** combining cell lists with partial energy caching may offer the best of both worlds for intermediate system sizes.

## References

1. Shah, J. K., et al. "Cassandra: An open source Monte Carlo package for molecular simulation." *Journal of Computational Chemistry* 38, 1727–1739 (2017).
2. Sukeník, L., Lund, M., Vácha, R. "Hastier Monte Carlo with cached state energies" (2026).
3. Chaimovich, A., Shell, M. S. "Accelerating molecular Monte Carlo simulations using distance and orientation dependent energy tables." *Molecular Physics* 109, 367-382 (2011).
4. Hajjar, E., et al. "'Dividing and Conquering' and 'Caching' in Molecular Modeling." *International Journal of Molecular Sciences* 22, 5053 (2021).
5. Lund, M., Trulsson, M., Persson, B. "Faunus: An object oriented framework for molecular simulation." *Source Code for Biology and Medicine* 3, 1 (2008).
6. Nejahi, Y., et al. "GOMC: GPU Optimized Monte Carlo for the simulation of phase equilibria and physical properties of complex fluids." *SoftwareX* 9, 20-27 (2019).
7. Martin, M. G. "MCCCS Towhee: a tool for Monte Carlo molecular simulation." *Molecular Simulation* 39, 1212-1222 (2013).
8. Howard, M. P., et al. "Efficient neighbor list calculation for molecular simulation of colloidal systems using graphics processing units." *Computer Physics Communications* 203, 45-52 (2016).
9. Chen, B., Siepmann, J. I. "Efficient Monte Carlo methods for the computer simulation of biological molecules." *Methods in Molecular Biology* 443, 25-46 (2008).
