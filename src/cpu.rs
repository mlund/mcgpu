//! CPU/SIMD backend for energy calculations with pairwise caching

use crate::types::{SiteParams, System};
use rayon::prelude::*;
use wide::{f32x4, CmpGt, CmpLt};

/// CPU backend for computing interaction energies using SIMD with pairwise caching
pub struct CpuEnergyBackend {
    cutoff_sq: f32,

    // Cache: pairwise[i * n_molecules + j] = E_ij
    pairwise_cache: Vec<f64>,
    // Cached molecule energies (row sums)
    mol_energies: Vec<f64>,
    // Track which molecules need recomputation
    dirty: Vec<bool>,
    cache_initialized: bool,
    n_molecules: usize,
}

impl CpuEnergyBackend {
    pub fn new(cutoff: f32) -> Self {
        Self {
            cutoff_sq: cutoff * cutoff,
            pairwise_cache: Vec::new(),
            mol_energies: Vec::new(),
            dirty: Vec::new(),
            cache_initialized: false,
            n_molecules: 0,
        }
    }

    /// Initialize cache by computing all pairwise energies
    fn initialize_cache(&mut self, system: &System) {
        let n = system.n_molecules();
        self.n_molecules = n;
        self.pairwise_cache = vec![0.0; n * n];
        self.mol_energies = vec![0.0; n];
        self.dirty = vec![false; n];

        // Compute all pairwise energies in parallel
        let pairwise: Vec<f64> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            0.0
                        } else {
                            self.compute_pair_energy(system, i, j)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        self.pairwise_cache = pairwise;

        // Compute row sums
        for i in 0..n {
            self.mol_energies[i] = (0..n).map(|j| self.pairwise_cache[i * n + j]).sum();
        }

        self.cache_initialized = true;
    }

    /// Update cache for a single molecule that moved
    fn update_row(&mut self, system: &System, mol_i: usize) {
        let n = self.n_molecules;

        // Save old row values
        let old_row: Vec<f64> = (0..n)
            .map(|j| self.pairwise_cache[mol_i * n + j])
            .collect();

        // Compute new row in parallel
        let new_row: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|j| {
                if j == mol_i {
                    0.0
                } else {
                    self.compute_pair_energy(system, mol_i, j)
                }
            })
            .collect();

        // Update cache
        for j in 0..n {
            let old_val = old_row[j];
            let new_val = new_row[j];
            let delta = new_val - old_val;

            // Update row i
            self.pairwise_cache[mol_i * n + j] = new_val;
            // Update column i (symmetry: E_ji = E_ij)
            self.pairwise_cache[j * n + mol_i] = new_val;

            // Update affected molecule energies
            if j != mol_i {
                self.mol_energies[j] += delta;
            }
        }

        // Recalculate energy for molecule i
        self.mol_energies[mol_i] = new_row.iter().sum();
        self.dirty[mol_i] = false;
    }

    /// Compute interaction energy of molecule `mol_idx` with all other molecules
    pub fn molecule_energy(&mut self, system: &System, mol_idx: usize) -> f64 {
        if !self.cache_initialized {
            self.initialize_cache(system);
        } else if self.dirty[mol_idx] {
            self.update_row(system, mol_idx);
        }
        self.mol_energies[mol_idx]
    }

    /// Invalidate cache for molecule that is about to move
    pub fn invalidate_molecule(&mut self, mol_idx: usize) {
        if self.cache_initialized {
            self.dirty[mol_idx] = true;
        }
    }

    /// Compute total system energy (inter-molecular only)
    pub fn total_energy(&mut self, system: &System) -> f64 {
        if !self.cache_initialized {
            self.initialize_cache(system);
        }
        // Update any dirty molecules
        for i in 0..self.n_molecules {
            if self.dirty[i] {
                self.update_row(system, i);
            }
        }
        // Sum all molecule energies and divide by 2 (each pair counted twice)
        self.mol_energies.iter().sum::<f64>() * 0.5
    }

    /// Compute energy between two molecules using SIMD
    fn compute_pair_energy(&self, system: &System, mol_i: usize, mol_j: usize) -> f64 {
        let mol_type = &system.mol_type;
        let sites_i = &system.molecules[mol_i].sites_lab;
        let sites_j = &system.molecules[mol_j].sites_lab;
        let box_len = system.box_length as f32;

        // Sum over all site pairs between molecules i and j
        sites_i
            .iter()
            .enumerate()
            .map(|(si, pos_i)| {
                let par_i = &mol_type.site_params[si];
                self.site_pair_energy(pos_i, par_i, sites_j, &mol_type.site_params, box_len)
            })
            .sum()
    }

    /// Compute energy of one site with all sites in another molecule using SIMD
    fn site_pair_energy(
        &self,
        pos_i: &[f32; 4],
        par_i: &SiteParams,
        other_sites: &[[f32; 4]],
        other_params: &[SiteParams],
        box_len: f32,
    ) -> f64 {
        let mut total = 0.0f64;

        // SIMD constants (f32x4 = 128-bit, native on both ARM NEON and x86 SSE)
        let xi = f32x4::splat(pos_i[0]);
        let yi = f32x4::splat(pos_i[1]);
        let zi = f32x4::splat(pos_i[2]);
        let box_v = f32x4::splat(box_len);
        let inv_box = f32x4::splat(1.0 / box_len);
        let cutoff_sq_v = f32x4::splat(self.cutoff_sq);
        let eps_i = f32x4::splat(par_i.epsilon);
        let sig_i = f32x4::splat(par_i.sigma);
        let four = f32x4::splat(4.0);
        let half = f32x4::splat(0.5);
        let one = f32x4::splat(1.0);
        let tiny = f32x4::splat(1e-6);
        let zero = f32x4::splat(0.0);

        // Process 4 sites at a time (128-bit SIMD)
        let chunks = other_sites.chunks_exact(4);
        let remainder_start = other_sites.len() - chunks.remainder().len();

        for (chunk_idx, chunk) in chunks.enumerate() {
            // Load 4 positions
            let xj = f32x4::new([chunk[0][0], chunk[1][0], chunk[2][0], chunk[3][0]]);
            let yj = f32x4::new([chunk[0][1], chunk[1][1], chunk[2][1], chunk[3][1]]);
            let zj = f32x4::new([chunk[0][2], chunk[1][2], chunk[2][2], chunk[3][2]]);

            // Minimum image convention
            let mut dx = xj - xi;
            let mut dy = yj - yi;
            let mut dz = zj - zi;
            dx = dx - box_v * (dx * inv_box).round();
            dy = dy - box_v * (dy * inv_box).round();
            dz = dz - box_v * (dz * inv_box).round();

            let r_sq = dx * dx + dy * dy + dz * dz;

            // Load parameters for 4 sites
            let base = chunk_idx * 4;
            let eps_j = f32x4::new([
                other_params[base].epsilon,
                other_params[base + 1].epsilon,
                other_params[base + 2].epsilon,
                other_params[base + 3].epsilon,
            ]);
            let sig_j = f32x4::new([
                other_params[base].sigma,
                other_params[base + 1].sigma,
                other_params[base + 2].sigma,
                other_params[base + 3].sigma,
            ]);

            // Lorentz-Berthelot combining rules
            let eps = (eps_i * eps_j).sqrt();
            let sig = (sig_i + sig_j) * half;

            // LJ: 4ε[(σ/r)^12 - (σ/r)^6]
            let s2 = sig * sig / r_sq;
            let s6 = s2 * s2 * s2;
            let lj = four * eps * s6 * (s6 - one);

            // Mask: within cutoff and not too close
            let in_cutoff = r_sq.cmp_lt(cutoff_sq_v);
            let not_tiny = r_sq.cmp_gt(tiny);
            let mask = in_cutoff & not_tiny;

            // Apply mask (blend with zero)
            let lj_masked = mask.blend(lj, zero);

            // Sum the 4 values
            let arr: [f32; 4] = lj_masked.into();
            total += arr.iter().map(|&x| x as f64).sum::<f64>();
        }

        // Handle remainder with scalar code
        for j in remainder_start..other_sites.len() {
            total +=
                self.scalar_pair_energy(pos_i, par_i, &other_sites[j], &other_params[j], box_len);
        }

        total
    }

    /// Scalar pair energy calculation for remainder
    #[inline]
    fn scalar_pair_energy(
        &self,
        pos_i: &[f32; 4],
        par_i: &SiteParams,
        pos_j: &[f32; 4],
        par_j: &SiteParams,
        box_len: f32,
    ) -> f64 {
        let mut dx = pos_j[0] - pos_i[0];
        let mut dy = pos_j[1] - pos_i[1];
        let mut dz = pos_j[2] - pos_i[2];

        // Minimum image
        dx -= box_len * (dx / box_len).round();
        dy -= box_len * (dy / box_len).round();
        dz -= box_len * (dz / box_len).round();

        let r_sq = dx * dx + dy * dy + dz * dz;
        if r_sq >= self.cutoff_sq || r_sq < 1e-6 {
            return 0.0;
        }

        // Lorentz-Berthelot
        let eps = (par_i.epsilon * par_j.epsilon).sqrt();
        let sig = 0.5 * (par_i.sigma + par_j.sigma);

        // LJ
        let s2 = sig * sig / r_sq;
        let s6 = s2 * s2 * s2;

        (4.0 * eps * s6 * (s6 - 1.0)) as f64
    }
}
