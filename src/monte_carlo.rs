use crate::energy::EnergyBackend;
use crate::types::{Quat, System, Vec3};
use nalgebra::Unit;
use rand::Rng;
use rand_distr::{Distribution, UnitSphere};

/// Type of Monte Carlo move
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveType {
    Translation,
    Rotation,
}

/// Monte Carlo simulator for rigid molecules
pub struct MonteCarlo {
    pub system: System,
    pub energy: EnergyBackend,
    pub temperature: f64, // in energy units (kT = 1 means T in reduced units)
    pub max_trans: f64,   // maximum translation displacement
    pub max_rot: f64,     // maximum rotation angle (radians)
    pub rng: rand::rngs::SmallRng,

    // Per-move-type statistics
    pub trans_accepted: u64,
    pub trans_rejected: u64,
    pub rot_accepted: u64,
    pub rot_rejected: u64,
}

impl MonteCarlo {
    pub fn new(
        system: System,
        energy: EnergyBackend,
        temperature: f64,
        max_trans: f64,
        max_rot: f64,
        seed: u64,
    ) -> Self {
        use rand::SeedableRng;
        Self {
            system,
            energy,
            temperature,
            max_trans,
            max_rot,
            rng: rand::rngs::SmallRng::seed_from_u64(seed),
            trans_accepted: 0,
            trans_rejected: 0,
            rot_accepted: 0,
            rot_rejected: 0,
        }
    }

    /// Perform a single MC step: randomly select molecule and move type
    pub async fn step(&mut self) -> (MoveType, bool) {
        let i = self.rng.gen_range(0..self.system.n_molecules());

        // 50% translation, 50% rotation
        let move_type = if self.rng.gen::<f64>() < 0.5 {
            MoveType::Translation
        } else {
            MoveType::Rotation
        };

        let mol = &self.system.molecules[i];
        let old_com = mol.com;
        let old_q = mol.orientation;
        let old_sites = mol.sites_lab.clone();

        let e_old = self.energy.molecule_energy(&self.system, i).await;

        // Invalidate cache BEFORE moving so e_new is computed fresh
        self.energy.invalidate_molecule(i);

        match move_type {
            MoveType::Translation => self.propose_translation(i),
            MoveType::Rotation => self.propose_rotation(i),
        }
        self.system.molecules[i].update_sites(&self.system.mol_type);

        let e_new = self.energy.molecule_energy(&self.system, i).await;
        let accepted = self.metropolis(e_new - e_old);

        if accepted {
            // Notify backend that molecule moved (GPU uses this to invalidate cache)
            self.energy.notify_move_accepted(i);
            match move_type {
                MoveType::Translation => self.trans_accepted += 1,
                MoveType::Rotation => self.rot_accepted += 1,
            }
        } else {
            // Revert to old state
            let mol = &mut self.system.molecules[i];
            mol.com = old_com;
            mol.orientation = old_q;
            mol.sites_lab = old_sites;

            // Invalidate cache since we reverted positions (GPU cached wrong energy)
            self.energy.invalidate_molecule(i);

            match move_type {
                MoveType::Translation => self.trans_rejected += 1,
                MoveType::Rotation => self.rot_rejected += 1,
            }
        }

        (move_type, accepted)
    }

    /// Blocking version of step()
    pub fn step_blocking(&mut self) -> (MoveType, bool) {
        pollster::block_on(self.step())
    }

    /// Propose a random translation for molecule i
    fn propose_translation(&mut self, i: usize) {
        let dir: [f64; 3] = UnitSphere.sample(&mut self.rng);
        let mag = self.rng.gen::<f64>() * self.max_trans;

        let new_com = {
            let mol = &mut self.system.molecules[i];
            mol.com += mag * Vec3::from_row_slice(&dir);
            mol.com
        };

        let wrapped = self.system.wrap_pbc(new_com);
        self.system.molecules[i].com = wrapped;
    }

    /// Propose a random rotation for molecule i
    fn propose_rotation(&mut self, i: usize) {
        let axis: [f64; 3] = UnitSphere.sample(&mut self.rng);
        let angle = (self.rng.gen::<f64>() - 0.5) * 2.0 * self.max_rot;

        let mol = &mut self.system.molecules[i];
        let dq = Quat::from_axis_angle(&Unit::new_normalize(Vec3::from_row_slice(&axis)), angle);
        mol.orientation = dq * mol.orientation;
    }

    /// Metropolis acceptance criterion
    #[inline]
    fn metropolis(&mut self, delta_e: f64) -> bool {
        delta_e < 0.0 || self.rng.gen::<f64>() < (-delta_e / self.temperature).exp()
    }

    /// Get acceptance ratio for a specific move type
    pub fn acceptance_ratio(&self, move_type: MoveType) -> f64 {
        match move_type {
            MoveType::Translation => {
                let total = self.trans_accepted + self.trans_rejected;
                if total == 0 {
                    0.0
                } else {
                    self.trans_accepted as f64 / total as f64
                }
            }
            MoveType::Rotation => {
                let total = self.rot_accepted + self.rot_rejected;
                if total == 0 {
                    0.0
                } else {
                    self.rot_accepted as f64 / total as f64
                }
            }
        }
    }

    /// Get overall acceptance ratio
    pub fn total_acceptance_ratio(&self) -> f64 {
        let accepted = self.trans_accepted + self.rot_accepted;
        let total = accepted + self.trans_rejected + self.rot_rejected;
        if total == 0 {
            0.0
        } else {
            accepted as f64 / total as f64
        }
    }

    /// Reset statistics counters
    pub fn reset_statistics(&mut self) {
        self.trans_accepted = 0;
        self.trans_rejected = 0;
        self.rot_accepted = 0;
        self.rot_rejected = 0;
    }

    /// Adjust step sizes to target ~40% acceptance (call periodically during equilibration)
    pub fn adjust_step_sizes(&mut self, target_acceptance: f64) {
        let factor = |ratio: f64| -> f64 {
            if ratio < target_acceptance - 0.05 {
                0.95 // decrease step size
            } else if ratio > target_acceptance + 0.05 {
                1.05 // increase step size
            } else {
                1.0
            }
        };

        self.max_trans *= factor(self.acceptance_ratio(MoveType::Translation));
        self.max_rot *= factor(self.acceptance_ratio(MoveType::Rotation));

        // Clamp to reasonable bounds
        self.max_trans = self.max_trans.clamp(0.01, self.system.box_length * 0.5);
        self.max_rot = self.max_rot.clamp(0.01, std::f64::consts::PI);

        self.reset_statistics();
    }

    /// Total number of attempted moves
    pub fn total_steps(&self) -> u64 {
        self.trans_accepted + self.trans_rejected + self.rot_accepted + self.rot_rejected
    }
}
