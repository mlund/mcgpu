//! Energy backend abstraction

use crate::cpu::CpuEnergyBackend;
use crate::gpu::GpuEnergyBackend;
use crate::types::System;

/// Unified energy backend that can use either GPU or CPU
pub enum EnergyBackend {
    Gpu(GpuEnergyBackend),
    Cpu(CpuEnergyBackend),
}

impl EnergyBackend {
    /// Compute interaction energy of molecule `mol_idx` with all other molecules
    pub async fn molecule_energy(&mut self, system: &System, mol_idx: usize) -> f64 {
        match self {
            EnergyBackend::Gpu(gpu) => gpu.molecule_energy(system, mol_idx).await,
            EnergyBackend::Cpu(cpu) => cpu.molecule_energy(system, mol_idx),
        }
    }

    /// Compute total system energy (inter-molecular only)
    pub async fn total_energy(&mut self, system: &System) -> f64 {
        match self {
            EnergyBackend::Gpu(gpu) => gpu.total_energy(system).await,
            EnergyBackend::Cpu(cpu) => cpu.total_energy(system),
        }
    }

    /// Notify that a move was accepted for molecule mol_idx
    /// (Currently no-op for both backends)
    pub fn notify_move_accepted(&mut self, mol_idx: usize) {
        match self {
            EnergyBackend::Gpu(gpu) => gpu.notify_move_accepted(mol_idx),
            EnergyBackend::Cpu(_) => {}
        }
    }

    /// Invalidate cache for molecule that is about to move
    /// Call BEFORE proposing move so e_new is computed fresh
    pub fn invalidate_molecule(&mut self, mol_idx: usize) {
        match self {
            EnergyBackend::Gpu(gpu) => gpu.invalidate_molecule(mol_idx),
            EnergyBackend::Cpu(_) => {} // no-op for CPU
        }
    }

    /// Get backend name for display
    pub fn name(&self) -> &'static str {
        match self {
            EnergyBackend::Gpu(_) => "GPU",
            EnergyBackend::Cpu(_) => "CPU/SIMD",
        }
    }
}
