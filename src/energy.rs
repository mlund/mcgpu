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
    pub async fn molecule_energy(&self, system: &System, mol_idx: usize) -> f64 {
        match self {
            EnergyBackend::Gpu(gpu) => gpu.molecule_energy(system, mol_idx).await,
            EnergyBackend::Cpu(cpu) => cpu.molecule_energy(system, mol_idx),
        }
    }

    /// Compute total system energy (inter-molecular only)
    pub async fn total_energy(&self, system: &System) -> f64 {
        match self {
            EnergyBackend::Gpu(gpu) => gpu.total_energy(system).await,
            EnergyBackend::Cpu(cpu) => cpu.total_energy(system),
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
