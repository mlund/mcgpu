//! Command-line interface for the Monte Carlo simulator

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// Energy calculation backend
#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum Backend {
    /// GPU backend with pairwise caching
    #[default]
    Gpu,
    /// GPU backend without caching (recomputes every call)
    GpuUncached,
    /// CPU backend with SIMD and pairwise caching
    Cpu,
}

/// GPU-accelerated Monte Carlo simulation of rigid molecules
#[derive(Parser, Debug)]
#[command(name = "mc_simulator")]
#[command(version, about, long_about = None)]
pub struct Args {
    /// Input PQR file containing the molecular structure
    #[arg(short = 'm', long, default_value = "molecule.pqr")]
    pub molecule: PathBuf,

    /// Temperature in kT units
    #[arg(short = 't', long, default_value_t = 1.0)]
    pub temperature: f64,

    /// Number of MC steps
    #[arg(short = 'n', long, default_value_t = 10000)]
    pub steps: u64,

    /// Output PQR file for final configuration
    #[arg(short = 'o', long, default_value = "confout.pqr")]
    pub output: PathBuf,

    /// Restart file to write (and optionally read from)
    #[arg(short = 'r', long, default_value = "restart.dat")]
    pub restart: PathBuf,

    /// Continue from existing restart file (production run)
    #[arg(short = 'c', long)]
    pub continue_from_restart: bool,

    /// Number of molecules
    #[arg(long, default_value_t = 50)]
    pub n_molecules: usize,

    /// Spacing between molecules in Ångströms
    #[arg(long, default_value_t = 100.0)]
    pub spacing: f64,

    /// LJ well depth epsilon in kJ/mol
    #[arg(long, default_value_t = 0.01)]
    pub epsilon: f32,

    /// Cutoff distance in Ångströms
    #[arg(long, default_value_t = 10.0)]
    pub cutoff: f64,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Maximum translation displacement in Ångströms
    #[arg(long, default_value_t = 5.0)]
    pub max_trans: f64,

    /// Maximum rotation angle in radians
    #[arg(long, default_value_t = 0.5)]
    pub max_rot: f64,

    /// Energy calculation backend (gpu or cpu)
    #[arg(long, value_enum, default_value_t = Backend::Gpu)]
    pub backend: Backend,

    /// Number of CPU threads for CPU backend (0 = all available)
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
}

impl Args {
    pub fn parse_args() -> Self {
        Args::parse()
    }
}
