use mc_simulator::{
    read_pqr_molecule, read_restart, write_pqr, write_restart,
    Args, Backend, CpuEnergyBackend, EnergyBackend, GpuEnergyBackend,
    GpuUncachedEnergyBackend, MonteCarlo, MoveType, System, Vec3,
};
use nalgebra::UnitQuaternion;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let args = Args::parse_args();
    pollster::block_on(run(args));
}

async fn run(args: Args) {
    // Configure rayon thread pool for CPU backend
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("Failed to configure thread pool");
    }

    // Load molecule structure from PQR file
    let mol_type = Arc::new(
        read_pqr_molecule(&args.molecule, "MOL", args.epsilon)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", args.molecule.display(), e))
    );
    println!("Loaded molecule with {} sites from {}", mol_type.n_sites(), args.molecule.display());

    // Create system
    let n_mol = args.n_molecules;
    let spacing = args.spacing;
    let box_length = spacing * (n_mol as f64).cbrt();
    let mut system = System::new(mol_type.clone(), box_length);

    // Place molecules on a grid
    let n_per_side = (n_mol as f64).cbrt().ceil() as usize;
    for ix in 0..n_per_side {
        for iy in 0..n_per_side {
            for iz in 0..n_per_side {
                if system.n_molecules() >= n_mol {
                    break;
                }
                let com = Vec3::new(
                    (ix as f64 + 0.5) * spacing,
                    (iy as f64 + 0.5) * spacing,
                    (iz as f64 + 0.5) * spacing,
                );
                system.add_molecule(com, UnitQuaternion::identity());
            }
        }
    }

    // Load restart file if continuing
    if args.continue_from_restart {
        read_restart(&mut system, &args.restart)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", args.restart.display(), e));
        println!("Loaded restart from {}", args.restart.display());
    }

    println!(
        "System: {} molecules, {} atoms, box={:.1} Å, spacing={:.1} Å",
        system.n_molecules(),
        system.n_atoms(),
        box_length,
        spacing
    );

    // Initialize energy backend
    let energy = match args.backend {
        Backend::Gpu => {
            let gpu = GpuEnergyBackend::new_async(&system, args.cutoff as f32).await;
            EnergyBackend::Gpu(gpu)
        }
        Backend::GpuUncached => {
            let gpu = GpuUncachedEnergyBackend::new_async(&system, args.cutoff as f32).await;
            EnergyBackend::GpuUncached(gpu)
        }
        Backend::Cpu => {
            let cpu = CpuEnergyBackend::new(args.cutoff as f32);
            EnergyBackend::Cpu(cpu)
        }
    };
    match &energy {
        EnergyBackend::Gpu(_) => println!("Using GPU backend (cached)"),
        EnergyBackend::GpuUncached(_) => println!("Using GPU backend (uncached)"),
        EnergyBackend::Cpu(_) => println!("Using CPU/SIMD backend ({} threads)", rayon::current_num_threads()),
    }

    // Create MC sampler
    let mut mc = MonteCarlo::new(
        system,
        energy,
        args.temperature,
        args.max_trans,
        args.max_rot,
        args.seed,
    );

    let steps = args.steps;

    let start_time = Instant::now();

    if args.continue_from_restart {
        // Production run (continuing from restart)
        println!("Production run ({} steps)...", steps);

        for step in 0..steps {
            mc.step().await;

            if (step + 1) % 1000 == 0 {
                let energy = mc.energy.total_energy(&mc.system).await;
                println!(
                    "  Step {}: E={:.1} acc={:.1}%",
                    step + 1,
                    energy,
                    mc.total_acceptance_ratio() * 100.0
                );
            }
        }
    } else {
        // Equilibration run
        println!("Equilibration ({} steps)...", steps);

        for step in 0..steps {
            mc.step().await;

            if (step + 1) % 100 == 0 {
                let energy = mc.energy.total_energy(&mc.system).await;
                println!(
                    "  Step {}: E={:.1} trans={:.1}% rot={:.1}% d={:.3} θ={:.3}",
                    step + 1,
                    energy,
                    mc.acceptance_ratio(MoveType::Translation) * 100.0,
                    mc.acceptance_ratio(MoveType::Rotation) * 100.0,
                    mc.max_trans,
                    mc.max_rot
                );
                mc.adjust_step_sizes(0.4);
            }
        }
    }

    // Save outputs
    write_pqr(&mc.system, &args.output, 1.0)
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", args.output.display(), e));
    write_restart(&mc.system, &args.restart)
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", args.restart.display(), e));
    println!("Wrote {} and {}", args.output.display(), args.restart.display());

    // Final statistics (production run only)
    if args.continue_from_restart {
        let elapsed = start_time.elapsed().as_secs_f64();
        let steps_per_sec = steps as f64 / elapsed;

        println!("\n=== Final Statistics ===");
        println!(
            "Translation: {:.1}% ({}/{})",
            mc.acceptance_ratio(MoveType::Translation) * 100.0,
            mc.trans_accepted,
            mc.trans_accepted + mc.trans_rejected
        );
        println!(
            "Rotation:    {:.1}% ({}/{})",
            mc.acceptance_ratio(MoveType::Rotation) * 100.0,
            mc.rot_accepted,
            mc.rot_accepted + mc.rot_rejected
        );
        println!(
            "Overall:     {:.1}%",
            mc.total_acceptance_ratio() * 100.0
        );
        println!(
            "Performance: {:.1} steps/s ({:.2}s total)",
            steps_per_sec,
            elapsed
        );
    }
}
