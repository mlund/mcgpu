# Metropolis Monte Carlo Simulation of Rigid Bodies

- Number of rigid bodies: ~ 100
- Atoms in each rigid body: ~1000
- Cuboidal box with periodic boundary conditions
- Atom-atom pair potential: Lennard-Jones + Yukawa / Screened coulomb
- Propagate using random rigid body rotation or rotation with Metropolis Monte Carlo algorithm
- Units: length = Å; energy = kJ/mol
- Use `wgpu` crate for GPU acceleration
- Use `nalgebra` for geometrical transformations
