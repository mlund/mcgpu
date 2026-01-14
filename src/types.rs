use nalgebra::{Vector3, UnitQuaternion};
use std::sync::Arc;

pub type Vec3 = Vector3<f64>;
pub type Quat = UnitQuaternion<f64>;

/// LJ + Yukawa parameters per interaction site (GPU-aligned)
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SiteParams {
    pub epsilon: f32,   // LJ well depth
    pub sigma: f32,     // LJ diameter
    pub yukawa_a: f32,  // Yukawa amplitude
    pub kappa: f32,     // Inverse Debye length
    pub charge: f32,    // Partial charge
    pub _pad: f32,      // Padding for GPU alignment
}

/// Immutable molecule template (shared across all molecules of same type)
#[derive(Clone)]
pub struct MoleculeType {
    pub name: String,
    pub sites_body: Vec<Vec3>,        // Site positions in body frame
    pub site_params: Vec<SiteParams>, // Parameters per site
}

impl MoleculeType {
    pub fn new(name: impl Into<String>, sites_body: Vec<Vec3>, site_params: Vec<SiteParams>) -> Self {
        assert_eq!(sites_body.len(), site_params.len(), "sites and params must match");
        Self {
            name: name.into(),
            sites_body,
            site_params,
        }
    }

    pub fn n_sites(&self) -> usize {
        self.sites_body.len()
    }
}

/// Single molecule instance with cached lab-frame positions
pub struct Molecule {
    pub com: Vec3,                  // Center of mass position
    pub orientation: Quat,          // Orientation quaternion
    pub sites_lab: Vec<[f32; 4]>,   // Cached lab-frame positions (vec4 for GPU alignment)
}

impl Molecule {
    pub fn new(com: Vec3, orientation: Quat, mol_type: &MoleculeType) -> Self {
        let mut mol = Self {
            com,
            orientation,
            sites_lab: vec![[0.0; 4]; mol_type.n_sites()],
        };
        mol.update_sites(mol_type);
        mol
    }

    /// Transform body-frame sites to lab frame using current CoM and orientation
    pub fn update_sites(&mut self, mol_type: &MoleculeType) {
        for (lab, body) in self.sites_lab.iter_mut().zip(&mol_type.sites_body) {
            let p = self.com + self.orientation * body;
            *lab = [p.x as f32, p.y as f32, p.z as f32, 0.0];
        }
    }
}

/// Simulation box containing molecules
pub struct System {
    pub mol_type: Arc<MoleculeType>,
    pub molecules: Vec<Molecule>,
    pub box_length: f64,
}

impl System {
    pub fn new(mol_type: Arc<MoleculeType>, box_length: f64) -> Self {
        Self {
            mol_type,
            molecules: Vec::new(),
            box_length,
        }
    }

    pub fn add_molecule(&mut self, com: Vec3, orientation: Quat) {
        let mol = Molecule::new(com, orientation, &self.mol_type);
        self.molecules.push(mol);
    }

    pub fn n_molecules(&self) -> usize {
        self.molecules.len()
    }

    pub fn n_atoms(&self) -> usize {
        self.molecules.len() * self.mol_type.n_sites()
    }

    /// Flat array of all positions for GPU upload
    pub fn positions_flat(&self) -> Vec<[f32; 4]> {
        self.molecules
            .iter()
            .flat_map(|m| m.sites_lab.iter().copied())
            .collect()
    }

    /// Apply periodic boundary conditions to a position
    #[inline]
    pub fn wrap_pbc(&self, mut pos: Vec3) -> Vec3 {
        let l = self.box_length;
        pos.x -= l * (pos.x / l).floor();
        pos.y -= l * (pos.y / l).floor();
        pos.z -= l * (pos.z / l).floor();
        pos
    }

    /// Minimum image vector from a to b
    #[inline]
    pub fn min_image(&self, a: Vec3, b: Vec3) -> Vec3 {
        let mut dr = b - a;
        let l = self.box_length;
        dr.x -= l * (dr.x / l).round();
        dr.y -= l * (dr.y / l).round();
        dr.z -= l * (dr.z / l).round();
        dr
    }
}
