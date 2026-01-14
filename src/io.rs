//! File I/O for molecular structures

use crate::{MoleculeType, SiteParams, System, Vec3};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Atom data read from a PQR file
#[derive(Debug, Clone)]
pub struct PqrAtom {
    pub position: Vec3,
    pub charge: f32,
    pub radius: f32,
    pub name: String,
}

/// Read a single-molecule PQR file and create a MoleculeType.
///
/// The molecule's geometric center is computed and subtracted to get body-frame coordinates.
/// Radius is converted to sigma (diameter = 2 * radius) for LJ parameters.
///
/// # Arguments
/// * `path` - Path to PQR file containing a single molecule
/// * `name` - Name for the molecule type
/// * `epsilon` - LJ well depth to use for all sites (PQR doesn't contain this)
pub fn read_pqr_molecule<P: AsRef<Path>>(
    path: P,
    name: &str,
    epsilon: f32,
) -> io::Result<MoleculeType> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let atoms = parse_pqr(reader)?;

    if atoms.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "No atoms found in PQR file"));
    }

    // Compute geometric center
    let center: Vec3 = atoms.iter().map(|a| a.position).sum::<Vec3>() / atoms.len() as f64;

    // Convert to body-frame coordinates and create SiteParams
    let sites_body: Vec<Vec3> = atoms.iter().map(|a| a.position - center).collect();

    let site_params: Vec<SiteParams> = atoms
        .iter()
        .map(|a| SiteParams {
            epsilon,
            sigma: a.radius * 2.0, // diameter = 2 * radius
            yukawa_a: 0.0,
            kappa: 0.0,
            charge: a.charge,
            _pad: 0.0,
        })
        .collect();

    Ok(MoleculeType::new(name, sites_body, site_params))
}

/// Parse PQR format from a reader, returning atom data.
fn parse_pqr<R: BufRead>(reader: R) -> io::Result<Vec<PqrAtom>> {
    let mut atoms = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if !line.starts_with("ATOM") && !line.starts_with("HETATM") {
            continue;
        }

        // PQR format is whitespace-separated after the record type
        // ATOM  index name altloc resname chain resnum icode x y z charge radius
        let parts: Vec<&str> = line.split_whitespace().collect();

        // Need at least: ATOM index name resname chain resnum x y z charge radius
        // Some PQR files omit altloc/icode, so we parse from the end
        if parts.len() < 10 {
            continue;
        }

        // Parse from the end (most reliable for varying PQR formats)
        let n = parts.len();
        let radius: f32 = parts[n - 1].parse().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid radius: {}", e))
        })?;
        let charge: f32 = parts[n - 2].parse().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid charge: {}", e))
        })?;
        let z: f64 = parts[n - 3].parse().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid z: {}", e))
        })?;
        let y: f64 = parts[n - 4].parse().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid y: {}", e))
        })?;
        let x: f64 = parts[n - 5].parse().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Invalid x: {}", e))
        })?;
        let name = parts[2].to_string();

        atoms.push(PqrAtom {
            position: Vec3::new(x, y, z),
            charge,
            radius,
            name,
        });
    }

    Ok(atoms)
}

/// Write a restart file with molecule positions and orientations.
///
/// Format (text-based):
/// ```
/// RESTART
/// box_length <value>
/// n_molecules <value>
/// mol <index> <com_x> <com_y> <com_z> <q_i> <q_j> <q_k> <q_w>
/// ...
/// END
/// ```
pub fn write_restart<P: AsRef<Path>>(system: &System, path: P) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "RESTART")?;
    writeln!(writer, "box_length {:.6}", system.box_length)?;
    writeln!(writer, "n_molecules {}", system.n_molecules())?;

    for (i, mol) in system.molecules.iter().enumerate() {
        let q = &mol.orientation;
        writeln!(
            writer,
            "mol {} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            i, mol.com.x, mol.com.y, mol.com.z, q.i, q.j, q.k, q.w
        )?;
    }

    writeln!(writer, "END")?;
    Ok(())
}

/// Read a restart file and update molecule positions/orientations in the system.
///
/// The system must already have the correct number of molecules.
pub fn read_restart<P: AsRef<Path>>(system: &mut System, path: P) -> io::Result<()> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "mol" if parts.len() >= 9 => {
                let idx: usize = parts[1].parse().map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid mol index: {}", e))
                })?;

                if idx >= system.n_molecules() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Molecule index {} out of range", idx),
                    ));
                }

                let x: f64 = parts[2].parse().map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid x: {}", e))
                })?;
                let y: f64 = parts[3].parse().map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid y: {}", e))
                })?;
                let z: f64 = parts[4].parse().map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid z: {}", e))
                })?;
                let qi: f64 = parts[5].parse().map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid qi: {}", e))
                })?;
                let qj: f64 = parts[6].parse().map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid qj: {}", e))
                })?;
                let qk: f64 = parts[7].parse().map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid qk: {}", e))
                })?;
                let qw: f64 = parts[8].parse().map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, format!("Invalid qw: {}", e))
                })?;

                let mol = &mut system.molecules[idx];
                mol.com = Vec3::new(x, y, z);
                mol.orientation = nalgebra::UnitQuaternion::from_quaternion(
                    nalgebra::Quaternion::new(qw, qi, qj, qk)
                );
                mol.update_sites(&system.mol_type);
            }
            _ => {}
        }
    }

    Ok(())
}

/// Write a PQR structure file of the entire system.
///
/// PQR format extends PDB with charge and radius fields. Each atom line:
/// ATOM  index name altloc resname chain resnum icode   x       y       z     charge radius
///
/// # Arguments
/// * `system` - The simulation system to write
/// * `path` - Output file path
/// * `scale` - Scale factor for coordinates (default 1.0 for Ångströms)
pub fn write_pqr<P: AsRef<Path>>(system: &System, path: P, scale: f64) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_pqr_to(&mut writer, system, scale)
}

/// Write PQR format to any writer.
pub fn write_pqr_to<W: Write>(writer: &mut W, system: &System, scale: f64) -> io::Result<()> {
    let mol_type = &system.mol_type;

    // Write CRYST1 record with box dimensions (cubic box, orthogonal)
    let box_len = scale * system.box_length;
    writeln!(
        writer,
        "CRYST1{:9.3}{:9.3}{:9.3}{:7.2}{:7.2}{:7.2} P 1           1",
        box_len, box_len, box_len, 90.0, 90.0, 90.0
    )?;

    // Truncate molecule name to 3 chars for residue name
    let res_name: String = mol_type.name.chars().take(3).collect();
    let res_name = if res_name.is_empty() { "MOL" } else { &res_name };

    let mut atom_index = 0usize;

    for (mol_idx, molecule) in system.molecules.iter().enumerate() {
        let chain = 'A';
        let res_num = (mol_idx % 9999) + 1; // PQR residue number field is 4 digits

        for (site_idx, pos) in molecule.sites_lab.iter().enumerate() {
            atom_index += 1;
            let params = &mol_type.site_params[site_idx];

            // Generate atom name from site index (e.g., "A001", "A002", ...)
            let atom_name = format!("A{:03}", (site_idx % 1000));

            let x = scale * pos[0] as f64;
            let y = scale * pos[1] as f64;
            let z = scale * pos[2] as f64;
            let charge = scale * params.charge as f64;
            let radius = scale * params.sigma as f64 * 0.5; // sigma is diameter, radius is half

            // PQR format: ATOM  index name altloc resname chain resnum icode   x y z charge radius
            writeln!(
                writer,
                "{:6}{:5} {:^4}{:1}{:3} {:1}{:4}{:1}   {:8.3}{:8.3}{:8.3}{:6.2}{:6.2}",
                "ATOM",
                atom_index % 100000, // 5 digit field
                atom_name,
                " ", // altloc
                res_name,
                chain,
                res_num,
                " ", // icode
                x,
                y,
                z,
                charge,
                radius
            )?;
        }
    }

    writeln!(writer, "END")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MoleculeType, SiteParams, Vec3};
    use nalgebra::UnitQuaternion;
    use std::sync::Arc;

    #[test]
    fn test_write_pqr_format() {
        // Create a simple test system
        let sites_body = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let site_params = vec![
            SiteParams {
                epsilon: 1.0,
                sigma: 3.0,
                yukawa_a: 0.0,
                kappa: 0.0,
                charge: 0.5,
                _pad: 0.0,
            },
            SiteParams {
                epsilon: 1.0,
                sigma: 2.5,
                yukawa_a: 0.0,
                kappa: 0.0,
                charge: -0.5,
                _pad: 0.0,
            },
        ];
        let mol_type = Arc::new(MoleculeType::new("WAT", sites_body, site_params));
        let mut system = System::new(mol_type, 10.0);
        system.add_molecule(Vec3::new(5.0, 5.0, 5.0), UnitQuaternion::identity());

        let mut output = Vec::new();
        write_pqr_to(&mut output, &system, 1.0).unwrap();
        let result = String::from_utf8(output).unwrap();

        assert!(result.contains("ATOM"));
        assert!(result.contains("WAT"));
        assert!(result.contains("END"));
        // Check that we have 2 atom lines plus END
        assert_eq!(result.lines().count(), 3);
    }
}
