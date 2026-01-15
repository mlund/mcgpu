//! Uncached GPU backend for energy calculations
//!
//! This backend recomputes energies from scratch on every call.
//! Useful for comparison with the cached version.

use crate::types::{SiteParams, System};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Push constants for the compute shader
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Uniforms {
    n_sites_i: u32,
    n_sites_other: u32,
    box_length: f32,
    cutoff_sq: f32,
}

/// Uncached GPU backend for computing single-molecule interaction energy
pub struct GpuUncachedEnergyBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Static buffers (params don't change)
    params_i: wgpu::Buffer,
    params_other: wgpu::Buffer,

    // Dynamic buffers (positions change each step)
    positions_i: wgpu::Buffer,
    positions_other: wgpu::Buffer,

    // Output
    energy_out: wgpu::Buffer,
    staging: wgpu::Buffer,

    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,

    n_sites_per_mol: u32,
    n_molecules: u32,
    workgroup_size: u32,
    cutoff_sq: f32,
}

impl GpuUncachedEnergyBackend {
    pub async fn new_async(system: &System, cutoff: f32) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("Failed to find GPU adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("MC Simulator"),
                    required_features: wgpu::Features::PUSH_CONSTANTS,
                    required_limits: wgpu::Limits {
                        max_push_constant_size: 16,
                        ..Default::default()
                    },
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        Self::with_device(Arc::new(device), Arc::new(queue), system, cutoff)
    }

    pub fn with_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        system: &System,
        cutoff: f32,
    ) -> Self {
        let n_sites = system.mol_type.n_sites() as u32;
        let n_mol = system.n_molecules() as u32;
        let n_other_sites = (n_mol - 1) * n_sites;
        let workgroup_size = 256u32;
        let n_workgroups = n_sites.div_ceil(workgroup_size);

        // Position buffer for molecule being evaluated
        let positions_i = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pos_i"),
            size: (n_sites as usize * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Position buffer for all other molecules
        let positions_other = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pos_other"),
            size: (n_other_sites as usize * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Parameters for single molecule (static)
        let params_i = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_i"),
            contents: bytemuck::cast_slice(&system.mol_type.site_params),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Parameters for N-1 molecules (all identical, static)
        let params_other_data: Vec<SiteParams> = (0..n_mol - 1)
            .flat_map(|_| system.mol_type.site_params.iter().copied())
            .collect();
        let params_other = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_other"),
            contents: bytemuck::cast_slice(&params_other_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Output: one partial sum per workgroup
        let energy_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("energy_out"),
            size: (n_workgroups * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer for CPU readback
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n_workgroups * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("energy_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("energy.wgsl").into()),
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("energy_bgl"),
            entries: &[
                bgl_entry(0, true),  // positions_i
                bgl_entry(1, true),  // positions_other
                bgl_entry(2, true),  // params_i
                bgl_entry(3, true),  // params_other
                bgl_entry(4, false), // output
            ],
        });

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("energy_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("energy_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            params_i,
            params_other,
            positions_i,
            positions_other,
            energy_out,
            staging,
            pipeline,
            bind_group_layout,
            n_sites_per_mol: n_sites,
            n_molecules: n_mol,
            workgroup_size,
            cutoff_sq: cutoff * cutoff,
        }
    }

    /// Compute interaction energy of molecule `mol_idx` with all other molecules
    pub async fn molecule_energy(&self, system: &System, mol_idx: usize) -> f64 {
        // Upload positions of molecule i
        self.queue.write_buffer(
            &self.positions_i,
            0,
            bytemuck::cast_slice(&system.molecules[mol_idx].sites_lab),
        );

        // Upload positions of all OTHER molecules (excluding mol_idx)
        let other_positions: Vec<[f32; 4]> = system
            .molecules
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != mol_idx)
            .flat_map(|(_, m)| m.sites_lab.iter().copied())
            .collect();
        self.queue.write_buffer(
            &self.positions_other,
            0,
            bytemuck::cast_slice(&other_positions),
        );

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.positions_i.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.positions_other.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_i.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_other.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.energy_out.as_entire_binding(),
                },
            ],
        });

        let uniforms = Uniforms {
            n_sites_i: self.n_sites_per_mol,
            n_sites_other: (self.n_molecules - 1) * self.n_sites_per_mol,
            box_length: system.box_length as f32,
            cutoff_sq: self.cutoff_sq,
        };

        let n_workgroups = self.n_sites_per_mol.div_ceil(self.workgroup_size);

        // Encode and submit
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("energy_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("energy_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&uniforms));
            pass.dispatch_workgroups(n_workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.energy_out,
            0,
            &self.staging,
            0,
            (n_workgroups * 4) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Readback
        let slice = self.staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().expect("Buffer mapping failed");

        let data = slice.get_mapped_range();
        let partials: &[f32] = bytemuck::cast_slice(&data);
        let total: f64 = partials.iter().map(|&x| x as f64).sum();
        drop(data);
        self.staging.unmap();

        total
    }

    /// Invalidate cache for molecule (no-op for uncached backend)
    pub fn invalidate_molecule(&mut self, _mol_idx: usize) {
        // No-op: uncached backend recomputes everything
    }

    /// Blocking version for convenience
    pub fn molecule_energy_blocking(&self, system: &System, mol_idx: usize) -> f64 {
        pollster::block_on(self.molecule_energy(system, mol_idx))
    }

    /// Compute total system energy (inter-molecular only)
    pub async fn total_energy(&self, system: &System) -> f64 {
        let mut total = 0.0;
        for i in 0..system.n_molecules() {
            total += self.molecule_energy(system, i).await;
        }
        // Each pair counted twice (i-j and j-i), so divide by 2
        total * 0.5
    }

    /// Blocking version of total_energy
    pub fn total_energy_blocking(&self, system: &System) -> f64 {
        pollster::block_on(self.total_energy(system))
    }
}

fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
