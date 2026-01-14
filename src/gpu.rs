use crate::types::System;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Push constants for pairwise compute shader
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct PairwiseUniforms {
    n_sites: u32,
    n_molecules: u32,
    mol_i: u32,
    box_length: f32,
    cutoff_sq: f32,
    _pad: u32,
}

/// GPU backend for computing interaction energies with pairwise caching
pub struct GpuEnergyBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Buffers for pairwise computation
    all_positions: wgpu::Buffer,
    all_params: wgpu::Buffer,
    pairwise_output: wgpu::Buffer,
    pairwise_staging: wgpu::Buffer,

    pairwise_pipeline: wgpu::ComputePipeline,
    pairwise_bind_group: wgpu::BindGroup,

    n_sites_per_mol: u32,
    n_molecules: u32,
    cutoff_sq: f32,

    // Cache: pairwise[i * n_molecules + j] = E_ij
    pairwise_cache: Vec<f64>,
    // Cached molecule energies (row sums)
    mol_energies: Vec<f64>,
    // Track which molecules need recomputation
    dirty: Vec<bool>,
    cache_initialized: bool,
}

impl GpuEnergyBackend {
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
                        max_push_constant_size: 24,
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
        let total_sites = n_mol * n_sites;

        // Buffer for ALL molecule positions
        let all_positions = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("all_positions"),
            size: (total_sites as usize * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Parameters (same for all molecules, just one copy)
        let all_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("all_params"),
            contents: bytemuck::cast_slice(&system.mol_type.site_params),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Output: one f32 per molecule for pairwise energies
        let pairwise_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pairwise_output"),
            size: (n_mol * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer for CPU readback
        let pairwise_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pairwise_staging"),
            size: (n_mol * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("energy_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("energy.wgsl").into()),
        });

        // Bind group layout for pairwise computation (bindings 5-7)
        let pairwise_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pairwise_bgl"),
            entries: &[
                bgl_entry(5, true),  // all_positions
                bgl_entry(6, true),  // all_params
                bgl_entry(7, false), // pairwise_output
            ],
        });

        // Pairwise pipeline
        let pairwise_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pairwise_pipeline_layout"),
            bind_group_layouts: &[&pairwise_bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..24,
            }],
        });

        let pairwise_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pairwise_pipeline"),
            layout: Some(&pairwise_pipeline_layout),
            module: &shader,
            entry_point: Some("compute_pairwise"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind group (bindings 5-7)
        let pairwise_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pairwise_bind_group"),
            layout: &pairwise_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: all_positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: all_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: pairwise_output.as_entire_binding(),
                },
            ],
        });

        // Initialize cache
        let cache_size = (n_mol * n_mol) as usize;
        let pairwise_cache = vec![0.0; cache_size];
        let mol_energies = vec![0.0; n_mol as usize];
        let dirty = vec![true; n_mol as usize]; // All dirty initially

        Self {
            device,
            queue,
            all_positions,
            all_params,
            pairwise_output,
            pairwise_staging,
            pairwise_pipeline,
            pairwise_bind_group,
            n_sites_per_mol: n_sites,
            n_molecules: n_mol,
            cutoff_sq: cutoff * cutoff,
            pairwise_cache,
            mol_energies,
            dirty,
            cache_initialized: false,
        }
    }

    /// Upload all positions to GPU
    fn upload_positions(&self, system: &System) {
        let all_pos: Vec<[f32; 4]> = system
            .molecules
            .iter()
            .flat_map(|m| m.sites_lab.iter().copied())
            .collect();
        self.queue.write_buffer(&self.all_positions, 0, bytemuck::cast_slice(&all_pos));
    }

    /// Compute row i of the pairwise matrix (E_ij for all j)
    async fn compute_row(&self, system: &System, mol_i: usize) -> Vec<f64> {
        let uniforms = PairwiseUniforms {
            n_sites: self.n_sites_per_mol,
            n_molecules: self.n_molecules,
            mol_i: mol_i as u32,
            box_length: system.box_length as f32,
            cutoff_sq: self.cutoff_sq,
            _pad: 0,
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pairwise_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pairwise_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pairwise_pipeline);
            pass.set_bind_group(0, &self.pairwise_bind_group, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&uniforms));
            // Dispatch N workgroups (one per target molecule)
            pass.dispatch_workgroups(self.n_molecules, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.pairwise_output,
            0,
            &self.pairwise_staging,
            0,
            (self.n_molecules * 4) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Readback
        let slice = self.pairwise_staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().expect("Buffer mapping failed");

        let data = slice.get_mapped_range();
        let row: Vec<f64> = bytemuck::cast_slice::<_, f32>(&data)
            .iter()
            .map(|&x| x as f64)
            .collect();
        drop(data);
        self.pairwise_staging.unmap();

        row
    }

    /// Initialize cache by computing all pairwise energies
    async fn initialize_cache(&mut self, system: &System) {
        self.upload_positions(system);

        let n = self.n_molecules as usize;
        for i in 0..n {
            let row = self.compute_row(system, i).await;
            for j in 0..n {
                self.pairwise_cache[i * n + j] = row[j];
            }
            self.mol_energies[i] = row.iter().sum();
            self.dirty[i] = false;
        }
        self.cache_initialized = true;
    }

    /// Update cache for a single molecule that moved
    async fn update_row(&mut self, system: &System, mol_i: usize) {
        self.upload_positions(system);

        let n = self.n_molecules as usize;

        // Save old row values
        let old_row: Vec<f64> = (0..n)
            .map(|j| self.pairwise_cache[mol_i * n + j])
            .collect();

        // Compute new row
        let new_row = self.compute_row(system, mol_i).await;

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
    pub async fn molecule_energy(&mut self, system: &System, mol_idx: usize) -> f64 {
        if !self.cache_initialized {
            self.initialize_cache(system).await;
        } else if self.dirty[mol_idx] {
            self.update_row(system, mol_idx).await;
        }
        self.mol_energies[mol_idx]
    }

    /// Notify that a move was accepted for molecule mol_idx
    /// (Currently unused since we invalidate before computing new energy)
    pub fn notify_move_accepted(&mut self, _mol_idx: usize) {
        // No-op: cache is already updated in molecule_energy
    }

    /// Invalidate cache for molecule that is about to move
    /// Call this BEFORE proposing a move to ensure e_new is recomputed
    pub fn invalidate_molecule(&mut self, mol_idx: usize) {
        self.dirty[mol_idx] = true;
    }

    /// Blocking version for convenience
    pub fn molecule_energy_blocking(&mut self, system: &System, mol_idx: usize) -> f64 {
        pollster::block_on(self.molecule_energy(system, mol_idx))
    }

    /// Compute total system energy (inter-molecular only)
    pub async fn total_energy(&mut self, system: &System) -> f64 {
        if !self.cache_initialized {
            self.initialize_cache(system).await;
        }
        // Update any dirty molecules
        for i in 0..self.n_molecules as usize {
            if self.dirty[i] {
                self.update_row(system, i).await;
            }
        }
        // Sum all molecule energies and divide by 2 (each pair counted twice)
        self.mol_energies.iter().sum::<f64>() * 0.5
    }

    /// Blocking version of total_energy
    pub fn total_energy_blocking(&mut self, system: &System) -> f64 {
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
