struct Params {
    epsilon: f32,
    sigma: f32,
    yukawa_a: f32,
    kappa: f32,
    charge: f32,
    _pad: f32,
}

struct Uniforms {
    n_sites_i: u32,
    n_sites_other: u32,
    box_length: f32,
    cutoff_sq: f32,
}

@group(0) @binding(0) var<storage, read> pos_i: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> pos_other: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> params_i: array<Params>;
@group(0) @binding(3) var<storage, read> params_other: array<Params>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<push_constant> u: Uniforms;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let i = gid.x;  // site index in molecule i
    var energy = 0.0f;

    if (i < u.n_sites_i) {
        let pi = pos_i[i].xyz;
        let par_i = params_i[i];
        let box_len = u.box_length;

        // Loop over all sites in other molecules
        for (var j = 0u; j < u.n_sites_other; j++) {
            var dr = pos_other[j].xyz - pi;

            // Minimum image convention
            dr -= box_len * round(dr / box_len);

            let r_sq = dot(dr, dr);
            if (r_sq < u.cutoff_sq && r_sq > 1e-6) {
                let r = sqrt(r_sq);
                let par_j = params_other[j];

                // Lorentz-Berthelot combining rules
                let eps = sqrt(par_i.epsilon * par_j.epsilon);
                let sig = 0.5 * (par_i.sigma + par_j.sigma);

                // Lennard-Jones: 4ε[(σ/r)^12 - (σ/r)^6]
                let s2 = sig * sig / r_sq;
                let s6 = s2 * s2 * s2;
                energy += 4.0 * eps * s6 * (s6 - 1.0);

                // Yukawa: A * exp(-κr) / r
                let A = sqrt(par_i.yukawa_a * par_j.yukawa_a);
                if (A > 0.0) {
                    energy += A * exp(-par_i.kappa * r) / r;
                }
            }
        }
    }

    // Workgroup parallel reduction
    shared_sum[lid.x] = energy;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (lid.x < s) {
            shared_sum[lid.x] += shared_sum[lid.x + s];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        output[wid.x] = shared_sum[0];
    }
}
