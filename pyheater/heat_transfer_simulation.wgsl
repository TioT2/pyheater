// Heat transfer simulation shader

/// Environment
struct Env {
    step: f32,
    env_temp: f32,
    delta_time: f32,
    cond: f32,

    cap: f32,
    env_heat_xchg: f32,
    _pad0: f32,
    _pad1: f32,

    map_shape: vec3<u32>,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> env: Env;
@group(0) @binding(1) var<storage, read> cell_flags: array<u32>;
@group(0) @binding(2) var<storage, read_write> temp: array<f32>;
@group(0) @binding(3) var<storage, read_write> temp_dst: array<f32>;

fn is_cell(idx: vec3<u32>) -> bool {
    let i = at(idx);
    return (cell_flags[i / 32] & (1u << (i % 32))) != 0;
}

fn at(idx: vec3<u32>) -> u32 {
    return env.map_shape.x * (env.map_shape.y * idx.z + idx.y) + idx.x;
}

// Heat exchange between cell and environment
fn xchg_heat_cell_env(ctemp: f32) -> f32 {
    return -env.env_heat_xchg * (env.env_temp - ctemp) * (env.step * 0.01) * (env.step * 0.01);
}

// Exchange cell's heat
fn xchg_heat_cell(id_i: vec3<u32>, ctemp: f32) -> f32 {
    if is_cell(id_i) {
        // cell -> cell
        let ctemp_i: f32 = temp[at(id_i)];

        // Q_i = -C * (dT / di) * di^2
        let q = -env.cond * (ctemp_i - ctemp) * (env.step * 0.01); // step is cm
        temp_dst[at(id_i)] = max(temp_dst[at(id_i)] + q / env.cap, 0.0);
        return q;
    } else {
        // cell -> env
        return xchg_heat_cell_env(ctemp);
    }
}

// Exchange env's heat
fn xchg_heat_env(id_i: vec3<u32>) {
    if is_cell(id_i) {
        // env -> cell
        temp_dst[at(id_i)] = max(temp_dst[at(id_i)] - xchg_heat_cell_env(temp[at(id_i)]) / env.cap, 0.0);
    } else {
        // env -> env exchange isn't needed
    }
}

@compute @workgroup_size(1, 1, 1)
fn main(
    @builtin(workgroup_id) id: vec3<u32>
) {
    // Help isosurface builder
    if is_cell(id) {
        let ctemp: f32 = temp[at(id)];
        var q_total: f32 = 0.0
            - xchg_heat_cell(id + vec3(1, 0, 0), ctemp)
            - xchg_heat_cell(id + vec3(0, 1, 0), ctemp)
            - xchg_heat_cell(id + vec3(0, 0, 1), ctemp);

        temp_dst[at(id)] = max(temp_dst[at(id)] + q_total / env.cap, 0.0);
    } else {
        xchg_heat_env(id + vec3(1, 0, 0));
        xchg_heat_env(id + vec3(0, 1, 0));
        xchg_heat_env(id + vec3(0, 0, 1));

        temp_dst[at(id)] = env.env_temp;
    }
}
