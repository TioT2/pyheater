// Heat transfer simulation shader

/// Environment
struct Env {
    step: f32,
    env_temp: f32,
    delta_time: f32,
    _pad0: f32,
    map_shape: vec3<u32>,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> env: Env;

@group(0) @binding(1) var<storage, read_write> capacity: array<f32>;
@group(0) @binding(2) var<storage, read_write> conductivity: array<f32>;
@group(0) @binding(3) var<storage, read_write> temp: array<f32>;
@group(0) @binding(4) var<storage, read_write> temp_dst: array<f32>;

fn at(idx: vec3<u32>) -> u32 {
   return env.map_shape.x * (env.map_shape.y * idx.z + idx.y) + idx.x;
}

@compute @workgroup_size(1, 1, 1)
fn main(
   @builtin(workgroup_id) id: vec3<u32>
) {
    // Skip negative conductivity
    let cond: f32 = conductivity[at(id)];
    if cond <= 0 {
        return;
    }

    let ctemp: f32 = temp[at(id)];
    var q_total: f32 = 0;

    let id_x = id + vec3(1, 0, 0);
    let id_y = id + vec3(0, 1, 0);
    let id_z = id + vec3(0, 0, 1);

    let cond_x: f32 = conductivity[at(id_x)];
    if cond_x > 0 {
       let ctemp_x: f32 = temp[at(id_x)];
       // let q = -cond_x * (env.step * env.step) * (ctemp_x - ctemp) / env.step;
       let q = -cond_x * (env.step * 0.01) * (ctemp_x - ctemp); // step is cm

       q_total -= q;
       temp_dst[at(id_x)] += q / capacity[at(id_x)];
    }

    let cond_y: f32 = conductivity[at(id_y)];
    if cond_y > 0 {
       let ctemp_y: f32 = temp[at(id_y)];
       // let q = -cond_y * (env.step * env.step) * (ctemp_y - ctemp) / env.step;
       let q = -cond_y * (env.step * 0.01) * (ctemp_y - ctemp); // step is cm

       q_total -= q;
       temp_dst[at(id_y)] += q / capacity[at(id_y)];
    }

    let cond_z: f32 = conductivity[at(id_z)];
    if cond_z > 0 {
       let ctemp_z: f32 = temp[at(id_z)];
       // let q = -cond_z * (env.step * env.step) * (ctemp_z - ctemp) / env.step;
       let q = -cond_z * (env.step * 0.01) * (ctemp_z - ctemp); // step is cm

       q_total -= q;
       temp_dst[at(id_z)] += q / capacity[at(id_z)];
    }

    temp_dst[at(id)] += q_total / capacity[at(id)];
}
