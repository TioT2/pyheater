// Isosurface compute shader

/// Environment
struct Env {
    world_view_projection: mat4x4<f32>,
    level: f32,
    width: u32,
    height: u32,
    depth: u32,
}

@group(0) @binding(0) var<uniform> env: Env;
@group(0) @binding(1) var<storage, read_write> sample: array<f32>;
@group(0) @binding(2) var<storage, read_write> vertex: array<vec3<f32>>;

fn at(idx: vec3<u32>) -> u32 {
   return env.width * (env.height * idx.z + idx.y) + idx.x;
}

fn edgep(c0: vec3<u32>, c1: vec3<u32>) -> vec4<f32> {
    let w0 = sample[at(c0)] - env.level;
    let w1 = sample[at(c1)] - env.level;

    if w0 * w1 >= 0 {
        return vec4<f32>(0.0);
    }

    let t = clamp(w0 / (w0 - w1), 0.0, 1.0);
    return vec4<f32>(vec3<f32>(c0) * (1.0 - t) + vec3<f32>(c1) * t, 1.0);
}

@compute @workgroup_size(1, 1, 1)
fn main(
   @builtin(workgroup_id) id: vec3<u32>
) {
    let sps = vec4<f32>(0.0)
        + edgep(id + vec3(0, 0, 0), id + vec3(0, 0, 1))
        + edgep(id + vec3(0, 1, 0), id + vec3(0, 1, 1))
        + edgep(id + vec3(1, 0, 0), id + vec3(1, 0, 1))
        + edgep(id + vec3(1, 1, 0), id + vec3(1, 1, 1))

        + edgep(id + vec3(0, 0, 0), id + vec3(0, 1, 0))
        + edgep(id + vec3(0, 0, 1), id + vec3(0, 1, 1))
        + edgep(id + vec3(1, 0, 0), id + vec3(1, 1, 0))
        + edgep(id + vec3(1, 0, 1), id + vec3(1, 1, 1))

        + edgep(id + vec3(0, 0, 0), id + vec3(1, 0, 0))
        + edgep(id + vec3(0, 0, 1), id + vec3(1, 0, 1))
        + edgep(id + vec3(0, 1, 0), id + vec3(1, 1, 0))
        + edgep(id + vec3(0, 1, 1), id + vec3(1, 1, 1))
        ;

    vertex[at(id)] = sps.xyz / sps.w;
}
