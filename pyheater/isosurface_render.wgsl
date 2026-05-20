// Isosurface rendering shader

/// Environment
struct Env {
    world_view_projection: mat4x4<f32>,
    level: f32,
    width: u32,
    height: u32,
    depth: u32,
}

@group(0) @binding(0) var<uniform> env: Env;
@group(0) @binding(1) var<storage, read> vertex: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read> sample: array<f32>;

fn at(idx: vec3<u32>) -> u32 {
   return env.width * (env.height * idx.z + idx.y) + idx.x;
}

const off_array = array<vec3<u32>, 3>(
    vec3<u32>(1, 0, 0),
    vec3<u32>(0, 1, 0),
    vec3<u32>(0, 0, 1),
);
const ind_map = array<u32, 6>(0, 1, 2, 1, 2, 3);

struct FSData {
    @builtin(position) out_pos: vec4f,
    @location(0) pos: vec3f,
}

@vertex
fn vs_main(@builtin(vertex_index) gpu_vertex: u32) -> FSData {
    let face = gpu_vertex / 18;
    let vbase = vec3<u32>(
        1 + face % (env.width - 2),
        1 + (face / (env.width - 2)) % (env.height - 2),
        1 + (face / (env.width - 2)) / (env.height - 2),
    );
    let dir = (gpu_vertex % 18) / 6;

    // Check if edge have signchange
    let da1 = off_array[(dir + 1) % 3];
    let da2 = off_array[(dir + 2) % 3];
    if (sample[at(vbase + da1 + da2)] - env.level) * (sample[at(vbase + vec3(1))] - env.level) > 0 {
        return FSData(
            vec4<f32>(0.0, 0.0, 0.0, 0.0),
            vec3<f32>(0.0, 0.0, 0.0)
        );
    }

    // Extract face vertex index and vertex itself
    let face_vt = ind_map[gpu_vertex % 6];
    let vt = vertex[at(vbase + da1 * (face_vt / 2) + da2 * (face_vt % 2))];

    return FSData(
        env.world_view_projection * vec4<f32>(vt, 1.0),
        vt
    );
}

@fragment
fn fs_main(fsd: FSData) -> @location(0) vec4<f32> {
    let n = (normalize(cross(dpdx(fsd.pos), dpdy(fsd.pos))) + 1) / 2;
    return vec4(n, 0);
}
