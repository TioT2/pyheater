//! MeshModel shader

/// Environment
struct Env {
    world_view_projection: mat4x4<f32>,
    world_inv: mat4x4<f32>,
    color: vec3<f32>,
}

@group(0) @binding(0) var<uniform> env: Env;

struct VSData {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

struct FSData {
    @builtin(position) out_pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
}

@vertex
fn vs_main(d: VSData) -> FSData {
    return FSData(
        env.world_view_projection * vec4(d.position, 1.0),
        (env.world_inv * vec4(d.normal, 0.0)).xyz,
    );
}

@fragment
fn fs_main(d: FSData) -> @location(0) vec4<f32> {
    return vec4(env.color * abs(dot(normalize(d.normal), normalize(vec3<f32>(0.30, 0.47, 0.80)))), 0.0);
}
