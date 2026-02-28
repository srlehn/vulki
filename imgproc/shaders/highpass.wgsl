// High-pass emphasis filter: (1 - cos(pi*x/W)) * (1 - cos(pi*y/H))
// Applied in-place to magnitude spectrum.

struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> params: Params;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }
    let x = idx % params.width;
    let y = idx / params.width;
    let hx = 1.0 - cos(PI * f32(x) / f32(params.width));
    let hy = 1.0 - cos(PI * f32(y) / f32(params.height));
    data[idx] = data[idx] * hx * hy;
}
