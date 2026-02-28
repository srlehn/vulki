// High-pass emphasis filter that only zeroes DC, not entire frequency axes.
// Uses h(x,y) = 1 - (1-hx)*(1-hy) where hx,hy are scaled Hann components.
// This zeroes only (0,0) while preserving the u=0 and v=0 axes.

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
    let hx = 0.5 * (1.0 - cos(2.0 * PI * f32(x) / f32(params.width)));
    let hy = 0.5 * (1.0 - cos(2.0 * PI * f32(y) / f32(params.height)));
    // Only zero at DC (0,0), not along entire axes.
    let h = 1.0 - (1.0 - hx) * (1.0 - hy);
    data[idx] = data[idx] * h;
}
