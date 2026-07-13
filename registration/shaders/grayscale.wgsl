// Convert RGBA u32 pixels to f32 grayscale using ITU-R BT.601 weights.
// Each u32 is packed RGBA (R in low byte).

struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }
    let rgba = input[idx];
    let r = f32(rgba & 0xffu) / 255.0;
    let g = f32((rgba >> 8u) & 0xffu) / 255.0;
    let b = f32((rgba >> 16u) & 0xffu) / 255.0;
    output[idx] = 0.299 * r + 0.587 * g + 0.114 * b;
}
