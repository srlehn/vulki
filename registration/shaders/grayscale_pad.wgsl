// Convert RGBA u32 pixels to complex f32 grayscale with square crop + center padding.
// Reads raw RGBA from source image, crops centered square, zero-pads to pow2.

struct Params {
    src_w: u32,
    src_h: u32,
    src_stride: u32, // in pixels (bytes / 4)
    pad_size: u32,
    crop_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = params.pad_size * params.pad_size;
    if idx >= total {
        return;
    }

    let px = idx % params.pad_size;
    let py = idx / params.pad_size;

    // Map output position to crop coordinates.
    let pad_off = (params.pad_size - params.crop_size) / 2u;
    let cx = i32(px) - i32(pad_off);
    let cy = i32(py) - i32(pad_off);

    // Check if within crop region.
    if cx < 0 || cx >= i32(params.crop_size) || cy < 0 || cy >= i32(params.crop_size) {
        output[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    // Map crop to source image (centered).
    let src_off_x = (i32(params.src_w) - i32(params.crop_size)) / 2;
    let src_off_y = (i32(params.src_h) - i32(params.crop_size)) / 2;
    let sx = cx + src_off_x;
    let sy = cy + src_off_y;

    // Bounds check.
    if sx < 0 || sx >= i32(params.src_w) || sy < 0 || sy >= i32(params.src_h) {
        output[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    let rgba = input[u32(sy) * params.src_stride + u32(sx)];
    let r = f32(rgba & 0xffu) / 255.0;
    let g = f32((rgba >> 8u) & 0xffu) / 255.0;
    let b = f32((rgba >> 16u) & 0xffu) / 255.0;
    let gray = 0.299 * r + 0.587 * g + 0.114 * b;

    output[idx] = vec2<f32>(gray, 0.0);
}
