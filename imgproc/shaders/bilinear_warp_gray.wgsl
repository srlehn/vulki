// Bilinear warp: inverse rotation+scale of RGBA source, output as complex f32.
// Reads cos/sin/scale from the GPU result buffer (written by peak_find Phase 1).
// Combines BilinearWarp + grayPad + realToComplex into one shader.

struct Params {
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    pad_size: u32,
    crop_size: u32,
    warp_slot: u32, // 0: result[4,5] (cos,sin), 1: result[6,7] (-cos,-sin)
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> params: Params;
@group(0) @binding(3) var<storage, read> result: array<f32>;

fn rgba_to_gray(rgba: u32) -> f32 {
    let r = f32(rgba & 0xffu) / 255.0;
    let g = f32((rgba >> 8u) & 0xffu) / 255.0;
    let b = f32((rgba >> 16u) & 0xffu) / 255.0;
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

fn sample_gray(x: f32, y: f32) -> f32 {
    let cx = clamp(x, 0.0, f32(params.src_w - 1u));
    let cy = clamp(y, 0.0, f32(params.src_h - 1u));

    let x0 = u32(floor(cx));
    let y0 = u32(floor(cy));
    let x1 = min(x0 + 1u, params.src_w - 1u);
    let y1 = min(y0 + 1u, params.src_h - 1u);
    let fx = cx - floor(cx);
    let fy = cy - floor(cy);

    let v00 = rgba_to_gray(input[y0 * params.src_stride + x0]);
    let v10 = rgba_to_gray(input[y0 * params.src_stride + x1]);
    let v01 = rgba_to_gray(input[y1 * params.src_stride + x0]);
    let v11 = rgba_to_gray(input[y1 * params.src_stride + x1]);

    let top = v00 * (1.0 - fx) + v10 * fx;
    let bot = v01 * (1.0 - fx) + v11 * fx;
    return top * (1.0 - fy) + bot * fy;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = params.pad_size * params.pad_size;
    if idx >= total {
        return;
    }

    let px = idx % params.pad_size;
    let py = idx / params.pad_size;

    // Map output to crop coordinates.
    let pad_off = (params.pad_size - params.crop_size) / 2u;
    let cx = i32(px) - i32(pad_off);
    let cy = i32(py) - i32(pad_off);

    if cx < 0 || cx >= i32(params.crop_size) || cy < 0 || cy >= i32(params.crop_size) {
        output[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    // Map crop to source image coordinates (centered).
    let src_off_x = (i32(params.src_w) - i32(params.crop_size)) / 2;
    let src_off_y = (i32(params.src_h) - i32(params.crop_size)) / 2;
    let ix = f32(cx + src_off_x);
    let iy = f32(cy + src_off_y);

    // Read warp parameters from result buffer.
    let cos_slot = 4u + params.warp_slot * 2u; // slot 0: [4,5], slot 1: [6,7]
    let cos_a = result[cos_slot];
    // Phase 2 applies the inverse detected rotation, matching the CPU
    // BilinearWarp(imgA, -angle, scale) reference path.
    let sin_a = -result[cos_slot + 1u];
    let scale = result[3]; // scale is always at index 3

    // Inverse rotation+scale around image center.
    let img_cx = f32(params.src_w) / 2.0;
    let img_cy = f32(params.src_h) / 2.0;
    let dx = ix - img_cx;
    let dy = iy - img_cy;
    let sx = (dx * cos_a + dy * sin_a) / scale + img_cx;
    let sy = (-dx * sin_a + dy * cos_a) / scale + img_cy;

    let gray = sample_gray(sx, sy);
    output[idx] = vec2<f32>(gray, 0.0);
}
