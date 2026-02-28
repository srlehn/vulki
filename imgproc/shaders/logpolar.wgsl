// Remap magnitude spectrum to log-polar coordinates with bilinear sampling.

struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    log_rmax: f32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> params: Params;

const PI: f32 = 3.14159265358979323846;

fn sample_bilinear(x: f32, y: f32) -> f32 {
    // Clamp to valid range (matching scikit-image behavior on fftshifted data).
    let cx = clamp(x, 0.0, f32(params.src_w - 1u));
    let cy = clamp(y, 0.0, f32(params.src_h - 1u));

    let x0 = u32(floor(cx));
    let y0 = u32(floor(cy));
    let x1 = min(x0 + 1u, params.src_w - 1u);
    let y1 = min(y0 + 1u, params.src_h - 1u);
    let fx = cx - floor(cx);
    let fy = cy - floor(cy);

    let cx0 = x0;
    let cx1 = x1;
    let cy0 = y0;
    let cy1 = y1;

    let v00 = src[cy0 * params.src_w + cx0];
    let v10 = src[cy0 * params.src_w + cx1];
    let v01 = src[cy1 * params.src_w + cx0];
    let v11 = src[cy1 * params.src_w + cx1];

    let top = v00 * (1.0 - fx) + v10 * fx;
    let bot = v01 * (1.0 - fx) + v11 * fx;
    return top * (1.0 - fy) + bot * fy;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = params.dst_w * params.dst_h;
    if idx >= total {
        return;
    }

    let xi = idx % params.dst_w;
    let yi = idx / params.dst_w;

    // Log-polar mapping (180° range; magnitude spectrum has 180° symmetry).
    let log_r = f32(xi) / f32(params.dst_w) * params.log_rmax;
    let theta = f32(yi) / f32(params.dst_h) * PI;
    let r = exp(log_r);

    // Sample from center of fftshifted magnitude spectrum.
    let cx = f32(params.src_w) / 2.0;
    let cy = f32(params.src_h) / 2.0;
    let sx = cx + r * cos(theta);
    let sy = cy + r * sin(theta);

    let val = sample_bilinear(sx, sy);

    // Store as complex with zero imaginary part (no windowing, matching scikit-image).
    dst[idx] = vec2<f32>(val, 0.0);
}
