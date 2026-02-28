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
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = x0 + 1u;
    let y1 = y0 + 1u;
    let fx = x - floor(x);
    let fy = y - floor(y);

    // Wrap to valid range.
    let cx0 = x0 % params.src_w;
    let cx1 = x1 % params.src_w;
    let cy0 = y0 % params.src_h;
    let cy1 = y1 % params.src_h;

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

    // Log-polar mapping (full 360° range, matching scikit-image convention).
    let log_r = f32(xi) / f32(params.dst_w) * params.log_rmax;
    let theta = f32(yi) / f32(params.dst_h) * 2.0 * PI;
    let r = exp(log_r);

    // Sample from DC (origin) of magnitude spectrum with wraparound.
    let sx = ((r * cos(theta)) % f32(params.src_w) + f32(params.src_w)) % f32(params.src_w);
    let sy = ((r * sin(theta)) % f32(params.src_h) + f32(params.src_h)) % f32(params.src_h);

    let val = sample_bilinear(sx, sy);

    // Apply Hann window in the log-r (x) direction to suppress spectral leakage.
    // The theta (y) direction is periodic due to magnitude spectrum symmetry.
    let wx = 0.5 * (1.0 - cos(2.0 * PI * f32(xi) / f32(params.dst_w)));

    // Store as complex with zero imaginary part.
    dst[idx] = vec2<f32>(val * wx, 0.0);
}
