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

    // Clamp to valid range.
    let cx0 = min(x0, params.src_w - 1u);
    let cx1 = min(x1, params.src_w - 1u);
    let cy0 = min(y0, params.src_h - 1u);
    let cy1 = min(y1, params.src_h - 1u);

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

    // Log-polar mapping.
    let log_r = f32(xi) / f32(params.dst_w) * params.log_rmax;
    let theta = f32(yi) / f32(params.dst_h) * PI;
    let r = exp(log_r);

    // Sample from center of magnitude spectrum (implicit FFT shift).
    let cx = f32(params.src_w) * 0.5;
    let cy = f32(params.src_h) * 0.5;
    let sx = cx + r * cos(theta);
    let sy = cy + r * sin(theta);

    // Boundary check — output 0 if out of range.
    var val = 0.0;
    if sx >= 0.0 && sx < f32(params.src_w) - 1.0 && sy >= 0.0 && sy < f32(params.src_h) - 1.0 {
        val = sample_bilinear(sx, sy);
    }

    // Store as complex with zero imaginary part.
    dst[idx] = vec2<f32>(val, 0.0);
}
