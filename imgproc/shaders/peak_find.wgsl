// Peak finding with subpixel refinement over complex correlation surface.
// Single workgroup, thread 0 does serial scan (TODO: parallel reduction).
//
// Mode 0 (logpolar): wraparound, angle/scale conversion, writes result[0..7]
// Mode 1 (translation): wraparound, negate, writes result[offset..offset+2]

struct Params {
    width: u32,
    height: u32,
    mode: u32,         // 0 = logpolar, 1 = translation
    result_offset: u32, // write offset in result array (mode 1 only)
    log_rmax: f32,     // log(maxRadius), mode 0 only
}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<storage, read> params: Params;

const PI: f32 = 3.14159265358979323846;

fn cmag(idx: u32) -> f32 {
    let c = input[idx];
    return sqrt(c.x * c.x + c.y * c.y);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.width * params.height;

    // Serial scan to find peak.
    var best_mag: f32 = -1.0;
    var best_idx: u32 = 0u;
    for (var i: u32 = 0u; i < total; i = i + 1u) {
        let m = cmag(i);
        if m > best_mag {
            best_mag = m;
            best_idx = i;
        }
    }

    let max_x = i32(best_idx % params.width);
    let max_y = i32(best_idx / params.width);
    let w = i32(params.width);
    let h = i32(params.height);

    var peak_x = f32(max_x);
    var peak_y = f32(max_y);

    // Subpixel X refinement with wraparound.
    let lx = ((max_x - 1) + w) % w;
    let rx = (max_x + 1) % w;
    let mag_l = cmag(u32(max_y * w + lx));
    let mag_c = best_mag;
    let mag_r = cmag(u32(max_y * w + rx));
    let denom_x = 2.0 * mag_c - mag_l - mag_r;
    if denom_x > 1e-10 {
        peak_x += (mag_l - mag_r) / (2.0 * denom_x);
    }

    // Subpixel Y refinement with wraparound.
    let uy = ((max_y - 1) + h) % h;
    let dy = (max_y + 1) % h;
    let mag_u = cmag(u32(uy * w + max_x));
    let mag_d = cmag(u32(dy * w + max_x));
    let denom_y = 2.0 * mag_c - mag_u - mag_d;
    if denom_y > 1e-10 {
        peak_y += (mag_u - mag_d) / (2.0 * denom_y);
    }

    // Wraparound: peaks in second half represent negative shifts.
    if peak_x > f32(w) / 2.0 {
        peak_x -= f32(w);
    }
    if peak_y > f32(h) / 2.0 {
        peak_y -= f32(h);
    }

    if params.mode == 0u {
        // Logpolar mode: convert peak to angle and scale.
        let angle_deg = peak_y / f32(h) * 180.0;
        let klog = f32(w) / params.log_rmax;
        let scale = exp(peak_x / klog);
        let angle_rad = angle_deg * PI / 180.0;

        result[0] = peak_x;
        result[1] = peak_y;
        result[2] = angle_deg;
        result[3] = scale;
        result[4] = cos(angle_rad);
        result[5] = sin(angle_rad);
        result[6] = -cos(angle_rad);  // 180° variant
        result[7] = -sin(angle_rad);
    } else {
        // Translation mode: negate (cross-power peak is at -translation).
        let off = params.result_offset;
        result[off] = -peak_x;
        result[off + 1u] = -peak_y;
        result[off + 2u] = best_mag;
    }
}
