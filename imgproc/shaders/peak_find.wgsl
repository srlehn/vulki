// Parallel peak finding with subpixel refinement over complex correlation surface.
// Single workgroup of 256 threads: parallel scan → tree reduction → subpixel → write results.
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

const WG_SIZE: u32 = 256u;
const PI: f32 = 3.14159265358979323846;

var<workgroup> shared_mag: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

fn cmag(idx: u32) -> f32 {
    let c = input[idx];
    return sqrt(c.x * c.x + c.y * c.y);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let total = params.width * params.height;

    // Phase 1: Each thread scans its portion of the array.
    var best_mag: f32 = -1.0;
    var best_idx: u32 = 0u;
    var i = tid;
    loop {
        if i >= total {
            break;
        }
        let m = cmag(i);
        if m > best_mag {
            best_mag = m;
            best_idx = i;
        }
        i += WG_SIZE;
    }

    shared_mag[tid] = best_mag;
    shared_idx[tid] = best_idx;
    workgroupBarrier();

    // Phase 2: Tree reduction.
    var s = WG_SIZE / 2u;
    loop {
        if s == 0u {
            break;
        }
        if tid < s {
            if shared_mag[tid + s] > shared_mag[tid] {
                shared_mag[tid] = shared_mag[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
            }
        }
        workgroupBarrier();
        s = s / 2u;
    }

    // Phase 3: Thread 0 does subpixel refinement and writes results.
    if tid != 0u {
        return;
    }

    let max_idx = shared_idx[0];
    let peak_mag = shared_mag[0];
    let max_x = i32(max_idx % params.width);
    let max_y = i32(max_idx / params.width);
    let w = i32(params.width);
    let h = i32(params.height);

    var peak_x = f32(max_x);
    var peak_y = f32(max_y);

    // Subpixel X refinement with wraparound.
    let lx = ((max_x - 1) + w) % w;
    let rx = (max_x + 1) % w;
    let mag_l = cmag(u32(max_y * w + lx));
    let mag_c = cmag(u32(max_y * w + max_x));
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
        result[off + 2u] = peak_mag;
    }
}
