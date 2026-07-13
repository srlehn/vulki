// Refine and convert the peak selected by peak_find.wgsl.

struct Params {
    width: u32,
    height: u32,
    mode: u32,          // 0 = log-polar, 1 = translation
    result_offset: u32, // write offset in result array for translation
    log_rmax: f32,      // log(maxRadius), used for log-polar conversion
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
fn main() {
    let max_idx = u32(result[15]);
    let global_mag = result[11];
    let confidence = global_mag / f32(params.width * params.height);
    let max_x = i32(max_idx % params.width);
    let max_y = i32(max_idx / params.width);
    let width = i32(params.width);
    let height = i32(params.height);

    var peak_x = f32(max_x);
    var peak_y = f32(max_y);

    let left_x = ((max_x - 1) + width) % width;
    let right_x = (max_x + 1) % width;
    let mag_left = cmag(u32(max_y * width + left_x));
    let mag_right = cmag(u32(max_y * width + right_x));
    let denom_x = 2.0 * global_mag - mag_left - mag_right;
    if denom_x > 1e-10 {
        peak_x += (mag_left - mag_right) / (2.0 * denom_x);
    }

    let upper_y = ((max_y - 1) + height) % height;
    let lower_y = (max_y + 1) % height;
    let mag_upper = cmag(u32(upper_y * width + max_x));
    let mag_lower = cmag(u32(lower_y * width + max_x));
    let denom_y = 2.0 * global_mag - mag_upper - mag_lower;
    if denom_y > 1e-10 {
        peak_y += (mag_upper - mag_lower) / (2.0 * denom_y);
    }

    if peak_x > f32(width) / 2.0 {
        peak_x -= f32(width);
    }
    if peak_y > f32(height) / 2.0 {
        peak_y -= f32(height);
    }

    if params.mode == 0u {
        let angle_deg = peak_y / f32(height) * 180.0;
        let klog = f32(width) / params.log_rmax;
        let scale = exp(peak_x / klog);
        let angle_rad = angle_deg * PI / 180.0;

        result[0] = confidence;
        result[1] = peak_y;
        result[2] = angle_deg;
        result[3] = scale;
        result[4] = cos(angle_rad);
        result[5] = sin(angle_rad);
        result[6] = -cos(angle_rad);
        result[7] = -sin(angle_rad);
    } else {
        let offset = params.result_offset;
        result[offset] = -peak_x;
        result[offset + 1u] = -peak_y;
        result[offset + 2u] = confidence;
    }
}
