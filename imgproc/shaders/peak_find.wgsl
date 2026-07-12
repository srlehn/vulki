// Scan a complex correlation surface for its largest magnitude.
//
// The running maximum is stored in result padding slots. A separate finalize
// dispatch performs subpixel refinement because older naga versions can lower
// reads after loop-mutated state incorrectly within the same shader invocation.

struct Params {
    width: u32,
    height: u32,
    mode: u32,
    result_offset: u32,
    log_rmax: f32,
}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<storage, read> params: Params;

fn cmag(idx: u32) -> f32 {
    let c = input[idx];
    return sqrt(c.x * c.x + c.y * c.y);
}

@compute @workgroup_size(1)
fn main() {
    let total = params.width * params.height;

    // result[11] and result[15] are padding slots in the public result layout.
    result[11] = -1.0;
    result[15] = 0.0;

    var i = 0u;
    loop {
        if i >= total {
            break;
        }
        let magnitude = cmag(i);
        if magnitude > result[11] {
            result[11] = magnitude;
            result[15] = f32(i);
        }
        i += 1u;
    }
}
