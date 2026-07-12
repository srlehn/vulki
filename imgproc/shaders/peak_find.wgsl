// Scan a complex correlation surface for its largest magnitude in parallel.
// Each workgroup reduces its portion to one scratch entry. A separate shader
// reduces those entries before peak_finalize performs subpixel refinement.

struct Params {
    width: u32,
    height: u32,
    mode: u32,
    result_offset: u32,
    log_rmax: f32,
}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> scratch: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> params: Params;

var<workgroup> partial: array<vec2<f32>, 64>;

fn cmag2(idx: u32) -> f32 {
    let c = input[idx];
    return c.x * c.x + c.y * c.y;
}

fn reduce_at(local_index: u32, offset: u32) {
    if local_index < offset {
        let other = partial[local_index + offset];
        if other.x > partial[local_index].x {
            partial[local_index] = other;
        }
    }
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) workgroup_count: vec3<u32>,
) {
    let total = params.width * params.height;
    let local_index = local_id.x;

    // Keep loop-mutated state in workgroup memory for compatibility with the
    // naga version used by this project.
    partial[local_index] = vec2<f32>(-1.0, 0.0);

    var i = global_id.x;
    let stride = workgroup_count.x * 64u;
    loop {
        if i >= total {
            break;
        }
        let magnitude_squared = cmag2(i);
        if magnitude_squared > partial[local_index].x {
            partial[local_index] = vec2<f32>(magnitude_squared, f32(i));
        }
        i += stride;
    }

    workgroupBarrier();
    reduce_at(local_index, 32u);
    workgroupBarrier();
    reduce_at(local_index, 16u);
    workgroupBarrier();
    reduce_at(local_index, 8u);
    workgroupBarrier();
    reduce_at(local_index, 4u);
    workgroupBarrier();
    reduce_at(local_index, 2u);
    workgroupBarrier();
    reduce_at(local_index, 1u);
    workgroupBarrier();

    if local_index == 0u {
        scratch[workgroup_id.x] = partial[0];
    }
}
