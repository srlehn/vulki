// Reduce the per-workgroup maxima produced by peak_find.wgsl.

@group(0) @binding(0) var<storage, read> scratch: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

var<workgroup> partial: array<vec2<f32>, 64>;

fn reduce_at(local_index: u32, offset: u32) {
    if local_index < offset {
        let other = partial[local_index + offset];
        if other.x > partial[local_index].x {
            partial[local_index] = other;
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let local_index = local_id.x;
    partial[local_index] = scratch[local_index];

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
        // result[11] and result[15] are scratch slots in the result layout.
        result[11] = sqrt(max(partial[0].x, 0.0));
        result[15] = partial[0].y;
    }
}
