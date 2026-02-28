// Normalized cross-power spectrum: A * conj(B) / (|A * conj(B)| + eps)

struct Params {
    count: u32,
}

@group(0) @binding(0) var<storage, read> a: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.count {
        return;
    }

    let va = a[idx];
    let vb = b[idx];

    // A * conj(B)
    let re = va.x * vb.x + va.y * vb.y;
    let im = va.y * vb.x - va.x * vb.y;

    let mag = sqrt(re * re + im * im) + 1e-10;
    out[idx] = vec2<f32>(re / mag, im / mag);
}
