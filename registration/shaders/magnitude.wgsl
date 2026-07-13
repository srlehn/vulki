// Compute log-magnitude log(1 + |z|) from complex array.
// Per Reddy & Chatterji (1996) §III.A: "Fourier log-magnitude spectra
// is used instead of Fourier magnitude spectra for the log-polar conversion."

struct Params {
    count: u32,
}

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx >= params.count {
        return;
    }
    let c = input[idx];
    let mag = sqrt(c.x * c.x + c.y * c.y);
    output[idx] = log(1.0 + mag);
}
