// Highpass emphasis filter per Reddy & Chatterji (1996) eq. 23-24.
// Applied to fftshifted magnitude spectrum (DC at center).
//
// Paper formula (centered coordinates -0.5 ≤ ξ,η ≤ 0.5):
//   X(ξ,η) = cos(πξ) * cos(πη)
//   H(ξ,η) = (1.0 - X) * (2.0 - X)
//
// In pixel coordinates with DC at (w/2, h/2):
//   ξ = (x - w/2) / w,  η = (y - h/2) / h

struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> params: Params;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let total = params.width * params.height;
    if idx >= total {
        return;
    }
    let x = idx % params.width;
    let y = idx / params.width;

    // Centered normalized coordinates: ξ ∈ [-0.5, 0.5], η ∈ [-0.5, 0.5]
    let xi = f32(x) / f32(params.width) - 0.5;
    let eta = f32(y) / f32(params.height) - 0.5;

    let X = cos(PI * xi) * cos(PI * eta);
    let H = (1.0 - X) * (2.0 - X);

    data[idx] = data[idx] * H;
}
