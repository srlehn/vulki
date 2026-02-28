// In-place fftshift: swap quadrants of a 2D float buffer.
// Quadrant layout: [Q0 Q1] -> [Q3 Q2]
//                  [Q2 Q3]    [Q1 Q0]

struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    let half_total = (params.width * params.height) / 2u;
    if idx >= half_total {
        return;
    }

    let hw = params.width / 2u;
    let hh = params.height / 2u;

    // Map linear index to (x, y) in the top half of the image.
    let x = idx % params.width;
    let y = idx / params.width;

    // Swap with diagonally opposite position.
    let sx = (x + hw) % params.width;
    let sy = (y + hh) % params.height;

    let idx_a = y * params.width + x;
    let idx_b = sy * params.width + sx;

    let tmp = data[idx_a];
    data[idx_a] = data[idx_b];
    data[idx_b] = tmp;
}
