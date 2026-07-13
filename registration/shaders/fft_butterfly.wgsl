// In-place Cooley-Tukey DIT butterfly for one FFT stage.
// One thread per butterfly pair.

struct Params {
    stage: u32,     // current stage (0-indexed)
    n: u32,         // line length (power of 2)
    num_lines: u32, // number of lines to process
    axis: u32,      // 0 = row-wise, 1 = column-wise
    stride: u32,    // stride between elements along the other axis
    inverse: u32,   // 0 = forward FFT, 1 = inverse FFT
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> params: Params;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = id.x;
    // Each stage has n/2 butterflies per line.
    let pairs_per_line = params.n / 2u;
    let total = pairs_per_line * params.num_lines;
    if tid >= total {
        return;
    }

    let line = tid / pairs_per_line;
    let pair_idx = tid % pairs_per_line;

    let block_size = 1u << (params.stage + 1u);
    let half_block = 1u << params.stage;
    let block = pair_idx / half_block;
    let offset = pair_idx % half_block;

    let i_pos = block * block_size + offset;
    let j_pos = i_pos + half_block;

    // Compute linear indices based on axis.
    var idx_i: u32;
    var idx_j: u32;
    if params.axis == 0u {
        idx_i = line * params.stride + i_pos;
        idx_j = line * params.stride + j_pos;
    } else {
        idx_i = i_pos * params.stride + line;
        idx_j = j_pos * params.stride + line;
    }

    // Twiddle factor: exp(-2πi * offset / block_size) for forward,
    //                 exp(+2πi * offset / block_size) for inverse.
    var angle = -2.0 * PI * f32(offset) / f32(block_size);
    if params.inverse != 0u {
        angle = -angle;
    }
    let tw = vec2<f32>(cos(angle), sin(angle));

    let u = data[idx_i];
    let v = data[idx_j];

    // Complex multiply: tw * v
    let tv = vec2<f32>(tw.x * v.x - tw.y * v.y, tw.x * v.y + tw.y * v.x);

    data[idx_i] = u + tv;
    data[idx_j] = u - tv;
}
