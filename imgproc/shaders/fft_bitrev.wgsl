// In-place bit-reversal permutation for FFT.
// Swaps element pairs where bitrev(i) > i within each line.

struct Params {
    n: u32,         // line length (power of 2)
    num_lines: u32, // number of lines to process
    log2n: u32,     // log2(n)
    axis: u32,      // 0 = row-wise, 1 = column-wise
    stride: u32,    // for axis=0: width; for axis=1: width (stride between rows)
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> params: Params;

fn bit_reverse(x: u32, bits: u32) -> u32 {
    var v = x;
    var r = 0u;
    for (var i = 0u; i < bits; i = i + 1u) {
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return r;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let tid = id.x;
    let total = params.n * params.num_lines;
    if tid >= total {
        return;
    }

    let line = tid / params.n;
    let i = tid % params.n;
    let j = bit_reverse(i, params.log2n);

    if j <= i {
        return;
    }

    // Compute linear indices based on axis.
    var idx_i: u32;
    var idx_j: u32;
    if params.axis == 0u {
        // Row-wise: line is the row index.
        idx_i = line * params.stride + i;
        idx_j = line * params.stride + j;
    } else {
        // Column-wise: line is the column index.
        idx_i = i * params.stride + line;
        idx_j = j * params.stride + line;
    }

    let tmp = data[idx_i];
    data[idx_i] = data[idx_j];
    data[idx_j] = tmp;
}
