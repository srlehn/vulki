package registration

import (
	"testing"

	"github.com/srlehn/vulki/vk"
)

func TestFftshift(t *testing.T) {
	ctx := testContext(t)

	// Create a simple 4x4 array where value = row*4+col.
	w, h := 4, 4
	n := w * h
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i)
	}
	// Before fftshift:
	// [0  1  2  3 ]
	// [4  5  6  7 ]
	// [8  9  10 11]
	// [12 13 14 15]

	// After fftshift (swap quadrants):
	// [10 11 8  9 ]
	// [14 15 12 13]
	// [2  3  0  1 ]
	// [6  7  4  5 ]

	usage := vk.BufferUsageStorageBufferBit
	buf, err := newTestTypedBuffer[float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer buf.Destroy(ctx)

	paramsBuf, err := ctx.CreateBuffer(32, vk.BufferUsageStorageBufferBit|vk.BufferUsageTransferDstBit)
	if err != nil {
		t.Fatal(err)
	}
	defer paramsBuf.Close()

	pipe := compilePipeline(t, ctx, fftshiftWGSL, []testBufferBinding{
		bb(0, buf.Buf),
		bb(1, paramsBuf),
	})

	if err := buf.UploadSlice(ctx, data); err != nil {
		t.Fatal(err)
	}

	params := encodeU32Params(uint32(w), uint32(h))
	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	rec.UpdateBuffer(paramsBuf, 0, params)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(pipe)
	rec.Dispatch(uint32(n/2+63)/64, 1, 1)
	rec.Barrier(buf.Buf)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	result, err := buf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}

	expected := []float32{
		10, 11, 8, 9,
		14, 15, 12, 13,
		2, 3, 0, 1,
		6, 7, 4, 5,
	}

	t.Logf("Result:   %v", result)
	t.Logf("Expected: %v", expected)

	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("mismatch at %d: got %f, want %f", i, result[i], expected[i])
		}
	}
}
