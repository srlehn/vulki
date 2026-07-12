package imgproc

import (
	"errors"
	"fmt"
	"image"
	"image/png"
	"math"
	"math/cmplx"
	"os"
	"testing"

	"github.com/srlehn/vulki/compute"
	"github.com/srlehn/vulki/shader"
	"github.com/srlehn/vulki/vk"
)

// testContext creates a Vulkan compute context, skipping if unavailable.
func testContext(t *testing.T) *compute.Context {
	t.Helper()
	ctx, err := compute.NewContext()
	if err != nil {
		t.Skipf("no Vulkan compute context: %v", err)
	}
	t.Cleanup(func() { ctx.Close() })
	return ctx
}

// compilePipeline compiles a WGSL shader and creates a pipeline.
func compilePipeline(t *testing.T, ctx *compute.Context, wgsl string, bindings []compute.BufferBinding) *compute.Pipeline {
	t.Helper()
	spirv, err := shader.Compile(wgsl)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	pipe, err := ctx.CreateComputePipeline(spirv, bindings)
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { pipe.Destroy(ctx) })
	return pipe
}

func bb(binding uint32, buf *compute.Buffer) compute.BufferBinding {
	return compute.BufferBinding{Binding: binding, Buffer: buf}
}

// --- FFT round-trip test ---

func TestFFT_RoundTrip(t *testing.T) {
	ctx := testContext(t)

	w, h := 16, 16
	n := w * h
	usage := vk.BufferUsageStorageBufferBit

	dataBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer dataBuf.Destroy(ctx)

	paramsBuf, err := ctx.CreateBuffer(32, vk.BufferUsageStorageBufferBit|vk.BufferUsageTransferDstBit)
	if err != nil {
		t.Fatal(err)
	}
	defer paramsBuf.Destroy(ctx)

	// Create known input signal.
	input := make([][2]float32, n)
	for i := range input {
		x := i % w
		y := i / w
		input[i] = [2]float32{float32(math.Sin(2*math.Pi*float64(x)/float64(w)) + math.Cos(2*math.Pi*float64(y)/float64(h))), 0}
	}
	if err := dataBuf.UploadSlice(ctx, input); err != nil {
		t.Fatal(err)
	}

	bitrevPipe := compilePipeline(t, ctx, fftBitrevWGSL, []compute.BufferBinding{
		bb(0, dataBuf.Buf), bb(1, paramsBuf),
	})
	butterflyPipe := compilePipeline(t, ctx, fftButterflyWGSL, []compute.BufferBinding{
		bb(0, dataBuf.Buf), bb(1, paramsBuf),
	})

	// Forward FFT.
	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	recordFFT2D(rec, dataBuf.Buf.DeviceBuffer, paramsBuf.DeviceBuffer, bitrevPipe, butterflyPipe, w, h, false)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	// Inverse FFT.
	rec, err = ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	recordFFT2D(rec, dataBuf.Buf.DeviceBuffer, paramsBuf.DeviceBuffer, bitrevPipe, butterflyPipe, w, h, true)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	// Download and normalize.
	output, err := dataBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}
	normalizeComplex(output, n)

	// Compare.
	maxErr := float64(0)
	for i := range input {
		dr := math.Abs(float64(input[i][0] - output[i][0]))
		di := math.Abs(float64(input[i][1] - output[i][1]))
		maxErr = math.Max(maxErr, math.Max(dr, di))
	}
	t.Logf("FFT round-trip max error: %e", maxErr)
	if maxErr > 1e-3 {
		t.Errorf("FFT round-trip error too large: %e", maxErr)
	}
}

// --- FFT known spectrum test ---

func TestFFT_KnownSpectrum(t *testing.T) {
	ctx := testContext(t)

	// 1D-like test: 8x1 FFT of a known signal.
	// Use 8x1 = single row.
	w, h := 8, 1
	n := w * h
	usage := vk.BufferUsageStorageBufferBit

	dataBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer dataBuf.Destroy(ctx)

	paramsBuf, err := ctx.CreateBuffer(32, vk.BufferUsageStorageBufferBit|vk.BufferUsageTransferDstBit)
	if err != nil {
		t.Fatal(err)
	}
	defer paramsBuf.Destroy(ctx)

	// Delta at position 0: FFT should be all ones.
	input := make([][2]float32, n)
	input[0] = [2]float32{1, 0}
	if err := dataBuf.UploadSlice(ctx, input); err != nil {
		t.Fatal(err)
	}

	bitrevPipe := compilePipeline(t, ctx, fftBitrevWGSL, []compute.BufferBinding{
		bb(0, dataBuf.Buf), bb(1, paramsBuf),
	})
	butterflyPipe := compilePipeline(t, ctx, fftButterflyWGSL, []compute.BufferBinding{
		bb(0, dataBuf.Buf), bb(1, paramsBuf),
	})

	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	recordFFT2D(rec, dataBuf.Buf.DeviceBuffer, paramsBuf.DeviceBuffer, bitrevPipe, butterflyPipe, w, h, false)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	output, err := dataBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}

	t.Log("FFT of delta: ", output[:n])
	// All values should be (1, 0).
	for i := 0; i < n; i++ {
		if math.Abs(float64(output[i][0])-1) > 1e-5 || math.Abs(float64(output[i][1])) > 1e-5 {
			t.Errorf("FFT[%d] = (%v, %v), want (1, 0)", i, output[i][0], output[i][1])
		}
	}
}

// --- Magnitude test ---

func TestMagnitude(t *testing.T) {
	ctx := testContext(t)

	n := 8
	usage := vk.BufferUsageStorageBufferBit

	complexBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer complexBuf.Destroy(ctx)

	magBuf, err := compute.NewTypedBuffer[float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer magBuf.Destroy(ctx)

	paramsBuf, err := ctx.CreateBuffer(32, vk.BufferUsageStorageBufferBit|vk.BufferUsageTransferDstBit)
	if err != nil {
		t.Fatal(err)
	}
	defer paramsBuf.Destroy(ctx)

	input := [][2]float32{
		{3, 4},   // |z| = 5
		{1, 0},   // |z| = 1
		{0, 1},   // |z| = 1
		{0, 0},   // |z| = 0
		{-3, 4},  // |z| = 5
		{1, 1},   // |z| = sqrt(2)
		{5, 12},  // |z| = 13
		{-1, -1}, // |z| = sqrt(2)
	}
	if err := complexBuf.UploadSlice(ctx, input); err != nil {
		t.Fatal(err)
	}

	pipe := compilePipeline(t, ctx, magnitudeWGSL, []compute.BufferBinding{
		bb(0, complexBuf.Buf), bb(1, magBuf.Buf), bb(2, paramsBuf),
	})

	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	params := encodeU32Params(uint32(n))
	rec.UpdateBuffer(paramsBuf.DeviceBuffer, 0, params)
	rec.BarrierTransferToCompute(paramsBuf.DeviceBuffer)
	rec.Bind(pipe)
	rec.Dispatch(1, 1, 1)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	output, err := magBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}

	expected := []float64{
		math.Log1p(5),
		math.Log1p(1),
		math.Log1p(1),
		0,
		math.Log1p(5),
		math.Log1p(math.Sqrt(2)),
		math.Log1p(13),
		math.Log1p(math.Sqrt(2)),
	}
	for i, want := range expected {
		if math.Abs(float64(output[i])-want) > 1e-4 {
			t.Errorf("mag[%d] = %v, want %v", i, output[i], want)
		}
	}
}

// --- Highpass test ---

func TestHighpass(t *testing.T) {
	ctx := testContext(t)

	w, h := 4, 4
	n := w * h
	usage := vk.BufferUsageStorageBufferBit

	dataBuf, err := compute.NewTypedBuffer[float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer dataBuf.Destroy(ctx)

	paramsBuf, err := ctx.CreateBuffer(32, vk.BufferUsageStorageBufferBit|vk.BufferUsageTransferDstBit)
	if err != nil {
		t.Fatal(err)
	}
	defer paramsBuf.Destroy(ctx)

	// Fill with 1.0 everywhere.
	input := make([]float32, n)
	for i := range input {
		input[i] = 1.0
	}
	if err := dataBuf.UploadSlice(ctx, input); err != nil {
		t.Fatal(err)
	}

	pipe := compilePipeline(t, ctx, highpassWGSL, []compute.BufferBinding{
		bb(0, dataBuf.Buf), bb(1, paramsBuf),
	})

	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	params := encodeU32Params(uint32(w), uint32(h))
	rec.UpdateBuffer(paramsBuf.DeviceBuffer, 0, params)
	rec.BarrierTransferToCompute(paramsBuf.DeviceBuffer)
	rec.Bind(pipe)
	rec.Dispatch(1, 1, 1)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	output, err := dataBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// Verify the paper's H(xi,eta) = (1-X)(2-X), where
	// X = cos(pi*xi)cos(pi*eta) and the shifted DC is at the center.
	for y := 0; y < h; y++ {
		eta := float64(y)/float64(h) - 0.5
		for x := 0; x < w; x++ {
			xi := float64(x)/float64(w) - 0.5
			spectralX := math.Cos(math.Pi*xi) * math.Cos(math.Pi*eta)
			want := (1 - spectralX) * (2 - spectralX)
			if math.Abs(float64(output[y*w+x])-want) > 1e-5 {
				t.Errorf("highpass[%d,%d] = %v, want %v", x, y, output[y*w+x], want)
			}
		}
	}

	t.Logf("highpass output:\n")
	for y := 0; y < h; y++ {
		t.Logf("  row %d: %v", y, output[y*w:(y+1)*w])
	}
}

// --- Cross-power spectrum test ---

func TestCrosspower_IdenticalSignals(t *testing.T) {
	ctx := testContext(t)

	n := 16
	usage := vk.BufferUsageStorageBufferBit

	aBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer aBuf.Destroy(ctx)

	bBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer bBuf.Destroy(ctx)

	outBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer outBuf.Destroy(ctx)

	paramsBuf, err := ctx.CreateBuffer(32, vk.BufferUsageStorageBufferBit|vk.BufferUsageTransferDstBit)
	if err != nil {
		t.Fatal(err)
	}
	defer paramsBuf.Destroy(ctx)

	// Two identical non-trivial signals.
	signal := make([][2]float32, n)
	for i := range signal {
		signal[i] = [2]float32{float32(i + 1), float32(i * 2)}
	}
	if err := aBuf.UploadSlice(ctx, signal); err != nil {
		t.Fatal(err)
	}
	if err := bBuf.UploadSlice(ctx, signal); err != nil {
		t.Fatal(err)
	}

	pipe := compilePipeline(t, ctx, crosspowerWGSL, []compute.BufferBinding{
		bb(0, aBuf.Buf), bb(1, bBuf.Buf), bb(2, outBuf.Buf), bb(3, paramsBuf),
	})

	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	params := encodeU32Params(uint32(n), 1)
	rec.UpdateBuffer(paramsBuf.DeviceBuffer, 0, params)
	rec.BarrierTransferToCompute(paramsBuf.DeviceBuffer)
	rec.Bind(pipe)
	rec.Dispatch(1, 1, 1)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	output, err := outBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// A * conj(A) = |A|^2 (real, positive), so normalized result should be (1, 0).
	for i := 0; i < n; i++ {
		if math.Abs(float64(output[i][0])-1) > 1e-3 || math.Abs(float64(output[i][1])) > 1e-3 {
			t.Errorf("crosspower[%d] = (%v, %v), want (1, 0)", i, output[i][0], output[i][1])
		}
	}
}

func TestPeakFind_KnownLocation(t *testing.T) {
	ctx := testContext(t)

	const w, h = 16, 16
	usage := vk.BufferUsageStorageBufferBit
	input, err := compute.NewTypedBuffer[[2]float32](ctx, w*h, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Destroy(ctx)
	result, err := compute.NewTypedBuffer[float32](ctx, 16, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer result.Destroy(ctx)
	scratch, err := compute.NewTypedBuffer[[2]float32](ctx, int(peakWorkgroups), usage)
	if err != nil {
		t.Fatal(err)
	}
	defer scratch.Destroy(ctx)
	params, err := ctx.CreateBuffer(32, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer params.Destroy(ctx)

	values := make([][2]float32, w*h)
	values[7*w+5] = [2]float32{100, 0}
	if err := input.UploadSlice(ctx, values); err != nil {
		t.Fatal(err)
	}
	if err := params.Upload(ctx, encodePeakFindParams(w, h, 1, 0, 0)); err != nil {
		t.Fatal(err)
	}
	pipe := compilePipeline(t, ctx, peakFindWGSL, []compute.BufferBinding{
		bb(0, input.Buf), bb(1, scratch.Buf), bb(2, params),
	})
	reducePipe := compilePipeline(t, ctx, peakReduceWGSL, []compute.BufferBinding{
		bb(0, scratch.Buf), bb(1, result.Buf),
	})
	finalizePipe := compilePipeline(t, ctx, peakFinalizeWGSL, []compute.BufferBinding{
		bb(0, input.Buf), bb(1, result.Buf), bb(2, params),
	})

	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	rec.Bind(pipe)
	rec.Dispatch(peakWorkgroups, 1, 1)
	rec.Barrier(scratch.Buf.DeviceBuffer)
	rec.Bind(reducePipe)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(scratch.Buf.DeviceBuffer, result.Buf.DeviceBuffer)
	rec.Bind(finalizePipe)
	rec.Dispatch(1, 1, 1)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}
	got, err := result.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(got[0])+5) > 1e-4 || math.Abs(float64(got[1])+7) > 1e-4 {
		t.Fatalf("peak translation = (%v,%v), want (-5,-7); scratch magnitude=%v index=%v", got[0], got[1], got[11], got[15])
	}
	wantMagnitude := 100.0 / float64(w*h)
	if math.Abs(float64(got[2])-wantMagnitude) > 1e-4 {
		t.Fatalf("peak magnitude = %v, want %v", got[2], wantMagnitude)
	}
}

// --- Phase correlation of identical signals should peak at (0,0) ---

func TestCrosspower_IFFT_PeaksAtOrigin(t *testing.T) {
	ctx := testContext(t)

	w, h := 16, 16
	n := w * h
	usage := vk.BufferUsageStorageBufferBit

	aBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer aBuf.Destroy(ctx)

	bBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer bBuf.Destroy(ctx)

	outBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer outBuf.Destroy(ctx)

	paramsBuf, err := ctx.CreateBuffer(32, vk.BufferUsageStorageBufferBit|vk.BufferUsageTransferDstBit)
	if err != nil {
		t.Fatal(err)
	}
	defer paramsBuf.Destroy(ctx)

	// Create two identical signals, FFT them, cross-power, IFFT.
	signal := make([][2]float32, n)
	for i := range signal {
		x := i % w
		y := i / w
		signal[i] = [2]float32{float32(math.Sin(2*math.Pi*float64(x)/float64(w)) + 0.5*math.Cos(4*math.Pi*float64(y)/float64(h))), 0}
	}
	if err := aBuf.UploadSlice(ctx, signal); err != nil {
		t.Fatal(err)
	}
	if err := bBuf.UploadSlice(ctx, signal); err != nil {
		t.Fatal(err)
	}

	bitrevA := compilePipeline(t, ctx, fftBitrevWGSL, []compute.BufferBinding{bb(0, aBuf.Buf), bb(1, paramsBuf)})
	butterflyA := compilePipeline(t, ctx, fftButterflyWGSL, []compute.BufferBinding{bb(0, aBuf.Buf), bb(1, paramsBuf)})
	bitrevB := compilePipeline(t, ctx, fftBitrevWGSL, []compute.BufferBinding{bb(0, bBuf.Buf), bb(1, paramsBuf)})
	butterflyB := compilePipeline(t, ctx, fftButterflyWGSL, []compute.BufferBinding{bb(0, bBuf.Buf), bb(1, paramsBuf)})
	bitrevOut := compilePipeline(t, ctx, fftBitrevWGSL, []compute.BufferBinding{bb(0, outBuf.Buf), bb(1, paramsBuf)})
	butterflyOut := compilePipeline(t, ctx, fftButterflyWGSL, []compute.BufferBinding{bb(0, outBuf.Buf), bb(1, paramsBuf)})
	cpPipe := compilePipeline(t, ctx, crosspowerWGSL, []compute.BufferBinding{
		bb(0, aBuf.Buf), bb(1, bBuf.Buf), bb(2, outBuf.Buf), bb(3, paramsBuf),
	})

	// FFT both signals.
	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	recordFFT2D(rec, aBuf.Buf.DeviceBuffer, paramsBuf.DeviceBuffer, bitrevA, butterflyA, w, h, false)
	recordFFT2D(rec, bBuf.Buf.DeviceBuffer, paramsBuf.DeviceBuffer, bitrevB, butterflyB, w, h, false)

	// Cross-power.
	cpParams := encodeU32Params(uint32(n), 1)
	rec.UpdateBuffer(paramsBuf.DeviceBuffer, 0, cpParams)
	rec.BarrierTransferToCompute(paramsBuf.DeviceBuffer)
	rec.Bind(cpPipe)
	rec.Dispatch(uint32((n+63)/64), 1, 1)
	rec.Barrier(outBuf.Buf.DeviceBuffer)

	// IFFT.
	recordFFT2D(rec, outBuf.Buf.DeviceBuffer, paramsBuf.DeviceBuffer, bitrevOut, butterflyOut, w, h, true)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	output, err := outBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}
	normalizeComplex(output, n)

	px, py := find2DPeak(output, w, h)
	t.Logf("Cross-power peak of identical signals: (%v, %v)", px, py)
	if math.Abs(px) > 1 || math.Abs(py) > 1 {
		t.Errorf("peak at (%v, %v), want near (0, 0)", px, py)
	}
}

// --- Phase correlation with known translation ---

func TestPhaseCorrelation_KnownTranslation(t *testing.T) {
	ctx := testContext(t)

	w, h := 32, 32
	n := w * h
	usage := vk.BufferUsageStorageBufferBit

	aBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer aBuf.Destroy(ctx)

	bBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer bBuf.Destroy(ctx)

	outBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer outBuf.Destroy(ctx)

	paramsBuf, err := ctx.CreateBuffer(32, vk.BufferUsageStorageBufferBit|vk.BufferUsageTransferDstBit)
	if err != nil {
		t.Fatal(err)
	}
	defer paramsBuf.Destroy(ctx)

	// Signal A: rich signal with many frequency components.
	sigA := make([][2]float32, n)
	for i := range sigA {
		x := i % w
		y := i / w
		val := math.Sin(2*math.Pi*float64(x)/float64(w)) +
			0.7*math.Cos(4*math.Pi*float64(y)/float64(h)) +
			0.5*math.Sin(6*math.Pi*float64(x)/float64(w)) +
			0.3*math.Cos(2*math.Pi*float64(x+y)/float64(w)) +
			0.2*math.Sin(2*math.Pi*float64(x*y+1)/float64(w*h))
		sigA[i] = [2]float32{float32(val), 0}
	}

	// Signal B: same pattern shifted by (3, 5) with wraparound.
	shiftX, shiftY := 3, 5
	sigB := make([][2]float32, n)
	for i := range sigB {
		x := i % w
		y := i / w
		srcX := ((x-shiftX)%w + w) % w
		srcY := ((y-shiftY)%h + h) % h
		sigB[i] = sigA[srcY*w+srcX]
	}

	if err := aBuf.UploadSlice(ctx, sigA); err != nil {
		t.Fatal(err)
	}
	if err := bBuf.UploadSlice(ctx, sigB); err != nil {
		t.Fatal(err)
	}

	bitrevA := compilePipeline(t, ctx, fftBitrevWGSL, []compute.BufferBinding{bb(0, aBuf.Buf), bb(1, paramsBuf)})
	butterflyA := compilePipeline(t, ctx, fftButterflyWGSL, []compute.BufferBinding{bb(0, aBuf.Buf), bb(1, paramsBuf)})
	bitrevB := compilePipeline(t, ctx, fftBitrevWGSL, []compute.BufferBinding{bb(0, bBuf.Buf), bb(1, paramsBuf)})
	butterflyB := compilePipeline(t, ctx, fftButterflyWGSL, []compute.BufferBinding{bb(0, bBuf.Buf), bb(1, paramsBuf)})
	bitrevOut := compilePipeline(t, ctx, fftBitrevWGSL, []compute.BufferBinding{bb(0, outBuf.Buf), bb(1, paramsBuf)})
	butterflyOut := compilePipeline(t, ctx, fftButterflyWGSL, []compute.BufferBinding{bb(0, outBuf.Buf), bb(1, paramsBuf)})
	// B*conj(A) so IFFT peaks at +shift.
	cpPipe := compilePipeline(t, ctx, crosspowerWGSL, []compute.BufferBinding{
		bb(0, bBuf.Buf), bb(1, aBuf.Buf), bb(2, outBuf.Buf), bb(3, paramsBuf),
	})

	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	recordFFT2D(rec, aBuf.Buf.DeviceBuffer, paramsBuf.DeviceBuffer, bitrevA, butterflyA, w, h, false)
	recordFFT2D(rec, bBuf.Buf.DeviceBuffer, paramsBuf.DeviceBuffer, bitrevB, butterflyB, w, h, false)

	cpParams := encodeU32Params(uint32(n), 1)
	rec.UpdateBuffer(paramsBuf.DeviceBuffer, 0, cpParams)
	rec.BarrierTransferToCompute(paramsBuf.DeviceBuffer)
	rec.Bind(cpPipe)
	rec.Dispatch(uint32((n+63)/64), 1, 1)
	rec.Barrier(outBuf.Buf.DeviceBuffer)

	recordFFT2D(rec, outBuf.Buf.DeviceBuffer, paramsBuf.DeviceBuffer, bitrevOut, butterflyOut, w, h, true)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	output, err := outBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}
	normalizeComplex(output, n)

	px, py := find2DPeak(output, w, h)
	t.Logf("Translation peak: (%v, %v), expected (%d, %d)", px, py, shiftX, shiftY)
	if math.Abs(px-float64(shiftX)) > 1 || math.Abs(py-float64(shiftY)) > 1 {
		t.Errorf("peak at (%v, %v), want near (%d, %d)", px, py, shiftX, shiftY)
	}
}

// --- Log-polar remap test ---

func TestLogPolar_SamplesFromDC(t *testing.T) {
	ctx := testContext(t)

	// Create a magnitude spectrum with a known pattern:
	// Put a large value at (0,0) and unique values at specific locations.
	srcW, srcH := 16, 16
	srcN := srcW * srcH
	dstW, dstH := 16, 16
	dstN := dstW * dstH
	usage := vk.BufferUsageStorageBufferBit

	srcBuf, err := compute.NewTypedBuffer[float32](ctx, srcN, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer srcBuf.Destroy(ctx)

	dstBuf, err := compute.NewTypedBuffer[[2]float32](ctx, dstN, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer dstBuf.Destroy(ctx)

	paramsBuf, err := ctx.CreateBuffer(32, vk.BufferUsageStorageBufferBit|vk.BufferUsageTransferDstBit)
	if err != nil {
		t.Fatal(err)
	}
	defer paramsBuf.Destroy(ctx)

	// Create a spectrum where DC (0,0) has value 100 and all else is 0.
	spectrum := make([]float32, srcN)
	spectrum[0] = 100.0
	if err := srcBuf.UploadSlice(ctx, spectrum); err != nil {
		t.Fatal(err)
	}

	pipe := compilePipeline(t, ctx, logpolarWGSL, []compute.BufferBinding{
		bb(0, srcBuf.Buf), bb(1, dstBuf.Buf), bb(2, paramsBuf),
	})

	maxRadius := math.Sqrt(float64(srcW*srcW+srcH*srcH)) / 2.0
	logRmax := float32(math.Log(maxRadius))
	params := encodeLogPolarParams(uint32(srcW), uint32(srcH), uint32(dstW), uint32(dstH), logRmax)

	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	rec.UpdateBuffer(paramsBuf.DeviceBuffer, 0, params)
	rec.BarrierTransferToCompute(paramsBuf.DeviceBuffer)
	rec.Bind(pipe)
	rec.Dispatch(uint32((dstN+63)/64), 1, 1)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	output, err := dstBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// The log-polar output should sample near DC for small radii.
	// At (xi=0, yi=0): r = exp(0) = 1, theta = 0
	//   → samples spectrum at (1, 0) with wraparound.
	// At (xi=0, yi=dstH/2): r = 1, theta = π/2
	//   → samples spectrum at (0, 1) with wraparound.

	// Since only (0,0) has a non-zero value, most of the log-polar output
	// should be 0 (bilinear interpolation from the sole non-zero pixel).
	// The key test: log-polar should NOT sample centered at (srcW/2, srcH/2).

	// Put value at Nyquist (srcW/2, srcH/2) instead and verify it's NOT picked up at small radii.
	spectrum2 := make([]float32, srcN)
	spectrum2[(srcH/2)*srcW+(srcW/2)] = 100.0
	if err := srcBuf.UploadSlice(ctx, spectrum2); err != nil {
		t.Fatal(err)
	}

	rec, err = ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	rec.UpdateBuffer(paramsBuf.DeviceBuffer, 0, params)
	rec.BarrierTransferToCompute(paramsBuf.DeviceBuffer)
	rec.Bind(pipe)
	rec.Dispatch(uint32((dstN+63)/64), 1, 1)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	output2, err := dstBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// With DC-centered sampling, the Nyquist point should only appear at
	// large radii (near xi=dstW-1), not at small radii.
	// Check that (xi=0, yi=0) is near zero (not sampling from Nyquist).
	val00 := output2[0]
	t.Logf("logpolar[0,0] from Nyquist-only spectrum: real=%v", val00[0])
	if val00[0] > 1 {
		t.Errorf("logpolar[0,0] = %v, should be ~0 (not sampling from Nyquist center)", val00[0])
	}

	// Log some of the output for debugging.
	t.Log("Log-polar from DC-only spectrum (first row):")
	for x := 0; x < dstW; x++ {
		t.Logf("  lp[%d,0] = %v", x, output[x][0])
	}
	t.Log("Log-polar from Nyquist-only spectrum (first row):")
	for x := 0; x < dstW; x++ {
		t.Logf("  lp[%d,0] = %v", x, output2[x][0])
	}
}

// --- Full registration tests with real image content ---

func TestPhase1_RotationDetection(t *testing.T) {
	ctx := testContext(t)

	imgA := loadTestImage(t, "../testdata/snake.png")
	imgB := BilinearWarp(imgA, -12, 1)
	if fixture := os.Getenv("VULKI_ROTATION_FIXTURE"); fixture != "" {
		imgB = loadTestImage(t, fixture)
	}
	corr, err := newVulkanCorrelator(ctx, imgA.Bounds().Dx(), imgA.Bounds().Dy())
	if err != nil {
		t.Fatal(err)
	}
	defer corr.Close()

	result, err := corr.PhaseCorrelate(imgA, imgB)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("registration result: angle=%.3f scale=%.5f translation=(%.3f, %.3f)",
		result.Angle, result.Scale, result.Tx, result.Ty)

	angleDelta := math.Mod(result.Angle-12+180, 360) - 180
	if math.Abs(angleDelta) > 2 {
		t.Errorf("angle = %v degrees, want about 12 degrees", result.Angle)
	}
	if math.Abs(result.Scale-1) > 0.1 {
		t.Errorf("scale = %v, want about 1", result.Scale)
	}
	if math.Abs(result.Tx) > 1 || math.Abs(result.Ty) > 1 {
		t.Errorf("translation = (%v,%v), want about (0,0)", result.Tx, result.Ty)
	}
	if result.RotationConfidence <= minimumMatchConfidence || result.TranslationConfidence <= minimumMatchConfidence {
		t.Errorf("unexpected confidence: rotation=%v translation=%v",
			result.RotationConfidence, result.TranslationConfidence)
	}

	blank := image.NewRGBA(imgA.Bounds())
	if _, err := corr.PhaseCorrelate(blank, blank); !errors.Is(err, ErrLowConfidence) {
		t.Fatalf("blank-image error = %v, want ErrLowConfidence", err)
	}
}

func TestPhaseCorrelateGPU_KnownTransform(t *testing.T) {
	imgA := loadTestImage(t, "../testdata/snake.png")
	corr, err := NewCorrelator(
		imgA.Bounds().Dx(),
		imgA.Bounds().Dy(),
		WithBackend(BackendVulkan),
	)
	if err != nil {
		t.Skipf("no Vulkan GPU backend: %v", err)
	}
	defer corr.Close()
	if corr.Backend() != BackendVulkan {
		t.Fatalf("backend = %q, want %q", corr.Backend(), BackendVulkan)
	}
	before := corr.ctx.SubmissionCount()
	assertKnownTransform(t, corr, imgA)
	if got := corr.ctx.SubmissionCount() - before; got != 1 {
		t.Fatalf("PhaseCorrelate queue submissions = %d, want 1", got)
	}
}

func assertKnownTransform(t *testing.T, corr *Correlator, imgA *image.RGBA) {
	t.Helper()
	const (
		wantAngle = 12.0
		wantScale = 1.15
		wantTx    = 15
		wantTy    = -20
	)
	imgB := translateImageForTest(BilinearWarp(imgA, -wantAngle, wantScale), wantTx, wantTy)

	result, err := corr.PhaseCorrelate(imgA, imgB)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("registration result: angle=%.3f scale=%.5f translation=(%.3f, %.3f)",
		result.Angle, result.Scale, result.Tx, result.Ty)

	angleDelta := math.Mod(result.Angle-wantAngle+180, 360) - 180
	if math.Abs(angleDelta) > 2 {
		t.Errorf("angle = %v degrees, want about %v degrees", result.Angle, wantAngle)
	}
	if math.Abs(result.Scale-wantScale) > 0.05 {
		t.Errorf("scale = %v, want about %v", result.Scale, wantScale)
	}
	if math.Abs(result.Tx-wantTx) > 3 || math.Abs(result.Ty-wantTy) > 3 {
		t.Errorf("translation = (%v,%v), want about (%v,%v)", result.Tx, result.Ty, wantTx, wantTy)
	}
}

func translateImageForTest(src *image.RGBA, tx, ty int) *image.RGBA {
	bounds := src.Bounds()
	dst := image.NewRGBA(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			sx := min(max(x-tx, bounds.Min.X), bounds.Max.X-1)
			sy := min(max(y-ty, bounds.Min.Y), bounds.Max.Y-1)
			dst.SetRGBA(x, y, src.RGBAAt(sx, sy))
		}
	}
	return dst
}

func loadTestImage(t *testing.T, path string) *image.RGBA {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	b := img.Bounds()
	rgba := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}
	return rgba
}

// --- Reference CPU FFT to validate GPU FFT ---

func cpuFFT1D(data []complex128) {
	n := len(data)
	if n <= 1 {
		return
	}
	// Bit-reversal.
	bits := 0
	for v := n; v > 1; v >>= 1 {
		bits++
	}
	for i := 0; i < n; i++ {
		j := 0
		for b := 0; b < bits; b++ {
			j = (j << 1) | ((i >> b) & 1)
		}
		if j > i {
			data[i], data[j] = data[j], data[i]
		}
	}
	// Butterfly.
	for s := 1; s <= bits; s++ {
		m := 1 << s
		wm := cmplx.Exp(complex(0, -2*math.Pi/float64(m)))
		for k := 0; k < n; k += m {
			w := complex(1, 0)
			for j := 0; j < m/2; j++ {
				u := data[k+j]
				v := w * data[k+j+m/2]
				data[k+j] = u + v
				data[k+j+m/2] = u - v
				w *= wm
			}
		}
	}
}

func TestFFT_MatchesCPUReference(t *testing.T) {
	ctx := testContext(t)

	w := 16
	n := w
	usage := vk.BufferUsageStorageBufferBit

	dataBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer dataBuf.Destroy(ctx)

	paramsBuf, err := ctx.CreateBuffer(32, vk.BufferUsageStorageBufferBit|vk.BufferUsageTransferDstBit)
	if err != nil {
		t.Fatal(err)
	}
	defer paramsBuf.Destroy(ctx)

	// Known input.
	input := make([][2]float32, n)
	cpuInput := make([]complex128, n)
	for i := range input {
		val := float64(i*i+1) * 0.1
		input[i] = [2]float32{float32(val), 0}
		cpuInput[i] = complex(val, 0)
	}

	// CPU reference FFT.
	cpuFFT1D(cpuInput)

	// GPU FFT.
	if err := dataBuf.UploadSlice(ctx, input); err != nil {
		t.Fatal(err)
	}

	bitrev := compilePipeline(t, ctx, fftBitrevWGSL, []compute.BufferBinding{bb(0, dataBuf.Buf), bb(1, paramsBuf)})
	butterfly := compilePipeline(t, ctx, fftButterflyWGSL, []compute.BufferBinding{bb(0, dataBuf.Buf), bb(1, paramsBuf)})

	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	// 1D FFT: w=16, h=1 → only row-wise.
	recordFFT2D(rec, dataBuf.Buf.DeviceBuffer, paramsBuf.DeviceBuffer, bitrev, butterfly, w, 1, false)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	output, err := dataBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}

	maxErr := 0.0
	for i := 0; i < n; i++ {
		dr := math.Abs(float64(output[i][0]) - real(cpuInput[i]))
		di := math.Abs(float64(output[i][1]) - imag(cpuInput[i]))
		maxErr = math.Max(maxErr, math.Max(dr, di))
		if dr > 0.1 || di > 0.1 {
			t.Errorf("FFT[%d]: GPU=(%v,%v) CPU=(%v,%v)", i, output[i][0], output[i][1], real(cpuInput[i]), imag(cpuInput[i]))
		}
	}
	t.Logf("GPU vs CPU FFT max error: %e", maxErr)
}

// --- Full pipeline debug: dump intermediate values ---

func TestPipeline_DumpIntermediates(t *testing.T) {
	ctx := testContext(t)

	w, h := 8, 8
	n := w * h
	usage := vk.BufferUsageStorageBufferBit

	complexBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer complexBuf.Destroy(ctx)

	magBuf, err := compute.NewTypedBuffer[float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer magBuf.Destroy(ctx)

	logPolBuf, err := compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		t.Fatal(err)
	}
	defer logPolBuf.Destroy(ctx)

	paramsBuf, err := ctx.CreateBuffer(32, vk.BufferUsageStorageBufferBit|vk.BufferUsageTransferDstBit)
	if err != nil {
		t.Fatal(err)
	}
	defer paramsBuf.Destroy(ctx)

	// Create a simple signal with known frequency content.
	input := make([][2]float32, n)
	for i := range input {
		x := i % w
		y := i / w
		input[i] = [2]float32{float32(math.Sin(2*math.Pi*float64(x)/float64(w)) + 0.5*math.Cos(4*math.Pi*float64(y)/float64(h))), 0}
	}
	if err := complexBuf.UploadSlice(ctx, input); err != nil {
		t.Fatal(err)
	}

	bitrev := compilePipeline(t, ctx, fftBitrevWGSL, []compute.BufferBinding{bb(0, complexBuf.Buf), bb(1, paramsBuf)})
	butterfly := compilePipeline(t, ctx, fftButterflyWGSL, []compute.BufferBinding{bb(0, complexBuf.Buf), bb(1, paramsBuf)})
	magPipe := compilePipeline(t, ctx, magnitudeWGSL, []compute.BufferBinding{
		bb(0, complexBuf.Buf), bb(1, magBuf.Buf), bb(2, paramsBuf),
	})
	logPolPipe := compilePipeline(t, ctx, logpolarWGSL, []compute.BufferBinding{
		bb(0, magBuf.Buf), bb(1, logPolBuf.Buf), bb(2, paramsBuf),
	})

	// FFT.
	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	recordFFT2D(rec, complexBuf.Buf.DeviceBuffer, paramsBuf.DeviceBuffer, bitrev, butterfly, w, h, false)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	fftData, err := complexBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("=== FFT output (first row) ===")
	for x := 0; x < w; x++ {
		t.Logf("  FFT[%d,0] = (%v, %v)", x, fftData[x][0], fftData[x][1])
	}

	// Magnitude.
	rec, err = ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	magParams := encodeU32Params(uint32(n))
	rec.UpdateBuffer(paramsBuf.DeviceBuffer, 0, magParams)
	rec.BarrierTransferToCompute(paramsBuf.DeviceBuffer)
	rec.Bind(magPipe)
	rec.Dispatch(uint32((n+63)/64), 1, 1)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	magData, err := magBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("=== Magnitude (all) ===")
	for y := 0; y < h; y++ {
		row := fmt.Sprintf("  mag[*,%d] =", y)
		for x := 0; x < w; x++ {
			row += fmt.Sprintf(" %.2f", magData[y*w+x])
		}
		t.Log(row)
	}

	// Verify DC is at (0,0).
	dcMag := magData[0]
	nyquistMag := magData[(h/2)*w+(w/2)]
	t.Logf("DC magnitude (0,0): %v", dcMag)
	t.Logf("Nyquist magnitude (%d,%d): %v", w/2, h/2, nyquistMag)

	// Log-polar remap.
	rec, err = ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	maxRadius := math.Sqrt(float64(w*w+h*h)) / 2.0
	logRmax := float32(math.Log(maxRadius))
	lpParams := encodeLogPolarParams(uint32(w), uint32(h), uint32(w), uint32(h), logRmax)
	rec.UpdateBuffer(paramsBuf.DeviceBuffer, 0, lpParams)
	rec.BarrierTransferToCompute(paramsBuf.DeviceBuffer)
	rec.Bind(logPolPipe)
	rec.Dispatch(uint32((n+63)/64), 1, 1)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	lpData, err := logPolBuf.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("=== Log-polar output ===")
	for y := 0; y < h; y++ {
		row := fmt.Sprintf("  lp[*,%d] =", y)
		for x := 0; x < w; x++ {
			row += fmt.Sprintf(" %.2f", lpData[y*w+x][0])
		}
		t.Log(row)
	}
}
