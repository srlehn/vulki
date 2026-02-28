package imgproc

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"math/cmplx"
	"os"
	"testing"

	"vkpg/compute"
	"vkpg/shader"
	"vkpg/vk"
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
	spirv, err := shader.Compile(wgsl, nil)
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
	usage := uint32(vk.BufferUsageStorageBufferBit)

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
	usage := uint32(vk.BufferUsageStorageBufferBit)

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
	usage := uint32(vk.BufferUsageStorageBufferBit)

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

	expected := []float64{5, 1, 1, 0, 5, math.Sqrt(2), 13, math.Sqrt(2)}
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
	usage := uint32(vk.BufferUsageStorageBufferBit)

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

	// At (0,0): hx = 1 - cos(0) = 0, so output should be 0.
	if output[0] != 0 {
		t.Errorf("highpass[0,0] = %v, want 0 (DC should be zeroed)", output[0])
	}

	// At (0, y) for any y: hx = 0, so output should be 0.
	for y := 0; y < h; y++ {
		if output[y*w] != 0 {
			t.Errorf("highpass[0,%d] = %v, want 0 (DC column)", y, output[y*w])
		}
	}

	// At (x, 0) for any x: hy = 0, so output should be 0.
	for x := 0; x < w; x++ {
		if output[x] != 0 {
			t.Errorf("highpass[%d,0] = %v, want 0 (DC row)", x, output[x])
		}
	}

	// Non-DC locations should be non-zero.
	for y := 1; y < h; y++ {
		for x := 1; x < w; x++ {
			if output[y*w+x] == 0 {
				t.Errorf("highpass[%d,%d] = 0, want non-zero", x, y)
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
	usage := uint32(vk.BufferUsageStorageBufferBit)

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
	params := encodeU32Params(uint32(n))
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

// --- Phase correlation of identical signals should peak at (0,0) ---

func TestCrosspower_IFFT_PeaksAtOrigin(t *testing.T) {
	ctx := testContext(t)

	w, h := 16, 16
	n := w * h
	usage := uint32(vk.BufferUsageStorageBufferBit)

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
	cpParams := encodeU32Params(uint32(n))
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
	usage := uint32(vk.BufferUsageStorageBufferBit)

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
		srcX := ((x - shiftX) % w + w) % w
		srcY := ((y - shiftY) % h + h) % h
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

	cpParams := encodeU32Params(uint32(n))
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
	usage := uint32(vk.BufferUsageStorageBufferBit)

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

// --- Phase 1 debug: test rotation detection with real images ---

func TestPhase1_RotationDetection(t *testing.T) {
	ctx := testContext(t)

	// Load real images.
	imgA := loadTestImage(t, "testdata/snake.png")
	imgB := loadTestImage(t, "testdata/snake_rotated.png") // 12° rotation

	maxW := imgA.Bounds().Dx()
	maxH := imgA.Bounds().Dy()

	corr, err := NewCorrelator(ctx, maxW, maxH)
	if err != nil {
		t.Fatal(err)
	}
	defer corr.Close()

	// Run Phase 1 manually and dump intermediates.
	wgSize := uint32(64)
	n := uint32(corr.w * corr.h)
	groups := (n + wgSize - 1) / wgSize
	paramsBuf := corr.params.DeviceBuffer

	// Upload images.
	pixA := padImageToRGBA(imgA, corr.w, corr.h)
	pixB := padImageToRGBA(imgB, corr.w, corr.h)
	if err := corr.rgbaA.UploadSlice(ctx, pixA); err != nil {
		t.Fatal(err)
	}
	if err := corr.rgbaB.UploadSlice(ctx, pixB); err != nil {
		t.Fatal(err)
	}

	whParams := encodeU32Params(uint32(corr.w), uint32(corr.h))

	// Grayscale + Hann.
	rec, err := ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	rec.UpdateBuffer(paramsBuf, 0, whParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(corr.pipeGrayscaleA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(corr.grayA.Buf.DeviceBuffer)
	rec.Bind(corr.pipeGrayscaleB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(corr.grayB.Buf.DeviceBuffer)
	rec.Bind(corr.pipeHannA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(corr.grayA.Buf.DeviceBuffer)
	rec.Bind(corr.pipeHannB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(corr.grayB.Buf.DeviceBuffer)
	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	grayAData, err := corr.grayA.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}
	grayBData, err := corr.grayB.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// Check grayscale differences.
	grayDiff := float64(0)
	for i := range grayAData {
		d := math.Abs(float64(grayAData[i] - grayBData[i]))
		if d > grayDiff {
			grayDiff = d
		}
	}
	t.Logf("Max grayscale diff (A vs B): %v", grayDiff)

	// FFT.
	if err := corr.complexA.UploadSlice(ctx, realToComplex(grayAData)); err != nil {
		t.Fatal(err)
	}
	if err := corr.complexB.UploadSlice(ctx, realToComplex(grayBData)); err != nil {
		t.Fatal(err)
	}

	rec, err = ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	recordFFT2D(rec, corr.complexA.Buf.DeviceBuffer, paramsBuf, corr.pipeBitrevA, corr.pipeButterflyA, corr.w, corr.h, false)
	recordFFT2D(rec, corr.complexB.Buf.DeviceBuffer, paramsBuf, corr.pipeBitrevB, corr.pipeButterflyB, corr.w, corr.h, false)

	// Magnitude.
	magParams := encodeU32Params(n)
	rec.UpdateBuffer(paramsBuf, 0, magParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(corr.pipeMagA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(corr.magA.Buf.DeviceBuffer)
	rec.Bind(corr.pipeMagB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(corr.magB.Buf.DeviceBuffer)

	// Highpass.
	rec.UpdateBuffer(paramsBuf, 0, whParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(corr.pipeHighpassA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(corr.magA.Buf.DeviceBuffer)
	rec.Bind(corr.pipeHighpassB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(corr.magB.Buf.DeviceBuffer)

	// Log-polar.
	maxRadius := math.Sqrt(float64(corr.w*corr.w+corr.h*corr.h)) / 2.0
	logRmax := float32(math.Log(maxRadius))
	lpParams := encodeLogPolarParams(uint32(corr.w), uint32(corr.h), uint32(corr.lpW), uint32(corr.lpH), logRmax)
	rec.UpdateBuffer(paramsBuf, 0, lpParams)
	rec.BarrierTransferToCompute(paramsBuf)
	lpN := uint32(corr.lpW * corr.lpH)
	lpGroups := (lpN + wgSize - 1) / wgSize
	rec.Bind(corr.pipeLogPolA)
	rec.Dispatch(lpGroups, 1, 1)
	rec.Barrier(corr.logPolA.Buf.DeviceBuffer)
	rec.Bind(corr.pipeLogPolB)
	rec.Dispatch(lpGroups, 1, 1)
	rec.Barrier(corr.logPolB.Buf.DeviceBuffer)

	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	// Download magnitude spectra.
	magAData, err := corr.magA.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}
	magBData, err := corr.magB.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// Dump some mag stats.
	magAMax, magBMax := float32(0), float32(0)
	magASum, magBSum := float64(0), float64(0)
	for i := range magAData {
		if magAData[i] > magAMax {
			magAMax = magAData[i]
		}
		if magBData[i] > magBMax {
			magBMax = magBData[i]
		}
		magASum += float64(magAData[i])
		magBSum += float64(magBData[i])
	}
	t.Logf("Magnitude A: max=%v, sum=%v", magAMax, magASum)
	t.Logf("Magnitude B: max=%v, sum=%v", magBMax, magBSum)

	// Download log-polar images.
	lpAData, err := corr.logPolA.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}
	lpBData, err := corr.logPolB.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}

	// Stats on log-polar.
	lpAMax, lpBMax := float32(0), float32(0)
	lpANonzero, lpBNonzero := 0, 0
	lpDiffMax := float32(0)
	for i := range lpAData {
		a := lpAData[i][0]
		b := lpBData[i][0]
		if a > lpAMax {
			lpAMax = a
		}
		if b > lpBMax {
			lpBMax = b
		}
		if a != 0 {
			lpANonzero++
		}
		if b != 0 {
			lpBNonzero++
		}
		d := float32(math.Abs(float64(a - b)))
		if d > lpDiffMax {
			lpDiffMax = d
		}
	}
	t.Logf("Log-polar A: max=%v, nonzero=%d/%d", lpAMax, lpANonzero, len(lpAData))
	t.Logf("Log-polar B: max=%v, nonzero=%d/%d", lpBMax, lpBNonzero, len(lpBData))
	t.Logf("Log-polar max diff: %v", lpDiffMax)
	t.Logf("Log-polar dims: %dx%d", corr.lpW, corr.lpH)

	// Run Phase 1 cross-power + IFFT.
	rec, err = ctx.NewCommandRecorder()
	if err != nil {
		t.Fatal(err)
	}
	recordFFT2D(rec, corr.logPolA.Buf.DeviceBuffer, paramsBuf, corr.pipeBitrevLPA, corr.pipeButterflyLPA, corr.lpW, corr.lpH, false)
	recordFFT2D(rec, corr.logPolB.Buf.DeviceBuffer, paramsBuf, corr.pipeBitrevLPB, corr.pipeButterflyLPB, corr.lpW, corr.lpH, false)

	cpParams := encodeU32Params(lpN)
	rec.UpdateBuffer(paramsBuf, 0, cpParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(corr.pipeCrosspowerLP)
	rec.Dispatch(lpGroups, 1, 1)
	rec.Barrier(corr.crossPow.Buf.DeviceBuffer)

	recordFFT2D(rec, corr.crossPow.Buf.DeviceBuffer, paramsBuf, corr.pipeBitrevCP, corr.pipeButterflyCP, corr.lpW, corr.lpH, true)

	if err := rec.Submit(); err != nil {
		t.Fatal(err)
	}

	cpData, err := corr.crossPow.DownloadSlice(ctx)
	if err != nil {
		t.Fatal(err)
	}
	normalizeComplex(cpData[:lpN], corr.lpW*corr.lpH)
	peakX, peakY := find2DPeak(cpData[:lpN], corr.lpW, corr.lpH)

	if peakX > float64(corr.lpW)/2 {
		peakX -= float64(corr.lpW)
	}
	if peakY > float64(corr.lpH)/2 {
		peakY -= float64(corr.lpH)
	}

	angle, scale := logPolarToAngleScale(peakX, peakY, corr.lpW, corr.lpH, maxRadius)
	t.Logf("Phase 1 peak: (%v, %v)", peakX, peakY)
	t.Logf("Angle: %v°, Scale: %v", angle, scale)

	// Dump top-10 peaks in cross-power surface.
	type peak struct {
		x, y int
		mag  float64
	}
	var peaks []peak
	for y := 0; y < corr.lpH; y++ {
		for x := 0; x < corr.lpW; x++ {
			c := cpData[y*corr.lpW+x]
			m := math.Sqrt(float64(c[0]*c[0] + c[1]*c[1]))
			peaks = append(peaks, peak{x, y, m})
		}
	}
	// Sort by magnitude descending.
	for i := 0; i < 10 && i < len(peaks); i++ {
		for j := i + 1; j < len(peaks); j++ {
			if peaks[j].mag > peaks[i].mag {
				peaks[i], peaks[j] = peaks[j], peaks[i]
			}
		}
	}
	t.Log("Top 10 correlation peaks:")
	for i := 0; i < 10 && i < len(peaks); i++ {
		p := peaks[i]
		py := float64(p.y)
		px := float64(p.x)
		if px > float64(corr.lpW)/2 {
			px -= float64(corr.lpW)
		}
		if py > float64(corr.lpH)/2 {
			py -= float64(corr.lpH)
		}
		a, s := logPolarToAngleScale(px, py, corr.lpW, corr.lpH, maxRadius)
		t.Logf("  #%d: (%d,%d) mag=%.6f → angle=%.2f° scale=%.4f", i+1, p.x, p.y, p.mag, a, s)
	}

	if math.Abs(angle-12) > 2 {
		t.Errorf("angle = %v°, want ~12°", angle)
	}
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
	usage := uint32(vk.BufferUsageStorageBufferBit)

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
	usage := uint32(vk.BufferUsageStorageBufferBit)

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
