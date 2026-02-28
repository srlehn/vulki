package imgproc

import (
	_ "embed"
	"fmt"
	"image"
	"math"

	"vkpg/compute"
	"vkpg/shader"
	"vkpg/vk"
)

//go:embed shaders/grayscale.wgsl
var grayscaleWGSL string

//go:embed shaders/hann.wgsl
var hannWGSL string

//go:embed shaders/fft_bitrev.wgsl
var fftBitrevWGSL string

//go:embed shaders/fft_butterfly.wgsl
var fftButterflyWGSL string

//go:embed shaders/magnitude.wgsl
var magnitudeWGSL string

//go:embed shaders/highpass.wgsl
var highpassWGSL string

//go:embed shaders/logpolar.wgsl
var logpolarWGSL string

//go:embed shaders/fftshift.wgsl
var fftshiftWGSL string

//go:embed shaders/crosspower.wgsl
var crosspowerWGSL string

// Result holds the recovered rotation, scale, and translation.
type Result struct {
	Angle float64 // degrees
	Scale float64 // multiplier
	Tx    float64 // translation in pixels (x)
	Ty    float64 // translation in pixels (y)
}

// Correlator performs log-polar phase correlation on the GPU.
type Correlator struct {
	ctx *compute.Context

	// Padded image dimensions (power of 2).
	w, h int
	// Log-polar dimensions.
	lpW, lpH int

	// Compiled SPIR-V (one per shader type).
	spirvGrayscale  []byte
	spirvHann       []byte
	spirvBitrev     []byte
	spirvButterfly  []byte
	spirvMagnitude  []byte
	spirvHighpass   []byte
	spirvLogpolar   []byte
	spirvFftshift   []byte
	spirvCrosspower []byte

	// Working buffers.
	rgbaA, rgbaB         *compute.TypedBuffer[uint32]
	grayA, grayB         *compute.TypedBuffer[float32]
	complexA, complexB   *compute.TypedBuffer[[2]float32]
	magA, magB           *compute.TypedBuffer[float32]
	logPolA, logPolB     *compute.TypedBuffer[[2]float32]
	crossPow             *compute.TypedBuffer[[2]float32]
	params               *compute.Buffer // shared small params buffer

	// Pipelines (one per unique buffer binding configuration).
	pipeGrayscaleA, pipeGrayscaleB   *compute.Pipeline
	pipeHannA, pipeHannB             *compute.Pipeline
	pipeBitrevA, pipeBitrevB         *compute.Pipeline
	pipeButterflyA, pipeButterflyB   *compute.Pipeline
	pipeMagA, pipeMagB               *compute.Pipeline
	pipeFftshiftA, pipeFftshiftB     *compute.Pipeline
	pipeHighpassA, pipeHighpassB     *compute.Pipeline
	pipeLogPolA, pipeLogPolB         *compute.Pipeline
	pipeBitrevLPA, pipeBitrevLPB     *compute.Pipeline
	pipeButterflyLPA, pipeButterflyLPB *compute.Pipeline
	pipeBitrevCP, pipeButterflyCP    *compute.Pipeline
	pipeCrosspowerLP                 *compute.Pipeline
	pipeCrosspowerTrans              *compute.Pipeline

	allPipelines []*compute.Pipeline
}

// NewCorrelator creates a Correlator for images up to maxW x maxH pixels.
func NewCorrelator(ctx *compute.Context, maxW, maxH int) (*Correlator, error) {
	c := &Correlator{ctx: ctx}

	// Square crop + minimal padding: crop to min(w,h) then pad to next power of 2.
	// This preserves spectral symmetry and avoids excessive zero-padding.
	padSize := nextPow2(min(maxW, maxH))
	c.w = padSize
	c.h = padSize
	c.lpW = c.w
	c.lpH = c.h

	n := c.w * c.h
	lpN := c.lpW * c.lpH

	// Compile all shaders.
	var err error
	for _, entry := range []struct {
		dst  *[]byte
		src  string
		name string
	}{
		{&c.spirvGrayscale, grayscaleWGSL, "grayscale"},
		{&c.spirvHann, hannWGSL, "hann"},
		{&c.spirvBitrev, fftBitrevWGSL, "fft_bitrev"},
		{&c.spirvButterfly, fftButterflyWGSL, "fft_butterfly"},
		{&c.spirvMagnitude, magnitudeWGSL, "magnitude"},
		{&c.spirvFftshift, fftshiftWGSL, "fftshift"},
		{&c.spirvHighpass, highpassWGSL, "highpass"},
		{&c.spirvLogpolar, logpolarWGSL, "logpolar"},
		{&c.spirvCrosspower, crosspowerWGSL, "crosspower"},
	} {
		*entry.dst, err = shader.Compile(entry.src, nil)
		if err != nil {
			return nil, fmt.Errorf("imgproc: compile %s: %w", entry.name, err)
		}
	}

	usage := uint32(vk.BufferUsageStorageBufferBit)

	// Allocate working buffers.
	c.rgbaA, err = compute.NewTypedBuffer[uint32](ctx, n, usage)
	if err != nil {
		return nil, fmt.Errorf("imgproc: alloc rgbaA: %w", err)
	}
	c.rgbaB, err = compute.NewTypedBuffer[uint32](ctx, n, usage)
	if err != nil {
		c.Close()
		return nil, fmt.Errorf("imgproc: alloc rgbaB: %w", err)
	}
	c.grayA, err = compute.NewTypedBuffer[float32](ctx, n, usage)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.grayB, err = compute.NewTypedBuffer[float32](ctx, n, usage)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.complexA, err = compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.complexB, err = compute.NewTypedBuffer[[2]float32](ctx, n, usage)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.magA, err = compute.NewTypedBuffer[float32](ctx, n, usage)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.magB, err = compute.NewTypedBuffer[float32](ctx, n, usage)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.logPolA, err = compute.NewTypedBuffer[[2]float32](ctx, lpN, usage)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.logPolB, err = compute.NewTypedBuffer[[2]float32](ctx, lpN, usage)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.crossPow, err = compute.NewTypedBuffer[[2]float32](ctx, max(n, lpN), usage)
	if err != nil {
		c.Close()
		return nil, err
	}

	// Params buffer: 24 bytes is enough for all shader params.
	c.params, err = ctx.CreateBuffer(32, vk.BufferUsageStorageBufferBit|vk.BufferUsageTransferDstBit)
	if err != nil {
		c.Close()
		return nil, err
	}

	// Create pipelines.
	if err := c.createPipelines(); err != nil {
		c.Close()
		return nil, err
	}

	return c, nil
}

func (c *Correlator) createPipelines() error {
	var err error
	bb := func(binding uint32, buf *compute.Buffer) compute.BufferBinding {
		return compute.BufferBinding{Binding: binding, Buffer: buf}
	}
	mk := func(spirv []byte, bindings []compute.BufferBinding) (*compute.Pipeline, error) {
		p, e := c.ctx.CreateComputePipeline(spirv, bindings)
		if e != nil {
			return nil, e
		}
		c.allPipelines = append(c.allPipelines, p)
		return p, nil
	}

	paramBuf := c.params

	// Grayscale: binding 0=rgba, 1=gray, 2=params
	c.pipeGrayscaleA, err = mk(c.spirvGrayscale, []compute.BufferBinding{
		bb(0, c.rgbaA.Buf), bb(1, c.grayA.Buf), bb(2, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeGrayscaleB, err = mk(c.spirvGrayscale, []compute.BufferBinding{
		bb(0, c.rgbaB.Buf), bb(1, c.grayB.Buf), bb(2, paramBuf),
	})
	if err != nil {
		return err
	}

	// Hann: binding 0=gray(rw), 1=params
	c.pipeHannA, err = mk(c.spirvHann, []compute.BufferBinding{
		bb(0, c.grayA.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeHannB, err = mk(c.spirvHann, []compute.BufferBinding{
		bb(0, c.grayB.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}

	// FFT bitrev + butterfly for complexA, complexB
	c.pipeBitrevA, err = mk(c.spirvBitrev, []compute.BufferBinding{
		bb(0, c.complexA.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeButterflyA, err = mk(c.spirvButterfly, []compute.BufferBinding{
		bb(0, c.complexA.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeBitrevB, err = mk(c.spirvBitrev, []compute.BufferBinding{
		bb(0, c.complexB.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeButterflyB, err = mk(c.spirvButterfly, []compute.BufferBinding{
		bb(0, c.complexB.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}

	// Magnitude: binding 0=complex(read), 1=mag(write), 2=params
	c.pipeMagA, err = mk(c.spirvMagnitude, []compute.BufferBinding{
		bb(0, c.complexA.Buf), bb(1, c.magA.Buf), bb(2, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeMagB, err = mk(c.spirvMagnitude, []compute.BufferBinding{
		bb(0, c.complexB.Buf), bb(1, c.magB.Buf), bb(2, paramBuf),
	})
	if err != nil {
		return err
	}

	// Fftshift: binding 0=mag(rw), 1=params
	c.pipeFftshiftA, err = mk(c.spirvFftshift, []compute.BufferBinding{
		bb(0, c.magA.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeFftshiftB, err = mk(c.spirvFftshift, []compute.BufferBinding{
		bb(0, c.magB.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}

	// Highpass: binding 0=mag(rw), 1=params
	c.pipeHighpassA, err = mk(c.spirvHighpass, []compute.BufferBinding{
		bb(0, c.magA.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeHighpassB, err = mk(c.spirvHighpass, []compute.BufferBinding{
		bb(0, c.magB.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}

	// Logpolar: binding 0=mag(read), 1=logpol(write), 2=params
	c.pipeLogPolA, err = mk(c.spirvLogpolar, []compute.BufferBinding{
		bb(0, c.magA.Buf), bb(1, c.logPolA.Buf), bb(2, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeLogPolB, err = mk(c.spirvLogpolar, []compute.BufferBinding{
		bb(0, c.magB.Buf), bb(1, c.logPolB.Buf), bb(2, paramBuf),
	})
	if err != nil {
		return err
	}

	// FFT bitrev + butterfly for logPolA, logPolB, crossPow
	c.pipeBitrevLPA, err = mk(c.spirvBitrev, []compute.BufferBinding{
		bb(0, c.logPolA.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeButterflyLPA, err = mk(c.spirvButterfly, []compute.BufferBinding{
		bb(0, c.logPolA.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeBitrevLPB, err = mk(c.spirvBitrev, []compute.BufferBinding{
		bb(0, c.logPolB.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeButterflyLPB, err = mk(c.spirvButterfly, []compute.BufferBinding{
		bb(0, c.logPolB.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeBitrevCP, err = mk(c.spirvBitrev, []compute.BufferBinding{
		bb(0, c.crossPow.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeButterflyCP, err = mk(c.spirvButterfly, []compute.BufferBinding{
		bb(0, c.crossPow.Buf), bb(1, paramBuf),
	})
	if err != nil {
		return err
	}

	// Crosspower: binding 0=a(read), 1=b(read), 2=out(write), 3=params
	// We compute A*conj(B) following scikit-image convention.
	c.pipeCrosspowerLP, err = mk(c.spirvCrosspower, []compute.BufferBinding{
		bb(0, c.logPolA.Buf), bb(1, c.logPolB.Buf), bb(2, c.crossPow.Buf), bb(3, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeCrosspowerTrans, err = mk(c.spirvCrosspower, []compute.BufferBinding{
		bb(0, c.complexA.Buf), bb(1, c.complexB.Buf), bb(2, c.crossPow.Buf), bb(3, paramBuf),
	})
	if err != nil {
		return err
	}

	return nil
}

// PhaseCorrelate recovers rotation, scale, and translation between two images.
func (c *Correlator) PhaseCorrelate(imgA, imgB *image.RGBA) (*Result, error) {
	// Determine square crop size from the smaller dimension of each image.
	cropSize := min(imgA.Bounds().Dx(), imgA.Bounds().Dy(),
		imgB.Bounds().Dx(), imgB.Bounds().Dy())

	wgSize := uint32(64)
	n := uint32(c.w * c.h)
	groups := (n + wgSize - 1) / wgSize
	paramsBuf := c.params.DeviceBuffer

	// Crop images to centered squares, apply DoG+Hann, pad to power-of-2.
	grayAData := cropDogHannPad(imgA, cropSize, c.w)
	grayBData := cropDogHannPad(imgB, cropSize, c.w)

	// ---- Phase 1: Rotation & Scale ----
	// Upload preprocessed grayscale as complex for FFT.
	if err := c.complexA.UploadSlice(c.ctx, realToComplex(grayAData)); err != nil {
		return nil, err
	}
	if err := c.complexB.UploadSlice(c.ctx, realToComplex(grayBData)); err != nil {
		return nil, err
	}

	// FFT2D on complexA and complexB.
	rec, err := c.ctx.NewCommandRecorder()
	if err != nil {
		return nil, err
	}
	recordFFT2D(rec, c.complexA.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevA, c.pipeButterflyA, c.w, c.h, false)
	recordFFT2D(rec, c.complexB.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevB, c.pipeButterflyB, c.w, c.h, false)

	// Magnitude.
	magParams := encodeU32Params(n)
	rec.UpdateBuffer(paramsBuf, 0, magParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeMagA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.magA.Buf.DeviceBuffer)
	rec.Bind(c.pipeMagB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.magB.Buf.DeviceBuffer)

	// Fftshift magnitude spectra so DC is at center.
	shiftParams := encodeU32Params(uint32(c.w), uint32(c.h))
	rec.UpdateBuffer(paramsBuf, 0, shiftParams)
	rec.BarrierTransferToCompute(paramsBuf)
	halfN := n / 2
	shiftGroups := (halfN + wgSize - 1) / wgSize
	rec.Bind(c.pipeFftshiftA)
	rec.Dispatch(shiftGroups, 1, 1)
	rec.Barrier(c.magA.Buf.DeviceBuffer)
	rec.Bind(c.pipeFftshiftB)
	rec.Dispatch(shiftGroups, 1, 1)
	rec.Barrier(c.magB.Buf.DeviceBuffer)

	// Log-polar remap. Limited radius following scikit-image (shape[0]//8).
	maxRadius := float64(cropSize) / 8.0
	logRmax := float32(math.Log(maxRadius))
	lpParams := encodeLogPolarParams(uint32(c.w), uint32(c.h), uint32(c.lpW), uint32(c.lpH), logRmax)
	rec.UpdateBuffer(paramsBuf, 0, lpParams)
	rec.BarrierTransferToCompute(paramsBuf)
	lpN := uint32(c.lpW * c.lpH)
	lpGroups := (lpN + wgSize - 1) / wgSize
	rec.Bind(c.pipeLogPolA)
	rec.Dispatch(lpGroups, 1, 1)
	rec.Barrier(c.logPolA.Buf.DeviceBuffer)
	rec.Bind(c.pipeLogPolB)
	rec.Dispatch(lpGroups, 1, 1)
	rec.Barrier(c.logPolB.Buf.DeviceBuffer)

	// FFT2D on log-polar buffers.
	recordFFT2D(rec, c.logPolA.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevLPA, c.pipeButterflyLPA, c.lpW, c.lpH, false)
	recordFFT2D(rec, c.logPolB.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevLPB, c.pipeButterflyLPB, c.lpW, c.lpH, false)

	// Cross-power spectrum (raw, no phase normalization for log-polar).
	cpParams := encodeU32Params(lpN, 0) // normalize=0
	rec.UpdateBuffer(paramsBuf, 0, cpParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeCrosspowerLP)
	rec.Dispatch(lpGroups, 1, 1)
	rec.Barrier(c.crossPow.Buf.DeviceBuffer)

	// IFFT2D on cross-power result.
	recordFFT2D(rec, c.crossPow.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevCP, c.pipeButterflyCP, c.lpW, c.lpH, true)

	if err := rec.Submit(); err != nil {
		return nil, fmt.Errorf("imgproc: phase1 FFT: %w", err)
	}

	// Download and find peak.
	cpData, err := c.crossPow.DownloadSlice(c.ctx)
	if err != nil {
		return nil, err
	}
	normalizeComplex(cpData[:lpN], c.lpW*c.lpH)
	peakX, peakY := find2DPeak(cpData[:lpN], c.lpW, c.lpH)
	fmt.Printf("DEBUG: raw peak (%f, %f) lpW=%d lpH=%d maxR=%.2f cropSize=%d\n",
		peakX, peakY, c.lpW, c.lpH, maxRadius, cropSize)

	// Handle wraparound: if peak is in second half, subtract N.
	if peakX > float64(c.lpW)/2 {
		peakX -= float64(c.lpW)
	}
	if peakY > float64(c.lpH)/2 {
		peakY -= float64(c.lpH)
	}

	angle, scale := logPolarToAngleScale(peakX, peakY, c.lpW, c.lpH, maxRadius)

	// ---- Phase 2: Translation ----
	// Rotate+scale imgB to undo the found transform on CPU.
	warpedB := BilinearWarp(imgB, angle, scale)

	// Preprocess warped B with same square crop + DoG + Hann + pad pipeline.
	grayBData = cropDogHannPad(warpedB, cropSize, c.w)

	// Upload both as complex for translation FFT.
	// grayAData is still valid from phase 1.
	if err := c.complexA.UploadSlice(c.ctx, realToComplex(grayAData)); err != nil {
		return nil, err
	}
	if err := c.complexB.UploadSlice(c.ctx, realToComplex(grayBData)); err != nil {
		return nil, err
	}

	// FFT2D, cross-power, IFFT2D for translation.
	rec, err = c.ctx.NewCommandRecorder()
	if err != nil {
		return nil, err
	}
	recordFFT2D(rec, c.complexA.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevA, c.pipeButterflyA, c.w, c.h, false)
	recordFFT2D(rec, c.complexB.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevB, c.pipeButterflyB, c.w, c.h, false)

	transParams := encodeU32Params(n, 1) // normalize=1 for translation
	rec.UpdateBuffer(paramsBuf, 0, transParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeCrosspowerTrans)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.crossPow.Buf.DeviceBuffer)

	recordFFT2D(rec, c.crossPow.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevCP, c.pipeButterflyCP, c.w, c.h, true)

	if err := rec.Submit(); err != nil {
		return nil, fmt.Errorf("imgproc: phase2 FFT: %w", err)
	}

	// Download and find translation peak.
	transData, err := c.crossPow.DownloadSlice(c.ctx)
	if err != nil {
		return nil, err
	}
	normalizeComplex(transData[:n], c.w*c.h)
	tx, ty := find2DPeak(transData[:n], c.w, c.h)
	fmt.Printf("DEBUG: translation raw peak (%f, %f) w=%d h=%d\n", tx, ty, c.w, c.h)

	// Handle wraparound for translation.
	if tx > float64(c.w)/2 {
		tx -= float64(c.w)
	}
	if ty > float64(c.h)/2 {
		ty -= float64(c.h)
	}

	// Cross-correlation gives shift to align B→A; negate to get B's displacement.
	tx = -tx
	ty = -ty

	// The detected translation is in the de-rotated/de-scaled frame.
	// Apply R(-angle)*scale to recover the original SRT translation.
	rad := angle * math.Pi / 180.0
	cosA := math.Cos(rad)
	sinA := math.Sin(rad)
	origTx := (tx*cosA + ty*sinA) * scale
	origTy := (-tx*sinA + ty*cosA) * scale

	return &Result{
		Angle: angle,
		Scale: scale,
		Tx:    origTx,
		Ty:    origTy,
	}, nil
}

// Close releases all GPU resources.
func (c *Correlator) Close() {
	for _, p := range c.allPipelines {
		p.Destroy(c.ctx)
	}
	if c.rgbaA != nil {
		c.rgbaA.Destroy(c.ctx)
	}
	if c.rgbaB != nil {
		c.rgbaB.Destroy(c.ctx)
	}
	if c.grayA != nil {
		c.grayA.Destroy(c.ctx)
	}
	if c.grayB != nil {
		c.grayB.Destroy(c.ctx)
	}
	if c.complexA != nil {
		c.complexA.Destroy(c.ctx)
	}
	if c.complexB != nil {
		c.complexB.Destroy(c.ctx)
	}
	if c.magA != nil {
		c.magA.Destroy(c.ctx)
	}
	if c.magB != nil {
		c.magB.Destroy(c.ctx)
	}
	if c.logPolA != nil {
		c.logPolA.Destroy(c.ctx)
	}
	if c.logPolB != nil {
		c.logPolB.Destroy(c.ctx)
	}
	if c.crossPow != nil {
		c.crossPow.Destroy(c.ctx)
	}
	if c.params != nil {
		c.params.Destroy(c.ctx)
	}
}
