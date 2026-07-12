package imgproc

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"math"

	"github.com/srlehn/vulki/compute"
	"github.com/srlehn/vulki/shader"
	"github.com/srlehn/vulki/vk"
)

//go:embed shaders/grayscale_pad.wgsl
var grayscalePadWGSL string

//go:embed shaders/bilinear_warp_gray.wgsl
var bilinearWarpGrayWGSL string

//go:embed shaders/peak_find.wgsl
var peakFindWGSL string

//go:embed shaders/peak_reduce.wgsl
var peakReduceWGSL string

//go:embed shaders/peak_finalize.wgsl
var peakFinalizeWGSL string

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
	Angle                 float64 // degrees
	Scale                 float64 // multiplier
	Tx                    float64 // translation in pixels (x)
	Ty                    float64 // translation in pixels (y)
	RotationConfidence    float64 // normalized log-polar IFFT peak
	TranslationConfidence float64 // normalized translation IFFT peak
}

// ErrLowConfidence indicates that at least one phase-correlation peak did not
// meet the validity threshold described by Reddy and Chatterji.
var ErrLowConfidence = errors.New("imgproc: phase-correlation match confidence is too low")

const minimumMatchConfidence = 0.03

const maxExactFloat32Index = 1 << 24

const peakWorkgroups uint32 = 64

// Correlator performs log-polar phase correlation on a GPU or CPU backend.
type Correlator struct {
	ctx *compute.Context

	backend        Backend
	ownsContext    bool
	fallbackReason error
	closed         bool

	// Padded image dimensions (power of 2).
	w, h int
	// Log-polar dimensions.
	lpW, lpH int
	// Max image dimensions (for RGBA buffer sizing).
	maxW, maxH int

	// Compiled SPIR-V.
	spirvGrayPad    []byte
	spirvWarpGray   []byte
	spirvPeakFind   []byte
	spirvPeakReduce []byte
	spirvPeakFinal  []byte
	spirvBitrev     []byte
	spirvButterfly  []byte
	spirvMagnitude  []byte
	spirvHighpass   []byte
	spirvLogpolar   []byte
	spirvFftshift   []byte
	spirvCrosspower []byte

	// Working buffers.
	rgbaA, rgbaB       *compute.TypedBuffer[uint32]     // raw RGBA (full image)
	complexA, complexB *compute.TypedBuffer[[2]float32] // padSize x padSize
	magA, magB         *compute.TypedBuffer[float32]    // padSize x padSize
	logPolA, logPolB   *compute.TypedBuffer[[2]float32] // lpW x lpH
	crossPow           *compute.TypedBuffer[[2]float32] // max(n, lpN)
	peakScratch        *compute.TypedBuffer[[2]float32] // one maximum per peak-scan workgroup
	result             *compute.TypedBuffer[float32]    // 16 floats = 64 bytes
	params             *compute.Buffer                  // shared small params buffer

	// Pipelines.
	pipeGrayPadA, pipeGrayPadB         *compute.Pipeline
	pipeWarpA                          *compute.Pipeline
	pipeBitrevA, pipeBitrevB           *compute.Pipeline
	pipeButterflyA, pipeButterflyB     *compute.Pipeline
	pipeMagA, pipeMagB                 *compute.Pipeline
	pipeFftshiftA, pipeFftshiftB       *compute.Pipeline
	pipeHighpassA, pipeHighpassB       *compute.Pipeline
	pipeLogPolA, pipeLogPolB           *compute.Pipeline
	pipeBitrevLPA, pipeBitrevLPB       *compute.Pipeline
	pipeButterflyLPA, pipeButterflyLPB *compute.Pipeline
	pipeBitrevCP, pipeButterflyCP      *compute.Pipeline
	pipeCrosspowerLP                   *compute.Pipeline
	pipeCrosspowerTrans                *compute.Pipeline
	pipePeakFind                       *compute.Pipeline
	pipePeakReduce                     *compute.Pipeline
	pipePeakFinal                      *compute.Pipeline

	allPipelines []*compute.Pipeline
}

// NewCorrelator creates a GPU Correlator using a caller-owned Vulkan context.
func NewCorrelator(ctx *compute.Context, maxW, maxH int) (*Correlator, error) {
	if ctx == nil || ctx.DevFuncs == nil || ctx.Device == 0 {
		return nil, fmt.Errorf("imgproc: invalid compute context")
	}
	if maxW < 2 || maxH < 2 {
		return nil, fmt.Errorf("imgproc: maximum dimensions must both be at least 2 pixels")
	}
	if uint64(maxW) > math.MaxUint32 || uint64(maxH) > math.MaxUint32 {
		return nil, fmt.Errorf("imgproc: maximum dimensions exceed shader limits")
	}
	maxInt := int(^uint(0) >> 1)
	if maxW > maxInt/maxH {
		return nil, fmt.Errorf("imgproc: maximum image area overflows int")
	}
	if uint64(maxW)*uint64(maxH) > math.MaxUint32 {
		return nil, fmt.Errorf("imgproc: maximum image area exceeds shader indexing limits")
	}

	c := &Correlator{ctx: ctx, backend: BackendGPU, maxW: maxW, maxH: maxH}

	// Square crop + minimal padding: crop to min(w,h) then pad to next power of 2.
	// This preserves spectral symmetry and avoids excessive zero-padding.
	padSize, ok := nextPow2Checked(min(maxW, maxH))
	if !ok || padSize > maxInt/padSize {
		return nil, fmt.Errorf("imgproc: padded image area overflows int")
	}
	if uint64(padSize)*uint64(padSize) > maxExactFloat32Index {
		return nil, fmt.Errorf("imgproc: padded image area exceeds peak-index precision limit")
	}
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
		{&c.spirvGrayPad, grayscalePadWGSL, "grayscale_pad"},
		{&c.spirvWarpGray, bilinearWarpGrayWGSL, "bilinear_warp_gray"},
		{&c.spirvPeakFind, peakFindWGSL, "peak_find"},
		{&c.spirvPeakReduce, peakReduceWGSL, "peak_reduce"},
		{&c.spirvPeakFinal, peakFinalizeWGSL, "peak_finalize"},
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

	// RGBA buffers: hold full raw image pixels.
	rgbaSize := maxW * maxH
	c.rgbaA, err = compute.NewTypedBuffer[uint32](ctx, rgbaSize, usage)
	if err != nil {
		return nil, fmt.Errorf("imgproc: alloc rgbaA: %w", err)
	}
	c.rgbaB, err = compute.NewTypedBuffer[uint32](ctx, rgbaSize, usage)
	if err != nil {
		c.Close()
		return nil, fmt.Errorf("imgproc: alloc rgbaB: %w", err)
	}

	// Complex working buffers: padSize x padSize.
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

	// Magnitude buffers.
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

	// Log-polar buffers.
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

	// Cross-power buffer.
	c.crossPow, err = compute.NewTypedBuffer[[2]float32](ctx, max(n, lpN), usage)
	if err != nil {
		c.Close()
		return nil, err
	}

	c.peakScratch, err = compute.NewTypedBuffer[[2]float32](ctx, int(peakWorkgroups), usage)
	if err != nil {
		c.Close()
		return nil, err
	}

	// Result buffer: 16 x float32 = 64 bytes.
	// Layout: [0] rotation confidence [1] rawPeakY [2] angle_deg [3] scale
	//         [4] cos(angle) [5] sin(angle) [6] -cos(angle) [7] -sin(angle)
	//         [8] tx1 [9] ty1 [10] confidence1 [11] peak-scan scratch
	//         [12] tx2 [13] ty2 [14] confidence2 [15] peak-scan scratch
	c.result, err = compute.NewTypedBuffer[float32](ctx, 16, usage)
	if err != nil {
		c.Close()
		return nil, err
	}

	// Params buffer: 32 bytes covers all shader param structs.
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

	// GrayscalePad: binding 0=rgba, 1=complex, 2=params
	c.pipeGrayPadA, err = mk(c.spirvGrayPad, []compute.BufferBinding{
		bb(0, c.rgbaA.Buf), bb(1, c.complexA.Buf), bb(2, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipeGrayPadB, err = mk(c.spirvGrayPad, []compute.BufferBinding{
		bb(0, c.rgbaB.Buf), bb(1, c.complexB.Buf), bb(2, paramBuf),
	})
	if err != nil {
		return err
	}

	// BilinearWarpGray: binding 0=rgba, 1=complex, 2=params, 3=result
	c.pipeWarpA, err = mk(c.spirvWarpGray, []compute.BufferBinding{
		bb(0, c.rgbaA.Buf), bb(1, c.complexA.Buf), bb(2, paramBuf), bb(3, c.result.Buf),
	})
	if err != nil {
		return err
	}

	// FFT bitrev + butterfly for complexA, complexB.
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

	// FFT bitrev + butterfly for logPolA, logPolB, crossPow.
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

	// Crosspower: binding 0=a(read), 1=b(read), 2=out(write), 3=params.
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

	// Peak scan: binding 0=input(complex), 1=per-workgroup scratch, 2=params.
	c.pipePeakFind, err = mk(c.spirvPeakFind, []compute.BufferBinding{
		bb(0, c.crossPow.Buf), bb(1, c.peakScratch.Buf), bb(2, paramBuf),
	})
	if err != nil {
		return err
	}
	c.pipePeakReduce, err = mk(c.spirvPeakReduce, []compute.BufferBinding{
		bb(0, c.peakScratch.Buf), bb(1, c.result.Buf),
	})
	if err != nil {
		return err
	}
	c.pipePeakFinal, err = mk(c.spirvPeakFinal, []compute.BufferBinding{
		bb(0, c.crossPow.Buf), bb(1, c.result.Buf), bb(2, paramBuf),
	})
	if err != nil {
		return err
	}

	return nil
}

// packRGBA copies an image into a tightly packed row-major pixel buffer. The
// copy is required because RGBA subimages can have non-zero bounds and a stride
// inherited from a larger parent image.
func packRGBA(img *image.RGBA) ([]uint32, error) {
	if img == nil {
		return nil, fmt.Errorf("imgproc: nil RGBA image")
	}
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("imgproc: RGBA image must have non-empty bounds")
	}
	maxInt := int(^uint(0) >> 1)
	if w > maxInt/4 || w > maxInt/h {
		return nil, fmt.Errorf("imgproc: RGBA image dimensions overflow int")
	}
	rowBytes := w * 4
	if img.Stride < rowBytes {
		return nil, fmt.Errorf("imgproc: RGBA stride %d is smaller than row size %d", img.Stride, rowBytes)
	}
	if h-1 > (maxInt-rowBytes)/img.Stride {
		return nil, fmt.Errorf("imgproc: RGBA layout overflows int")
	}
	required := (h-1)*img.Stride + rowBytes
	if len(img.Pix) < required {
		return nil, fmt.Errorf("imgproc: RGBA pixel data has %d bytes, need %d", len(img.Pix), required)
	}

	pixels := make([]uint32, w*h)
	for y := 0; y < h; y++ {
		row := img.Pix[y*img.Stride : y*img.Stride+rowBytes]
		for x := 0; x < w; x++ {
			off := x * 4
			pixels[y*w+x] = uint32(row[off]) |
				uint32(row[off+1])<<8 |
				uint32(row[off+2])<<16 |
				uint32(row[off+3])<<24
		}
	}
	return pixels, nil
}

// PhaseCorrelate recovers rotation, scale, and translation between two images.
// Following Reddy & Chatterji (1996):
//
//	Phase 1: FFT → magnitude → highpass → log-polar → FFT → cross-power → IFFT → peak → angle/scale
//	Phase 2: transform image A by detected angle/scale, phase correlate with B for translation
//	         Try both angle and angle+180° (magnitude spectrum has 180° symmetry), pick higher peak.
//
// Stages packed RGBA pixels once per image, then submits both uploads, the
// entire GPU pipeline, and a 64-byte result readback as one queue operation.
func (c *Correlator) PhaseCorrelate(imgA, imgB *image.RGBA) (*Result, error) {
	if c == nil || c.closed {
		return nil, fmt.Errorf("imgproc: invalid correlator")
	}
	if c.backend == BackendCPU {
		return c.phaseCorrelateCPU(imgA, imgB)
	}
	if c.backend != BackendGPU || c.ctx == nil {
		return nil, fmt.Errorf("imgproc: invalid correlator backend")
	}
	if imgA == nil || imgB == nil {
		return nil, fmt.Errorf("imgproc: input images must not be nil")
	}
	wA, hA := imgA.Bounds().Dx(), imgA.Bounds().Dy()
	wB, hB := imgB.Bounds().Dx(), imgB.Bounds().Dy()
	if wA != wB || hA != hB {
		return nil, fmt.Errorf("imgproc: input images must have equal dimensions, got %dx%d and %dx%d", wA, hA, wB, hB)
	}
	if wA < 2 || hA < 2 {
		return nil, fmt.Errorf("imgproc: input dimensions must both be at least 2 pixels")
	}
	if wA > c.maxW || hA > c.maxH {
		return nil, fmt.Errorf("imgproc: input dimensions %dx%d exceed correlator maximum %dx%d", wA, hA, c.maxW, c.maxH)
	}
	pixelsA, err := packRGBA(imgA)
	if err != nil {
		return nil, fmt.Errorf("imgproc: image A: %w", err)
	}
	pixelsB, err := packRGBA(imgB)
	if err != nil {
		return nil, fmt.Errorf("imgproc: image B: %w", err)
	}

	// Determine square crop size from the smaller dimension of each image.
	cropSize := min(wA, hA)

	wgSize := uint32(64)
	n := uint32(c.w * c.h)
	groups := (n + wgSize - 1) / wgSize
	paramsBuf := c.params.DeviceBuffer

	// Image dimensions for params.
	srcWA := uint32(wA)
	srcHA := uint32(hA)
	srcStrideA := srcWA
	srcWB := uint32(wB)
	srcHB := uint32(hB)
	srcStrideB := srcWB
	padSize := uint32(c.w)
	cropU32 := uint32(cropSize)

	// ---- Stage tightly packed RGBA pixels once per image ----
	if err := c.rgbaA.StageUploadSlice(c.ctx, pixelsA); err != nil {
		return nil, fmt.Errorf("imgproc: stage imgA: %w", err)
	}
	if err := c.rgbaB.StageUploadSlice(c.ctx, pixelsB); err != nil {
		return nil, fmt.Errorf("imgproc: stage imgB: %w", err)
	}

	// ---- Single command buffer for uploads, compute, and readback ----
	rec, err := c.ctx.NewCommandRecorder()
	if err != nil {
		return nil, err
	}
	defer rec.Abort()
	if err := rec.CopyToDevice(c.rgbaA.Buf, uint64(len(pixelsA))*4); err != nil {
		return nil, fmt.Errorf("imgproc: record imgA upload: %w", err)
	}
	if err := rec.CopyToDevice(c.rgbaB.Buf, uint64(len(pixelsB))*4); err != nil {
		return nil, fmt.Errorf("imgproc: record imgB upload: %w", err)
	}

	// ==== Phase 1: Rotation & Scale (per Reddy & Chatterji §III) ====

	// GrayscalePad A: RGBA → complex (crop centered square + zero-pad to pow2).
	// No DoG, no Hann — paper relies on highpass filter of magnitude spectrum.
	gpParamsA := encodeGrayPadParams(srcWA, srcHA, srcStrideA, padSize, cropU32)
	rec.UpdateBuffer(paramsBuf, 0, gpParamsA)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeGrayPadA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexA.Buf.DeviceBuffer)

	// GrayscalePad B: RGBA → complex.
	gpParamsB := encodeGrayPadParams(srcWB, srcHB, srcStrideB, padSize, cropU32)
	rec.UpdateBuffer(paramsBuf, 0, gpParamsB)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeGrayPadB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexB.Buf.DeviceBuffer)

	// FFT2D on complexA and complexB.
	recordFFT2D(rec, c.complexA.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevA, c.pipeButterflyA, c.w, c.h, false)
	recordFFT2D(rec, c.complexB.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevB, c.pipeButterflyB, c.w, c.h, false)

	// Magnitude: log(1 + |F|) per paper §III.A.
	magParams := encodeU32Params(n)
	rec.UpdateBuffer(paramsBuf, 0, magParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeMagA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.magA.Buf.DeviceBuffer)
	rec.Bind(c.pipeMagB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.magB.Buf.DeviceBuffer)

	// Fftshift magnitude spectra so DC is at center (paper §III.A, implied).
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

	// Highpass emphasis filter per paper §III.B eq. 23-24:
	// H(ξ,η) = (1 − X)(2 − X) where X = cos(πξ)cos(πη).
	// Reuse shiftParams (same width/height layout).
	rec.Bind(c.pipeHighpassA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.magA.Buf.DeviceBuffer)
	rec.Bind(c.pipeHighpassB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.magB.Buf.DeviceBuffer)

	// Log-polar remap (paper §III.A, §III.C).
	// Radius = cropSize * 1.1 / 2 following imreg_dft.
	maxRadius := float64(cropSize) * 1.1 / 2.0
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

	// Cross-power spectrum with phase normalization per paper eq. (3):
	// F·F'* / |F·F'*|
	cpParams := encodeU32Params(lpN, 1) // normalize=1
	rec.UpdateBuffer(paramsBuf, 0, cpParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeCrosspowerLP)
	rec.Dispatch(lpGroups, 1, 1)
	rec.Barrier(c.crossPow.Buf.DeviceBuffer)

	// IFFT2D on cross-power result.
	recordFFT2D(rec, c.crossPow.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevCP, c.pipeButterflyCP, c.lpW, c.lpH, true)

	// Peak find (logpolar mode): find the IFFT peak, then convert it to
	// confidence, angle, scale, and the two rotation candidates in result[0..7].
	peakLPParams := encodePeakFindParams(uint32(c.lpW), uint32(c.lpH), 0, 0, logRmax)
	rec.UpdateBuffer(paramsBuf, 0, peakLPParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipePeakFind)
	rec.Dispatch(peakWorkgroups, 1, 1)
	rec.Barrier(c.peakScratch.Buf.DeviceBuffer)
	rec.Bind(c.pipePeakReduce)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(c.peakScratch.Buf.DeviceBuffer, c.result.Buf.DeviceBuffer)
	rec.Bind(c.pipePeakFinal)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(c.result.Buf.DeviceBuffer)

	// ==== Phase 2 Try 1: Translation (angle θ₀) ====
	// Per paper §III end: "the image with the highest resolution is scaled and
	// rotated by amounts a and θ₀, respectively, and the amount of translational
	// movement is found out using phase correlation technique."

	// BilinearWarpGray A (slot=0: uses cos/sin from result[4,5]) → complexA.
	// Inverse rotation+scale of imgA to align with imgB's frame.
	warpParams0 := encodeWarpParams(srcWA, srcHA, srcStrideA, padSize, cropU32, 0)
	rec.UpdateBuffer(paramsBuf, 0, warpParams0)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeWarpA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexA.Buf.DeviceBuffer)

	// GrayscalePad B → complexB (fresh, since complexB was overwritten by Phase 1).
	rec.UpdateBuffer(paramsBuf, 0, gpParamsB)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeGrayPadB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexB.Buf.DeviceBuffer)

	// FFT2D, crosspower (phase-normalized), IFFT2D.
	recordFFT2D(rec, c.complexA.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevA, c.pipeButterflyA, c.w, c.h, false)
	recordFFT2D(rec, c.complexB.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevB, c.pipeButterflyB, c.w, c.h, false)
	transParams := encodeU32Params(n, 1) // normalize=1
	rec.UpdateBuffer(paramsBuf, 0, transParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeCrosspowerTrans)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.crossPow.Buf.DeviceBuffer)
	recordFFT2D(rec, c.crossPow.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevCP, c.pipeButterflyCP, c.w, c.h, true)

	// Peak find (translation mode, offset=8): result[8..10] is tx1, ty1, confidence1.
	peakT1Params := encodePeakFindParams(uint32(c.w), uint32(c.h), 1, 8, 0)
	rec.UpdateBuffer(paramsBuf, 0, peakT1Params)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipePeakFind)
	rec.Dispatch(peakWorkgroups, 1, 1)
	rec.Barrier(c.peakScratch.Buf.DeviceBuffer)
	rec.Bind(c.pipePeakReduce)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(c.peakScratch.Buf.DeviceBuffer, c.result.Buf.DeviceBuffer)
	rec.Bind(c.pipePeakFinal)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(c.result.Buf.DeviceBuffer)

	// ==== Phase 2 Try 2: Translation (angle θ₀ + 180°) ====
	// Paper p. 1268: "We then rotate the spectrum of Image 2 by (180° + θ₀)
	// and again compute the translation."

	// BilinearWarpGray A (slot=1: uses -cos/-sin from result[6,7]) → complexA.
	warpParams1 := encodeWarpParams(srcWA, srcHA, srcStrideA, padSize, cropU32, 1)
	rec.UpdateBuffer(paramsBuf, 0, warpParams1)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeWarpA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexA.Buf.DeviceBuffer)

	// GrayscalePad B → complexB (fresh again).
	rec.UpdateBuffer(paramsBuf, 0, gpParamsB)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeGrayPadB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexB.Buf.DeviceBuffer)

	// FFT2D, crosspower (phase-normalized), IFFT2D.
	recordFFT2D(rec, c.complexA.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevA, c.pipeButterflyA, c.w, c.h, false)
	recordFFT2D(rec, c.complexB.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevB, c.pipeButterflyB, c.w, c.h, false)
	rec.UpdateBuffer(paramsBuf, 0, transParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeCrosspowerTrans)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.crossPow.Buf.DeviceBuffer)
	recordFFT2D(rec, c.crossPow.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevCP, c.pipeButterflyCP, c.w, c.h, true)

	// Peak find (translation mode, offset=12): result[12..14] is tx2, ty2, confidence2.
	peakT2Params := encodePeakFindParams(uint32(c.w), uint32(c.h), 1, 12, 0)
	rec.UpdateBuffer(paramsBuf, 0, peakT2Params)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipePeakFind)
	rec.Dispatch(peakWorkgroups, 1, 1)
	rec.Barrier(c.peakScratch.Buf.DeviceBuffer)
	rec.Bind(c.pipePeakReduce)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(c.peakScratch.Buf.DeviceBuffer, c.result.Buf.DeviceBuffer)
	rec.Bind(c.pipePeakFinal)
	rec.Dispatch(1, 1, 1)
	if err := rec.CopyToStaging(c.result.Buf, c.result.Buf.Size()); err != nil {
		return nil, fmt.Errorf("imgproc: record result readback: %w", err)
	}

	// ==== Submit uploads, pipeline, and readback together ====
	if err := rec.Submit(); err != nil {
		return nil, fmt.Errorf("imgproc: GPU pipeline: %w", err)
	}

	// ---- Read the 64 bytes copied to staging by the completed submission ----
	res, err := c.result.ReadStagedSlice(c.ctx)
	if err != nil {
		return nil, fmt.Errorf("imgproc: download results: %w", err)
	}

	// Parse results.
	angleDeg := float64(res[2])
	scale := float64(res[3])
	rotationConfidence := float64(res[0])
	tx1 := float64(res[8])
	ty1 := float64(res[9])
	confidence1 := float64(res[10])
	tx2 := float64(res[12])
	ty2 := float64(res[13])
	confidence2 := float64(res[14])

	if math.IsNaN(angleDeg) || math.IsInf(angleDeg, 0) ||
		math.IsNaN(scale) || math.IsInf(scale, 0) || scale <= 0 ||
		math.IsNaN(rotationConfidence) || math.IsInf(rotationConfidence, 0) ||
		math.IsNaN(tx1) || math.IsInf(tx1, 0) || math.IsNaN(ty1) || math.IsInf(ty1, 0) ||
		math.IsNaN(tx2) || math.IsInf(tx2, 0) || math.IsNaN(ty2) || math.IsInf(ty2, 0) ||
		math.IsNaN(confidence1) || math.IsInf(confidence1, 0) ||
		math.IsNaN(confidence2) || math.IsInf(confidence2, 0) {
		return nil, fmt.Errorf("imgproc: GPU pipeline returned a non-finite transform")
	}

	// Disambiguate: pick the 180° variant with higher peak magnitude.
	// Paper p. 1268: "If the value of the peak of the IFFT of the cross-power
	// spectrum phase is greater when the angle is θ₀, then the true angle of
	// rotation is θ₀, otherwise (180° + θ₀) is the true angle of rotation."
	bestAngle := angleDeg
	bestTx, bestTy := tx1, ty1
	translationConfidence := confidence1
	if confidence2 > confidence1 {
		bestAngle = angleDeg + 180
		bestTx, bestTy = tx2, ty2
		translationConfidence = confidence2
	}
	if rotationConfidence <= minimumMatchConfidence || translationConfidence <= minimumMatchConfidence {
		return nil, fmt.Errorf("%w: rotation %.5f, translation %.5f, minimum %.2f",
			ErrLowConfidence, rotationConfidence, translationConfidence, minimumMatchConfidence)
	}

	return &Result{
		Angle:                 bestAngle,
		Scale:                 scale,
		Tx:                    bestTx,
		Ty:                    bestTy,
		RotationConfidence:    rotationConfidence,
		TranslationConfidence: translationConfidence,
	}, nil
}

// Close releases all GPU resources.
func (c *Correlator) Close() {
	if c == nil || c.closed {
		return
	}
	c.closed = true
	if c.backend == BackendCPU {
		return
	}

	for _, p := range c.allPipelines {
		p.Destroy(c.ctx)
	}
	if c.rgbaA != nil {
		c.rgbaA.Destroy(c.ctx)
	}
	if c.rgbaB != nil {
		c.rgbaB.Destroy(c.ctx)
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
	if c.peakScratch != nil {
		c.peakScratch.Destroy(c.ctx)
	}
	if c.result != nil {
		c.result.Destroy(c.ctx)
	}
	if c.params != nil {
		c.params.Destroy(c.ctx)
	}
	if c.ownsContext && c.ctx != nil {
		c.ctx.Close()
		c.ctx = nil
	}
}

// encodeGrayPadParams encodes parameters for the grayscale_pad shader.
func encodeGrayPadParams(srcW, srcH, srcStride, padSize, cropSize uint32) []byte {
	buf := make([]byte, 20)
	binary.LittleEndian.PutUint32(buf[0:4], srcW)
	binary.LittleEndian.PutUint32(buf[4:8], srcH)
	binary.LittleEndian.PutUint32(buf[8:12], srcStride)
	binary.LittleEndian.PutUint32(buf[12:16], padSize)
	binary.LittleEndian.PutUint32(buf[16:20], cropSize)
	return buf
}

// encodeWarpParams encodes parameters for the bilinear_warp_gray shader.
func encodeWarpParams(srcW, srcH, srcStride, padSize, cropSize, warpSlot uint32) []byte {
	buf := make([]byte, 24)
	binary.LittleEndian.PutUint32(buf[0:4], srcW)
	binary.LittleEndian.PutUint32(buf[4:8], srcH)
	binary.LittleEndian.PutUint32(buf[8:12], srcStride)
	binary.LittleEndian.PutUint32(buf[12:16], padSize)
	binary.LittleEndian.PutUint32(buf[16:20], cropSize)
	binary.LittleEndian.PutUint32(buf[20:24], warpSlot)
	return buf
}

// encodePeakFindParams encodes parameters for the peak_find shader.
func encodePeakFindParams(width, height, mode, resultOffset uint32, logRmax float32) []byte {
	buf := make([]byte, 20)
	binary.LittleEndian.PutUint32(buf[0:4], width)
	binary.LittleEndian.PutUint32(buf[4:8], height)
	binary.LittleEndian.PutUint32(buf[8:12], mode)
	binary.LittleEndian.PutUint32(buf[12:16], resultOffset)
	binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(logRmax))
	return buf
}
