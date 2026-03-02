package imgproc

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"image"
	"math"
	"unsafe"

	"vkpg/compute"
	"vkpg/shader"
	"vkpg/vk"
)

//go:embed shaders/grayscale_pad.wgsl
var grayscalePadWGSL string

//go:embed shaders/bilinear_warp_gray.wgsl
var bilinearWarpGrayWGSL string

//go:embed shaders/peak_find.wgsl
var peakFindWGSL string

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
	// Max image dimensions (for RGBA buffer sizing).
	maxW, maxH int

	// Compiled SPIR-V.
	spirvGrayPad      []byte
	spirvWarpGray     []byte
	spirvPeakFind     []byte
	spirvBitrev       []byte
	spirvButterfly    []byte
	spirvMagnitude    []byte
	spirvHighpass     []byte
	spirvLogpolar     []byte
	spirvFftshift     []byte
	spirvCrosspower   []byte

	// Working buffers.
	rgbaA, rgbaB       *compute.TypedBuffer[uint32]   // raw RGBA (full image)
	complexA, complexB *compute.TypedBuffer[[2]float32] // padSize x padSize
	magA, magB         *compute.TypedBuffer[float32]    // padSize x padSize
	logPolA, logPolB   *compute.TypedBuffer[[2]float32] // lpW x lpH
	crossPow           *compute.TypedBuffer[[2]float32] // max(n, lpN)
	result             *compute.TypedBuffer[float32]    // 16 floats = 64 bytes
	params             *compute.Buffer                  // shared params

	// Pipelines.
	pipeGrayPadA, pipeGrayPadB     *compute.Pipeline
	pipeWarpA                      *compute.Pipeline
	pipeBitrevA, pipeBitrevB       *compute.Pipeline
	pipeButterflyA, pipeButterflyB *compute.Pipeline
	pipeMagA, pipeMagB             *compute.Pipeline
	pipeFftshiftA, pipeFftshiftB   *compute.Pipeline
	pipeHighpassA, pipeHighpassB   *compute.Pipeline
	pipeLogPolA, pipeLogPolB       *compute.Pipeline
	pipeBitrevLPA, pipeBitrevLPB   *compute.Pipeline
	pipeButterflyLPA, pipeButterflyLPB *compute.Pipeline
	pipeBitrevCP, pipeButterflyCP  *compute.Pipeline
	pipeCrosspowerLP               *compute.Pipeline
	pipeCrosspowerTrans            *compute.Pipeline
	pipePeakFind                   *compute.Pipeline

	allPipelines []*compute.Pipeline
}

// NewCorrelator creates a Correlator for images up to maxW x maxH pixels.
func NewCorrelator(ctx *compute.Context, maxW, maxH int) (*Correlator, error) {
	c := &Correlator{ctx: ctx, maxW: maxW, maxH: maxH}

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
		{&c.spirvGrayPad, grayscalePadWGSL, "grayscale_pad"},
		{&c.spirvWarpGray, bilinearWarpGrayWGSL, "bilinear_warp_gray"},
		{&c.spirvPeakFind, peakFindWGSL, "peak_find"},
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

	// Result buffer: 16 x float32 = 64 bytes.
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

	// PeakFind: binding 0=input(complex), 1=result(f32), 2=params
	c.pipePeakFind, err = mk(c.spirvPeakFind, []compute.BufferBinding{
		bb(0, c.crossPow.Buf), bb(1, c.result.Buf), bb(2, paramBuf),
	})
	if err != nil {
		return err
	}

	return nil
}

// uploadRGBA uploads raw RGBA pixel data to the GPU buffer.
func uploadRGBA(ctx *compute.Context, buf *compute.TypedBuffer[uint32], img *image.RGBA) error {
	pix := img.Pix
	// Upload raw bytes (reinterpreted as packed RGBA u32 on GPU).
	return buf.Buf.Upload(ctx, pix)
}

// PhaseCorrelate recovers rotation, scale, and translation between two images.
// Uploads raw RGBA once, runs entire pipeline on GPU, downloads only 64 bytes of results.
func (c *Correlator) PhaseCorrelate(imgA, imgB *image.RGBA) (*Result, error) {
	cropSize := min(imgA.Bounds().Dx(), imgA.Bounds().Dy(),
		imgB.Bounds().Dx(), imgB.Bounds().Dy())

	wgSize := uint32(64)
	n := uint32(c.w * c.h)
	groups := (n + wgSize - 1) / wgSize
	paramsBuf := c.params.DeviceBuffer

	// Image dimensions for params.
	srcWA := uint32(imgA.Bounds().Dx())
	srcHA := uint32(imgA.Bounds().Dy())
	srcStrideA := uint32(imgA.Stride / 4)
	srcWB := uint32(imgB.Bounds().Dx())
	srcHB := uint32(imgB.Bounds().Dy())
	srcStrideB := uint32(imgB.Stride / 4)
	padSize := uint32(c.w)
	cropU32 := uint32(cropSize)

	// ---- Upload raw RGBA pixels (once) ----
	if err := uploadRGBA(c.ctx, c.rgbaA, imgA); err != nil {
		return nil, fmt.Errorf("imgproc: upload imgA: %w", err)
	}
	if err := uploadRGBA(c.ctx, c.rgbaB, imgB); err != nil {
		return nil, fmt.Errorf("imgproc: upload imgB: %w", err)
	}

	// ---- Single command buffer for entire pipeline ----
	rec, err := c.ctx.NewCommandRecorder()
	if err != nil {
		return nil, err
	}

	// ==== Phase 1: Rotation & Scale ====

	// GrayscalePad A: RGBA → complex (crop + pad)
	gpParamsA := encodeGrayPadParams(srcWA, srcHA, srcStrideA, padSize, cropU32)
	rec.UpdateBuffer(paramsBuf, 0, gpParamsA)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeGrayPadA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexA.Buf.DeviceBuffer)

	// GrayscalePad B: RGBA → complex (crop + pad)
	gpParamsB := encodeGrayPadParams(srcWB, srcHB, srcStrideB, padSize, cropU32)
	rec.UpdateBuffer(paramsBuf, 0, gpParamsB)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeGrayPadB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexB.Buf.DeviceBuffer)

	// DEBUG: Submit and verify grayscale_pad output.
	if err := rec.Submit(); err != nil {
		return nil, fmt.Errorf("imgproc: grayscale_pad submit: %w", err)
	}
	{
		dbg, err := c.complexA.DownloadSlice(c.ctx)
		if err != nil {
			return nil, fmt.Errorf("imgproc: debug download: %w", err)
		}
		// Check center pixel and some stats.
		center := c.w/2*c.w + c.w/2
		nonzero := 0
		maxVal := float32(0)
		for _, v := range dbg {
			if v[0] != 0 {
				nonzero++
			}
			if v[0] > maxVal {
				maxVal = v[0]
			}
		}
		fmt.Printf("DEBUG grayscale_pad: center=(%f,%f) nonzero=%d/%d max=%f\n",
			dbg[center][0], dbg[center][1], nonzero, len(dbg), maxVal)
	}

	rec, err = c.ctx.NewCommandRecorder()
	if err != nil {
		return nil, err
	}

	// FFT2D on complexA and complexB.
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

	// Fftshift magnitude spectra.
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

	// Highpass.
	rec.Bind(c.pipeHighpassA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.magA.Buf.DeviceBuffer)
	rec.Bind(c.pipeHighpassB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.magB.Buf.DeviceBuffer)

	// Log-polar remap.
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

	// Cross-power spectrum (phase-normalized).
	cpParams := encodeU32Params(lpN, 1) // normalize=1
	rec.UpdateBuffer(paramsBuf, 0, cpParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeCrosspowerLP)
	rec.Dispatch(lpGroups, 1, 1)
	rec.Barrier(c.crossPow.Buf.DeviceBuffer)

	// IFFT2D on cross-power result.
	recordFFT2D(rec, c.crossPow.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevCP, c.pipeButterflyCP, c.lpW, c.lpH, true)

	// DEBUG: Download IFFT result and find peak on CPU.
	if err := rec.Submit(); err != nil {
		return nil, fmt.Errorf("imgproc: phase1 IFFT: %w", err)
	}
	{
		cpData, err := c.crossPow.DownloadSlice(c.ctx)
		if err != nil {
			return nil, err
		}
		// CPU peak finding (inline).
		maxVal := float64(0)
		maxX, maxY := 0, 0
		for y := 0; y < c.lpH; y++ {
			for x := 0; x < c.lpW; x++ {
				cv := cpData[y*c.lpW+x]
				mag := math.Sqrt(float64(cv[0]*cv[0] + cv[1]*cv[1]))
				if mag > maxVal {
					maxVal = mag
					maxX = x
					maxY = y
				}
			}
		}
		fmt.Printf("DEBUG CPU peak: (%d, %d) mag=%f (lpW=%d lpH=%d)\n", maxX, maxY, maxVal, c.lpW, c.lpH)
		// Also check DC for comparison.
		dc := cpData[0]
		dcMag := math.Sqrt(float64(dc[0]*dc[0] + dc[1]*dc[1]))
		fmt.Printf("DEBUG CPU DC mag=%f, ratio peak/DC=%f\n", dcMag, maxVal/dcMag)
	}

	rec, err = c.ctx.NewCommandRecorder()
	if err != nil {
		return nil, err
	}

	// Peak find (logpolar mode) → result[0..7]
	peakLPParams := encodePeakFindParams(uint32(c.lpW), uint32(c.lpH), 0, 0, logRmax)
	rec.UpdateBuffer(paramsBuf, 0, peakLPParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipePeakFind)
	rec.Dispatch(1, 1, 1) // single workgroup
	rec.Barrier(c.result.Buf.DeviceBuffer)

	// ==== Phase 2 Try 1: Translation (angle) ====

	// GrayscalePad B → complexB (fresh)
	rec.UpdateBuffer(paramsBuf, 0, gpParamsB)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeGrayPadB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexB.Buf.DeviceBuffer)

	// BilinearWarpGray A (slot=0) → complexA
	warpParams0 := encodeWarpParams(srcWA, srcHA, srcStrideA, padSize, cropU32, 0)
	rec.UpdateBuffer(paramsBuf, 0, warpParams0)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeWarpA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexA.Buf.DeviceBuffer)

	// FFT2D, crosspower, IFFT2D
	recordFFT2D(rec, c.complexA.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevA, c.pipeButterflyA, c.w, c.h, false)
	recordFFT2D(rec, c.complexB.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevB, c.pipeButterflyB, c.w, c.h, false)
	transParams := encodeU32Params(n, 1) // normalize=1
	rec.UpdateBuffer(paramsBuf, 0, transParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeCrosspowerTrans)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.crossPow.Buf.DeviceBuffer)
	recordFFT2D(rec, c.crossPow.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevCP, c.pipeButterflyCP, c.w, c.h, true)

	// Peak find (translation mode, offset=8) → result[8..10]
	peakT1Params := encodePeakFindParams(uint32(c.w), uint32(c.h), 1, 8, 0)
	rec.UpdateBuffer(paramsBuf, 0, peakT1Params)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipePeakFind)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(c.result.Buf.DeviceBuffer)

	// ==== Phase 2 Try 2: Translation (angle + 180°) ====

	// GrayscalePad B → complexB (fresh again)
	rec.UpdateBuffer(paramsBuf, 0, gpParamsB)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeGrayPadB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexB.Buf.DeviceBuffer)

	// BilinearWarpGray A (slot=1, 180° variant) → complexA
	warpParams1 := encodeWarpParams(srcWA, srcHA, srcStrideA, padSize, cropU32, 1)
	rec.UpdateBuffer(paramsBuf, 0, warpParams1)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeWarpA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexA.Buf.DeviceBuffer)

	// FFT2D, crosspower, IFFT2D
	recordFFT2D(rec, c.complexA.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevA, c.pipeButterflyA, c.w, c.h, false)
	recordFFT2D(rec, c.complexB.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevB, c.pipeButterflyB, c.w, c.h, false)
	rec.UpdateBuffer(paramsBuf, 0, transParams)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipeCrosspowerTrans)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.crossPow.Buf.DeviceBuffer)
	recordFFT2D(rec, c.crossPow.Buf.DeviceBuffer, paramsBuf, c.pipeBitrevCP, c.pipeButterflyCP, c.w, c.h, true)

	// Peak find (translation mode, offset=12) → result[12..14]
	peakT2Params := encodePeakFindParams(uint32(c.w), uint32(c.h), 1, 12, 0)
	rec.UpdateBuffer(paramsBuf, 0, peakT2Params)
	rec.BarrierTransferToCompute(paramsBuf)
	rec.Bind(c.pipePeakFind)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(c.result.Buf.DeviceBuffer)

	// ==== Submit everything ====
	if err := rec.Submit(); err != nil {
		return nil, fmt.Errorf("imgproc: GPU pipeline: %w", err)
	}

	// ---- Download 64 bytes of results ----
	res, err := c.result.DownloadSlice(c.ctx)
	if err != nil {
		return nil, fmt.Errorf("imgproc: download results: %w", err)
	}

	// Parse results.
	rawPeakX := float64(res[0])
	rawPeakY := float64(res[1])
	angleDeg := float64(res[2])
	scale := float64(res[3])
	tx1 := float64(res[8])
	ty1 := float64(res[9])
	mag1 := float64(res[10])
	tx2 := float64(res[12])
	ty2 := float64(res[13])
	mag2 := float64(res[14])

	fmt.Printf("DEBUG: raw peak (%f, %f) lpW=%d lpH=%d maxR=%.2f cropSize=%d\n",
		rawPeakX, rawPeakY, c.lpW, c.lpH, maxRadius, cropSize)
	fmt.Printf("DEBUG: angle=%.2f° scale=%.4f\n", angleDeg, scale)
	fmt.Printf("DEBUG: translation try1 (%.2f, %.2f) mag=%f\n", tx1, ty1, mag1)
	fmt.Printf("DEBUG: translation try2 (%.2f, %.2f) mag=%f\n", tx2, ty2, mag2)

	// Disambiguate: pick the 180° variant with higher peak magnitude.
	bestAngle := angleDeg
	bestTx, bestTy := tx1, ty1
	if mag2 > mag1 {
		bestAngle = angleDeg + 180
		bestTx, bestTy = tx2, ty2
	}

	fmt.Printf("DEBUG: chose angle=%.2f° (peak1=%f peak2=%f)\n", bestAngle, mag1, mag2)

	return &Result{
		Angle: bestAngle,
		Scale: scale,
		Tx:    bestTx,
		Ty:    bestTy,
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
	if c.result != nil {
		c.result.Destroy(c.ctx)
	}
	if c.params != nil {
		c.params.Destroy(c.ctx)
	}
}

// pixToU32 reinterprets RGBA pixel bytes as packed uint32 without copying.
func pixToU32(pix []byte) []uint32 {
	if len(pix) == 0 {
		return nil
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&pix[0])), len(pix)/4)
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
