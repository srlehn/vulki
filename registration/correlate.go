package registration

import (
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/draw"
	"math"
	"reflect"

	"github.com/srlehn/vulki"
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

// Result describes the transform that maps image A to image B. Apply Scale and
// the counterclockwise Angle around the image center, then apply Tx and Ty.
type Result struct {
	// Angle is counterclockwise rotation in degrees, normalized to [-180, 180).
	Angle float64
	// Scale is the image-size multiplier and is greater than zero.
	Scale float64
	// Tx is horizontal translation in pixels; positive values move right.
	Tx float64
	// Ty is vertical translation in pixels; positive values move down.
	Ty float64
	// RotationConfidence is the normalized log-polar correlation peak.
	RotationConfidence float64
	// TranslationConfidence is the normalized translation correlation peak.
	TranslationConfidence float64
}

// ErrLowConfidence indicates that at least one phase-correlation peak did not
// meet the validity threshold described by Reddy and Chatterji.
var ErrLowConfidence = errors.New("registration: phase-correlation match confidence is too low")

const minimumMatchConfidence = 0.03

const maxExactFloat32Index = 1 << 24

const peakWorkgroups uint32 = 64

// Correlator performs log-polar phase correlation on a GPU or CPU backend.
type Correlator struct {
	device *vulki.Device

	backend        Backend
	ownsDevice     bool
	fallbackReason error
	closed         bool

	// Padded image dimensions (power of 2).
	w, h int
	// Log-polar dimensions.
	lpW, lpH int
	// Max image dimensions (for RGBA buffer sizing).
	maxW, maxH int

	// Working buffers.
	rgbaA, rgbaB       *vulki.Buffer // raw RGBA (full image)
	complexA, complexB *vulki.Buffer // padSize x padSize complex values
	magA, magB         *vulki.Buffer // padSize x padSize float values
	logPolA, logPolB   *vulki.Buffer // lpW x lpH complex values
	crossPow           *vulki.Buffer // max(n, lpN) complex values
	peakScratch        *vulki.Buffer // one complex maximum per peak-scan workgroup
	result             *vulki.Buffer // 16 float values = 64 bytes
	params             *vulki.Buffer // shared small params buffer

	// Pipelines.
	pipeGrayPadA, pipeGrayPadB         *gpuPipeline
	pipeWarpA                          *gpuPipeline
	pipeBitrevA, pipeBitrevB           *gpuPipeline
	pipeButterflyA, pipeButterflyB     *gpuPipeline
	pipeMagA, pipeMagB                 *gpuPipeline
	pipeFftshiftA, pipeFftshiftB       *gpuPipeline
	pipeHighpassA, pipeHighpassB       *gpuPipeline
	pipeLogPolA, pipeLogPolB           *gpuPipeline
	pipeBitrevLPA, pipeBitrevLPB       *gpuPipeline
	pipeButterflyLPA, pipeButterflyLPB *gpuPipeline
	pipeBitrevCP, pipeButterflyCP      *gpuPipeline
	pipeCrosspowerLP                   *gpuPipeline
	pipeCrosspowerTrans                *gpuPipeline
	pipePeakFind                       *gpuPipeline
	pipePeakReduce                     *gpuPipeline
	pipePeakFinal                      *gpuPipeline

	kernels     map[string]*vulki.Kernel
	kernelOrder []*vulki.Kernel
	bindings    []*vulki.BindingSet
}

func newVulkanCorrelator(device *vulki.Device, maxW, maxH int) (*Correlator, error) {
	if device == nil || device.Closed() {
		return nil, fmt.Errorf("registration: invalid Vulkan device")
	}
	if maxW < 2 || maxH < 2 {
		return nil, fmt.Errorf("registration: maximum dimensions must both be at least 2 pixels")
	}
	if uint64(maxW) > math.MaxUint32 || uint64(maxH) > math.MaxUint32 {
		return nil, fmt.Errorf("registration: maximum dimensions exceed shader limits")
	}
	maxInt := int(^uint(0) >> 1)
	if maxW > maxInt/maxH {
		return nil, fmt.Errorf("registration: maximum image area overflows int")
	}
	if uint64(maxW)*uint64(maxH) > math.MaxUint32 {
		return nil, fmt.Errorf("registration: maximum image area exceeds shader indexing limits")
	}

	c := &Correlator{
		device: device, backend: BackendVulkan, maxW: maxW, maxH: maxH,
		kernels: make(map[string]*vulki.Kernel),
	}

	// Square crop + minimal padding: crop to min(w,h) then pad to next power of 2.
	// This preserves spectral symmetry and avoids excessive zero-padding.
	padSize, ok := nextPow2Checked(min(maxW, maxH))
	if !ok || padSize > maxInt/padSize {
		return nil, fmt.Errorf("registration: padded image area overflows int")
	}
	if uint64(padSize)*uint64(padSize) > maxExactFloat32Index {
		return nil, fmt.Errorf("registration: padded image area exceeds peak-index precision limit")
	}
	c.w = padSize
	c.h = padSize
	c.lpW = c.w
	c.lpH = c.h

	n := c.w * c.h
	lpN := c.lpW * c.lpH

	var err error
	// RGBA buffers: hold full raw image pixels.
	rgbaSize := maxW * maxH
	c.rgbaA, err = newElementBuffer[uint32](device, rgbaSize)
	if err != nil {
		return nil, fmt.Errorf("registration: alloc rgbaA: %w", err)
	}
	c.rgbaB, err = newElementBuffer[uint32](device, rgbaSize)
	if err != nil {
		c.Close()
		return nil, fmt.Errorf("registration: alloc rgbaB: %w", err)
	}

	// Complex working buffers: padSize x padSize.
	c.complexA, err = newElementBuffer[[2]float32](device, n)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.complexB, err = newElementBuffer[[2]float32](device, n)
	if err != nil {
		c.Close()
		return nil, err
	}

	// Magnitude buffers.
	c.magA, err = newElementBuffer[float32](device, n)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.magB, err = newElementBuffer[float32](device, n)
	if err != nil {
		c.Close()
		return nil, err
	}

	// Log-polar buffers.
	c.logPolA, err = newElementBuffer[[2]float32](device, lpN)
	if err != nil {
		c.Close()
		return nil, err
	}
	c.logPolB, err = newElementBuffer[[2]float32](device, lpN)
	if err != nil {
		c.Close()
		return nil, err
	}

	// Cross-power buffer.
	c.crossPow, err = newElementBuffer[[2]float32](device, max(n, lpN))
	if err != nil {
		c.Close()
		return nil, err
	}

	c.peakScratch, err = newElementBuffer[[2]float32](device, int(peakWorkgroups))
	if err != nil {
		c.Close()
		return nil, err
	}

	// Result buffer: 16 x float32 = 64 bytes.
	// Layout: [0] rotation confidence [1] rawPeakY [2] angle_deg [3] scale
	//         [4] cos(angle) [5] sin(angle) [6] -cos(angle) [7] -sin(angle)
	//         [8] tx1 [9] ty1 [10] confidence1 [11] peak-scan scratch
	//         [12] tx2 [13] ty2 [14] confidence2 [15] peak-scan scratch
	c.result, err = newElementBuffer[float32](device, 16)
	if err != nil {
		c.Close()
		return nil, err
	}

	// Params buffer: 32 bytes covers all shader param structs.
	c.params, err = device.NewBuffer(32)
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
	mk := func(name, source string, bindings ...gpuBinding) (*gpuPipeline, error) {
		return c.newPipeline(name, source, bindings)
	}
	paramBuf := c.params

	// GrayscalePad: binding 0=rgba, 1=complex, 2=params
	c.pipeGrayPadA, err = mk("grayscale_pad", grayscalePadWGSL,
		readBinding(0, c.rgbaA), readWriteBinding(1, c.complexA), readBinding(2, paramBuf))
	if err != nil {
		return err
	}
	c.pipeGrayPadB, err = mk("grayscale_pad", grayscalePadWGSL,
		readBinding(0, c.rgbaB), readWriteBinding(1, c.complexB), readBinding(2, paramBuf))
	if err != nil {
		return err
	}

	// BilinearWarpGray: binding 0=rgba, 1=complex, 2=params, 3=result
	c.pipeWarpA, err = mk("bilinear_warp_gray", bilinearWarpGrayWGSL,
		readBinding(0, c.rgbaA), readWriteBinding(1, c.complexA),
		readBinding(2, paramBuf), readBinding(3, c.result))
	if err != nil {
		return err
	}

	// FFT bitrev + butterfly for complexA, complexB.
	c.pipeBitrevA, err = mk("fft_bitrev", fftBitrevWGSL,
		readWriteBinding(0, c.complexA), readBinding(1, paramBuf))
	if err != nil {
		return err
	}
	c.pipeButterflyA, err = mk("fft_butterfly", fftButterflyWGSL,
		readWriteBinding(0, c.complexA), readBinding(1, paramBuf))
	if err != nil {
		return err
	}
	c.pipeBitrevB, err = mk("fft_bitrev", fftBitrevWGSL,
		readWriteBinding(0, c.complexB), readBinding(1, paramBuf))
	if err != nil {
		return err
	}
	c.pipeButterflyB, err = mk("fft_butterfly", fftButterflyWGSL,
		readWriteBinding(0, c.complexB), readBinding(1, paramBuf))
	if err != nil {
		return err
	}

	// Magnitude: binding 0=complex(read), 1=mag(write), 2=params
	c.pipeMagA, err = mk("magnitude", magnitudeWGSL,
		readBinding(0, c.complexA), readWriteBinding(1, c.magA), readBinding(2, paramBuf))
	if err != nil {
		return err
	}
	c.pipeMagB, err = mk("magnitude", magnitudeWGSL,
		readBinding(0, c.complexB), readWriteBinding(1, c.magB), readBinding(2, paramBuf))
	if err != nil {
		return err
	}

	// Fftshift: binding 0=mag(rw), 1=params
	c.pipeFftshiftA, err = mk("fftshift", fftshiftWGSL,
		readWriteBinding(0, c.magA), readBinding(1, paramBuf))
	if err != nil {
		return err
	}
	c.pipeFftshiftB, err = mk("fftshift", fftshiftWGSL,
		readWriteBinding(0, c.magB), readBinding(1, paramBuf))
	if err != nil {
		return err
	}

	// Highpass: binding 0=mag(rw), 1=params
	c.pipeHighpassA, err = mk("highpass", highpassWGSL,
		readWriteBinding(0, c.magA), readBinding(1, paramBuf))
	if err != nil {
		return err
	}
	c.pipeHighpassB, err = mk("highpass", highpassWGSL,
		readWriteBinding(0, c.magB), readBinding(1, paramBuf))
	if err != nil {
		return err
	}

	// Logpolar: binding 0=mag(read), 1=logpol(write), 2=params
	c.pipeLogPolA, err = mk("logpolar", logpolarWGSL,
		readBinding(0, c.magA), readWriteBinding(1, c.logPolA), readBinding(2, paramBuf))
	if err != nil {
		return err
	}
	c.pipeLogPolB, err = mk("logpolar", logpolarWGSL,
		readBinding(0, c.magB), readWriteBinding(1, c.logPolB), readBinding(2, paramBuf))
	if err != nil {
		return err
	}

	// FFT bitrev + butterfly for logPolA, logPolB, crossPow.
	c.pipeBitrevLPA, err = mk("fft_bitrev", fftBitrevWGSL,
		readWriteBinding(0, c.logPolA), readBinding(1, paramBuf))
	if err != nil {
		return err
	}
	c.pipeButterflyLPA, err = mk("fft_butterfly", fftButterflyWGSL,
		readWriteBinding(0, c.logPolA), readBinding(1, paramBuf))
	if err != nil {
		return err
	}
	c.pipeBitrevLPB, err = mk("fft_bitrev", fftBitrevWGSL,
		readWriteBinding(0, c.logPolB), readBinding(1, paramBuf))
	if err != nil {
		return err
	}
	c.pipeButterflyLPB, err = mk("fft_butterfly", fftButterflyWGSL,
		readWriteBinding(0, c.logPolB), readBinding(1, paramBuf))
	if err != nil {
		return err
	}
	c.pipeBitrevCP, err = mk("fft_bitrev", fftBitrevWGSL,
		readWriteBinding(0, c.crossPow), readBinding(1, paramBuf))
	if err != nil {
		return err
	}
	c.pipeButterflyCP, err = mk("fft_butterfly", fftButterflyWGSL,
		readWriteBinding(0, c.crossPow), readBinding(1, paramBuf))
	if err != nil {
		return err
	}

	// Crosspower: binding 0=a(read), 1=b(read), 2=out(write), 3=params.
	// We compute A*conj(B) following scikit-image convention.
	c.pipeCrosspowerLP, err = mk("crosspower", crosspowerWGSL,
		readBinding(0, c.logPolA), readBinding(1, c.logPolB),
		readWriteBinding(2, c.crossPow), readBinding(3, paramBuf))
	if err != nil {
		return err
	}
	c.pipeCrosspowerTrans, err = mk("crosspower", crosspowerWGSL,
		readBinding(0, c.complexA), readBinding(1, c.complexB),
		readWriteBinding(2, c.crossPow), readBinding(3, paramBuf))
	if err != nil {
		return err
	}

	// Peak scan: binding 0=input(complex), 1=per-workgroup scratch, 2=params.
	c.pipePeakFind, err = mk("peak_find", peakFindWGSL,
		readBinding(0, c.crossPow), readWriteBinding(1, c.peakScratch), readBinding(2, paramBuf))
	if err != nil {
		return err
	}
	c.pipePeakReduce, err = mk("peak_reduce", peakReduceWGSL,
		readBinding(0, c.peakScratch), readWriteBinding(1, c.result))
	if err != nil {
		return err
	}
	c.pipePeakFinal, err = mk("peak_finalize", peakFinalizeWGSL,
		readBinding(0, c.crossPow), readWriteBinding(1, c.result), readBinding(2, paramBuf))
	if err != nil {
		return err
	}

	return nil
}

func rgbaLayout(img *image.RGBA) (width, height, rowBytes, required int, err error) {
	if img == nil {
		return 0, 0, 0, 0, fmt.Errorf("registration: nil RGBA image")
	}
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	if w <= 0 || h <= 0 {
		return 0, 0, 0, 0, fmt.Errorf("registration: RGBA image must have non-empty bounds")
	}
	maxInt := int(^uint(0) >> 1)
	if w > maxInt/4 {
		return 0, 0, 0, 0, fmt.Errorf("registration: RGBA row size overflows int")
	}
	rowBytes = w * 4
	if h > maxInt/rowBytes {
		return 0, 0, 0, 0, fmt.Errorf("registration: packed RGBA size overflows int")
	}
	if img.Stride < rowBytes {
		return 0, 0, 0, 0, fmt.Errorf("registration: RGBA stride %d is smaller than row size %d", img.Stride, rowBytes)
	}
	if h-1 > (maxInt-rowBytes)/img.Stride {
		return 0, 0, 0, 0, fmt.Errorf("registration: RGBA layout overflows int")
	}
	required = (h-1)*img.Stride + rowBytes
	if len(img.Pix) < required {
		return 0, 0, 0, 0, fmt.Errorf("registration: RGBA pixel data has %d bytes, need %d", len(img.Pix), required)
	}
	return w, h, rowBytes, required, nil
}

// packRGBA copies an image into tightly packed little-endian RGBA words for
// the CPU reference implementation.
func packRGBA(img *image.RGBA) ([]uint32, error) {
	w, h, rowBytes, _, err := rgbaLayout(img)
	if err != nil {
		return nil, err
	}

	pixels := make([]uint32, w*h)
	for y := range h {
		row := img.Pix[y*img.Stride : y*img.Stride+rowBytes]
		for x := range w {
			off := x * 4
			pixels[y*w+x] = uint32(row[off]) |
				uint32(row[off+1])<<8 |
				uint32(row[off+2])<<16 |
				uint32(row[off+3])<<24
		}
	}
	return pixels, nil
}

// packRGBABytes returns the shader's tightly packed R, G, B, A byte layout.
// Tight images are returned as a view because Recorder.Upload copies the input
// synchronously. Images with row padding are copied once into packed storage.
func packRGBABytes(img *image.RGBA) ([]byte, error) {
	_, h, rowBytes, required, err := rgbaLayout(img)
	if err != nil {
		return nil, err
	}
	packedSize := rowBytes * h
	if img.Stride == rowBytes {
		return img.Pix[:required], nil
	}
	packed := make([]byte, packedSize)
	for y := range h {
		copy(packed[y*rowBytes:(y+1)*rowBytes], img.Pix[y*img.Stride:y*img.Stride+rowBytes])
	}
	return packed, nil
}

func asRGBA(img image.Image) (*image.RGBA, error) {
	switch img := img.(type) {
	case nil:
		return nil, fmt.Errorf("registration: nil image")
	case *image.RGBA:
		if img == nil {
			return nil, fmt.Errorf("registration: nil RGBA image")
		}
		return img, nil
	default:
		bounds := img.Bounds()
		converted := image.NewRGBA(bounds)
		draw.Draw(converted, bounds, img, bounds.Min, draw.Src)
		return converted, nil
	}
}

func imageBounds(img image.Image) (image.Rectangle, error) {
	if img == nil {
		return image.Rectangle{}, fmt.Errorf("registration: nil image")
	}
	value := reflect.ValueOf(img)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map,
		reflect.Pointer, reflect.Slice:
		if value.IsNil() {
			return image.Rectangle{}, fmt.Errorf("registration: nil image")
		}
	}
	return img.Bounds(), nil
}

// PhaseCorrelate recovers the transform that maps image A to image B.
// Following Reddy & Chatterji (1996):
//
//	Phase 1: FFT → magnitude → highpass → log-polar → FFT → cross-power → IFFT → peak → angle/scale
//	Phase 2: transform image A by detected angle/scale, phase correlate with B for translation
//	         Try both angle and angle+180° (magnitude spectrum has 180° symmetry), pick higher peak.
//
// RGBA inputs are used directly. Other image implementations are converted to
// RGBA before processing. The Vulkan path stages packed pixels once per image,
// then submits both uploads, the entire GPU pipeline, and a 64-byte result
// readback as one queue operation.
//
// Inputs must have equal dimensions, both dimensions must be at least two, and
// neither dimension may exceed the maximum passed to NewCorrelator. Processing
// uses a centered square crop based on the smaller dimension and pads it to the
// next power of two. Source images are not mutated.
//
// Angle is counterclockwise in displayed image coordinates. Scale and rotation
// are applied around the image center before translation; positive Tx moves
// right and positive Ty moves down. Confidence values are normalized
// correlation peaks. A peak at or below 0.03 returns a nil Result and an error
// wrapping ErrLowConfidence.
func (c *Correlator) PhaseCorrelate(imgA, imgB image.Image) (*Result, error) {
	if c == nil || c.closed {
		return nil, fmt.Errorf("registration: invalid correlator")
	}
	boundsA, err := imageBounds(imgA)
	if err != nil {
		return nil, fmt.Errorf("registration: image A: %w", err)
	}
	boundsB, err := imageBounds(imgB)
	if err != nil {
		return nil, fmt.Errorf("registration: image B: %w", err)
	}
	wA, hA := boundsA.Dx(), boundsA.Dy()
	wB, hB := boundsB.Dx(), boundsB.Dy()
	if wA != wB || hA != hB {
		return nil, fmt.Errorf("registration: input images must have equal dimensions, got %dx%d and %dx%d", wA, hA, wB, hB)
	}
	if wA < 2 || hA < 2 {
		return nil, fmt.Errorf("registration: input dimensions must both be at least 2 pixels")
	}
	if wA > c.maxW || hA > c.maxH {
		return nil, fmt.Errorf("registration: input dimensions %dx%d exceed correlator maximum %dx%d", wA, hA, c.maxW, c.maxH)
	}
	rgbaA, err := asRGBA(imgA)
	if err != nil {
		return nil, fmt.Errorf("registration: image A: %w", err)
	}
	rgbaB, err := asRGBA(imgB)
	if err != nil {
		return nil, fmt.Errorf("registration: image B: %w", err)
	}
	if c.backend == BackendCPU {
		return c.phaseCorrelateCPU(rgbaA, rgbaB)
	}
	if c.backend != BackendVulkan || c.device == nil || c.device.Closed() {
		return nil, fmt.Errorf("registration: invalid correlator backend")
	}
	pixelsA, err := packRGBABytes(rgbaA)
	if err != nil {
		return nil, fmt.Errorf("registration: image A: %w", err)
	}
	pixelsB, err := packRGBABytes(rgbaB)
	if err != nil {
		return nil, fmt.Errorf("registration: image B: %w", err)
	}

	// Determine square crop size from the smaller dimension of each image.
	cropSize := min(wA, hA)

	wgSize := uint32(64)
	n := uint32(c.w * c.h)
	groups := (n + wgSize - 1) / wgSize
	paramsBuf := c.params

	// Image dimensions for params.
	srcWA := uint32(wA)
	srcHA := uint32(hA)
	srcStrideA := srcWA
	srcWB := uint32(wB)
	srcHB := uint32(hB)
	srcStrideB := srcWB
	padSize := uint32(c.w)
	cropU32 := uint32(cropSize)

	// ---- Single command buffer for uploads, compute, and readback ----
	rec, err := newGPURecorder(c.device)
	if err != nil {
		return nil, err
	}
	defer rec.Abort()
	if err := rec.Upload(c.rgbaA, pixelsA); err != nil {
		return nil, fmt.Errorf("registration: record imgA upload: %w", err)
	}
	if err := rec.Upload(c.rgbaB, pixelsB); err != nil {
		return nil, fmt.Errorf("registration: record imgB upload: %w", err)
	}

	// ==== Phase 1: Rotation & Scale (per Reddy & Chatterji §III) ====

	// GrayscalePad A: RGBA → complex (crop centered square + zero-pad to pow2).
	// No DoG or Hann: the paper relies on a high-pass filter of the magnitude spectrum.
	gpParamsA := encodeGrayPadParams(srcWA, srcHA, srcStrideA, padSize, cropU32)
	rec.Update(paramsBuf, gpParamsA)
	rec.Bind(c.pipeGrayPadA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexA)

	// GrayscalePad B: RGBA → complex.
	gpParamsB := encodeGrayPadParams(srcWB, srcHB, srcStrideB, padSize, cropU32)
	rec.Update(paramsBuf, gpParamsB)
	rec.Bind(c.pipeGrayPadB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexB)

	// FFT2D on complexA and complexB.
	recordFFT2D(rec, c.complexA, paramsBuf, c.pipeBitrevA, c.pipeButterflyA, c.w, c.h, false)
	recordFFT2D(rec, c.complexB, paramsBuf, c.pipeBitrevB, c.pipeButterflyB, c.w, c.h, false)

	// Magnitude: log(1 + |F|) per paper §III.A.
	magParams := encodeU32Params(n)
	rec.Update(paramsBuf, magParams)
	rec.Bind(c.pipeMagA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.magA)
	rec.Bind(c.pipeMagB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.magB)

	// Fftshift magnitude spectra so DC is at center (paper §III.A, implied).
	shiftParams := encodeU32Params(uint32(c.w), uint32(c.h))
	rec.Update(paramsBuf, shiftParams)
	halfN := n / 2
	shiftGroups := (halfN + wgSize - 1) / wgSize
	rec.Bind(c.pipeFftshiftA)
	rec.Dispatch(shiftGroups, 1, 1)
	rec.Barrier(c.magA)
	rec.Bind(c.pipeFftshiftB)
	rec.Dispatch(shiftGroups, 1, 1)
	rec.Barrier(c.magB)

	// Highpass emphasis filter per paper §III.B eq. 23-24:
	// H(ξ,η) = (1 − X)(2 − X) where X = cos(πξ)cos(πη).
	// Reuse shiftParams (same width/height layout).
	rec.Bind(c.pipeHighpassA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.magA)
	rec.Bind(c.pipeHighpassB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.magB)

	// Log-polar remap (paper §III.A, §III.C).
	// Radius = cropSize * 1.1 / 2 following imreg_dft.
	maxRadius := float64(cropSize) * 1.1 / 2.0
	logRmax := float32(math.Log(maxRadius))
	lpParams := encodeLogPolarParams(uint32(c.w), uint32(c.h), uint32(c.lpW), uint32(c.lpH), logRmax)
	rec.Update(paramsBuf, lpParams)
	lpN := uint32(c.lpW * c.lpH)
	lpGroups := (lpN + wgSize - 1) / wgSize
	rec.Bind(c.pipeLogPolA)
	rec.Dispatch(lpGroups, 1, 1)
	rec.Barrier(c.logPolA)
	rec.Bind(c.pipeLogPolB)
	rec.Dispatch(lpGroups, 1, 1)
	rec.Barrier(c.logPolB)

	// FFT2D on log-polar buffers.
	recordFFT2D(rec, c.logPolA, paramsBuf, c.pipeBitrevLPA, c.pipeButterflyLPA, c.lpW, c.lpH, false)
	recordFFT2D(rec, c.logPolB, paramsBuf, c.pipeBitrevLPB, c.pipeButterflyLPB, c.lpW, c.lpH, false)

	// Cross-power spectrum with phase normalization per paper eq. (3):
	// F·F'* / |F·F'*|
	cpParams := encodeU32Params(lpN, 1) // normalize=1
	rec.Update(paramsBuf, cpParams)
	rec.Bind(c.pipeCrosspowerLP)
	rec.Dispatch(lpGroups, 1, 1)
	rec.Barrier(c.crossPow)

	// IFFT2D on cross-power result.
	recordFFT2D(rec, c.crossPow, paramsBuf, c.pipeBitrevCP, c.pipeButterflyCP, c.lpW, c.lpH, true)

	// Peak find (logpolar mode): find the IFFT peak, then convert it to
	// confidence, angle, scale, and the two rotation candidates in result[0..7].
	peakLPParams := encodePeakFindParams(uint32(c.lpW), uint32(c.lpH), 0, 0, logRmax)
	rec.Update(paramsBuf, peakLPParams)
	rec.Bind(c.pipePeakFind)
	rec.Dispatch(peakWorkgroups, 1, 1)
	rec.Barrier(c.peakScratch)
	rec.Bind(c.pipePeakReduce)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(c.peakScratch, c.result)
	rec.Bind(c.pipePeakFinal)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(c.result)

	// ==== Phase 2 Try 1: Translation (angle θ₀) ====
	// Per paper §III end: "the image with the highest resolution is scaled and
	// rotated by amounts a and θ₀, respectively, and the amount of translational
	// movement is found out using phase correlation technique."

	// BilinearWarpGray A (slot=0: uses cos/sin from result[4,5]) → complexA.
	// Inverse rotation+scale of imgA to align with imgB's frame.
	warpParams0 := encodeWarpParams(srcWA, srcHA, srcStrideA, padSize, cropU32, 0)
	rec.Update(paramsBuf, warpParams0)
	rec.Bind(c.pipeWarpA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexA)

	// GrayscalePad B → complexB (fresh, since complexB was overwritten by Phase 1).
	rec.Update(paramsBuf, gpParamsB)
	rec.Bind(c.pipeGrayPadB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexB)

	// FFT2D, crosspower (phase-normalized), IFFT2D.
	recordFFT2D(rec, c.complexA, paramsBuf, c.pipeBitrevA, c.pipeButterflyA, c.w, c.h, false)
	recordFFT2D(rec, c.complexB, paramsBuf, c.pipeBitrevB, c.pipeButterflyB, c.w, c.h, false)
	transParams := encodeU32Params(n, 1) // normalize=1
	rec.Update(paramsBuf, transParams)
	rec.Bind(c.pipeCrosspowerTrans)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.crossPow)
	recordFFT2D(rec, c.crossPow, paramsBuf, c.pipeBitrevCP, c.pipeButterflyCP, c.w, c.h, true)

	// Peak find (translation mode, offset=8): result[8..10] is tx1, ty1, confidence1.
	peakT1Params := encodePeakFindParams(uint32(c.w), uint32(c.h), 1, 8, 0)
	rec.Update(paramsBuf, peakT1Params)
	rec.Bind(c.pipePeakFind)
	rec.Dispatch(peakWorkgroups, 1, 1)
	rec.Barrier(c.peakScratch)
	rec.Bind(c.pipePeakReduce)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(c.peakScratch, c.result)
	rec.Bind(c.pipePeakFinal)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(c.result)

	// ==== Phase 2 Try 2: Translation (angle θ₀ + 180°) ====
	// Paper p. 1268: "We then rotate the spectrum of Image 2 by (180° + θ₀)
	// and again compute the translation."

	// BilinearWarpGray A (slot=1: uses -cos/-sin from result[6,7]) → complexA.
	warpParams1 := encodeWarpParams(srcWA, srcHA, srcStrideA, padSize, cropU32, 1)
	rec.Update(paramsBuf, warpParams1)
	rec.Bind(c.pipeWarpA)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexA)

	// GrayscalePad B → complexB (fresh again).
	rec.Update(paramsBuf, gpParamsB)
	rec.Bind(c.pipeGrayPadB)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.complexB)

	// FFT2D, crosspower (phase-normalized), IFFT2D.
	recordFFT2D(rec, c.complexA, paramsBuf, c.pipeBitrevA, c.pipeButterflyA, c.w, c.h, false)
	recordFFT2D(rec, c.complexB, paramsBuf, c.pipeBitrevB, c.pipeButterflyB, c.w, c.h, false)
	rec.Update(paramsBuf, transParams)
	rec.Bind(c.pipeCrosspowerTrans)
	rec.Dispatch(groups, 1, 1)
	rec.Barrier(c.crossPow)
	recordFFT2D(rec, c.crossPow, paramsBuf, c.pipeBitrevCP, c.pipeButterflyCP, c.w, c.h, true)

	// Peak find (translation mode, offset=12): result[12..14] is tx2, ty2, confidence2.
	peakT2Params := encodePeakFindParams(uint32(c.w), uint32(c.h), 1, 12, 0)
	rec.Update(paramsBuf, peakT2Params)
	rec.Bind(c.pipePeakFind)
	rec.Dispatch(peakWorkgroups, 1, 1)
	rec.Barrier(c.peakScratch)
	rec.Bind(c.pipePeakReduce)
	rec.Dispatch(1, 1, 1)
	rec.Barrier(c.peakScratch, c.result)
	rec.Bind(c.pipePeakFinal)
	rec.Dispatch(1, 1, 1)
	resultBytes := make([]byte, c.result.Size())
	if err := rec.Download(c.result, resultBytes); err != nil {
		return nil, fmt.Errorf("registration: record result readback: %w", err)
	}

	// ==== Submit uploads, pipeline, and readback together ====
	if err := rec.SubmitAndWait(); err != nil {
		return nil, fmt.Errorf("registration: GPU pipeline: %w", err)
	}

	res := decodeFloat32Slice(resultBytes)

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
		return nil, fmt.Errorf("registration: GPU pipeline returned a non-finite transform")
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
		Angle:                 normalizeAngle(bestAngle),
		Scale:                 scale,
		Tx:                    bestTx,
		Ty:                    bestTy,
		RotationConfidence:    rotationConfidence,
		TranslationConfidence: translationConfidence,
	}, nil
}

// Close releases resources owned by the Correlator. Cleanup continues after an
// error, and repeated calls return nil. A borrowed Device is never closed.
func (c *Correlator) Close() error {
	if c == nil || c.closed {
		return nil
	}
	c.closed = true
	if c.backend == BackendCPU {
		return nil
	}

	var closeErrors []error
	for index := len(c.bindings) - 1; index >= 0; index-- {
		if err := c.bindings[index].Close(); err != nil {
			closeErrors = append(closeErrors, fmt.Errorf("registration: close binding set %d: %w", index, err))
		}
	}
	for index := len(c.kernelOrder) - 1; index >= 0; index-- {
		if err := c.kernelOrder[index].Close(); err != nil {
			closeErrors = append(closeErrors, fmt.Errorf("registration: close kernel %d: %w", index, err))
		}
	}
	buffers := []*vulki.Buffer{
		c.rgbaA, c.rgbaB, c.complexA, c.complexB, c.magA, c.magB,
		c.logPolA, c.logPolB, c.crossPow, c.peakScratch, c.result, c.params,
	}
	for index := len(buffers) - 1; index >= 0; index-- {
		if buffers[index] != nil {
			if err := buffers[index].Close(); err != nil {
				closeErrors = append(closeErrors, fmt.Errorf("registration: close buffer %d: %w", index, err))
			}
		}
	}
	if c.ownsDevice && c.device != nil {
		if err := c.device.Close(); err != nil {
			closeErrors = append(closeErrors, fmt.Errorf("registration: close owned device: %w", err))
		}
	}
	c.device = nil
	return errors.Join(closeErrors...)
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
