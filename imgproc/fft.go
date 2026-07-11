package imgproc

import (
	"encoding/binary"
	"math"

	"github.com/srlehn/vulki/compute"
	"github.com/srlehn/vulki/vk"
)

// fftParams holds the parameters written to the GPU params buffer for bit-reversal.
type fftBitrevParams struct {
	N        uint32
	NumLines uint32
	Log2N    uint32
	Axis     uint32
	Stride   uint32
}

// fftButterflyParams holds the parameters for one butterfly stage.
type fftButterflyParams struct {
	Stage    uint32
	N        uint32
	NumLines uint32
	Axis     uint32
	Stride   uint32
	Inverse  uint32
}

// recordFFT2D records a complete 2D FFT (or IFFT) into the command recorder.
// The data buffer contains W*H complex elements in row-major order.
// bitrevPipe and butterflyPipe must have the data buffer at binding 0 and params at binding 1.
func recordFFT2D(
	rec *compute.CommandRecorder,
	dataBuf vk.Buffer,
	paramsBuf vk.Buffer,
	bitrevPipe *compute.Pipeline,
	butterflyPipe *compute.Pipeline,
	w, h int,
	inverse bool,
) {
	wgSize := uint32(64)
	invFlag := uint32(0)
	if inverse {
		invFlag = 1
	}

	// Row-wise FFT: N=W, numLines=H, axis=0, stride=W
	recordFFT1D(rec, dataBuf, paramsBuf, bitrevPipe, butterflyPipe, w, h, 0, w, invFlag, wgSize)

	// Column-wise FFT: N=H, numLines=W, axis=1, stride=W
	recordFFT1D(rec, dataBuf, paramsBuf, bitrevPipe, butterflyPipe, h, w, 1, w, invFlag, wgSize)
}

func recordFFT1D(
	rec *compute.CommandRecorder,
	dataBuf vk.Buffer,
	paramsBuf vk.Buffer,
	bitrevPipe *compute.Pipeline,
	butterflyPipe *compute.Pipeline,
	n, numLines int,
	axis uint32,
	stride int,
	inverse uint32,
	wgSize uint32,
) {
	log2n := log2i(n)

	// Bit-reversal permutation.
	{
		p := fftBitrevParams{
			N:        uint32(n),
			NumLines: uint32(numLines),
			Log2N:    uint32(log2n),
			Axis:     axis,
			Stride:   uint32(stride),
		}
		data := encodeBitrevParams(p)
		rec.UpdateBuffer(paramsBuf, 0, data)
		rec.BarrierTransferToCompute(paramsBuf)
		rec.Bind(bitrevPipe)
		total := uint32(n * numLines)
		rec.Dispatch((total+wgSize-1)/wgSize, 1, 1)
		rec.Barrier(dataBuf)
	}

	// Butterfly stages.
	for s := 0; s < log2n; s++ {
		p := fftButterflyParams{
			Stage:    uint32(s),
			N:        uint32(n),
			NumLines: uint32(numLines),
			Axis:     axis,
			Stride:   uint32(stride),
			Inverse:  inverse,
		}
		data := encodeButterflyParams(p)
		rec.UpdateBuffer(paramsBuf, 0, data)
		rec.BarrierTransferToCompute(paramsBuf)
		rec.Bind(butterflyPipe)
		pairsPerLine := uint32(n) / 2
		total := pairsPerLine * uint32(numLines)
		rec.Dispatch((total+wgSize-1)/wgSize, 1, 1)
		rec.Barrier(dataBuf)
	}
}


func encodeBitrevParams(p fftBitrevParams) []byte {
	buf := make([]byte, 20)
	binary.LittleEndian.PutUint32(buf[0:4], p.N)
	binary.LittleEndian.PutUint32(buf[4:8], p.NumLines)
	binary.LittleEndian.PutUint32(buf[8:12], p.Log2N)
	binary.LittleEndian.PutUint32(buf[12:16], p.Axis)
	binary.LittleEndian.PutUint32(buf[16:20], p.Stride)
	return buf
}

func encodeButterflyParams(p fftButterflyParams) []byte {
	buf := make([]byte, 24)
	binary.LittleEndian.PutUint32(buf[0:4], p.Stage)
	binary.LittleEndian.PutUint32(buf[4:8], p.N)
	binary.LittleEndian.PutUint32(buf[8:12], p.NumLines)
	binary.LittleEndian.PutUint32(buf[12:16], p.Axis)
	binary.LittleEndian.PutUint32(buf[16:20], p.Stride)
	binary.LittleEndian.PutUint32(buf[20:24], p.Inverse)
	return buf
}

func encodeU32Params(vals ...uint32) []byte {
	buf := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:(i+1)*4], v)
	}
	return buf
}

func encodeLogPolarParams(srcW, srcH, dstW, dstH uint32, logRmax float32) []byte {
	buf := make([]byte, 20)
	binary.LittleEndian.PutUint32(buf[0:4], srcW)
	binary.LittleEndian.PutUint32(buf[4:8], srcH)
	binary.LittleEndian.PutUint32(buf[8:12], dstW)
	binary.LittleEndian.PutUint32(buf[12:16], dstH)
	binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(logRmax))
	return buf
}
