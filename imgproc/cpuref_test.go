package imgproc

import (
	"math"
	"math/cmplx"
	"testing"
)

// CPU-only FFT for reference testing.
func cpuFFT(data []complex128, n int) {
	if n <= 1 {
		return
	}
	even := make([]complex128, n/2)
	odd := make([]complex128, n/2)
	for i := 0; i < n/2; i++ {
		even[i] = data[2*i]
		odd[i] = data[2*i+1]
	}
	cpuFFT(even, n/2)
	cpuFFT(odd, n/2)
	for k := 0; k < n/2; k++ {
		t := cmplx.Rect(1, -2*math.Pi*float64(k)/float64(n)) * odd[k]
		data[k] = even[k] + t
		data[k+n/2] = even[k] - t
	}
}

func cpuFFT2D(data []complex128, w, h int) {
	// Row FFT
	for y := 0; y < h; y++ {
		row := make([]complex128, w)
		copy(row, data[y*w:(y+1)*w])
		cpuFFT(row, w)
		copy(data[y*w:], row)
	}
	// Column FFT
	for x := 0; x < w; x++ {
		col := make([]complex128, h)
		for y := 0; y < h; y++ {
			col[y] = data[y*w+x]
		}
		cpuFFT(col, h)
		for y := 0; y < h; y++ {
			data[y*w+x] = col[y]
		}
	}
}

func cpuIFFT2D(data []complex128, w, h int) {
	// Conjugate
	for i := range data {
		data[i] = cmplx.Conj(data[i])
	}
	cpuFFT2D(data, w, h)
	// Conjugate and scale
	n := float64(w * h)
	for i := range data {
		data[i] = cmplx.Conj(data[i]) / complex(n, 0)
	}
}

func TestCPURefPipeline(t *testing.T) {
	// Load snake images using file ops.
	imgA := loadTestImage(t, "../testdata/snake.png")
	imgB := loadTestImage(t, "../testdata/snake_rot_12deg.png")

	boundsA := imgA.Bounds()
	boundsB := imgB.Bounds()
	cropSize := min(boundsA.Dx(), boundsA.Dy(), boundsB.Dx(), boundsB.Dy())
	padSize := nextPow2(cropSize)
	t.Logf("cropSize=%d padSize=%d", cropSize, padSize)

	// Crop, DoG, Hann, pad (reusing our function)
	grayA := cropDogHannPad(imgA, cropSize, padSize)
	grayB := cropDogHannPad(imgB, cropSize, padSize)

	n := padSize * padSize

	// Convert to complex128
	cA := make([]complex128, n)
	cB := make([]complex128, n)
	for i := range grayA {
		cA[i] = complex(float64(grayA[i]), 0)
		cB[i] = complex(float64(grayB[i]), 0)
	}

	// FFT2D
	cpuFFT2D(cA, padSize, padSize)
	cpuFFT2D(cB, padSize, padSize)

	// Magnitude
	magA := make([]float64, n)
	magB := make([]float64, n)
	for i := range cA {
		magA[i] = cmplx.Abs(cA[i])
		magB[i] = cmplx.Abs(cB[i])
	}

	// fftshift
	half := padSize / 2
	fftshift := func(data []float64, w, h int) {
		hw, hh := w/2, h/2
		for y := 0; y < hh; y++ {
			for x := 0; x < w; x++ {
				sx := (x + hw) % w
				sy := (y + hh) % h
				ia := y*w + x
				ib := sy*w + sx
				data[ia], data[ib] = data[ib], data[ia]
			}
		}
	}
	fftshift(magA, padSize, padSize)
	fftshift(magB, padSize, padSize)

	// Verify DC is near center
	maxVal := 0.0
	maxIdx := 0
	for i, v := range magA {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	dcX := maxIdx % padSize
	dcY := maxIdx / padSize
	t.Logf("CPU mag DC at (%d, %d), expected near (%d, %d)", dcX, dcY, half, half)

	// Log-polar from center (matching our shader)
	maxRadius := float64(cropSize) / 8.0
	logRmax := math.Log(maxRadius)
	lpW, lpH := padSize, padSize

	// Sample bilinear from magnitude
	sampleBilinear := func(mag []float64, x, y float64) float64 {
		x = math.Max(0, math.Min(x, float64(padSize-1)))
		y = math.Max(0, math.Min(y, float64(padSize-1)))
		x0 := int(math.Floor(x))
		y0 := int(math.Floor(y))
		x1 := min(x0+1, padSize-1)
		y1 := min(y0+1, padSize-1)
		fx := x - float64(x0)
		fy := y - float64(y0)
		v00 := mag[y0*padSize+x0]
		v10 := mag[y0*padSize+x1]
		v01 := mag[y1*padSize+x0]
		v11 := mag[y1*padSize+x1]
		top := v00*(1-fx) + v10*fx
		bot := v01*(1-fx) + v11*fx
		return top*(1-fy) + bot*fy
	}

	lpA := make([]complex128, lpW*lpH)
	lpB := make([]complex128, lpW*lpH)
	cx := float64(padSize) / 2.0
	cy := float64(padSize) / 2.0

	for yi := 0; yi < lpH; yi++ {
		for xi := 0; xi < lpW; xi++ {
			logR := float64(xi) / float64(lpW) * logRmax
			theta := float64(yi) / float64(lpH) * math.Pi
			r := math.Exp(logR)
			sx := cx + r*math.Cos(theta)
			sy := cy + r*math.Sin(theta)
			lpA[yi*lpW+xi] = complex(sampleBilinear(magA, sx, sy), 0)
			lpB[yi*lpW+xi] = complex(sampleBilinear(magB, sx, sy), 0)
		}
	}

	// FFT log-polar
	cpuFFT2D(lpA, lpW, lpH)
	cpuFFT2D(lpB, lpW, lpH)

	// Cross-power (raw, A*conj(B))
	cross := make([]complex128, lpW*lpH)
	for i := range cross {
		cross[i] = lpA[i] * cmplx.Conj(lpB[i])
	}

	// IFFT
	cpuIFFT2D(cross, lpW, lpH)

	// Find peak
	maxMag := 0.0
	peakX, peakY := 0, 0
	for y := 0; y < lpH; y++ {
		for x := 0; x < lpW; x++ {
			m := cmplx.Abs(cross[y*lpW+x])
			if m > maxMag {
				maxMag = m
				peakX = x
				peakY = y
				maxMag = m
			}
		}
	}

	fpX := float64(peakX)
	fpY := float64(peakY)
	if fpX > float64(lpW)/2 {
		fpX -= float64(lpW)
	}
	if fpY > float64(lpH)/2 {
		fpY -= float64(lpH)
	}

	angle := fpY / float64(lpH) * 180.0
	klog := float64(lpW) / math.Log(maxRadius)
	scale := math.Exp(fpX / klog)

	t.Logf("Raw peak: (%d, %d), after wrap: (%.1f, %.1f)", peakX, peakY, fpX, fpY)
	t.Logf("Detected: angle=%.2f° scale=%.4f", angle, scale)
	t.Logf("Expected: angle≈12° scale≈1.0")

	t.Logf("CPU ref: angle=%.2f° scale=%.4f (expected ~12°, ~1.0)", angle, scale)
}
