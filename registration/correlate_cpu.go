package registration

import (
	"fmt"
	"image"
	"math"
	"math/cmplx"
)

func newCPUCorrelator(maxW, maxH int) (*Correlator, error) {
	if maxW < 2 || maxH < 2 {
		return nil, fmt.Errorf("registration: maximum dimensions must both be at least 2 pixels")
	}
	maxInt := int(^uint(0) >> 1)
	if maxW > maxInt/maxH {
		return nil, fmt.Errorf("registration: maximum image area overflows int")
	}
	padSize, ok := nextPow2Checked(min(maxW, maxH))
	if !ok || padSize > maxInt/padSize {
		return nil, fmt.Errorf("registration: padded image area overflows int")
	}
	if padSize*padSize > maxExactFloat32Index {
		return nil, fmt.Errorf("registration: padded image area exceeds CPU backend limit")
	}

	return &Correlator{
		backend: BackendCPU,
		maxW:    maxW,
		maxH:    maxH,
		w:       padSize,
		h:       padSize,
		lpW:     padSize,
		lpH:     padSize,
	}, nil
}

func (c *Correlator) phaseCorrelateCPU(imgA, imgB *image.RGBA) (*Result, error) {
	if imgA == nil || imgB == nil {
		return nil, fmt.Errorf("registration: input images must not be nil")
	}
	wA, hA := imgA.Bounds().Dx(), imgA.Bounds().Dy()
	wB, hB := imgB.Bounds().Dx(), imgB.Bounds().Dy()
	if wA != wB || hA != hB {
		return nil, fmt.Errorf("registration: input images must have equal dimensions, got %dx%d and %dx%d", wA, hA, wB, hB)
	}
	if wA < 2 || hA < 2 {
		return nil, fmt.Errorf("registration: input dimensions must both be at least 2 pixels")
	}
	if wA > c.maxW || hA > c.maxH {
		return nil, fmt.Errorf("registration: input dimensions %dx%d exceed correlator maximum %dx%d", wA, hA, c.maxW, c.maxH)
	}

	pixelsA, err := packRGBA(imgA)
	if err != nil {
		return nil, fmt.Errorf("registration: image A: %w", err)
	}
	pixelsB, err := packRGBA(imgB)
	if err != nil {
		return nil, fmt.Errorf("registration: image B: %w", err)
	}

	cropSize := min(wA, hA)
	grayA := grayCropPadCPU(pixelsA, wA, hA, cropSize, c.w)
	grayB := grayCropPadCPU(pixelsB, wB, hB, cropSize, c.w)

	magA := magnitudeSpectrumCPU(grayA, c.w, c.h)
	magB := magnitudeSpectrumCPU(grayB, c.w, c.h)
	maxRadius := float64(cropSize) * 1.1 / 2.0
	lpA := logPolarCPU(magA, c.w, c.h, c.lpW, c.lpH, maxRadius)
	lpB := logPolarCPU(magB, c.w, c.h, c.lpW, c.lpH, maxRadius)
	peakX, peakY, rotationConfidence := phaseCorrelationCPU(lpA, lpB, c.lpW, c.lpH)
	angleDeg, scale := logPolarToAngleScale(peakX, peakY, c.lpW, c.lpH, maxRadius)

	type translationResult struct {
		angle      float64
		tx, ty     float64
		confidence float64
	}
	tryTranslation := func(angle float64) (translationResult, error) {
		warped := bilinearWarp(imgA, -angle, scale)
		warpedPixels, err := packRGBA(warped)
		if err != nil {
			return translationResult{}, err
		}
		warpedGray := grayCropPadCPU(warpedPixels, wA, hA, cropSize, c.w)
		tx, ty, confidence := phaseCorrelationCPU(
			complexFromRealCPU(warpedGray),
			complexFromRealCPU(grayB),
			c.w,
			c.h,
		)
		return translationResult{
			angle:      angle,
			tx:         -tx,
			ty:         -ty,
			confidence: confidence,
		}, nil
	}

	first, err := tryTranslation(angleDeg)
	if err != nil {
		return nil, err
	}
	second, err := tryTranslation(angleDeg + 180)
	if err != nil {
		return nil, err
	}
	best := first
	if second.confidence > first.confidence {
		best = second
	}

	if !finiteCPU(angleDeg) || !finiteCPU(scale) || scale <= 0 ||
		!finiteCPU(rotationConfidence) || !finiteCPU(best.tx) ||
		!finiteCPU(best.ty) || !finiteCPU(best.confidence) {
		return nil, fmt.Errorf("registration: CPU pipeline returned a non-finite transform")
	}
	if rotationConfidence <= minimumMatchConfidence || best.confidence <= minimumMatchConfidence {
		return nil, fmt.Errorf("%w: rotation %.5f, translation %.5f, minimum %.2f",
			ErrLowConfidence, rotationConfidence, best.confidence, minimumMatchConfidence)
	}

	return &Result{
		Angle:                 normalizeAngle(best.angle),
		Scale:                 scale,
		Tx:                    best.tx,
		Ty:                    best.ty,
		RotationConfidence:    rotationConfidence,
		TranslationConfidence: best.confidence,
	}, nil
}

func finiteCPU(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func grayCropPadCPU(pixels []uint32, srcW, srcH, cropSize, padSize int) []float64 {
	output := make([]float64, padSize*padSize)
	srcOffsetX := (srcW - cropSize) / 2
	srcOffsetY := (srcH - cropSize) / 2
	padOffset := (padSize - cropSize) / 2
	for y := range cropSize {
		for x := range cropSize {
			rgba := pixels[(y+srcOffsetY)*srcW+x+srcOffsetX]
			r := float64(rgba&0xff) / 255.0
			g := float64((rgba>>8)&0xff) / 255.0
			b := float64((rgba>>16)&0xff) / 255.0
			output[(y+padOffset)*padSize+x+padOffset] = 0.299*r + 0.587*g + 0.114*b
		}
	}
	return output
}

func magnitudeSpectrumCPU(gray []float64, width, height int) []float64 {
	values := complexFromRealCPU(gray)
	fft2DCPU(values, width, height, false)
	magnitude := make([]float64, len(values))
	for i, value := range values {
		magnitude[i] = math.Log1p(cmplx.Abs(value))
	}
	fftShiftCPU(magnitude, width, height)
	for y := range height {
		eta := float64(y)/float64(height) - 0.5
		for x := range width {
			xi := float64(x)/float64(width) - 0.5
			spectralX := math.Cos(math.Pi*xi) * math.Cos(math.Pi*eta)
			highpass := (1 - spectralX) * (2 - spectralX)
			magnitude[y*width+x] *= highpass
		}
	}
	return magnitude
}

func fftShiftCPU(values []float64, width, height int) {
	halfWidth := width / 2
	halfHeight := height / 2
	for y := range halfHeight {
		for x := range width {
			oppositeX := (x + halfWidth) % width
			oppositeY := (y + halfHeight) % height
			first := y*width + x
			second := oppositeY*width + oppositeX
			values[first], values[second] = values[second], values[first]
		}
	}
}

func logPolarCPU(
	source []float64,
	srcW, srcH, dstW, dstH int,
	maxRadius float64,
) []complex128 {
	output := make([]complex128, dstW*dstH)
	logRadius := math.Log(maxRadius)
	centerX := float64(srcW) / 2.0
	centerY := float64(srcH) / 2.0
	for y := range dstH {
		theta := float64(y) / float64(dstH) * math.Pi
		for x := range dstW {
			radius := math.Exp(float64(x) / float64(dstW) * logRadius)
			sampleX := centerX + radius*math.Cos(theta)
			sampleY := centerY + radius*math.Sin(theta)
			output[y*dstW+x] = complex(sampleFloatCPU(source, srcW, srcH, sampleX, sampleY), 0)
		}
	}
	return output
}

func sampleFloatCPU(source []float64, width, height int, x, y float64) float64 {
	x = math.Max(0, math.Min(x, float64(width-1)))
	y = math.Max(0, math.Min(y, float64(height-1)))
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := min(x0+1, width-1)
	y1 := min(y0+1, height-1)
	fx := x - float64(x0)
	fy := y - float64(y0)
	top := source[y0*width+x0]*(1-fx) + source[y0*width+x1]*fx
	bottom := source[y1*width+x0]*(1-fx) + source[y1*width+x1]*fx
	return top*(1-fy) + bottom*fy
}

func complexFromRealCPU(values []float64) []complex128 {
	output := make([]complex128, len(values))
	for i, value := range values {
		output[i] = complex(value, 0)
	}
	return output
}

func phaseCorrelationCPU(a, b []complex128, width, height int) (float64, float64, float64) {
	fft2DCPU(a, width, height, false)
	fft2DCPU(b, width, height, false)
	crossPower := make([]complex128, len(a))
	for i := range crossPower {
		product := a[i] * cmplx.Conj(b[i])
		magnitude := cmplx.Abs(product) + 1e-10
		crossPower[i] = product / complex(magnitude, 0)
	}
	fft2DCPU(crossPower, width, height, true)
	return findPeakCPU(crossPower, width, height)
}

func findPeakCPU(values []complex128, width, height int) (float64, float64, float64) {
	maxMagnitude := -1.0
	maxX, maxY := 0, 0
	for y := range height {
		for x := range width {
			magnitude := cmplx.Abs(values[y*width+x])
			if magnitude > maxMagnitude {
				maxMagnitude = magnitude
				maxX, maxY = x, y
			}
		}
	}
	magnitudeAt := func(x, y int) float64 {
		x = ((x % width) + width) % width
		y = ((y % height) + height) % height
		return cmplx.Abs(values[y*width+x])
	}
	peakX := float64(maxX)
	peakY := float64(maxY)
	left := magnitudeAt(maxX-1, maxY)
	right := magnitudeAt(maxX+1, maxY)
	denomX := 2*maxMagnitude - left - right
	if denomX > 1e-10 {
		peakX += (left - right) / (2 * denomX)
	}
	upper := magnitudeAt(maxX, maxY-1)
	lower := magnitudeAt(maxX, maxY+1)
	denomY := 2*maxMagnitude - upper - lower
	if denomY > 1e-10 {
		peakY += (upper - lower) / (2 * denomY)
	}
	if peakX > float64(width)/2 {
		peakX -= float64(width)
	}
	if peakY > float64(height)/2 {
		peakY -= float64(height)
	}
	return peakX, peakY, maxMagnitude
}

func fft2DCPU(values []complex128, width, height int, inverse bool) {
	for y := range height {
		fft1DCPU(values[y*width:(y+1)*width], inverse)
	}
	column := make([]complex128, height)
	for x := range width {
		for y := range height {
			column[y] = values[y*width+x]
		}
		fft1DCPU(column, inverse)
		for y := range height {
			values[y*width+x] = column[y]
		}
	}
}

func fft1DCPU(values []complex128, inverse bool) {
	n := len(values)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			values[i], values[j] = values[j], values[i]
		}
	}
	for length := 2; length <= n; length <<= 1 {
		angle := -2 * math.Pi / float64(length)
		if inverse {
			angle = -angle
		}
		root := complex(math.Cos(angle), math.Sin(angle))
		for start := 0; start < n; start += length {
			factor := complex(1.0, 0)
			half := length / 2
			for offset := range half {
				even := values[start+offset]
				odd := values[start+offset+half] * factor
				values[start+offset] = even + odd
				values[start+offset+half] = even - odd
				factor *= root
			}
		}
	}
	if inverse {
		scale := complex(float64(n), 0)
		for i := range values {
			values[i] /= scale
		}
	}
}
