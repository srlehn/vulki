package registration

import (
	"image"
	"image/color"
	"math"
)

// find2DPeak finds the peak in a 2D correlation surface with subpixel parabolic refinement.
// Returns fractional (x, y) coordinates of the peak.
func find2DPeak(data [][2]float32, w, h int) (float64, float64) {
	// Find max magnitude.
	maxVal := float64(0)
	maxX, maxY := 0, 0
	for y := range h {
		for x := range w {
			c := data[y*w+x]
			mag := math.Sqrt(float64(c[0]*c[0] + c[1]*c[1]))
			if mag > maxVal {
				maxVal = mag
				maxX = x
				maxY = y
			}
		}
	}

	peakX := float64(maxX)
	peakY := float64(maxY)

	mag := func(x, y int) float64 {
		// Wrap around for periodic signal.
		x = ((x % w) + w) % w
		y = ((y % h) + h) % h
		c := data[y*w+x]
		return math.Sqrt(float64(c[0]*c[0] + c[1]*c[1]))
	}

	// Subpixel refinement via parabolic interpolation.
	if maxX > 0 && maxX < w-1 {
		l := mag(maxX-1, maxY)
		c := mag(maxX, maxY)
		r := mag(maxX+1, maxY)
		denom := 2*c - l - r
		if denom > 1e-10 {
			peakX += (l - r) / (2 * denom)
		}
	}
	if maxY > 0 && maxY < h-1 {
		u := mag(maxX, maxY-1)
		c := mag(maxX, maxY)
		d := mag(maxX, maxY+1)
		denom := 2*c - u - d
		if denom > 1e-10 {
			peakY += (u - d) / (2 * denom)
		}
	}

	return peakX, peakY
}

// find2DPeakWithMag is like find2DPeak but also returns the peak magnitude.
func find2DPeakWithMag(data [][2]float32, w, h int) (float64, float64, float64) {
	maxVal := float64(0)
	maxX, maxY := 0, 0
	for y := range h {
		for x := range w {
			c := data[y*w+x]
			mag := math.Sqrt(float64(c[0]*c[0] + c[1]*c[1]))
			if mag > maxVal {
				maxVal = mag
				maxX = x
				maxY = y
			}
		}
	}

	peakX := float64(maxX)
	peakY := float64(maxY)

	mag := func(x, y int) float64 {
		x = ((x % w) + w) % w
		y = ((y % h) + h) % h
		c := data[y*w+x]
		return math.Sqrt(float64(c[0]*c[0] + c[1]*c[1]))
	}

	if maxX > 0 && maxX < w-1 {
		l := mag(maxX-1, maxY)
		c := mag(maxX, maxY)
		r := mag(maxX+1, maxY)
		denom := 2*c - l - r
		if denom > 1e-10 {
			peakX += (l - r) / (2 * denom)
		}
	}
	if maxY > 0 && maxY < h-1 {
		u := mag(maxX, maxY-1)
		c := mag(maxX, maxY)
		d := mag(maxX, maxY+1)
		denom := 2*c - u - d
		if denom > 1e-10 {
			peakY += (u - d) / (2 * denom)
		}
	}

	return peakX, peakY, maxVal
}

// logPolarToAngleScale converts a peak in log-polar correlation space to angle and scale.
// Following scikit-image convention: A*conj(B) cross-correlation with 180° angle range.
func logPolarToAngleScale(peakX, peakY float64, lpW, lpH int, maxRadius float64) (angle, scale float64) {
	angle = peakY / float64(lpH) * 180.0
	klog := float64(lpW) / math.Log(maxRadius)
	scale = math.Exp(peakX / klog)
	return angle, scale
}

func normalizeAngle(angle float64) float64 {
	angle = math.Mod(angle+180, 360)
	if angle < 0 {
		angle += 360
	}
	return angle - 180
}

func bilinearWarp(src *image.RGBA, angleDeg, scale float64) *image.RGBA {
	bounds := src.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	dst := image.NewRGBA(bounds)

	cx := float64(w) / 2.0
	cy := float64(h) / 2.0
	rad := angleDeg * math.Pi / 180.0
	cosA := math.Cos(rad)
	sinA := math.Sin(rad)

	for y := range h {
		for x := range w {
			// Map destination back to source (inverse transform).
			dx := float64(x) - cx
			dy := float64(y) - cy
			// Inverse rotation and scale.
			sx := (dx*cosA+dy*sinA)/scale + cx
			sy := (-dx*sinA+dy*cosA)/scale + cy

			dst.SetRGBA(bounds.Min.X+x, bounds.Min.Y+y, sampleBilinear(src, sx, sy))
		}
	}
	return dst
}

func sampleBilinear(img *image.RGBA, x, y float64) color.RGBA {
	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()

	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	y1 := y0 + 1
	fx := x - float64(x0)
	fy := y - float64(y0)

	clampX := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= w {
			return w - 1
		}
		return v
	}
	clampY := func(v int) int {
		if v < 0 {
			return 0
		}
		if v >= h {
			return h - 1
		}
		return v
	}

	cx0 := clampX(x0)
	cx1 := clampX(x1)
	cy0 := clampY(y0)
	cy1 := clampY(y1)

	c00 := img.RGBAAt(cx0+bounds.Min.X, cy0+bounds.Min.Y)
	c10 := img.RGBAAt(cx1+bounds.Min.X, cy0+bounds.Min.Y)
	c01 := img.RGBAAt(cx0+bounds.Min.X, cy1+bounds.Min.Y)
	c11 := img.RGBAAt(cx1+bounds.Min.X, cy1+bounds.Min.Y)

	lerp := func(a, b uint8, t float64) uint8 {
		return uint8(float64(a)*(1-t) + float64(b)*t + 0.5)
	}

	top := color.RGBA{
		R: lerp(c00.R, c10.R, fx),
		G: lerp(c00.G, c10.G, fx),
		B: lerp(c00.B, c10.B, fx),
		A: lerp(c00.A, c10.A, fx),
	}
	bot := color.RGBA{
		R: lerp(c01.R, c11.R, fx),
		G: lerp(c01.G, c11.G, fx),
		B: lerp(c01.B, c11.B, fx),
		A: lerp(c01.A, c11.A, fx),
	}

	return color.RGBA{
		R: lerp(top.R, bot.R, fy),
		G: lerp(top.G, bot.G, fy),
		B: lerp(top.B, bot.B, fy),
		A: lerp(top.A, bot.A, fy),
	}
}

// nextPow2 returns the smallest power of 2 >= n.
func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// log2i returns floor(log2(n)) for positive n.
func log2i(n int) int {
	r := 0
	v := n
	for v > 1 {
		v >>= 1
		r++
	}
	return r
}

// padImageToRGBA pads an RGBA image to padW x padH with black pixels,
// centering the content so the Hann window is strongest over the image.
func padImageToRGBA(src *image.RGBA, padW, padH int) []uint32 {
	if src == nil || padW <= 0 || padH <= 0 {
		return nil
	}
	bounds := src.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()
	out := make([]uint32, padW*padH)

	// Center the image, cropping equally from opposite sides when needed.
	offX := (padW - srcW) / 2
	offY := (padH - srcH) / 2
	srcStartX, srcStartY := 0, 0
	dstStartX, dstStartY := offX, offY
	if dstStartX < 0 {
		srcStartX = -dstStartX
		dstStartX = 0
	}
	if dstStartY < 0 {
		srcStartY = -dstStartY
		dstStartY = 0
	}
	copyW := min(srcW-srcStartX, padW-dstStartX)
	copyH := min(srcH-srcStartY, padH-dstStartY)

	for y := range copyH {
		for x := range copyW {
			pixel := src.RGBAAt(bounds.Min.X+srcStartX+x, bounds.Min.Y+srcStartY+y)
			r := uint32(pixel.R)
			g := uint32(pixel.G)
			b := uint32(pixel.B)
			a := uint32(pixel.A)
			out[(y+dstStartY)*padW+(x+dstStartX)] = r | (g << 8) | (b << 16) | (a << 24)
		}
	}
	return out
}

func nextPow2Checked(n int) (int, bool) {
	if n <= 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	p := 1
	for p < n {
		if p > maxInt/2 {
			return 0, false
		}
		p <<= 1
	}
	return p, true
}

// realToComplex converts float32 grayscale to complex (real + 0i).
func realToComplex(data []float32) [][2]float32 {
	out := make([][2]float32, len(data))
	for i, v := range data {
		out[i] = [2]float32{v, 0}
	}
	return out
}

// grayPad converts an RGBA image to grayscale float32 and zero-pads to padSize.
// No DoG, Hann, or cropping is used for Phase 2 translation per Reddy & Chatterji.
func grayPad(img *image.RGBA, padSize int) []float32 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	out := make([]float32, padSize*padSize)
	offX := (padSize - w) / 2
	offY := (padSize - h) / 2
	if offX < 0 {
		offX = 0
	}
	if offY < 0 {
		offY = 0
	}

	for y := 0; y < h && y+offY < padSize; y++ {
		for x := 0; x < w && x+offX < padSize; x++ {
			sx := bounds.Min.X + x
			sy := bounds.Min.Y + y
			if sx >= bounds.Max.X || sy >= bounds.Max.Y {
				continue
			}
			i := (sy-bounds.Min.Y)*img.Stride + (sx-bounds.Min.X)*4
			r := float32(img.Pix[i]) / 255.0
			g := float32(img.Pix[i+1]) / 255.0
			b := float32(img.Pix[i+2]) / 255.0
			out[(y+offY)*padSize+(x+offX)] = 0.2989*r + 0.5870*g + 0.1140*b
		}
	}
	return out
}

// cropDogHannPad takes an RGBA image, crops a centered square of the given size,
// converts to grayscale, applies DoG bandpass + Hann window, then zero-pads to padSize.
func cropDogHannPad(img *image.RGBA, cropSize, padSize int) []float32 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// Center crop to square.
	offX := (w - cropSize) / 2
	offY := (h - cropSize) / 2
	if offX < 0 {
		offX = 0
	}
	if offY < 0 {
		offY = 0
	}

	// Convert crop to grayscale.
	gray := make([]float32, cropSize*cropSize)
	for y := range cropSize {
		for x := range cropSize {
			sx := bounds.Min.X + offX + x
			sy := bounds.Min.Y + offY + y
			if sx >= bounds.Max.X || sy >= bounds.Max.Y {
				continue
			}
			i := (sy-bounds.Min.Y)*img.Stride + (sx-bounds.Min.X)*4
			r := float32(img.Pix[i]) / 255.0
			g := float32(img.Pix[i+1]) / 255.0
			b := float32(img.Pix[i+2]) / 255.0
			gray[y*cropSize+x] = 0.2989*r + 0.5870*g + 0.1140*b
		}
	}

	// Apply DoG bandpass + Hann window.
	differenceOfGaussians(gray, cropSize, cropSize, 5, 20)
	applyHann2D(gray, cropSize, cropSize)

	// Embed centered in padded buffer.
	out := make([]float32, padSize*padSize)
	padOff := (padSize - cropSize) / 2
	for y := range cropSize {
		for x := range cropSize {
			out[(y+padOff)*padSize+(x+padOff)] = gray[y*cropSize+x]
		}
	}
	return out
}

// applyHann2D applies a separable 2D Hann window in-place.
func applyHann2D(data []float32, w, h int) {
	for y := range h {
		wy := float32(0.5 * (1 - math.Cos(2*math.Pi*float64(y)/float64(h))))
		for x := range w {
			wx := float32(0.5 * (1 - math.Cos(2*math.Pi*float64(x)/float64(w))))
			data[y*w+x] *= wx * wy
		}
	}
}

// differenceOfGaussians applies a spatial bandpass filter (DoG) in-place.
// Equivalent to scikit-image's difference_of_gaussians(image, sigmaLow, sigmaHigh).
func differenceOfGaussians(data []float32, w, h int, sigmaLow, sigmaHigh float64) {
	blurLow := gaussianBlur(data, w, h, sigmaLow)
	blurHigh := gaussianBlur(data, w, h, sigmaHigh)
	for i := range data {
		data[i] = blurLow[i] - blurHigh[i]
	}
}

// gaussianBlur performs a separable 2D Gaussian blur.
func gaussianBlur(data []float32, w, h int, sigma float64) []float32 {
	radius := int(math.Ceil(sigma * 4))
	kernel := make([]float64, 2*radius+1)
	sum := 0.0
	for i := range kernel {
		x := float64(i - radius)
		kernel[i] = math.Exp(-x * x / (2 * sigma * sigma))
		sum += kernel[i]
	}
	for i := range kernel {
		kernel[i] /= sum
	}

	// Horizontal pass.
	tmp := make([]float32, w*h)
	for y := range h {
		for x := range w {
			var v float64
			for k := -radius; k <= radius; k++ {
				sx := x + k
				if sx < 0 {
					sx = 0
				} else if sx >= w {
					sx = w - 1
				}
				v += float64(data[y*w+sx]) * kernel[k+radius]
			}
			tmp[y*w+x] = float32(v)
		}
	}

	// Vertical pass.
	out := make([]float32, w*h)
	for y := range h {
		for x := range w {
			var v float64
			for k := -radius; k <= radius; k++ {
				sy := y + k
				if sy < 0 {
					sy = 0
				} else if sy >= h {
					sy = h - 1
				}
				v += float64(tmp[sy*w+x]) * kernel[k+radius]
			}
			out[y*w+x] = float32(v)
		}
	}
	return out
}
