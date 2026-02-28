package imgproc

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
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
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

// logPolarToAngleScale converts a peak in log-polar correlation space to angle and scale.
// Following scikit-image convention: A*conj(B) cross-correlation with 360° angle range.
func logPolarToAngleScale(peakX, peakY float64, lpW, lpH int, maxRadius float64) (angle, scale float64) {
	angle = peakY / float64(lpH) * 360.0
	logRmax := math.Log(maxRadius)
	scale = math.Exp(peakX / float64(lpW) * logRmax)
	return angle, scale
}

// BilinearWarp rotates and scales an image around its center using bilinear interpolation.
// angleDeg is in degrees, scale is a multiplier.
func BilinearWarp(src *image.RGBA, angleDeg, scale float64) *image.RGBA {
	bounds := src.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()
	dst := image.NewRGBA(bounds)

	cx := float64(w) / 2.0
	cy := float64(h) / 2.0
	rad := angleDeg * math.Pi / 180.0
	cosA := math.Cos(rad)
	sinA := math.Sin(rad)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// Map destination back to source (inverse transform).
			dx := float64(x) - cx
			dy := float64(y) - cy
			// Inverse rotation and scale.
			sx := (dx*cosA+dy*sinA)/scale + cx
			sy := (-dx*sinA+dy*cosA)/scale + cy

			dst.SetRGBA(x, y, sampleBilinear(src, sx, sy))
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
	bounds := src.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()
	out := make([]uint32, padW*padH)

	// Center the image in the padded buffer.
	offX := (padW - srcW) / 2
	offY := (padH - srcH) / 2

	for y := 0; y < srcH && y+offY < padH; y++ {
		for x := 0; x < srcW && x+offX < padW; x++ {
			srcOff := (y+bounds.Min.Y)*src.Stride + (x+bounds.Min.X)*4
			r := uint32(src.Pix[srcOff])
			g := uint32(src.Pix[srcOff+1])
			b := uint32(src.Pix[srcOff+2])
			a := uint32(src.Pix[srcOff+3])
			out[(y+offY)*padW+(x+offX)] = r | (g << 8) | (b << 16) | (a << 24)
		}
	}
	return out
}

// realToComplex converts float32 grayscale to complex (real + 0i).
func realToComplex(data []float32) [][2]float32 {
	out := make([][2]float32, len(data))
	for i, v := range data {
		out[i] = [2]float32{v, 0}
	}
	return out
}
