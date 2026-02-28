//go:build ignore

package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
)

func main() {
	src, err := loadPNG("testdata/snake.png")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// 1. Shifted only: translate by (15, 25).
	shifted := translate(src, 15, 25)
	savePNG("testdata/snake_shifted.png", shifted)
	fmt.Println("wrote snake_shifted.png (tx=15, ty=25)")

	// 2. Scaled only: scale by 1.15.
	scaled := rotateScale(src, 0, 1.15)
	savePNG("testdata/snake_scaled.png", scaled)
	fmt.Println("wrote snake_scaled.png (scale=1.15)")

	// 3. Rotated only: rotate by 12 degrees.
	rotated := rotateScale(src, 12, 1.0)
	savePNG("testdata/snake_rotated.png", rotated)
	fmt.Println("wrote snake_rotated.png (angle=12°)")

	// 4. Rotated + scaled.
	rotScaled := rotateScale(src, 12, 1.15)
	savePNG("testdata/snake_rotscaled.png", rotScaled)
	fmt.Println("wrote snake_rotscaled.png (angle=12°, scale=1.15)")

	// 5. Shifted + rotated.
	shiftedRot := translate(rotated, 15, 25)
	savePNG("testdata/snake_shiftrot.png", shiftedRot)
	fmt.Println("wrote snake_shiftrot.png (angle=12°, tx=15, ty=25)")

	// 6. Shifted + scaled.
	shiftedScaled := translate(scaled, 15, 25)
	savePNG("testdata/snake_shiftscaled.png", shiftedScaled)
	fmt.Println("wrote snake_shiftscaled.png (scale=1.15, tx=15, ty=25)")
}

func loadPNG(path string) (*image.RGBA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	rgba := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}
	return rgba, nil
}

func savePNG(path string, img image.Image) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	png.Encode(f, img)
}

func translate(src *image.RGBA, dx, dy int) *image.RGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.NewRGBA(b)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			sx := x - dx
			sy := y - dy
			if sx >= 0 && sx < w && sy >= 0 && sy < h {
				dst.Set(x+b.Min.X, y+b.Min.Y, src.At(sx+b.Min.X, sy+b.Min.Y))
			}
		}
	}
	return dst
}

func rotateScale(src *image.RGBA, angleDeg, scale float64) *image.RGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.NewRGBA(b)
	cx := float64(w) / 2.0
	cy := float64(h) / 2.0
	rad := angleDeg * math.Pi / 180.0
	cosA := math.Cos(rad)
	sinA := math.Sin(rad)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// Inverse transform: where in src does this dst pixel come from?
			dx := float64(x) - cx
			dy := float64(y) - cy
			sx := (dx*cosA+dy*sinA)/scale + cx
			sy := (-dx*sinA+dy*cosA)/scale + cy
			dst.Set(x+b.Min.X, y+b.Min.Y, sampleBilinear(src, sx, sy))
		}
	}
	return dst
}

func sampleBilinear(img *image.RGBA, x, y float64) color.RGBA {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	y1 := y0 + 1
	fx := x - float64(x0)
	fy := y - float64(y0)

	clamp := func(v, max int) int {
		if v < 0 {
			return 0
		}
		if v >= max {
			return max - 1
		}
		return v
	}

	c00 := img.RGBAAt(clamp(x0, w)+b.Min.X, clamp(y0, h)+b.Min.Y)
	c10 := img.RGBAAt(clamp(x1, w)+b.Min.X, clamp(y0, h)+b.Min.Y)
	c01 := img.RGBAAt(clamp(x0, w)+b.Min.X, clamp(y1, h)+b.Min.Y)
	c11 := img.RGBAAt(clamp(x1, w)+b.Min.X, clamp(y1, h)+b.Min.Y)

	lerp := func(a, b uint8, t float64) uint8 {
		return uint8(float64(a)*(1-t) + float64(b)*t + 0.5)
	}

	return color.RGBA{
		R: lerp(lerp(c00.R, c10.R, fx), lerp(c01.R, c11.R, fx), fy),
		G: lerp(lerp(c00.G, c10.G, fx), lerp(c01.G, c11.G, fx), fy),
		B: lerp(lerp(c00.B, c10.B, fx), lerp(c01.B, c11.B, fx), fy),
		A: 255,
	}
}
