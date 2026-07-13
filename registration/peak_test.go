package registration

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestNextPow2(t *testing.T) {
	tests := []struct{ in, want int }{
		{1, 1}, {2, 2}, {3, 4}, {4, 4}, {5, 8}, {7, 8}, {8, 8},
		{9, 16}, {255, 256}, {256, 256}, {257, 512}, {1023, 1024},
	}
	for _, tt := range tests {
		if got := nextPow2(tt.in); got != tt.want {
			t.Errorf("nextPow2(%d) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

func TestLog2i(t *testing.T) {
	tests := []struct{ in, want int }{
		{1, 0}, {2, 1}, {4, 2}, {8, 3}, {16, 4}, {256, 8}, {1024, 10},
	}
	for _, tt := range tests {
		if got := log2i(tt.in); got != tt.want {
			t.Errorf("log2i(%d) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

func TestNormalizeAngle(t *testing.T) {
	tests := []struct {
		input float64
		want  float64
	}{
		{input: -540, want: -180},
		{input: -181, want: 179},
		{input: -180, want: -180},
		{input: -12, want: -12},
		{input: 0, want: 0},
		{input: 12, want: 12},
		{input: 179, want: 179},
		{input: 180, want: -180},
		{input: 181, want: -179},
		{input: 540, want: -180},
	}
	for _, test := range tests {
		if got := normalizeAngle(test.input); got != test.want {
			t.Errorf("normalizeAngle(%v) = %v, want %v", test.input, got, test.want)
		}
	}
}

func TestNormalizeComplex(t *testing.T) {
	data := [][2]float32{{4, 8}, {2, 6}, {10, 0}}
	normalizeComplex(data, 2)
	for i, d := range data {
		expected := [][2]float32{{2, 4}, {1, 3}, {5, 0}}
		if math.Abs(float64(d[0]-expected[i][0])) > 1e-6 || math.Abs(float64(d[1]-expected[i][1])) > 1e-6 {
			t.Errorf("data[%d] = %v, want %v", i, d, expected[i])
		}
	}
}

func TestFind2DPeak_DeltaAtOrigin(t *testing.T) {
	// A delta function at (0,0) has its peak at (0,0).
	w, h := 16, 16
	data := make([][2]float32, w*h)
	data[0] = [2]float32{100, 0}
	px, py := find2DPeak(data, w, h)
	if px != 0 || py != 0 {
		t.Errorf("peak at (%v, %v), want (0, 0)", px, py)
	}
}

func TestFind2DPeak_DeltaAtKnownLocation(t *testing.T) {
	w, h := 32, 32
	data := make([][2]float32, w*h)
	// Place peak at (5, 10).
	data[10*w+5] = [2]float32{100, 0}
	px, py := find2DPeak(data, w, h)
	if px != 5 || py != 10 {
		t.Errorf("peak at (%v, %v), want (5, 10)", px, py)
	}
}

func TestFind2DPeak_SubpixelRefinement(t *testing.T) {
	// Create a parabolic peak centered at (5.3, 10.2).
	// The discrete peak should be at (5,10) with subpixel refinement shifting it.
	w, h := 32, 32
	data := make([][2]float32, w*h)
	cx, cy := 5.3, 10.2
	for y := range h {
		for x := range w {
			dx := float64(x) - cx
			dy := float64(y) - cy
			val := float32(math.Max(0, 100-dx*dx-dy*dy))
			data[y*w+x] = [2]float32{val, 0}
		}
	}
	px, py := find2DPeak(data, w, h)
	if math.Abs(px-cx) > 0.7 || math.Abs(py-cy) > 0.7 {
		t.Errorf("peak at (%v, %v), want near (%.1f, %.1f)", px, py, cx, cy)
	}
}

func TestLogPolarToAngleScale_Identity(t *testing.T) {
	// Peak at (0,0) should give angle=0, scale=1.
	angle, scale := logPolarToAngleScale(0, 0, 256, 256, 100)
	if angle != 0 || scale != 1 {
		t.Errorf("angle=%v, scale=%v, want 0, 1", angle, scale)
	}
}

func TestLogPolarToAngleScale_KnownAngle(t *testing.T) {
	lpW, lpH := 256, 256
	maxRadius := 100.0

	// peakY = lpH * angle / 180 → for angle=12°: peakY = 256 * 12/180 = 17.066...
	peakY := float64(lpH) * 12.0 / 180.0
	angle, _ := logPolarToAngleScale(0, peakY, lpW, lpH, maxRadius)
	if math.Abs(angle-12.0) > 0.01 {
		t.Errorf("angle = %v, want ~12.0", angle)
	}
}

func TestLogPolarToAngleScale_KnownScale(t *testing.T) {
	lpW, lpH := 256, 256
	maxRadius := 100.0

	// scale = exp(peakX / lpW * log(maxRadius))
	// For scale=1.15: peakX = lpW * log(1.15) / log(maxRadius)
	wantScale := 1.15
	peakX := float64(lpW) * math.Log(wantScale) / math.Log(maxRadius)
	_, scale := logPolarToAngleScale(peakX, 0, lpW, lpH, maxRadius)
	if math.Abs(scale-wantScale) > 0.001 {
		t.Errorf("scale = %v, want ~%v", scale, wantScale)
	}
}

func TestLogPolarToAngleScale_NegativeAngle(t *testing.T) {
	lpW, lpH := 256, 256
	maxRadius := 100.0

	// After wraparound, peakY is negative → negative angle.
	peakY := -float64(lpH) * 12.0 / 180.0
	angle, _ := logPolarToAngleScale(0, peakY, lpW, lpH, maxRadius)
	if math.Abs(angle-(-12.0)) > 0.01 {
		t.Errorf("angle = %v, want ~-12.0", angle)
	}
}

func TestRealToComplex(t *testing.T) {
	data := []float32{1.0, 2.5, 3.0}
	out := realToComplex(data)
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3", len(out))
	}
	for i, v := range out {
		if v[0] != data[i] || v[1] != 0 {
			t.Errorf("out[%d] = %v, want {%v, 0}", i, v, data[i])
		}
	}
}

func TestPackRGBAHandlesSubimageStrideAndBounds(t *testing.T) {
	parent := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := range 4 {
		for x := range 4 {
			parent.SetRGBA(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}
	sub := parent.SubImage(image.Rect(1, 1, 3, 3)).(*image.RGBA)

	packed, err := packRGBA(sub)
	if err != nil {
		t.Fatal(err)
	}
	if len(packed) != 4 {
		t.Fatalf("packed length = %d, want 4", len(packed))
	}
	for y := range 2 {
		for x := range 2 {
			pixel := sub.RGBAAt(x+1, y+1)
			want := uint32(pixel.R) | uint32(pixel.G)<<8 | uint32(pixel.B)<<16 | uint32(pixel.A)<<24
			if got := packed[y*2+x]; got != want {
				t.Errorf("packed[%d,%d] = %#x, want %#x", x, y, got, want)
			}
		}
	}
}

func TestPackRGBARejectsShortPixelData(t *testing.T) {
	img := &image.RGBA{
		Pix:    make([]byte, 8),
		Stride: 8,
		Rect:   image.Rect(0, 0, 2, 2),
	}
	if _, err := packRGBA(img); err == nil {
		t.Fatal("packRGBA accepted short pixel data")
	}
}

func TestBilinearWarpPreservesNonZeroBounds(t *testing.T) {
	parent := image.NewRGBA(image.Rect(0, 0, 5, 5))
	for y := 1; y < 4; y++ {
		for x := 1; x < 4; x++ {
			parent.SetRGBA(x, y, color.RGBA{R: uint8(10*x + y), G: uint8(x + 10*y), A: 255})
		}
	}
	sub := parent.SubImage(image.Rect(1, 1, 4, 4)).(*image.RGBA)
	warped := bilinearWarp(sub, 0, 1)
	if warped.Bounds() != sub.Bounds() {
		t.Fatalf("warped bounds = %v, want %v", warped.Bounds(), sub.Bounds())
	}
	for y := sub.Bounds().Min.Y; y < sub.Bounds().Max.Y; y++ {
		for x := sub.Bounds().Min.X; x < sub.Bounds().Max.X; x++ {
			if got, want := warped.RGBAAt(x, y), sub.RGBAAt(x, y); got != want {
				t.Errorf("warped pixel (%d,%d) = %v, want %v", x, y, got, want)
			}
		}
	}
}

func TestPadImageToRGBAHandlesNonZeroBounds(t *testing.T) {
	parent := image.NewRGBA(image.Rect(0, 0, 3, 3))
	parent.SetRGBA(1, 1, color.RGBA{R: 1, G: 2, B: 3, A: 4})
	sub := parent.SubImage(image.Rect(1, 1, 2, 2)).(*image.RGBA)
	packed := padImageToRGBA(sub, 1, 1)
	if len(packed) != 1 || packed[0] != 0x04030201 {
		t.Fatalf("packed image = %#v, want [0x04030201]", packed)
	}
}
