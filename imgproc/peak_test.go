package imgproc

import (
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
	// A delta function at (0,0) — peak should be at (0,0).
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
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
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
