//go:build ignore

package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

type testImage struct {
	tx          int
	ty          int
	scaleX      float64
	scaleY      float64
	rotationDeg float64
}

func (t testImage) filename() string {
	var parts []string
	if t.tx != 0 || t.ty != 0 {
		parts = append(parts, fmt.Sprintf("shift_tx%d_ty%d", t.tx, t.ty))
	}
	if t.rotationDeg != 0 {
		parts = append(parts, fmt.Sprintf("rot_%sdeg", fmtf(t.rotationDeg)))
	}
	if t.scaleX != 1 && t.scaleX == t.scaleY {
		parts = append(parts, fmt.Sprintf("scale_%s", fmtf(t.scaleX)))
	} else {
		if t.scaleX != 1 {
			parts = append(parts, fmt.Sprintf("sx_%s", fmtf(t.scaleX)))
		}
		if t.scaleY != 1 {
			parts = append(parts, fmt.Sprintf("sy_%s", fmtf(t.scaleY)))
		}
	}
	return fmt.Sprintf("testdata/snake_%s.png", strings.Join(parts, "_"))
}

func fmtf(f float64) string {
	return strconv.FormatFloat(f, 'f', -1, 64)
}

func (t testImage) convertArgs(w, h int) []string {
	// SRT format: ox,oy scaleX,scaleY angle tx,ty
	// Use image center as origin so rotation/scale preserve content.
	cx, cy := w/2, h/2
	srt := fmt.Sprintf("%d,%d %s,%s %s %d,%d",
		cx, cy,
		fmtf(t.scaleX), fmtf(t.scaleY),
		fmtf(t.rotationDeg),
		t.tx, t.ty,
	)
	return []string{"-distort", "SRT", srt}
}

var images = []testImage{
	{tx: 15, ty: 25, scaleX: 1, scaleY: 1, rotationDeg: 0},
	{tx: 0, ty: 0, scaleX: 1.15, scaleY: 1.15, rotationDeg: 0},
	{tx: 0, ty: 0, scaleX: 1, scaleY: 1, rotationDeg: 12},
	{tx: 0, ty: 0, scaleX: 1.15, scaleY: 1.15, rotationDeg: 12},
	{tx: 15, ty: 25, scaleX: 1, scaleY: 1, rotationDeg: 12},
	{tx: 15, ty: 25, scaleX: 1.15, scaleY: 1.15, rotationDeg: 0},
}

func main() {
	src := "testdata/snake.png"

	// Get source image dimensions.
	out, err := exec.Command("identify", "-format", "%w %h", src).Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "identify %s: %v\n", src, err)
		os.Exit(1)
	}
	var srcW, srcH int
	fmt.Sscanf(string(out), "%d %d", &srcW, &srcH)

	for _, img := range images {
		name := img.filename()
		args := []string{src, "-virtual-pixel", "black", "-background", "black"}
		args = append(args, img.convertArgs(srcW, srcH)...)
		args = append(args, "+repage", name)
		cmd := exec.Command("convert", args...)
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", name, err)
			os.Exit(1)
		}
		fmt.Printf("wrote %s\n", name)
	}
}
