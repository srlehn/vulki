package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"strconv"
	"time"

	"github.com/srlehn/vulki/compute"
	"github.com/srlehn/vulki/imgproc"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	save := flag.Bool("save", false, "save stacked comparison image (self-test mode)")
	flag.Parse()
	args := flag.Args()

	switch len(args) {
	case 1:
		return runSelfTest(args[0], *save)
	case 2:
		return runCorrelate(args[0], args[1])
	default:
		return fmt.Errorf("usage: correlate [-save] <image.png> [imageB.png]")
	}
}

func runCorrelate(pathA, pathB string) error {
	imgA, err := loadRGBA(pathA)
	if err != nil {
		return fmt.Errorf("load %s: %w", pathA, err)
	}
	imgB, err := loadRGBA(pathB)
	if err != nil {
		return fmt.Errorf("load %s: %w", pathB, err)
	}

	fmt.Println("Creating Vulkan compute context...")
	ctx, err := compute.NewContext()
	if err != nil {
		return fmt.Errorf("compute context: %w", err)
	}
	defer ctx.Close()

	boundsA := imgA.Bounds()
	boundsB := imgB.Bounds()
	maxW := max(boundsA.Dx(), boundsB.Dx())
	maxH := max(boundsA.Dy(), boundsB.Dy())

	fmt.Printf("Image A: %dx%d, Image B: %dx%d\n", boundsA.Dx(), boundsA.Dy(), boundsB.Dx(), boundsB.Dy())
	fmt.Println("Creating correlator...")
	corr, err := imgproc.NewCorrelator(ctx, maxW, maxH)
	if err != nil {
		return fmt.Errorf("create correlator: %w", err)
	}
	defer corr.Close()

	fmt.Println("Running phase correlation...")
	result, err := corr.PhaseCorrelate(imgA, imgB)
	if err != nil {
		return fmt.Errorf("phase correlate: %w", err)
	}

	fmt.Println()
	fmt.Println("=== Result ===")
	fmt.Printf("Angle:       %.2f°\n", result.Angle)
	fmt.Printf("Scale:       %.4f\n", result.Scale)
	fmt.Printf("Translation: (%.2f, %.2f) px\n", result.Tx, result.Ty)
	fmt.Printf("Confidence:  rotation %.4f, translation %.4f\n",
		result.RotationConfidence, result.TranslationConfidence)

	return nil
}

func runSelfTest(path string, save bool) error {
	imgA, err := loadRGBA(path)
	if err != nil {
		return fmt.Errorf("load %s: %w", path, err)
	}
	bounds := imgA.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// Generate random transform params.
	gtTx := rand.IntN(101) - 50         // [-50, 50]
	gtTy := rand.IntN(101) - 50         // [-50, 50]
	gtRot := rand.Float64()*90 - 45     // [-45, 45] degrees
	gtScale := rand.Float64()*0.8 + 0.7 // [0.7, 1.5]

	fmt.Println("=== Self-test mode ===")
	fmt.Printf("Ground truth: tx=%d ty=%d rot=%.2f° scale=%.4f\n", gtTx, gtTy, gtRot, gtScale)

	// Use ImageMagick to create transformed image.
	tmpFile, err := os.CreateTemp("", "correlate-selftest-*.png")
	if err != nil {
		return fmt.Errorf("create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	tmpFile.Close()
	defer os.Remove(tmpPath)

	cx, cy := w/2, h/2
	// SRT format: cx,cy scaleX,scaleY angle newX,newY
	// Negate angle: ImageMagick SRT positive=CW, our convention positive=CCW.
	// newX,newY = cx+tx, cy+ty to keep center and apply translation.
	srt := fmt.Sprintf("%d,%d %s,%s %s %d,%d",
		cx, cy,
		fmtf(gtScale), fmtf(gtScale),
		fmtf(-gtRot),
		cx+gtTx, cy+gtTy,
	)
	cmd := exec.Command("convert", path,
		"-virtual-pixel", "edge",
		"-distort", "SRT", srt,
		"+repage", tmpPath,
	)
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ImageMagick convert: %w", err)
	}

	imgB, err := loadRGBA(tmpPath)
	if err != nil {
		return fmt.Errorf("load transformed: %w", err)
	}

	// Run correlator.
	fmt.Println("Creating Vulkan compute context...")
	ctx, err := compute.NewContext()
	if err != nil {
		return fmt.Errorf("compute context: %w", err)
	}
	defer ctx.Close()

	maxW := max(w, imgB.Bounds().Dx())
	maxH := max(h, imgB.Bounds().Dy())

	corr, err := imgproc.NewCorrelator(ctx, maxW, maxH)
	if err != nil {
		return fmt.Errorf("create correlator: %w", err)
	}
	defer corr.Close()

	fmt.Println("Running phase correlation...")
	t0 := time.Now()
	result, err := corr.PhaseCorrelate(imgA, imgB)
	if err != nil {
		return fmt.Errorf("phase correlate: %w", err)
	}
	elapsed := time.Since(t0)
	txError := result.Tx - float64(gtTx)
	tyError := result.Ty - float64(gtTy)
	angleError := math.Mod(result.Angle-gtRot+180, 360) - 180
	scaleError := result.Scale - gtScale

	fmt.Println()
	fmt.Printf("Phase correlation took %dms\n", elapsed.Milliseconds())
	fmt.Printf("Ground truth: tx=%d  ty=%d  rot=%.2f°  scale=%.4f\n", gtTx, gtTy, gtRot, gtScale)
	fmt.Printf("Detected:     tx=%.2f  ty=%.2f  rot=%.2f°  scale=%.4f\n", result.Tx, result.Ty, result.Angle, result.Scale)
	fmt.Printf("Error:        tx=%.2f  ty=%.2f  rot=%.2f°  scale=%.4f\n",
		txError, tyError, angleError, scaleError)

	if save {
		// Reconstruct: apply detected params via SRT (same as ground truth image).
		tmpDet, err := os.CreateTemp("", "correlate-detected-*.png")
		if err != nil {
			return fmt.Errorf("create temp file: %w", err)
		}
		tmpDetPath := tmpDet.Name()
		tmpDet.Close()
		defer os.Remove(tmpDetPath)

		detSrt := fmt.Sprintf("%d,%d %s,%s %s %s,%s",
			cx, cy,
			fmtf(result.Scale), fmtf(result.Scale),
			fmtf(-result.Angle),
			fmtf(float64(cx)+result.Tx), fmtf(float64(cy)+result.Ty),
		)
		detCmd := exec.Command("convert", path,
			"-virtual-pixel", "edge",
			"-distort", "SRT", detSrt,
			"+repage", tmpDetPath,
		)
		detCmd.Stderr = os.Stderr
		if err := detCmd.Run(); err != nil {
			return fmt.Errorf("ImageMagick reconstruct: %w", err)
		}
		cropped, err := loadRGBA(tmpDetPath)
		if err != nil {
			return fmt.Errorf("load reconstructed: %w", err)
		}

		savePath := fmt.Sprintf("selftest_gt_tx%d_ty%d_rot_%sdeg_scale_%s_det_tx%s_ty%s_rot_%sdeg_scale_%s.png",
			gtTx, gtTy, fmtf(gtRot), fmtf(gtScale),
			fmtf(result.Tx), fmtf(result.Ty), fmtf(result.Angle), fmtf(result.Scale),
		)

		stacked := stackVertical(imgA, imgB, cropped)
		if err := savePNG(savePath, stacked); err != nil {
			return fmt.Errorf("save stacked: %w", err)
		}
		fmt.Printf("Saved: %s\n", savePath)
	}

	const (
		maxTranslationError = 3.0
		maxAngleError       = 2.0
		maxScaleError       = 0.05
	)
	if math.Abs(txError) > maxTranslationError ||
		math.Abs(tyError) > maxTranslationError ||
		math.Abs(angleError) > maxAngleError ||
		math.Abs(scaleError) > maxScaleError {
		return fmt.Errorf("self-test exceeded error tolerances")
	}

	return nil
}

func fmtf(f float64) string {
	return strconv.FormatFloat(f, 'f', 2, 64)
}

// stackVertical stacks images vertically, using the widest width.
func stackVertical(imgs ...*image.RGBA) *image.RGBA {
	maxW := 0
	totalH := 0
	for _, img := range imgs {
		b := img.Bounds()
		if b.Dx() > maxW {
			maxW = b.Dx()
		}
		totalH += b.Dy()
	}

	dst := image.NewRGBA(image.Rect(0, 0, maxW, totalH))
	// Fill with dark gray separator lines are implicit (black bg from unwritten pixels).
	draw.Draw(dst, dst.Bounds(), &image.Uniform{color.RGBA{32, 32, 32, 255}}, image.Point{}, draw.Src)

	y := 0
	for _, img := range imgs {
		b := img.Bounds()
		draw.Draw(dst, image.Rect(0, y, b.Dx(), y+b.Dy()), img, b.Min, draw.Src)
		y += b.Dy()
	}
	return dst
}

func savePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func loadRGBA(path string) (*image.RGBA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, err := png.Decode(f)
	if err != nil {
		return nil, err
	}

	if rgba, ok := img.(*image.RGBA); ok {
		return rgba, nil
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
