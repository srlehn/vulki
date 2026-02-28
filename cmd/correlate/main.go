package main

import (
	"fmt"
	"image"
	"image/png"
	"os"

	"vkpg/compute"
	"vkpg/imgproc"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	if len(os.Args) != 3 {
		return fmt.Errorf("usage: correlate <imageA.png> <imageB.png>")
	}

	imgA, err := loadRGBA(os.Args[1])
	if err != nil {
		return fmt.Errorf("load %s: %w", os.Args[1], err)
	}
	imgB, err := loadRGBA(os.Args[2])
	if err != nil {
		return fmt.Errorf("load %s: %w", os.Args[2], err)
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

	return nil
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

	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}
	return rgba, nil
}
