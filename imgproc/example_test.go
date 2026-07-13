package imgproc_test

import (
	"image"

	"github.com/srlehn/vulki/imgproc"
)

func ExampleNewCorrelator() {
	correlator, err := imgproc.NewCorrelator(1024, 1024)
	if err != nil {
		return
	}
	defer correlator.Close()

	imageA := image.NewNRGBA(image.Rect(0, 0, 1024, 1024))
	imageB := image.NewNRGBA(image.Rect(0, 0, 1024, 1024))
	_, _ = correlator.PhaseCorrelate(imageA, imageB)
}

func ExampleNewCorrelator_cpu() {
	correlator, err := imgproc.NewCorrelator(
		1024,
		1024,
		imgproc.WithBackend(imgproc.BackendCPU),
	)
	if err != nil {
		return
	}
	defer correlator.Close()
}
