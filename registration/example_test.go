package registration_test

import (
	"image"

	"github.com/srlehn/vulki"
	"github.com/srlehn/vulki/registration"
)

func ExampleNewCorrelator() {
	correlator, err := registration.NewCorrelator(1024, 1024)
	if err != nil {
		return
	}
	defer correlator.Close()

	imageA := image.NewNRGBA(image.Rect(0, 0, 1024, 1024))
	imageB := image.NewNRGBA(image.Rect(0, 0, 1024, 1024))
	_, _ = correlator.PhaseCorrelate(imageA, imageB)
}

func ExampleNewCorrelator_cpu() {
	correlator, err := registration.NewCorrelator(
		1024,
		1024,
		registration.WithBackend(registration.BackendCPU),
	)
	if err != nil {
		return
	}
	defer correlator.Close()
}

func ExampleNewCorrelator_borrowedDevice() {
	device, err := vulki.Open()
	if err != nil {
		return
	}
	defer device.Close()

	correlator, err := registration.NewCorrelator(
		1024,
		1024,
		registration.WithDevice(device),
	)
	if err != nil {
		return
	}
	defer correlator.Close()
}
