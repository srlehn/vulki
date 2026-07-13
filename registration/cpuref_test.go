package registration

import (
	"errors"
	"image"
	"testing"

	"github.com/srlehn/vulki"
)

func TestPhaseCorrelateCPU_KnownTransform(t *testing.T) {
	imgA := loadTestDataImage(t, "snake.png")
	corr, err := NewCorrelator(
		imgA.Bounds().Dx(),
		imgA.Bounds().Dy(),
		WithBackend(BackendCPU),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer corr.Close()
	if corr.Backend() != BackendCPU {
		t.Fatalf("backend = %q, want %q", corr.Backend(), BackendCPU)
	}
	assertKnownTransform(t, corr, imgA)
}

func TestPhaseCorrelateCPU_RejectsBlankImages(t *testing.T) {
	const size = 64
	corr, err := NewCorrelator(size, size, WithBackend(BackendCPU))
	if err != nil {
		t.Fatal(err)
	}
	defer corr.Close()

	blank := image.NewRGBA(image.Rect(0, 0, size, size))
	if _, err := corr.PhaseCorrelate(blank, blank); !errors.Is(err, ErrLowConfidence) {
		t.Fatalf("blank-image error = %v, want ErrLowConfidence", err)
	}
}

func TestPhaseCorrelateCPU_ConvertsNonRGBAImages(t *testing.T) {
	const size = 64
	corr, err := NewCorrelator(size, size, WithBackend(BackendCPU))
	if err != nil {
		t.Fatal(err)
	}
	defer corr.Close()

	blank := image.NewNRGBA(image.Rect(0, 0, size, size))
	if _, err := corr.PhaseCorrelate(blank, blank); !errors.Is(err, ErrLowConfidence) {
		t.Fatalf("blank NRGBA error = %v, want ErrLowConfidence", err)
	}
}

func TestAsRGBAReusesRGBA(t *testing.T) {
	img := image.NewRGBA(image.Rect(3, 5, 11, 13))
	converted, err := asRGBA(img)
	if err != nil {
		t.Fatal(err)
	}
	if converted != img {
		t.Fatal("asRGBA copied an RGBA image")
	}
}

func TestPhaseCorrelateRejectsTypedNilImage(t *testing.T) {
	corr, err := NewCorrelator(64, 64, WithBackend(BackendCPU))
	if err != nil {
		t.Fatal(err)
	}
	defer corr.Close()

	var img *image.RGBA
	if _, err := corr.PhaseCorrelate(img, img); err == nil {
		t.Fatal("typed nil images were accepted")
	}
}

func TestAutoCorrelatorFallsBackToCPU(t *testing.T) {
	gpuErr := errors.New("test Vulkan failure")
	corr, err := newAutoCorrelator(64, 64, func() (*vulki.Device, error) {
		return nil, gpuErr
	})
	if err != nil {
		t.Fatal(err)
	}
	defer corr.Close()

	if corr.Backend() != BackendCPU {
		t.Fatalf("backend = %q, want %q", corr.Backend(), BackendCPU)
	}
	if !errors.Is(corr.FallbackReason(), gpuErr) {
		t.Fatalf("fallback reason = %v, want %v", corr.FallbackReason(), gpuErr)
	}
}

func TestNewCorrelatorRejectsUnknownBackend(t *testing.T) {
	if _, err := NewCorrelator(64, 64, WithBackend(Backend("invalid"))); err == nil {
		t.Fatal("unknown backend was accepted")
	}
}

func TestNewCorrelatorRejectsNilOption(t *testing.T) {
	if _, err := NewCorrelator(64, 64, nil); err == nil {
		t.Fatal("nil option was accepted")
	}
}

func TestNewCorrelatorRejectsDuplicateBackend(t *testing.T) {
	if _, err := NewCorrelator(
		64,
		64,
		WithBackend(BackendCPU),
		WithBackend(BackendCPU),
	); err == nil {
		t.Fatal("duplicate backend option was accepted")
	}
}

func TestNewCorrelatorRejectsNilAndClosedBorrowedDevice(t *testing.T) {
	if _, err := NewCorrelator(64, 64, WithDevice(nil)); err == nil {
		t.Fatal("nil borrowed device was accepted")
	}
	if _, err := NewCorrelator(64, 64, WithDevice(new(vulki.Device))); err == nil {
		t.Fatal("closed borrowed device was accepted")
	}
}

func TestNewCorrelatorBorrowsDevice(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	defer device.Close()

	corr, err := NewCorrelator(64, 64, WithDevice(device))
	if err != nil {
		t.Fatalf("NewCorrelator: %v", err)
	}
	if err := corr.Close(); err != nil {
		t.Fatalf("Correlator.Close: %v", err)
	}
	if err := corr.Close(); err != nil {
		t.Fatalf("second Correlator.Close: %v", err)
	}
	if device.Closed() {
		t.Fatal("Correlator.Close closed its borrowed device")
	}
	buffer, err := device.NewBuffer(4)
	if err != nil {
		t.Fatalf("borrowed device is unusable after Correlator.Close: %v", err)
	}
	_ = buffer.Close()
}

func TestNewCorrelatorRejectsConflictingAndRepeatedDeviceOptions(t *testing.T) {
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	defer device.Close()

	tests := []struct {
		name    string
		options []Option
	}{
		{name: "backend then device", options: []Option{WithBackend(BackendVulkan), WithDevice(device)}},
		{name: "device then backend", options: []Option{WithDevice(device), WithBackend(BackendVulkan)}},
		{name: "repeated device", options: []Option{WithDevice(device), WithDevice(device)}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := NewCorrelator(64, 64, test.options...); err == nil {
				t.Fatal("invalid option combination was accepted")
			}
		})
	}
}
