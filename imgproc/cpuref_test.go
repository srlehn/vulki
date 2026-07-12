package imgproc

import (
	"errors"
	"image"
	"testing"

	"github.com/srlehn/vulki/compute"
)

func TestPhaseCorrelateCPU_KnownTransform(t *testing.T) {
	imgA := loadTestImage(t, "../testdata/snake.png")
	corr, err := NewCPUCorrelator(imgA.Bounds().Dx(), imgA.Bounds().Dy())
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
	corr, err := NewCPUCorrelator(size, size)
	if err != nil {
		t.Fatal(err)
	}
	defer corr.Close()

	blank := image.NewRGBA(image.Rect(0, 0, size, size))
	if _, err := corr.PhaseCorrelate(blank, blank); !errors.Is(err, ErrLowConfidence) {
		t.Fatalf("blank-image error = %v, want ErrLowConfidence", err)
	}
}

func TestAutoCorrelatorFallsBackToCPU(t *testing.T) {
	gpuErr := errors.New("test Vulkan failure")
	corr, err := newAutoCorrelator(64, 64, func() (*compute.Context, error) {
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

func TestNewBackendCorrelatorRejectsUnknownBackend(t *testing.T) {
	if _, err := NewBackendCorrelator(Backend("invalid"), 64, 64); err == nil {
		t.Fatal("unknown backend was accepted")
	}
}
