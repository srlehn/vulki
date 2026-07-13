package imgproc

import (
	"errors"
	"image"
	"testing"

	"github.com/srlehn/vulki/compute"
)

func TestPhaseCorrelateCPU_KnownTransform(t *testing.T) {
	imgA := loadTestImage(t, "../testdata/snake.png")
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
