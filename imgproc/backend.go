package imgproc

import (
	"fmt"

	"github.com/srlehn/vulki/compute"
)

// Backend selects a phase-correlation implementation.
type Backend string

const (
	// BackendAuto prefers Vulkan and falls back to the CPU implementation.
	BackendAuto Backend = "auto"
	// BackendGPU requires a Vulkan compute backend.
	BackendGPU Backend = "gpu"
	// BackendCPU uses the CPU implementation without probing Vulkan.
	BackendCPU Backend = "cpu"
)

// NewBackendCorrelator creates a correlator using the requested backend.
func NewBackendCorrelator(backend Backend, maxW, maxH int) (*Correlator, error) {
	switch backend {
	case BackendAuto:
		return NewAutoCorrelator(maxW, maxH)
	case BackendGPU:
		return NewGPUCorrelator(maxW, maxH)
	case BackendCPU:
		return NewCPUCorrelator(maxW, maxH)
	default:
		return nil, fmt.Errorf("imgproc: unknown backend %q", backend)
	}
}

// NewAutoCorrelator prefers Vulkan and falls back to the CPU implementation
// when Vulkan initialization or GPU correlator creation fails.
func NewAutoCorrelator(maxW, maxH int) (*Correlator, error) {
	return newAutoCorrelator(maxW, maxH, compute.NewContext)
}

func newAutoCorrelator(
	maxW, maxH int,
	newContext func() (*compute.Context, error),
) (*Correlator, error) {
	cpu, err := NewCPUCorrelator(maxW, maxH)
	if err != nil {
		return nil, err
	}

	ctx, gpuErr := newContext()
	if gpuErr == nil {
		gpu, err := NewCorrelator(ctx, maxW, maxH)
		if err == nil {
			gpu.ownsContext = true
			return gpu, nil
		}
		ctx.Close()
		gpuErr = err
	}

	cpu.fallbackReason = gpuErr
	return cpu, nil
}

// NewGPUCorrelator creates and owns a Vulkan context for a GPU correlator.
func NewGPUCorrelator(maxW, maxH int) (*Correlator, error) {
	ctx, err := compute.NewContext()
	if err != nil {
		return nil, err
	}
	c, err := NewCorrelator(ctx, maxW, maxH)
	if err != nil {
		ctx.Close()
		return nil, err
	}
	c.ownsContext = true
	return c, nil
}

// Backend reports the implementation selected for this correlator.
func (c *Correlator) Backend() Backend {
	if c == nil {
		return ""
	}
	return c.backend
}

// FallbackReason reports why an automatic correlator selected the CPU. It is
// nil for explicit CPU correlators and correlators using the GPU.
func (c *Correlator) FallbackReason() error {
	if c == nil {
		return nil
	}
	return c.fallbackReason
}
