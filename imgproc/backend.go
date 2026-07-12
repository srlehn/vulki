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
	// BackendVulkan requires the direct Vulkan implementation.
	BackendVulkan Backend = "vulkan"
	// BackendCPU uses the CPU implementation without probing Vulkan.
	BackendCPU Backend = "cpu"
)

// Option configures NewCorrelator. Options are applied before any CPU or
// Vulkan resources are allocated.
type Option interface {
	apply(*correlatorConfig) error
}

type optionFunc func(*correlatorConfig) error

func (option optionFunc) apply(config *correlatorConfig) error {
	return option(config)
}

type correlatorConfig struct {
	backend    Backend
	backendSet bool
}

// WithBackend selects automatic fallback, direct Vulkan, or CPU execution.
// It may be supplied at most once.
func WithBackend(backend Backend) Option {
	return optionFunc(func(config *correlatorConfig) error {
		if config.backendSet {
			return fmt.Errorf("imgproc: backend option is already set")
		}
		switch backend {
		case BackendAuto, BackendVulkan, BackendCPU:
		default:
			return fmt.Errorf("imgproc: unknown backend %q", backend)
		}
		config.backend = backend
		config.backendSet = true
		return nil
	})
}

// NewCorrelator creates a correlator. With no options it prefers direct Vulkan
// and falls back to CPU if Vulkan initialization or resource creation fails.
func NewCorrelator(maxW, maxH int, options ...Option) (*Correlator, error) {
	config := correlatorConfig{backend: BackendAuto}
	for index, option := range options {
		if option == nil {
			return nil, fmt.Errorf("imgproc: option %d is nil", index)
		}
		if err := option.apply(&config); err != nil {
			return nil, fmt.Errorf("imgproc: option %d: %w", index, err)
		}
	}

	switch config.backend {
	case BackendAuto:
		return newAutoCorrelator(maxW, maxH, compute.NewContext)
	case BackendVulkan:
		return newOwnedVulkanCorrelator(maxW, maxH)
	case BackendCPU:
		return newCPUCorrelator(maxW, maxH)
	default:
		return nil, fmt.Errorf("imgproc: unknown backend %q", config.backend)
	}
}

func newAutoCorrelator(
	maxW, maxH int,
	newContext func() (*compute.Context, error),
) (*Correlator, error) {
	cpu, err := newCPUCorrelator(maxW, maxH)
	if err != nil {
		return nil, err
	}

	ctx, gpuErr := newContext()
	if gpuErr == nil {
		gpu, err := newVulkanCorrelator(ctx, maxW, maxH)
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

func newOwnedVulkanCorrelator(maxW, maxH int) (*Correlator, error) {
	ctx, err := compute.NewContext()
	if err != nil {
		return nil, err
	}
	c, err := newVulkanCorrelator(ctx, maxW, maxH)
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
