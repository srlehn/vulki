package registration

import (
	"fmt"

	"github.com/srlehn/vulki"
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
type Option func(*correlatorConfig) error

type correlatorConfig struct {
	backend    Backend
	backendSet bool
	device     *vulki.Device
	deviceSet  bool
}

// WithBackend selects automatic fallback, direct Vulkan, or CPU execution.
// It may be supplied at most once.
func WithBackend(backend Backend) Option {
	return func(config *correlatorConfig) error {
		if config.deviceSet {
			return fmt.Errorf("registration: backend and device options cannot be combined")
		}
		if config.backendSet {
			return fmt.Errorf("registration: backend option is already set")
		}
		switch backend {
		case BackendAuto, BackendVulkan, BackendCPU:
		default:
			return fmt.Errorf("registration: unknown backend %q", backend)
		}
		config.backend = backend
		config.backendSet = true
		return nil
	}
}

// WithDevice uses an existing Vulkan device without taking ownership. It
// cannot be combined with WithBackend and may be supplied at most once.
func WithDevice(device *vulki.Device) Option {
	return func(config *correlatorConfig) error {
		if config.backendSet {
			return fmt.Errorf("registration: backend and device options cannot be combined")
		}
		if config.deviceSet {
			return fmt.Errorf("registration: device option is already set")
		}
		if device == nil || device.Closed() {
			return fmt.Errorf("registration: borrowed device is closed")
		}
		config.device = device
		config.deviceSet = true
		return nil
	}
}

// NewCorrelator creates a correlator. With no options it prefers direct Vulkan
// and falls back to CPU if Vulkan initialization or resource creation fails.
func NewCorrelator(maxW, maxH int, options ...Option) (*Correlator, error) {
	config := correlatorConfig{backend: BackendAuto}
	for index, option := range options {
		if option == nil {
			return nil, fmt.Errorf("registration: option %d is nil", index)
		}
		if err := option(&config); err != nil {
			return nil, fmt.Errorf("registration: option %d: %w", index, err)
		}
	}

	if config.deviceSet {
		return newVulkanCorrelator(config.device, maxW, maxH)
	}
	switch config.backend {
	case BackendAuto:
		return newAutoCorrelator(maxW, maxH, vulki.Open)
	case BackendVulkan:
		return newOwnedVulkanCorrelator(maxW, maxH)
	case BackendCPU:
		return newCPUCorrelator(maxW, maxH)
	default:
		return nil, fmt.Errorf("registration: unknown backend %q", config.backend)
	}
}

func newAutoCorrelator(
	maxW, maxH int,
	openDevice func() (*vulki.Device, error),
) (*Correlator, error) {
	cpu, err := newCPUCorrelator(maxW, maxH)
	if err != nil {
		return nil, err
	}

	device, gpuErr := openDevice()
	if gpuErr == nil {
		gpu, err := newVulkanCorrelator(device, maxW, maxH)
		if err == nil {
			gpu.ownsDevice = true
			return gpu, nil
		}
		_ = device.Close()
		gpuErr = err
	}

	cpu.fallbackReason = gpuErr
	return cpu, nil
}

func newOwnedVulkanCorrelator(maxW, maxH int) (*Correlator, error) {
	device, err := vulki.Open()
	if err != nil {
		return nil, err
	}
	c, err := newVulkanCorrelator(device, maxW, maxH)
	if err != nil {
		_ = device.Close()
		return nil, err
	}
	c.ownsDevice = true
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
