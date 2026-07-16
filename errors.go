package vulki

import (
	"errors"

	"github.com/srlehn/vulki/vk"
)

// ErrOutOfDeviceMemory matches, via errors.Is, operations that failed with
// Vulkan's VK_ERROR_OUT_OF_DEVICE_MEMORY. It signals recoverable device
// memory pressure: releasing device resources and retrying can succeed.
var ErrOutOfDeviceMemory = errors.New("vulki: out of device memory")

// ErrDeviceLost matches, via errors.Is, operations that failed with Vulkan's
// VK_ERROR_DEVICE_LOST. A lost device does not recover; the caller should
// close it and open a new Device.
var ErrDeviceLost = errors.New("vulki: device lost")

// ErrDeviceUnavailable matches, via errors.Is, submissions refused because an
// earlier submitted fence wait failed and queue completion is unknown. The
// device stays unavailable until it is closed. Device.Err reports the cause.
var ErrDeviceUnavailable = errors.New("vulki: cannot submit while earlier queue completion is unknown")

// classifiedError attaches a sentinel to an underlying error for errors.Is
// without changing the error message.
type classifiedError struct {
	class error
	err   error
}

func (e *classifiedError) Error() string {
	return e.err.Error()
}

func (e *classifiedError) Is(target error) bool {
	return target == e.class
}

func (e *classifiedError) Unwrap() error {
	return e.err
}

// classifyDeviceError attaches the matching exported sentinel to a Vulkan
// failure. Unrecognized and already classified errors pass through unchanged.
func classifyDeviceError(err error) error {
	if err == nil {
		return nil
	}
	var classified *classifiedError
	if errors.As(err, &classified) {
		return err
	}
	var vkError *vk.Error
	if !errors.As(err, &vkError) {
		return err
	}
	switch vkError.Result {
	case vk.ErrorOutOfDeviceMemory:
		return &classifiedError{class: ErrOutOfDeviceMemory, err: err}
	case vk.ErrorDeviceLost:
		return &classifiedError{class: ErrDeviceLost, err: err}
	}
	return err
}
