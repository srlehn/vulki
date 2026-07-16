package vk

import (
	"errors"
	"strings"
	"testing"
)

func TestMemoryPropertyHostCachedBitValue(t *testing.T) {
	if got, want := MemoryPropertyHostCachedBit, MemoryPropertyFlags(0x00000008); got != want {
		t.Fatalf("MemoryPropertyHostCachedBit = %#x, want %#x", uint32(got), uint32(want))
	}
}

func TestMemoryHeapDeviceLocalBitValue(t *testing.T) {
	if got, want := MemoryHeapDeviceLocalBit, uint32(0x00000001); got != want {
		t.Fatalf("MemoryHeapDeviceLocalBit = %#x, want %#x", got, want)
	}
}

func TestResultString(t *testing.T) {
	if got, want := ErrorDeviceLost.String(), "VK_ERROR_DEVICE_LOST"; got != want {
		t.Fatalf("ErrorDeviceLost.String() = %q, want %q", got, want)
	}
	if got, want := Result(-999).String(), "VkResult(-999)"; got != want {
		t.Fatalf("unknown Result.String() = %q, want %q", got, want)
	}
}

func TestStatusErrorDoesNotCallTimeoutFailure(t *testing.T) {
	err := resultError("vkWaitForFences", Timeout)
	if strings.Contains(err.Error(), "failed") {
		t.Fatalf("timeout error = %q, must not call a status a failure", err)
	}
	var vkErr *Error
	if !errors.As(err, &vkErr) || vkErr.Result != Timeout {
		t.Fatalf("timeout error = %v, want inspectable Timeout", err)
	}
}

func TestErrorPreservesResult(t *testing.T) {
	err := resultError("vkQueueSubmit", ErrorDeviceLost)
	var vkErr *Error
	if !errors.As(err, &vkErr) {
		t.Fatalf("error = %v, want *Error", err)
	}
	if vkErr.Op != "vkQueueSubmit" {
		t.Fatalf("operation = %q, want vkQueueSubmit", vkErr.Op)
	}
	if vkErr.Result != ErrorDeviceLost {
		t.Fatalf("result = %v, want %v", vkErr.Result, ErrorDeviceLost)
	}
}
