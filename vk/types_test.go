package vk

import (
	"errors"
	"testing"
)

func TestResultString(t *testing.T) {
	if got, want := ErrorDeviceLost.String(), "VK_ERROR_DEVICE_LOST"; got != want {
		t.Fatalf("ErrorDeviceLost.String() = %q, want %q", got, want)
	}
	if got, want := Result(-999).String(), "VkResult(-999)"; got != want {
		t.Fatalf("unknown Result.String() = %q, want %q", got, want)
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
