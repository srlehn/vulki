package vulki

import (
	"testing"

	"github.com/srlehn/vulki/vk"
)

func TestTransferPoolLeasesExclusivelyAndReusesBestFit(t *testing.T) {
	device, _, createCount := fakeBufferDevice("")
	state := device.state

	small, err := device.acquireTransfer(state, 64)
	if err != nil {
		t.Fatalf("acquire small transfer: %v", err)
	}
	large, err := device.acquireTransfer(state, 128)
	if err != nil {
		t.Fatalf("acquire large transfer: %v", err)
	}
	if small.buffer == large.buffer || small.memory == large.memory {
		t.Fatal("simultaneous leases share native resources")
	}
	device.releaseTransfer(state, large)
	device.releaseTransfer(state, small)

	first, err := device.acquireTransfer(state, 32)
	if err != nil {
		t.Fatalf("reacquire first transfer: %v", err)
	}
	if first != small {
		t.Fatalf("best-fit lease = %p, want small resource %p", first, small)
	}
	second, err := device.acquireTransfer(state, 32)
	if err != nil {
		t.Fatalf("reacquire second transfer: %v", err)
	}
	if second != large {
		t.Fatalf("second lease = %p, want remaining resource %p", second, large)
	}
	if got := *createCount; got != 2 {
		t.Fatalf("transfer buffer creations = %d, want 2", got)
	}

	device.releaseTransfer(state, first)
	device.releaseTransfer(state, second)
	device.drainTransferPool(state)
}

func TestTransferPoolDoesNotReuseUndersizedResource(t *testing.T) {
	device, _, createCount := fakeBufferDevice("")
	state := device.state

	small, err := device.acquireTransfer(state, 16)
	if err != nil {
		t.Fatalf("acquire small transfer: %v", err)
	}
	device.releaseTransfer(state, small)
	larger, err := device.acquireTransfer(state, 32)
	if err != nil {
		t.Fatalf("acquire larger transfer: %v", err)
	}
	if larger == small {
		t.Fatal("undersized resource was reused")
	}
	if got := *createCount; got != 2 {
		t.Fatalf("transfer buffer creations = %d, want 2", got)
	}

	device.releaseTransfer(state, larger)
	device.drainTransferPool(state)
}

func TestTransferPoolBoundsIdleRetention(t *testing.T) {
	device, events, _ := fakeBufferDevice("")
	state := device.state
	resources := make([]*transferResource, maxIdleTransferResources+1)
	for index := range resources {
		var err error
		resources[index], err = device.acquireTransfer(state, 1)
		if err != nil {
			t.Fatalf("acquire transfer %d: %v", index, err)
		}
	}
	for _, resource := range resources {
		device.releaseTransfer(state, resource)
	}
	if got := len(device.idleTransfers); got != maxIdleTransferResources {
		t.Fatalf("idle transfers = %d, want %d", got, maxIdleTransferResources)
	}
	if got := device.idleTransferBytes; got != maxIdleTransferResources {
		t.Fatalf("idle transfer bytes = %d, want %d", got, maxIdleTransferResources)
	}
	if got := len(*events); got != 2 {
		t.Fatalf("overflow cleanup events = %v, want one buffer and memory pair", *events)
	}

	device.drainTransferPool(state)
	if got := len(*events); got != 2*(maxIdleTransferResources+1) {
		t.Fatalf("total cleanup event count = %d, want %d", got, 2*(maxIdleTransferResources+1))
	}
}

func TestTransferPoolRejectsResourceAboveByteBudget(t *testing.T) {
	device, events, _ := fakeBufferDevice("")
	state := device.state
	resource, err := device.acquireTransfer(state, maxIdleTransferBytes+1)
	if err != nil {
		t.Fatalf("acquire oversized transfer: %v", err)
	}
	device.releaseTransfer(state, resource)
	if len(device.idleTransfers) != 0 || device.idleTransferBytes != 0 {
		t.Fatal("oversized transfer was retained")
	}
	if got := len(*events); got != 2 {
		t.Fatalf("cleanup events = %v, want one buffer and memory pair", *events)
	}
}

func TestDeviceCloseDrainsTransferPool(t *testing.T) {
	device, events, _ := fakeBufferDevice("")
	device.state.hooks.destroyDevice = func(*vk.DeviceFuncs, vk.Device) {
		*events = append(*events, "destroy-device")
	}
	resource, err := device.acquireTransfer(device.state, 64)
	if err != nil {
		t.Fatalf("acquire transfer: %v", err)
	}
	device.releaseTransfer(device.state, resource)
	if err := device.Close(); err != nil {
		t.Fatalf("Device.Close: %v", err)
	}
	want := []string{"destroy-buffer:1", "free-memory:1", "destroy-device"}
	if len(*events) != len(want) {
		t.Fatalf("close events = %v, want %v", *events, want)
	}
	for index := range want {
		if (*events)[index] != want[index] {
			t.Fatalf("close events = %v, want %v", *events, want)
		}
	}
}
