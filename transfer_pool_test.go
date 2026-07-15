package vulki

import (
	"sync"
	"testing"

	"github.com/srlehn/vulki/vk"
)

func TestTransferPoolLeasesExclusivelyAndReusesBestFit(t *testing.T) {
	device, _, createCount := fakeBufferDevice("")
	state := device.state

	small, err := device.acquireTransfer(state, transferUpload, 64)
	if err != nil {
		t.Fatalf("acquire small transfer: %v", err)
	}
	large, err := device.acquireTransfer(state, transferUpload, 128)
	if err != nil {
		t.Fatalf("acquire large transfer: %v", err)
	}
	if small.buffer == large.buffer || small.memory == large.memory {
		t.Fatal("simultaneous leases share native resources")
	}
	device.releaseTransfer(state, large)
	device.releaseTransfer(state, small)

	first, err := device.acquireTransfer(state, transferUpload, 32)
	if err != nil {
		t.Fatalf("reacquire first transfer: %v", err)
	}
	if first != small {
		t.Fatalf("best-fit lease = %p, want small resource %p", first, small)
	}
	second, err := device.acquireTransfer(state, transferUpload, 32)
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
	for _, resource := range []*transferResource{small, large} {
		if resource.buffer != 0 || resource.memory != 0 || resource.capacity != 0 ||
			resource.allocationSize != 0 || resource.properties != 0 || resource.direction.valid() {
			t.Fatalf("discarded resource retained metadata: %#v", resource)
		}
	}
}

func TestTransferPoolDoesNotReuseUndersizedResource(t *testing.T) {
	device, _, createCount := fakeBufferDevice("")
	state := device.state

	small, err := device.acquireTransfer(state, transferUpload, 16)
	if err != nil {
		t.Fatalf("acquire small transfer: %v", err)
	}
	device.releaseTransfer(state, small)
	larger, err := device.acquireTransfer(state, transferUpload, 32)
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

func TestTransferPoolDoesNotReuseAcrossDirections(t *testing.T) {
	device, _, createCount := fakeBufferDevice("")
	state := device.state

	upload, err := device.acquireTransfer(state, transferUpload, 64)
	if err != nil {
		t.Fatalf("acquire upload transfer: %v", err)
	}
	device.releaseTransfer(state, upload)
	download, err := device.acquireTransfer(state, transferDownload, 64)
	if err != nil {
		t.Fatalf("acquire download transfer: %v", err)
	}
	if download == upload || download.buffer == upload.buffer || download.memory == upload.memory {
		t.Fatal("download reused an upload transfer resource")
	}
	if got := *createCount; got != 2 {
		t.Fatalf("transfer buffer creations = %d, want 2", got)
	}
	device.releaseTransfer(state, download)

	reusedUpload, err := device.acquireTransfer(state, transferUpload, 32)
	if err != nil {
		t.Fatalf("reacquire upload transfer: %v", err)
	}
	reusedDownload, err := device.acquireTransfer(state, transferDownload, 32)
	if err != nil {
		t.Fatalf("reacquire download transfer: %v", err)
	}
	if reusedUpload != upload || reusedDownload != download {
		t.Fatalf("directional reuse = upload %p download %p, want %p and %p",
			reusedUpload, reusedDownload, upload, download)
	}

	device.releaseTransfer(state, reusedUpload)
	device.releaseTransfer(state, reusedDownload)
	device.drainTransferPool(state)
}

func TestTransferPoolBoundsIdleRetention(t *testing.T) {
	device, events, _ := fakeBufferDevice("")
	state := device.state
	resources := make([]*transferResource, maxIdleTransferResources+1)
	for index := range resources {
		var err error
		direction := transferDirection(index % int(transferDirectionCount))
		resources[index], err = device.acquireTransfer(state, direction, 1)
		if err != nil {
			t.Fatalf("acquire transfer %d: %v", index, err)
		}
	}
	for _, resource := range resources {
		device.releaseTransfer(state, resource)
	}
	if got := idleTransferCountForTest(device); got != maxIdleTransferResources {
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
	resource, err := device.acquireTransfer(state, transferUpload, maxIdleTransferBytes+1)
	if err != nil {
		t.Fatalf("acquire oversized transfer: %v", err)
	}
	device.releaseTransfer(state, resource)
	if idleTransferCountForTest(device) != 0 || device.idleTransferBytes != 0 {
		t.Fatal("oversized transfer was retained")
	}
	if got := len(*events); got != 2 {
		t.Fatalf("cleanup events = %v, want one buffer and memory pair", *events)
	}
}

func TestTransferPoolAppliesByteBudgetAcrossDirections(t *testing.T) {
	device, events, _ := fakeBufferDevice("")
	state := device.state
	halfBudget := uint64(maxIdleTransferBytes / 2)
	upload, err := device.acquireTransfer(state, transferUpload, halfBudget)
	if err != nil {
		t.Fatalf("acquire upload transfer: %v", err)
	}
	download, err := device.acquireTransfer(state, transferDownload, halfBudget)
	if err != nil {
		t.Fatalf("acquire download transfer: %v", err)
	}
	overflow, err := device.acquireTransfer(state, transferUpload, 1)
	if err != nil {
		t.Fatalf("acquire overflow transfer: %v", err)
	}
	device.releaseTransfer(state, upload)
	device.releaseTransfer(state, download)
	if got := device.idleTransferBytes; got != maxIdleTransferBytes {
		t.Fatalf("idle transfer bytes = %d, want %d", got, maxIdleTransferBytes)
	}

	device.releaseTransfer(state, overflow)
	if got := idleTransferCountForTest(device); got != 2 {
		t.Fatalf("idle transfers = %d, want the two budget-filling resources", got)
	}
	if got := len(*events); got != 2 {
		t.Fatalf("overflow cleanup events = %v, want one buffer and memory pair", *events)
	}
	device.drainTransferPool(state)
}

func TestTransferPoolConcurrentLeasesRemainExclusive(t *testing.T) {
	const resourceCount = 16
	device := &Device{}
	state := &deviceState{}
	for index := range resourceCount {
		device.idleTransfers[transferUpload] = append(
			device.idleTransfers[transferUpload],
			&transferResource{
				buffer:         vk.Buffer(index + 1),
				memory:         vk.DeviceMemory(index + 1),
				capacity:       1,
				allocationSize: 1,
				direction:      transferUpload,
			},
		)
		device.idleTransferBytes++
	}

	leases := make(chan *transferResource, resourceCount)
	start := make(chan struct{})
	var wait sync.WaitGroup
	for range resourceCount {
		wait.Add(1)
		go func() {
			defer wait.Done()
			<-start
			resource, err := device.acquireTransfer(state, transferUpload, 1)
			if err != nil {
				t.Errorf("acquire transfer: %v", err)
				return
			}
			leases <- resource
		}()
	}
	close(start)
	wait.Wait()
	close(leases)

	seen := make(map[*transferResource]bool, resourceCount)
	for resource := range leases {
		if seen[resource] {
			t.Fatalf("resource %p was leased concurrently more than once", resource)
		}
		seen[resource] = true
	}
	if len(seen) != resourceCount {
		t.Fatalf("leased resources = %d, want %d", len(seen), resourceCount)
	}
	if got := idleTransferCountForTest(device); got != 0 {
		t.Fatalf("idle transfers during exclusive leases = %d, want 0", got)
	}

	for resource := range seen {
		device.releaseTransfer(state, resource)
	}
	if got := idleTransferCountForTest(device); got != resourceCount {
		t.Fatalf("idle transfers after release = %d, want %d", got, resourceCount)
	}
}

func TestDeviceCloseDrainsTransferPool(t *testing.T) {
	device, events, _ := fakeBufferDevice("")
	device.state.hooks.destroyDevice = func(*vk.DeviceFuncs, vk.Device) {
		*events = append(*events, "destroy-device")
	}
	resource, err := device.acquireTransfer(device.state, transferUpload, 64)
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

func idleTransferCountForTest(device *Device) int {
	device.transferMu.Lock()
	defer device.transferMu.Unlock()
	return device.idleTransferCountLocked()
}
