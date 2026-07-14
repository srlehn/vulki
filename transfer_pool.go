package vulki

import (
	"fmt"

	"github.com/srlehn/vulki/vk"
)

const (
	maxIdleTransferResources = 16
	maxIdleTransferBytes     = 64 * 1024 * 1024
)

// transferResource is exclusively owned by its current recorder or by the
// device's idle pool. It must not return to the pool while submitted commands
// may still reference it.
type transferResource struct {
	buffer   vk.Buffer
	memory   vk.DeviceMemory
	capacity uint64
}

func (d *Device) acquireTransfer(state *deviceState, size uint64) (*transferResource, error) {
	if d == nil || state == nil || size == 0 {
		return nil, fmt.Errorf("vulki: invalid transfer resource request")
	}

	d.transferMu.Lock()
	best := -1
	for index, resource := range d.idleTransfers {
		if resource == nil || resource.capacity < size {
			continue
		}
		if best < 0 || resource.capacity < d.idleTransfers[best].capacity {
			best = index
		}
	}
	if best >= 0 {
		resource := d.idleTransfers[best]
		last := len(d.idleTransfers) - 1
		d.idleTransfers[best] = d.idleTransfers[last]
		d.idleTransfers[last] = nil
		d.idleTransfers = d.idleTransfers[:last]
		d.idleTransferBytes -= resource.capacity
		d.transferMu.Unlock()
		return resource, nil
	}
	d.transferMu.Unlock()

	buffer, memory, err := createTransferResources(state, size)
	if err != nil {
		return nil, err
	}
	return &transferResource{buffer: buffer, memory: memory, capacity: size}, nil
}

func (d *Device) releaseTransfer(state *deviceState, resource *transferResource) {
	if resource == nil {
		return
	}

	d.transferMu.Lock()
	keep := len(d.idleTransfers) < maxIdleTransferResources &&
		resource.capacity <= maxIdleTransferBytes &&
		d.idleTransferBytes <= maxIdleTransferBytes-resource.capacity
	if keep {
		d.idleTransfers = append(d.idleTransfers, resource)
		d.idleTransferBytes += resource.capacity
	}
	d.transferMu.Unlock()
	if !keep {
		discardTransferResource(state, resource)
	}
}

func (d *Device) discardTransfer(state *deviceState, resource *transferResource) {
	if resource == nil {
		return
	}
	discardTransferResource(state, resource)
}

func (d *Device) drainTransferPool(state *deviceState) {
	if d == nil || state == nil {
		return
	}
	d.transferMu.Lock()
	resources := d.idleTransfers
	d.idleTransfers = nil
	d.idleTransferBytes = 0
	d.transferMu.Unlock()
	for index := len(resources) - 1; index >= 0; index-- {
		discardTransferResource(state, resources[index])
	}
}

func discardTransferResource(state *deviceState, resource *transferResource) {
	if state == nil || resource == nil {
		return
	}
	destroyTransferResources(state, resource.buffer, resource.memory)
	resource.buffer = 0
	resource.memory = 0
	resource.capacity = 0
}
