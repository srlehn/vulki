package vulki

import "fmt"

const (
	maxIdleTransferResources = 16
	maxIdleTransferBytes     = 64 * 1024 * 1024
)

func (d *Device) acquireTransfer(state *deviceState, direction transferDirection, size uint64) (*transferResource, error) {
	if d == nil || state == nil || !direction.valid() || size == 0 {
		return nil, fmt.Errorf("vulki: invalid transfer resource request")
	}

	d.transferMu.Lock()
	idle := d.idleTransfers[direction]
	best := -1
	for index, resource := range idle {
		if resource == nil || resource.capacity < size {
			continue
		}
		if best < 0 || resource.capacity < idle[best].capacity {
			best = index
		}
	}
	if best >= 0 {
		resource := idle[best]
		last := len(idle) - 1
		idle[best] = idle[last]
		idle[last] = nil
		d.idleTransfers[direction] = idle[:last]
		d.idleTransferBytes -= resource.retainedSize()
		d.transferMu.Unlock()
		return resource, nil
	}
	d.transferMu.Unlock()

	return createTransferResource(state, direction, size)
}

func (d *Device) releaseTransfer(state *deviceState, resource *transferResource) {
	if resource == nil {
		return
	}

	d.transferMu.Lock()
	retainedSize := resource.retainedSize()
	keep := resource.direction.valid() &&
		d.idleTransferCountLocked() < maxIdleTransferResources &&
		retainedSize <= maxIdleTransferBytes &&
		d.idleTransferBytes <= maxIdleTransferBytes-retainedSize
	if keep {
		d.idleTransfers[resource.direction] = append(d.idleTransfers[resource.direction], resource)
		d.idleTransferBytes += retainedSize
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
	resources := make([]*transferResource, 0, d.idleTransferCountLocked())
	for direction := transferUpload; direction < transferDirectionCount; direction++ {
		resources = append(resources, d.idleTransfers[direction]...)
		d.idleTransfers[direction] = nil
	}
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
	if resource.buffer != 0 {
		state.ops.destroyBuffer(state.deviceFns, state.device, resource.buffer)
	}
	if resource.memory != 0 {
		state.ops.freeMemory(state.deviceFns, state.device, resource.memory)
	}
	resource.buffer = 0
	resource.memory = 0
	resource.capacity = 0
	resource.allocationSize = 0
	resource.properties = 0
	resource.direction = transferDirectionCount
}

func (d *Device) idleTransferCountLocked() int {
	count := 0
	for direction := transferUpload; direction < transferDirectionCount; direction++ {
		count += len(d.idleTransfers[direction])
	}
	return count
}
