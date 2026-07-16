package vulki

import (
	"errors"
	"fmt"
	"sync"

	"github.com/srlehn/vulki/vk"
)

type submissionAccess uint8

const (
	submissionRead submissionAccess = iota + 1
	submissionWrite
)

type submissionResource struct {
	buffer *Buffer
	access submissionAccess
}

type submissionResources []submissionResource

func (resources *submissionResources) add(buffer *Buffer, access submissionAccess) {
	if buffer == nil {
		return
	}
	for index := range *resources {
		resource := &(*resources)[index]
		if resource.buffer != buffer {
			continue
		}
		if access == submissionWrite {
			resource.access = submissionWrite
		}
		return
	}
	*resources = append(*resources, submissionResource{buffer: buffer, access: access})
}

func (resources *submissionResources) merge(other submissionResources) {
	for _, resource := range other {
		resources.add(resource.buffer, resource.access)
	}
}

func submissionResourcesConflict(left, right submissionResources) bool {
	for _, leftResource := range left {
		for _, rightResource := range right {
			if leftResource.buffer == rightResource.buffer &&
				(leftResource.access == submissionWrite || rightResource.access == submissionWrite) {
				return true
			}
		}
	}
	return false
}

type submissionReservation struct {
	resources submissionResources
	active    bool
}

func (d *Device) acquireSubmission(resources submissionResources) (*submissionReservation, error) {
	reservation := &submissionReservation{resources: resources}
	d.submissionMu.Lock()
	if d.submissionCond == nil {
		d.submissionCond = sync.NewCond(&d.submissionMu)
	}
	if d.submissionErr != nil {
		err := d.submissionUnavailableErrorLocked()
		d.submissionMu.Unlock()
		return nil, err
	}
	d.pendingSubmissions = append(d.pendingSubmissions, reservation)
	for {
		if d.submissionErr != nil {
			d.pendingSubmissions = removeSubmissionReservation(d.pendingSubmissions, reservation)
			err := d.submissionUnavailableErrorLocked()
			d.submissionCond.Broadcast()
			d.submissionMu.Unlock()
			return nil, err
		}
		if d.canActivateSubmissionLocked(reservation) {
			d.pendingSubmissions = removeSubmissionReservation(d.pendingSubmissions, reservation)
			reservation.active = true
			d.activeSubmissions = append(d.activeSubmissions, reservation)
			d.submissionMu.Unlock()
			return reservation, nil
		}
		d.submissionCond.Wait()
	}
}

func (d *Device) canActivateSubmissionLocked(candidate *submissionReservation) bool {
	for _, active := range d.activeSubmissions {
		if submissionResourcesConflict(active.resources, candidate.resources) {
			return false
		}
	}
	for _, pending := range d.pendingSubmissions {
		if pending == candidate {
			break
		}
		if submissionResourcesConflict(pending.resources, candidate.resources) {
			return false
		}
	}
	return true
}

func (d *Device) releaseSubmission(reservation *submissionReservation) {
	if d == nil || reservation == nil {
		return
	}
	d.submissionMu.Lock()
	if reservation.active {
		d.activeSubmissions = removeSubmissionReservation(d.activeSubmissions, reservation)
		reservation.active = false
		if d.submissionCond != nil {
			d.submissionCond.Broadcast()
		}
	}
	d.submissionMu.Unlock()
}

func (d *Device) markSubmissionUnknown(reservation *submissionReservation, cause error) {
	if d == nil || reservation == nil {
		return
	}
	if cause == nil {
		cause = errors.New("submission completion is unknown")
	}
	d.submissionMu.Lock()
	d.submissionErr = errors.Join(d.submissionErr, cause)
	if d.submissionCond != nil {
		d.submissionCond.Broadcast()
	}
	d.submissionMu.Unlock()
}

func (d *Device) submissionUnavailableErrorLocked() error {
	return fmt.Errorf("vulki: cannot submit while earlier queue completion is unknown: %w", d.submissionErr)
}

func removeSubmissionReservation(
	reservations []*submissionReservation,
	target *submissionReservation,
) []*submissionReservation {
	for index, reservation := range reservations {
		if reservation != target {
			continue
		}
		copy(reservations[index:], reservations[index+1:])
		reservations[len(reservations)-1] = nil
		return reservations[:len(reservations)-1]
	}
	return reservations
}

// submitQueue serializes only host access to the shared Vulkan queue. Callers
// wait on their exclusively owned fences after this method releases queueMu.
func (d *Device) submitQueue(
	state *deviceState,
	submits []vk.SubmitInfo,
	fence vk.Fence,
) error {
	d.queueMu.Lock()
	defer d.queueMu.Unlock()
	return state.ops.queueSubmit(state.deviceFns, state.queue, submits, fence)
}

type transientSubmission struct {
	device      *Device
	pool        vk.CommandPool
	fence       vk.Fence
	reservation *submissionReservation
	buffer      *Buffer
	bindings    *BindingSet
}

func (submission *transientSubmission) retainBufferLocked(buffer *Buffer) {
	buffer.references++
	submission.buffer = buffer
}

func (submission *transientSubmission) releaseBufferLocked(buffer *Buffer) {
	if submission.buffer != buffer {
		return
	}
	if buffer.references > 0 {
		buffer.references--
	}
	submission.buffer = nil
}

func (submission *transientSubmission) retainBindingsLocked(bindings *BindingSet) {
	bindings.recorders++
	submission.bindings = bindings
}

// cleanup destroys the native resources, releases the reservation, and drops
// retained references. It locks the retained buffer and binding set, so a
// caller holding one of those mutexes must release that reference first.
func (submission *transientSubmission) cleanup(state *deviceState) {
	if submission == nil {
		return
	}
	if submission.fence != 0 {
		state.ops.destroyFence(state.deviceFns, state.device, submission.fence)
		submission.fence = 0
	}
	if submission.pool != 0 {
		state.ops.destroyCommandPool(state.deviceFns, state.device, submission.pool)
		submission.pool = 0
	}
	submission.device.releaseSubmission(submission.reservation)
	submission.reservation = nil
	if submission.bindings != nil {
		submission.bindings.mu.Lock()
		if submission.bindings.recorders > 0 {
			submission.bindings.recorders--
		}
		submission.bindings.mu.Unlock()
		submission.bindings = nil
	}
	if submission.buffer != nil {
		submission.buffer.mu.Lock()
		if submission.buffer.references > 0 {
			submission.buffer.references--
		}
		submission.buffer.mu.Unlock()
		submission.buffer = nil
	}
}

func (d *Device) retainUnknownTransientSubmission(
	submission *transientSubmission,
	cause error,
) {
	d.markSubmissionUnknown(submission.reservation, cause)
	d.submissionMu.Lock()
	d.unknownTransientSubmissions = append(d.unknownTransientSubmissions, submission)
	d.submissionMu.Unlock()
}

func (d *Device) cleanupUnknownTransientSubmissions(state *deviceState) {
	d.submissionMu.Lock()
	submissions := d.unknownTransientSubmissions
	d.unknownTransientSubmissions = nil
	d.submissionMu.Unlock()
	for index := len(submissions) - 1; index >= 0; index-- {
		submissions[index].cleanup(state)
	}
}
