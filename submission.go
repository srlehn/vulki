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
	cause = classifyDeviceError(cause)
	d.submissionMu.Lock()
	d.submissionErr = errors.Join(d.submissionErr, cause)
	if d.submissionCond != nil {
		d.submissionCond.Broadcast()
	}
	d.submissionMu.Unlock()
}

func (d *Device) submissionUnavailableErrorLocked() error {
	return fmt.Errorf("%w: %w", ErrDeviceUnavailable, d.submissionErr)
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
	return classifyDeviceError(state.ops.queueSubmit(state.deviceFns, state.queue, submits, fence))
}

type submissionState uint8

const (
	submissionPending submissionState = iota + 1
	submissionUnknown
	submissionDone
)

// Submission tracks one in-flight queue submission created by Recorder.Submit
// or Device.Submit. Completion must be observed through Wait or Poll before
// recorded download destinations are valid. Wait and Poll are safe for
// concurrent use; once completion is observed the result is final and repeated
// calls return it unchanged. A Submission that is never observed keeps its
// resources retained until the Device is closed.
type Submission struct {
	mu          sync.Mutex
	device      *Device
	childID     uint64
	fence       vk.Fence
	reservation *submissionReservation
	recorders   []*Recorder
	state       submissionState
	result      error
}

// Submit ends recording on every recorder and submits all recorded batches in
// argument order as one queue submission, without waiting for completion.
// Batches execute in submission order on the shared queue, observing the
// barriers they recorded. On success the recorders accept no further commands
// and the returned Submission owns completion tracking. On error every
// recorder is aborted and its resources are released.
//
// Each recorder must belong to this device, appear only once, and not be used
// concurrently by another goroutine, as documented on Recorder.
func (d *Device) Submit(recorders ...*Recorder) (*Submission, error) {
	if len(recorders) == 0 {
		return nil, fmt.Errorf("vulki: submit requires at least one recorder")
	}
	for index, recorder := range recorders {
		if recorder == nil || recorder.device != d {
			return nil, fmt.Errorf("vulki: submit recorder %d does not belong to this device", index)
		}
		for _, earlier := range recorders[:index] {
			if earlier == recorder {
				return nil, fmt.Errorf("vulki: submit recorder %d is duplicated", index)
			}
		}
	}
	state, err := d.beginOperation()
	if err != nil {
		return nil, err
	}
	defer d.endOperation()
	for _, recorder := range recorders {
		recorder.mu.Lock()
	}
	defer func() {
		for _, recorder := range recorders {
			recorder.mu.Unlock()
		}
	}()
	for _, recorder := range recorders {
		if err := recorder.requireRecording(); err != nil {
			return nil, err
		}
		if recorder.tsOpen {
			return nil, fmt.Errorf("vulki: timestamp span %q is not ended",
				recorder.tsLabels[len(recorder.tsLabels)-1])
		}
	}
	abortAll := func() {
		for _, recorder := range recorders {
			recorder.finish(state, recorderAborted, recorderRecycleTransfers)
		}
	}
	for _, recorder := range recorders {
		if err := state.ops.endCommandBuffer(state.deviceFns, recorder.command); err != nil {
			abortAll()
			return nil, fmt.Errorf("vulki: end recorder command buffer: %w", err)
		}
	}
	fenceInfo := vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}
	fence, err := state.ops.createFence(state.deviceFns, state.device, &fenceInfo)
	if err != nil {
		abortAll()
		return nil, fmt.Errorf("vulki: create submission fence: %w", err)
	}
	var merged submissionResources
	for _, recorder := range recorders {
		merged.merge(recorder.resources)
	}
	reservation, err := d.acquireSubmission(merged)
	if err != nil {
		state.ops.destroyFence(state.deviceFns, state.device, fence)
		abortAll()
		return nil, fmt.Errorf("vulki: reserve submission resources: %w", err)
	}
	commands := make([]vk.CommandBuffer, len(recorders))
	for index, recorder := range recorders {
		commands[index] = recorder.command
	}
	submit := vk.SubmitInfo{
		SType:              vk.StructureTypeSubmitInfo,
		CommandBufferCount: uint32(len(commands)),
		PCommandBuffers:    &commands[0],
	}
	if err := d.submitQueue(state, []vk.SubmitInfo{submit}, fence); err != nil {
		d.releaseSubmission(reservation)
		state.ops.destroyFence(state.deviceFns, state.device, fence)
		abortAll()
		return nil, fmt.Errorf("vulki: submit recorders: %w", err)
	}
	for _, recorder := range recorders {
		recorder.state = recorderCompletionUnknown
	}
	submission := &Submission{
		device:      d,
		fence:       fence,
		reservation: reservation,
		recorders:   append([]*Recorder(nil), recorders...),
		state:       submissionPending,
	}
	submission.childID, err = d.addChild(submission)
	if err != nil {
		// The batch is already on the queue and can no longer be tracked, so
		// retain the fence and reservation until device-idle cleanup.
		d.retainUnknownTransientSubmission(&transientSubmission{
			device:      d,
			fence:       fence,
			reservation: reservation,
		}, err)
		return nil, err
	}
	return submission, nil
}

// Wait blocks until the submission completes, fills recorded download
// destinations, releases the submitted recorders, and returns the submission
// result. If completion cannot be established, the affected resources remain
// retained until the Device is closed and later submissions are refused.
func (s *Submission) Wait() error {
	_, err := s.observe(true)
	return err
}

// Poll reports whether the submission has completed, without blocking. When it
// returns true the submission result is final and recorded download
// destinations are filled.
func (s *Submission) Poll() (bool, error) {
	return s.observe(false)
}

func (s *Submission) observe(block bool) (bool, error) {
	if s == nil {
		return false, fmt.Errorf("vulki: submission is not tracked")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.state == submissionDone || s.state == submissionUnknown {
		return true, s.result
	}
	state, err := s.device.beginOperation()
	if err != nil {
		return false, err
	}
	defer s.device.endOperation()
	timeout := uint64(0)
	if block {
		timeout = ^uint64(0)
	}
	err = state.ops.waitForFences(state.deviceFns, state.device, []vk.Fence{s.fence}, true, timeout)
	if err != nil {
		var vkErr *vk.Error
		if !block && errors.As(err, &vkErr) &&
			(vkErr.Result == vk.Timeout || vkErr.Result == vk.NotReady) {
			return false, nil
		}
		err = classifyDeviceError(err)
		s.device.markSubmissionUnknown(s.reservation, err)
		s.state = submissionUnknown
		s.result = fmt.Errorf("vulki: wait for submission completion: %w", err)
		return true, s.result
	}
	s.device.releaseSubmission(s.reservation)
	s.reservation = nil
	var firstErr error
	for _, recorder := range s.recorders {
		if err := recorder.completeSubmitted(state); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	s.recorders = nil
	state.ops.destroyFence(state.deviceFns, state.device, s.fence)
	s.fence = 0
	s.state = submissionDone
	s.result = firstErr
	s.device.removeChild(s.childID)
	return true, firstErr
}

func (s *Submission) closeFromDevice() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.state == submissionDone {
		return nil
	}
	s.device.mu.Lock()
	state := s.device.state
	s.device.mu.Unlock()
	if state == nil {
		return fmt.Errorf("vulki: submission owner is closed")
	}
	s.device.releaseSubmission(s.reservation)
	s.reservation = nil
	if s.fence != 0 {
		state.ops.destroyFence(state.deviceFns, state.device, s.fence)
		s.fence = 0
	}
	if s.state == submissionPending {
		s.result = fmt.Errorf("vulki: device closed before submission completion was observed")
	}
	s.state = submissionDone
	s.recorders = nil
	return nil
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
