package vulki

import (
	"fmt"
	"slices"
	"sync"
	"unsafe"

	"github.com/srlehn/vulki/vk"
)

type recorderState uint8

const (
	recorderRecording recorderState = iota + 1
	recorderSubmitted
	recorderAborted
)

// Recorder batches buffer updates, barriers, and dispatches into one queue
// submission. A Recorder is not safe for concurrent method calls.
type Recorder struct {
	mu          sync.Mutex
	device      *Device
	childID     uint64
	pool        vk.CommandPool
	command     vk.CommandBuffer
	state       recorderState
	buffers     []*Buffer
	bindingSets []*BindingSet
	transfers   []recorderTransfer
}

type recorderTransfer struct {
	buffer      vk.Buffer
	memory      vk.DeviceMemory
	destination []byte
}

// NewRecorder begins a command recording owned by d.
func (d *Device) NewRecorder() (*Recorder, error) {
	state, err := d.beginOperation()
	if err != nil {
		return nil, err
	}
	defer d.endOperation()

	poolInfo := vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: state.queueFamily,
	}
	pool, err := state.ops.createCommandPool(state.deviceFns, state.device, &poolInfo)
	if err != nil {
		return nil, fmt.Errorf("vulki: create recorder command pool: %w", err)
	}
	recorder := &Recorder{device: d, pool: pool, state: recorderRecording}
	allocation := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        pool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}
	commands, err := state.ops.allocateCommandBuffers(state.deviceFns, state.device, &allocation)
	if err != nil {
		recorder.closeNative(state)
		return nil, fmt.Errorf("vulki: allocate recorder command buffer: %w", err)
	}
	recorder.command = commands[0]
	begin := vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
		Flags: vk.CommandBufferUsageOneTimeSubmitBit,
	}
	if err := state.ops.beginCommandBuffer(state.deviceFns, recorder.command, &begin); err != nil {
		recorder.closeNative(state)
		return nil, fmt.Errorf("vulki: begin recorder command buffer: %w", err)
	}
	recorder.childID, err = d.addChild(recorder)
	if err != nil {
		recorder.closeNative(state)
		return nil, err
	}
	return recorder, nil
}

// Update records an inline buffer update. Offset and data length must be
// divisible by four, and data must not exceed 65536 bytes.
func (r *Recorder) Update(buffer *Buffer, offset uint64, data []byte) error {
	if r == nil {
		return fmt.Errorf("vulki: recorder is closed")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.requireRecording(); err != nil {
		return err
	}
	if buffer == nil || buffer.device != r.device {
		return fmt.Errorf("vulki: update buffer belongs to another device")
	}
	if len(data) == 0 {
		return nil
	}
	if offset%4 != 0 || len(data)%4 != 0 {
		return fmt.Errorf("vulki: update offset and size must be divisible by four")
	}
	if len(data) > 65536 {
		return fmt.Errorf("vulki: update size %d exceeds 65536 bytes", len(data))
	}
	buffer.mu.Lock()
	if err := buffer.validateRange(offset, len(data)); err != nil {
		buffer.mu.Unlock()
		return err
	}
	handle := buffer.buffer
	r.retainBufferLocked(buffer)
	buffer.mu.Unlock()
	words := make([]uint32, len(data)/4)
	aligned := unsafe.Slice((*byte)(unsafe.Pointer(&words[0])), len(data))
	copy(aligned, data)

	state, err := r.device.beginOperation()
	if err != nil {
		return err
	}
	defer r.device.endOperation()
	pre := vk.BufferMemoryBarrier{
		SType:               vk.StructureTypeBufferMemoryBarrier,
		SrcAccessMask:       vk.AccessShaderReadBit | vk.AccessShaderWriteBit,
		DstAccessMask:       vk.AccessTransferWriteBit,
		SrcQueueFamilyIndex: ^uint32(0), DstQueueFamilyIndex: ^uint32(0),
		Buffer: handle, Offset: offset, Size: uint64(len(data)),
	}
	state.ops.bufferBarriers(
		state.deviceFns, r.command,
		vk.PipelineStageComputeShaderBit, vk.PipelineStageTransferBit,
		[]vk.BufferMemoryBarrier{pre},
	)
	if err := state.ops.updateBuffer(state.deviceFns, r.command, handle, offset, aligned); err != nil {
		return fmt.Errorf("vulki: record buffer update: %w", err)
	}
	post := vk.BufferMemoryBarrier{
		SType:               vk.StructureTypeBufferMemoryBarrier,
		SrcAccessMask:       vk.AccessTransferWriteBit,
		DstAccessMask:       vk.AccessShaderReadBit | vk.AccessShaderWriteBit,
		SrcQueueFamilyIndex: ^uint32(0), DstQueueFamilyIndex: ^uint32(0),
		Buffer: handle, Offset: offset, Size: uint64(len(data)),
	}
	state.ops.bufferBarriers(
		state.deviceFns, r.command,
		vk.PipelineStageTransferBit, vk.PipelineStageComputeShaderBit,
		[]vk.BufferMemoryBarrier{post},
	)
	return nil
}

// Upload records an arbitrary-size host upload to buffer at byte offset. The
// input is copied before Upload returns and may be reused immediately.
func (r *Recorder) Upload(buffer *Buffer, offset uint64, data []byte) error {
	if r == nil {
		return fmt.Errorf("vulki: recorder is closed")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.requireRecording(); err != nil {
		return err
	}
	if buffer == nil || buffer.device != r.device {
		return fmt.Errorf("vulki: upload buffer belongs to another device")
	}
	buffer.mu.Lock()
	defer buffer.mu.Unlock()
	if err := buffer.validateRange(offset, len(data)); err != nil {
		return err
	}
	if len(data) == 0 {
		return nil
	}

	state, err := r.device.beginOperation()
	if err != nil {
		return err
	}
	defer r.device.endOperation()
	transferBuffer, transferMemory, err := createTransferResources(state, uint64(len(data)))
	if err != nil {
		return fmt.Errorf("vulki: prepare recorded upload: %w", err)
	}
	keepTransfer := false
	defer func() {
		if !keepTransfer {
			destroyTransferResources(state, transferBuffer, transferMemory)
		}
	}()
	pointer, err := state.ops.mapMemory(state.deviceFns, state.device, transferMemory, 0, uint64(len(data)))
	if err != nil {
		return fmt.Errorf("vulki: map recorded upload memory: %w", err)
	}
	if pointer == nil {
		state.ops.unmapMemory(state.deviceFns, state.device, transferMemory)
		return fmt.Errorf("vulki: map recorded upload memory returned a nil pointer")
	}
	copy(unsafe.Slice((*byte)(pointer), len(data)), data)
	state.ops.unmapMemory(state.deviceFns, state.device, transferMemory)

	pre := vk.BufferMemoryBarrier{
		SType:               vk.StructureTypeBufferMemoryBarrier,
		SrcAccessMask:       vk.AccessShaderReadBit | vk.AccessShaderWriteBit | vk.AccessTransferWriteBit,
		DstAccessMask:       vk.AccessTransferWriteBit,
		SrcQueueFamilyIndex: ^uint32(0), DstQueueFamilyIndex: ^uint32(0),
		Buffer: buffer.buffer, Offset: offset, Size: uint64(len(data)),
	}
	state.ops.bufferBarriers(
		state.deviceFns, r.command,
		vk.PipelineStageComputeShaderBit|vk.PipelineStageTransferBit,
		vk.PipelineStageTransferBit,
		[]vk.BufferMemoryBarrier{pre},
	)
	state.ops.copyBuffer(state.deviceFns, r.command, transferBuffer, buffer.buffer, []vk.BufferCopy{{
		DstOffset: offset,
		Size:      uint64(len(data)),
	}})
	post := vk.BufferMemoryBarrier{
		SType:               vk.StructureTypeBufferMemoryBarrier,
		SrcAccessMask:       vk.AccessTransferWriteBit,
		DstAccessMask:       vk.AccessShaderReadBit | vk.AccessShaderWriteBit,
		SrcQueueFamilyIndex: ^uint32(0), DstQueueFamilyIndex: ^uint32(0),
		Buffer: buffer.buffer, Offset: offset, Size: uint64(len(data)),
	}
	state.ops.bufferBarriers(
		state.deviceFns, r.command,
		vk.PipelineStageTransferBit, vk.PipelineStageComputeShaderBit,
		[]vk.BufferMemoryBarrier{post},
	)
	r.retainBufferLocked(buffer)
	r.transfers = append(r.transfers, recorderTransfer{buffer: transferBuffer, memory: transferMemory})
	keepTransfer = true
	return nil
}

// Download records a readback from buffer at byte offset. SubmitAndWait fills
// destination after GPU completion. The caller must not access or modify
// destination between Download and the return of SubmitAndWait.
func (r *Recorder) Download(buffer *Buffer, offset uint64, destination []byte) error {
	if r == nil {
		return fmt.Errorf("vulki: recorder is closed")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.requireRecording(); err != nil {
		return err
	}
	if buffer == nil || buffer.device != r.device {
		return fmt.Errorf("vulki: download buffer belongs to another device")
	}
	buffer.mu.Lock()
	defer buffer.mu.Unlock()
	if err := buffer.validateRange(offset, len(destination)); err != nil {
		return err
	}
	if len(destination) == 0 {
		return nil
	}

	state, err := r.device.beginOperation()
	if err != nil {
		return err
	}
	defer r.device.endOperation()
	transferBuffer, transferMemory, err := createTransferResources(state, uint64(len(destination)))
	if err != nil {
		return fmt.Errorf("vulki: prepare recorded download: %w", err)
	}
	pre := vk.BufferMemoryBarrier{
		SType:               vk.StructureTypeBufferMemoryBarrier,
		SrcAccessMask:       vk.AccessShaderWriteBit | vk.AccessTransferWriteBit,
		DstAccessMask:       vk.AccessTransferReadBit,
		SrcQueueFamilyIndex: ^uint32(0), DstQueueFamilyIndex: ^uint32(0),
		Buffer: buffer.buffer, Offset: offset, Size: uint64(len(destination)),
	}
	state.ops.bufferBarriers(
		state.deviceFns, r.command,
		vk.PipelineStageComputeShaderBit|vk.PipelineStageTransferBit,
		vk.PipelineStageTransferBit,
		[]vk.BufferMemoryBarrier{pre},
	)
	state.ops.copyBuffer(state.deviceFns, r.command, buffer.buffer, transferBuffer, []vk.BufferCopy{{
		SrcOffset: offset,
		Size:      uint64(len(destination)),
	}})
	host := vk.BufferMemoryBarrier{
		SType:               vk.StructureTypeBufferMemoryBarrier,
		SrcAccessMask:       vk.AccessTransferWriteBit,
		DstAccessMask:       vk.AccessHostReadBit,
		SrcQueueFamilyIndex: ^uint32(0), DstQueueFamilyIndex: ^uint32(0),
		Buffer: transferBuffer, Size: uint64(len(destination)),
	}
	state.ops.bufferBarriers(
		state.deviceFns, r.command,
		vk.PipelineStageTransferBit, vk.PipelineStageHostBit,
		[]vk.BufferMemoryBarrier{host},
	)
	r.retainBufferLocked(buffer)
	r.transfers = append(r.transfers, recorderTransfer{
		buffer: transferBuffer, memory: transferMemory, destination: destination,
	})
	return nil
}

// Barrier records a compute-to-compute memory dependency for buffers.
func (r *Recorder) Barrier(buffers ...*Buffer) error {
	if r == nil {
		return fmt.Errorf("vulki: recorder is closed")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.requireRecording(); err != nil {
		return err
	}
	if len(buffers) == 0 {
		return nil
	}
	barriers := make([]vk.BufferMemoryBarrier, len(buffers))
	for index, buffer := range buffers {
		if buffer == nil || buffer.device != r.device {
			return fmt.Errorf("vulki: barrier buffer %d belongs to another device", index)
		}
		buffer.mu.Lock()
		if buffer.closed || buffer.buffer == 0 {
			buffer.mu.Unlock()
			return fmt.Errorf("vulki: barrier buffer %d is closed", index)
		}
		handle := buffer.buffer
		r.retainBufferLocked(buffer)
		buffer.mu.Unlock()
		barriers[index] = vk.BufferMemoryBarrier{
			SType:               vk.StructureTypeBufferMemoryBarrier,
			SrcAccessMask:       vk.AccessShaderReadBit | vk.AccessShaderWriteBit,
			DstAccessMask:       vk.AccessShaderReadBit | vk.AccessShaderWriteBit,
			SrcQueueFamilyIndex: ^uint32(0), DstQueueFamilyIndex: ^uint32(0),
			Buffer: handle, Size: vk.WholeSize,
		}
	}
	state, err := r.device.beginOperation()
	if err != nil {
		return err
	}
	defer r.device.endOperation()
	state.ops.bufferBarriers(
		state.deviceFns, r.command,
		vk.PipelineStageComputeShaderBit, vk.PipelineStageComputeShaderBit,
		barriers,
	)
	return nil
}

// Dispatch records one dispatch using kernel and bindings.
func (r *Recorder) Dispatch(kernel *Kernel, bindings *BindingSet, groups Workgroups) error {
	if r == nil {
		return fmt.Errorf("vulki: recorder is closed")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.requireRecording(); err != nil {
		return err
	}
	if kernel == nil || bindings == nil || kernel.device != r.device || bindings.device != r.device || bindings.kernel != kernel {
		return fmt.Errorf("vulki: dispatch resources do not belong to this recorder and kernel")
	}
	if err := r.device.validateWorkgroups(groups); err != nil {
		return err
	}
	bindings.mu.Lock()
	if bindings.closed {
		bindings.mu.Unlock()
		return fmt.Errorf("vulki: binding set is closed")
	}
	state, err := r.device.beginOperation()
	if err != nil {
		bindings.mu.Unlock()
		return err
	}
	defer r.device.endOperation()
	r.retainBindingSet(bindings)
	state.kernelOps.bindPipeline(state.deviceFns, r.command, vk.PipelineBindPointCompute, kernel.pipeline)
	state.kernelOps.bindDescriptorSets(
		state.deviceFns, r.command, vk.PipelineBindPointCompute,
		kernel.pipelineLayout, 0, []vk.DescriptorSet{bindings.set},
	)
	state.kernelOps.dispatch(state.deviceFns, r.command, groups.X, groups.Y, groups.Z)
	bindings.mu.Unlock()
	return nil
}

// SubmitAndWait ends recording, submits once, waits for completion, and closes
// the Recorder. It may be called only while recording.
func (r *Recorder) SubmitAndWait() error {
	if r == nil {
		return fmt.Errorf("vulki: recorder is closed")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.requireRecording(); err != nil {
		return err
	}
	state, err := r.device.beginOperation()
	if err != nil {
		return err
	}
	defer r.device.endOperation()
	if err := state.ops.endCommandBuffer(state.deviceFns, r.command); err != nil {
		r.finish(state, recorderAborted)
		return fmt.Errorf("vulki: end recorder command buffer: %w", err)
	}
	fenceInfo := vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}
	fence, err := state.ops.createFence(state.deviceFns, state.device, &fenceInfo)
	if err != nil {
		r.finish(state, recorderAborted)
		return fmt.Errorf("vulki: create recorder fence: %w", err)
	}
	defer state.ops.destroyFence(state.deviceFns, state.device, fence)
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: &r.command}
	r.device.queueMu.Lock()
	err = state.ops.queueSubmit(state.deviceFns, state.queue, []vk.SubmitInfo{submit}, fence)
	if err == nil {
		err = state.ops.waitForFences(state.deviceFns, state.device, []vk.Fence{fence}, true, ^uint64(0))
	}
	r.device.queueMu.Unlock()
	if err != nil {
		r.finish(state, recorderSubmitted)
		return fmt.Errorf("vulki: submit recorder: %w", err)
	}
	err = r.readDownloads(state)
	r.finish(state, recorderSubmitted)
	if err != nil {
		return err
	}
	return nil
}

// Abort discards an unsubmitted recording and releases its resources.
// Repeated calls return nil.
func (r *Recorder) Abort() error {
	return r.abort(false)
}

// Close is equivalent to Abort for a recording and is otherwise a no-op.
func (r *Recorder) Close() error {
	return r.Abort()
}

func (r *Recorder) closeFromDevice() error {
	return r.abort(true)
}

func (r *Recorder) abort(fromDevice bool) error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.state != recorderRecording {
		return nil
	}
	var state *deviceState
	if fromDevice {
		r.device.mu.Lock()
		state = r.device.state
		r.device.mu.Unlock()
		if state == nil {
			return fmt.Errorf("vulki: recorder owner is closed")
		}
	} else {
		var err error
		state, err = r.device.beginOperation()
		if err != nil {
			return err
		}
		defer r.device.endOperation()
	}
	r.finish(state, recorderAborted)
	return nil
}

func (r *Recorder) requireRecording() error {
	if r.state != recorderRecording {
		return fmt.Errorf("vulki: recorder is not recording")
	}
	return nil
}

func (r *Recorder) retainBufferLocked(buffer *Buffer) {
	if slices.Contains(r.buffers, buffer) {
		return
	}
	buffer.references++
	r.buffers = append(r.buffers, buffer)
}

func (r *Recorder) retainBindingSet(set *BindingSet) {
	if slices.Contains(r.bindingSets, set) {
		return
	}
	set.recorders++
	r.bindingSets = append(r.bindingSets, set)
}

func (r *Recorder) finish(state *deviceState, final recorderState) {
	r.closeNative(state)
	r.releaseReferences()
	r.state = final
	r.device.removeChild(r.childID)
}

func (r *Recorder) closeNative(state *deviceState) {
	if r.pool != 0 {
		state.ops.destroyCommandPool(state.deviceFns, state.device, r.pool)
		r.pool = 0
		r.command = 0
	}
	for index := len(r.transfers) - 1; index >= 0; index-- {
		transfer := r.transfers[index]
		destroyTransferResources(state, transfer.buffer, transfer.memory)
	}
	r.transfers = nil
}

func (r *Recorder) readDownloads(state *deviceState) error {
	for _, transfer := range r.transfers {
		if transfer.destination == nil {
			continue
		}
		size := uint64(len(transfer.destination))
		pointer, err := state.ops.mapMemory(state.deviceFns, state.device, transfer.memory, 0, size)
		if err != nil {
			return fmt.Errorf("vulki: map recorded download memory: %w", err)
		}
		if pointer == nil {
			state.ops.unmapMemory(state.deviceFns, state.device, transfer.memory)
			return fmt.Errorf("vulki: map recorded download memory returned a nil pointer")
		}
		copy(transfer.destination, unsafe.Slice((*byte)(pointer), len(transfer.destination)))
		state.ops.unmapMemory(state.deviceFns, state.device, transfer.memory)
	}
	return nil
}

func (r *Recorder) releaseReferences() {
	for _, buffer := range r.buffers {
		buffer.mu.Lock()
		if buffer.references > 0 {
			buffer.references--
		}
		buffer.mu.Unlock()
	}
	r.buffers = nil
	for _, set := range r.bindingSets {
		set.mu.Lock()
		if set.recorders > 0 {
			set.recorders--
		}
		set.mu.Unlock()
	}
	r.bindingSets = nil
}
