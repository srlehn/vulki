package vulki

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/srlehn/vulki/vk"
)

// Buffer is a fixed-size storage buffer owned by its creating Device.
type Buffer struct {
	mu              sync.Mutex
	device          *Device
	childID         uint64
	size            uint64
	buffer          vk.Buffer
	memory          vk.DeviceMemory
	uploadStaging   *transferResource
	downloadStaging *transferResource
	references      int
	closed          bool
}

type deviceOps struct {
	createBuffer                 func(*vk.DeviceFuncs, vk.Device, *vk.BufferCreateInfo) (vk.Buffer, error)
	destroyBuffer                func(*vk.DeviceFuncs, vk.Device, vk.Buffer)
	bufferMemoryRequirements     func(*vk.DeviceFuncs, vk.Device, vk.Buffer) vk.MemoryRequirements
	allocateMemory               func(*vk.DeviceFuncs, vk.Device, *vk.MemoryAllocateInfo) (vk.DeviceMemory, error)
	freeMemory                   func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory)
	bindBufferMemory             func(*vk.DeviceFuncs, vk.Device, vk.Buffer, vk.DeviceMemory, uint64) error
	mapMemory                    func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error)
	unmapMemory                  func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory)
	invalidateMappedMemoryRanges func(*vk.DeviceFuncs, vk.Device, []vk.MappedMemoryRange) error
	createCommandPool            func(*vk.DeviceFuncs, vk.Device, *vk.CommandPoolCreateInfo) (vk.CommandPool, error)
	destroyCommandPool           func(*vk.DeviceFuncs, vk.Device, vk.CommandPool)
	allocateCommandBuffers       func(*vk.DeviceFuncs, vk.Device, *vk.CommandBufferAllocateInfo) ([]vk.CommandBuffer, error)
	beginCommandBuffer           func(*vk.DeviceFuncs, vk.CommandBuffer, *vk.CommandBufferBeginInfo) error
	bufferBarriers               func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineStageFlags, vk.PipelineStageFlags, []vk.BufferMemoryBarrier)
	copyBuffer                   func(*vk.DeviceFuncs, vk.CommandBuffer, vk.Buffer, vk.Buffer, []vk.BufferCopy)
	endCommandBuffer             func(*vk.DeviceFuncs, vk.CommandBuffer) error
	createFence                  func(*vk.DeviceFuncs, vk.Device, *vk.FenceCreateInfo) (vk.Fence, error)
	destroyFence                 func(*vk.DeviceFuncs, vk.Device, vk.Fence)
	queueSubmit                  func(*vk.DeviceFuncs, vk.Queue, []vk.SubmitInfo, vk.Fence) error
	waitForFences                func(*vk.DeviceFuncs, vk.Device, []vk.Fence, bool, uint64) error
	updateBuffer                 func(*vk.DeviceFuncs, vk.CommandBuffer, vk.Buffer, uint64, []byte) error
}

var directDeviceOps = deviceOps{
	createBuffer: func(functions *vk.DeviceFuncs, device vk.Device, info *vk.BufferCreateInfo) (vk.Buffer, error) {
		return functions.CreateBuffer(device, info)
	},
	destroyBuffer: func(functions *vk.DeviceFuncs, device vk.Device, buffer vk.Buffer) {
		functions.DestroyBuffer(device, buffer)
	},
	bufferMemoryRequirements: func(functions *vk.DeviceFuncs, device vk.Device, buffer vk.Buffer) vk.MemoryRequirements {
		return functions.GetBufferMemoryRequirements(device, buffer)
	},
	allocateMemory: func(functions *vk.DeviceFuncs, device vk.Device, info *vk.MemoryAllocateInfo) (vk.DeviceMemory, error) {
		return functions.AllocateMemory(device, info)
	},
	freeMemory: func(functions *vk.DeviceFuncs, device vk.Device, memory vk.DeviceMemory) {
		functions.FreeMemory(device, memory)
	},
	bindBufferMemory: func(functions *vk.DeviceFuncs, device vk.Device, buffer vk.Buffer, memory vk.DeviceMemory, offset uint64) error {
		return functions.BindBufferMemory(device, buffer, memory, offset)
	},
	mapMemory: func(functions *vk.DeviceFuncs, device vk.Device, memory vk.DeviceMemory, offset, size uint64) (unsafe.Pointer, error) {
		return functions.MapMemory(device, memory, offset, size)
	},
	unmapMemory: func(functions *vk.DeviceFuncs, device vk.Device, memory vk.DeviceMemory) {
		functions.UnmapMemory(device, memory)
	},
	invalidateMappedMemoryRanges: func(functions *vk.DeviceFuncs, device vk.Device, ranges []vk.MappedMemoryRange) error {
		return functions.InvalidateMappedMemoryRanges(device, ranges)
	},
	createCommandPool: func(functions *vk.DeviceFuncs, device vk.Device, info *vk.CommandPoolCreateInfo) (vk.CommandPool, error) {
		return functions.CreateCommandPool(device, info)
	},
	destroyCommandPool: func(functions *vk.DeviceFuncs, device vk.Device, pool vk.CommandPool) {
		functions.DestroyCommandPool(device, pool)
	},
	allocateCommandBuffers: func(functions *vk.DeviceFuncs, device vk.Device, info *vk.CommandBufferAllocateInfo) ([]vk.CommandBuffer, error) {
		return functions.AllocateCommandBuffers(device, info)
	},
	beginCommandBuffer: func(functions *vk.DeviceFuncs, buffer vk.CommandBuffer, info *vk.CommandBufferBeginInfo) error {
		return functions.BeginCommandBuffer(buffer, info)
	},
	bufferBarriers: func(functions *vk.DeviceFuncs, command vk.CommandBuffer, source, destination vk.PipelineStageFlags, barriers []vk.BufferMemoryBarrier) {
		functions.CmdPipelineBarrierBuffers(command, source, destination, barriers)
	},
	copyBuffer: func(functions *vk.DeviceFuncs, command vk.CommandBuffer, source, destination vk.Buffer, regions []vk.BufferCopy) {
		functions.CmdCopyBuffer(command, source, destination, regions)
	},
	endCommandBuffer: func(functions *vk.DeviceFuncs, buffer vk.CommandBuffer) error {
		return functions.EndCommandBuffer(buffer)
	},
	createFence: func(functions *vk.DeviceFuncs, device vk.Device, info *vk.FenceCreateInfo) (vk.Fence, error) {
		return functions.CreateFence(device, info)
	},
	destroyFence: func(functions *vk.DeviceFuncs, device vk.Device, fence vk.Fence) {
		functions.DestroyFence(device, fence)
	},
	queueSubmit: func(functions *vk.DeviceFuncs, queue vk.Queue, submits []vk.SubmitInfo, fence vk.Fence) error {
		return functions.QueueSubmit(queue, submits, fence)
	},
	waitForFences: func(functions *vk.DeviceFuncs, device vk.Device, fences []vk.Fence, waitAll bool, timeout uint64) error {
		return functions.WaitForFences(device, fences, waitAll, timeout)
	},
	updateBuffer: func(functions *vk.DeviceFuncs, command vk.CommandBuffer, buffer vk.Buffer, offset uint64, data []byte) error {
		return functions.CmdUpdateBuffer(command, buffer, offset, data)
	},
}

// NewBuffer creates a fixed-size storage buffer. Host-transfer resources are
// allocated lazily on the first upload or download.
func (d *Device) NewBuffer(size uint64) (*Buffer, error) {
	if size == 0 {
		return nil, fmt.Errorf("vulki: buffer size must be greater than zero")
	}
	limit := d.Info().Limits.MaxStorageBufferSize
	if limit != 0 && size > limit {
		return nil, fmt.Errorf("vulki: buffer size %d exceeds storage-buffer limit %d", size, limit)
	}
	state, err := d.beginOperation()
	if err != nil {
		return nil, err
	}
	defer d.endOperation()

	buffer := &Buffer{device: d, size: size}
	info := vk.BufferCreateInfo{
		SType:       vk.StructureTypeBufferCreateInfo,
		Size:        size,
		Usage:       vk.BufferUsageStorageBufferBit | vk.BufferUsageTransferSrcBit | vk.BufferUsageTransferDstBit,
		SharingMode: vk.SharingModeExclusive,
	}
	buffer.buffer, err = state.ops.createBuffer(state.deviceFns, state.device, &info)
	if err != nil {
		return nil, fmt.Errorf("vulki: create storage buffer: %w", err)
	}

	requirements := state.ops.bufferMemoryRequirements(state.deviceFns, state.device, buffer.buffer)
	memoryIndex, err := findMemoryType(state.memory, requirements.MemoryTypeBits, vk.MemoryPropertyDeviceLocalBit)
	if err != nil {
		buffer.closeNative(state)
		return nil, fmt.Errorf("vulki: select device-local buffer memory: %w", err)
	}
	allocation := vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  requirements.Size,
		MemoryTypeIndex: memoryIndex,
	}
	buffer.memory, err = state.ops.allocateMemory(state.deviceFns, state.device, &allocation)
	if err != nil {
		buffer.closeNative(state)
		return nil, fmt.Errorf("vulki: allocate storage buffer memory: %w", err)
	}
	if err := state.ops.bindBufferMemory(state.deviceFns, state.device, buffer.buffer, buffer.memory, 0); err != nil {
		buffer.closeNative(state)
		return nil, fmt.Errorf("vulki: bind storage buffer memory: %w", err)
	}

	buffer.childID, err = d.addChild(buffer)
	if err != nil {
		buffer.closeNative(state)
		return nil, err
	}
	return buffer, nil
}

// Size returns the fixed buffer size in bytes. It returns zero for a nil
// Buffer.
func (b *Buffer) Size() uint64 {
	if b == nil {
		return 0
	}
	return b.size
}

// Buffer transfers deliberately use Upload and Download rather than the io
// interfaces. A Buffer has no stream cursor, uses uint64 resource offsets, and
// reports completion for the entire requested GPU transfer instead of a
// partial byte count. The At variants use absolute byte offsets.

// Upload copies data to the beginning of the buffer and blocks until the copy
// completes.
func (b *Buffer) Upload(data []byte) error {
	return b.UploadAt(0, data)
}

// UploadAt copies data to byte offset and blocks until the copy completes.
func (b *Buffer) UploadAt(offset uint64, data []byte) error {
	if b == nil {
		return fmt.Errorf("vulki: buffer is closed")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.validateRange(offset, len(data)); err != nil {
		return err
	}
	if len(data) == 0 {
		return nil
	}
	state, err := b.device.beginOperation()
	if err != nil {
		return err
	}
	defer b.device.endOperation()
	staging, err := b.ensureStaging(state, transferUpload)
	if err != nil {
		return err
	}

	pointer, err := state.ops.mapMemory(state.deviceFns, state.device, staging.memory, 0, uint64(len(data)))
	if err != nil {
		return fmt.Errorf("vulki: map upload staging memory: %w", err)
	}
	if pointer == nil {
		state.ops.unmapMemory(state.deviceFns, state.device, staging.memory)
		return fmt.Errorf("vulki: map upload staging memory returned a nil pointer")
	}
	copy(unsafe.Slice((*byte)(pointer), len(data)), data)
	state.ops.unmapMemory(state.deviceFns, state.device, staging.memory)
	barrier := vk.BufferMemoryBarrier{
		SType:               vk.StructureTypeBufferMemoryBarrier,
		SrcAccessMask:       vk.AccessShaderReadBit | vk.AccessShaderWriteBit | vk.AccessTransferWriteBit,
		DstAccessMask:       vk.AccessTransferWriteBit,
		SrcQueueFamilyIndex: ^uint32(0),
		DstQueueFamilyIndex: ^uint32(0),
		Buffer:              b.buffer,
		Offset:              offset,
		Size:                uint64(len(data)),
	}
	if err := b.device.copyBufferAndWait(
		state,
		b,
		submissionWrite,
		staging.buffer,
		b.buffer,
		0,
		offset,
		uint64(len(data)),
		vk.PipelineStageComputeShaderBit|vk.PipelineStageTransferBit,
		vk.PipelineStageTransferBit,
		barrier,
	); err != nil {
		return fmt.Errorf("vulki: upload buffer: %w", err)
	}
	return nil
}

// Download copies bytes from the beginning of the buffer into destination and
// blocks until the copy completes.
func (b *Buffer) Download(destination []byte) error {
	return b.DownloadAt(0, destination)
}

// DownloadAt copies bytes from byte offset into destination and blocks until
// the copy completes.
func (b *Buffer) DownloadAt(offset uint64, destination []byte) error {
	if b == nil {
		return fmt.Errorf("vulki: buffer is closed")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.validateRange(offset, len(destination)); err != nil {
		return err
	}
	if len(destination) == 0 {
		return nil
	}
	state, err := b.device.beginOperation()
	if err != nil {
		return err
	}
	defer b.device.endOperation()
	staging, err := b.ensureStaging(state, transferDownload)
	if err != nil {
		return err
	}
	barrier := vk.BufferMemoryBarrier{
		SType:               vk.StructureTypeBufferMemoryBarrier,
		SrcAccessMask:       vk.AccessShaderWriteBit | vk.AccessTransferWriteBit,
		DstAccessMask:       vk.AccessTransferReadBit,
		SrcQueueFamilyIndex: ^uint32(0),
		DstQueueFamilyIndex: ^uint32(0),
		Buffer:              b.buffer,
		Offset:              offset,
		Size:                uint64(len(destination)),
	}
	if err := b.device.copyBufferAndWait(
		state,
		b,
		submissionRead,
		b.buffer,
		staging.buffer,
		offset,
		0,
		uint64(len(destination)),
		vk.PipelineStageComputeShaderBit|vk.PipelineStageTransferBit,
		vk.PipelineStageTransferBit,
		barrier,
	); err != nil {
		return fmt.Errorf("vulki: download buffer: %w", err)
	}

	if err := readTransferMemory(state, staging, 0, destination); err != nil {
		return fmt.Errorf("vulki: read download staging memory: %w", err)
	}
	return nil
}

// Bytes allocates and downloads the complete buffer contents.
func (b *Buffer) Bytes() ([]byte, error) {
	if b == nil {
		return nil, fmt.Errorf("vulki: buffer is closed")
	}
	if b.size > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("vulki: buffer size %d exceeds the platform slice limit", b.size)
	}
	contents := make([]byte, int(b.size))
	if err := b.Download(contents); err != nil {
		return nil, err
	}
	return contents, nil
}

// Close releases the buffer and any lazily allocated transfer resources.
// Repeated calls return nil.
func (b *Buffer) Close() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	if b.references > 0 {
		return fmt.Errorf("vulki: buffer is referenced by %d live resources", b.references)
	}
	if b.device == nil && b.buffer == 0 && b.memory == 0 {
		b.closed = true
		return nil
	}
	state, err := b.device.beginOperation()
	if err != nil {
		return err
	}
	b.closeNative(state)
	b.closed = true
	b.device.endOperation()
	b.device.removeChild(b.childID)
	return nil
}

func (b *Buffer) closeFromDevice() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	b.device.mu.Lock()
	state := b.device.state
	b.device.mu.Unlock()
	if state == nil {
		return fmt.Errorf("vulki: buffer owner is closed")
	}
	b.closeNative(state)
	b.closed = true
	return nil
}

func (b *Buffer) validateRange(offset uint64, length int) error {
	if b.closed {
		return fmt.Errorf("vulki: buffer is closed")
	}
	if length < 0 || offset > b.size || uint64(length) > b.size-offset {
		return fmt.Errorf("vulki: buffer range offset %d size %d exceeds buffer size %d", offset, length, b.size)
	}
	return nil
}

func (b *Buffer) ensureStaging(state *deviceState, direction transferDirection) (*transferResource, error) {
	var staging **transferResource
	switch direction {
	case transferUpload:
		staging = &b.uploadStaging
	case transferDownload:
		staging = &b.downloadStaging
	default:
		return nil, fmt.Errorf("vulki: invalid transfer direction %d", direction)
	}
	if *staging != nil {
		return *staging, nil
	}
	resource, err := createTransferResource(state, direction, b.size)
	if err != nil {
		return nil, err
	}
	*staging = resource
	return resource, nil
}

func (b *Buffer) closeNative(state *deviceState) {
	discardTransferResource(state, b.downloadStaging)
	b.downloadStaging = nil
	discardTransferResource(state, b.uploadStaging)
	b.uploadStaging = nil
	if b.buffer != 0 {
		state.ops.destroyBuffer(state.deviceFns, state.device, b.buffer)
		b.buffer = 0
	}
	if b.memory != 0 {
		state.ops.freeMemory(state.deviceFns, state.device, b.memory)
		b.memory = 0
	}
}

func (d *Device) copyBufferAndWait(
	state *deviceState,
	trackedBuffer *Buffer,
	access submissionAccess,
	source, destination vk.Buffer,
	sourceOffset, destinationOffset, size uint64,
	sourceStage, destinationStage vk.PipelineStageFlags,
	barrier vk.BufferMemoryBarrier,
) error {
	poolInfo := vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: state.queueFamily,
	}
	pool, err := state.ops.createCommandPool(state.deviceFns, state.device, &poolInfo)
	if err != nil {
		return err
	}
	defer func() {
		if pool != 0 {
			state.ops.destroyCommandPool(state.deviceFns, state.device, pool)
		}
	}()

	allocation := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        pool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}
	commands, err := state.ops.allocateCommandBuffers(state.deviceFns, state.device, &allocation)
	if err != nil {
		return err
	}
	command := commands[0]
	begin := vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
		Flags: vk.CommandBufferUsageOneTimeSubmitBit,
	}
	if err := state.ops.beginCommandBuffer(state.deviceFns, command, &begin); err != nil {
		return err
	}
	state.ops.bufferBarriers(
		state.deviceFns,
		command,
		sourceStage,
		destinationStage,
		[]vk.BufferMemoryBarrier{barrier},
	)
	state.ops.copyBuffer(state.deviceFns, command, source, destination, []vk.BufferCopy{{
		SrcOffset: sourceOffset,
		DstOffset: destinationOffset,
		Size:      size,
	}})
	if err := state.ops.endCommandBuffer(state.deviceFns, command); err != nil {
		return err
	}

	fenceInfo := vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}
	fence, err := state.ops.createFence(state.deviceFns, state.device, &fenceInfo)
	if err != nil {
		return err
	}
	defer func() {
		if fence != 0 {
			state.ops.destroyFence(state.deviceFns, state.device, fence)
		}
	}()
	submit := vk.SubmitInfo{
		SType:              vk.StructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    &command,
	}

	submission := &transientSubmission{device: d, pool: pool, fence: fence}
	pool = 0
	fence = 0
	submission.retainBufferLocked(trackedBuffer)
	resources := submissionResources{{buffer: trackedBuffer, access: access}}
	submission.reservation, err = d.acquireSubmission(resources)
	if err != nil {
		submission.releaseBufferLocked(trackedBuffer)
		submission.cleanup(state)
		return err
	}
	if err := d.submitQueue(state, []vk.SubmitInfo{submit}, submission.fence); err != nil {
		submission.releaseBufferLocked(trackedBuffer)
		submission.cleanup(state)
		return err
	}
	if err := state.ops.waitForFences(
		state.deviceFns,
		state.device,
		[]vk.Fence{submission.fence},
		true,
		^uint64(0),
	); err != nil {
		d.retainUnknownTransientSubmission(submission, err)
		return err
	}
	submission.releaseBufferLocked(trackedBuffer)
	submission.cleanup(state)
	return nil
}

func findMemoryType(properties vk.PhysicalDeviceMemoryProperties, typeBits uint32, required vk.MemoryPropertyFlags) (uint32, error) {
	count := min(properties.MemoryTypeCount, uint32(len(properties.MemoryTypes)))
	for index := range count {
		memoryType := properties.MemoryTypes[index]
		if typeBits&(1<<index) != 0 && memoryType.PropertyFlags&required == required {
			return index, nil
		}
	}
	return 0, fmt.Errorf("no memory type satisfies properties %#x", uint32(required))
}
