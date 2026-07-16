package vulki

import (
	"fmt"
	"unsafe"

	"github.com/srlehn/vulki/vk"
)

type transferDirection uint8

const (
	transferUpload transferDirection = iota
	transferDownload
	transferDirectionCount
)

func (direction transferDirection) valid() bool {
	return direction < transferDirectionCount
}

func (direction transferDirection) String() string {
	switch direction {
	case transferUpload:
		return "upload"
	case transferDownload:
		return "download"
	default:
		return fmt.Sprintf("transfer direction %d", direction)
	}
}

// transferResource is exclusively owned by a Buffer, an active Recorder, or
// the device's matching-direction idle pool. It must not return to the pool
// while submitted commands may still reference it.
type transferResource struct {
	buffer         vk.Buffer
	memory         vk.DeviceMemory
	capacity       uint64
	allocationSize uint64
	properties     vk.MemoryPropertyFlags
	direction      transferDirection
}

func (resource *transferResource) retainedSize() uint64 {
	if resource == nil {
		return 0
	}
	if resource.allocationSize != 0 {
		return resource.allocationSize
	}
	return resource.capacity
}

func createTransferResource(state *deviceState, direction transferDirection, size uint64) (*transferResource, error) {
	if state == nil || !direction.valid() || size == 0 {
		return nil, fmt.Errorf("vulki: invalid %s transfer resource request", direction)
	}
	usage := vk.BufferUsageTransferSrcBit
	if direction == transferDownload {
		usage = vk.BufferUsageTransferDstBit
	}
	info := vk.BufferCreateInfo{
		SType:       vk.StructureTypeBufferCreateInfo,
		Size:        size,
		Usage:       usage,
		SharingMode: vk.SharingModeExclusive,
	}
	buffer, err := state.ops.createBuffer(state.deviceFns, state.device, &info)
	if err != nil {
		return nil, fmt.Errorf("vulki: create %s transfer buffer: %w", direction, classifyDeviceError(err))
	}
	requirements := state.ops.bufferMemoryRequirements(state.deviceFns, state.device, buffer)
	memoryIndex, properties, err := selectTransferMemoryType(state.memory, requirements.MemoryTypeBits, direction)
	if err != nil {
		state.ops.destroyBuffer(state.deviceFns, state.device, buffer)
		return nil, err
	}
	allocation := vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  requirements.Size,
		MemoryTypeIndex: memoryIndex,
	}
	memory, err := state.ops.allocateMemory(state.deviceFns, state.device, &allocation)
	if err != nil {
		state.ops.destroyBuffer(state.deviceFns, state.device, buffer)
		return nil, fmt.Errorf("vulki: allocate %s transfer memory: %w", direction, classifyDeviceError(err))
	}
	if err := state.ops.bindBufferMemory(state.deviceFns, state.device, buffer, memory, 0); err != nil {
		state.ops.destroyBuffer(state.deviceFns, state.device, buffer)
		state.ops.freeMemory(state.deviceFns, state.device, memory)
		return nil, fmt.Errorf("vulki: bind %s transfer memory: %w", direction, classifyDeviceError(err))
	}
	return &transferResource{
		buffer:         buffer,
		memory:         memory,
		capacity:       size,
		allocationSize: requirements.Size,
		properties:     properties,
		direction:      direction,
	}, nil
}

func selectTransferMemoryType(
	properties vk.PhysicalDeviceMemoryProperties,
	typeBits uint32,
	direction transferDirection,
) (uint32, vk.MemoryPropertyFlags, error) {
	if !direction.valid() {
		return 0, 0, fmt.Errorf("vulki: invalid transfer direction %d", direction)
	}
	count := min(properties.MemoryTypeCount, uint32(len(properties.MemoryTypes)))
	bestIndex := uint32(0)
	bestRank := int(^uint(0) >> 1)
	found := false
	for index := range count {
		if typeBits&(uint32(1)<<index) == 0 {
			continue
		}
		flags := properties.MemoryTypes[index].PropertyFlags
		rank, compatible := transferMemoryRank(flags, direction)
		if !compatible || (found && rank >= bestRank) {
			continue
		}
		bestIndex = index
		bestRank = rank
		found = true
	}
	if !found {
		return 0, 0, fmt.Errorf(
			"vulki: no compatible %s transfer memory type for type bits %#x",
			direction,
			typeBits,
		)
	}
	return bestIndex, properties.MemoryTypes[bestIndex].PropertyFlags, nil
}

func transferMemoryRank(flags vk.MemoryPropertyFlags, direction transferDirection) (int, bool) {
	if flags&vk.MemoryPropertyHostVisibleBit == 0 {
		return 0, false
	}
	coherent := flags&vk.MemoryPropertyHostCoherentBit != 0
	cached := flags&vk.MemoryPropertyHostCachedBit != 0
	switch direction {
	case transferUpload:
		if !coherent {
			return 0, false
		}
		if !cached {
			return 0, true
		}
		return 1, true
	case transferDownload:
		switch {
		case cached && coherent:
			return 0, true
		case cached:
			return 1, true
		case coherent:
			return 2, true
		default:
			return 0, false
		}
	default:
		return 0, false
	}
}

func readTransferMemory(
	state *deviceState,
	resource *transferResource,
	offset uint64,
	destination []byte,
) error {
	if len(destination) == 0 {
		return nil
	}
	if state == nil || resource == nil || resource.direction != transferDownload {
		return fmt.Errorf("invalid download transfer resource")
	}
	length := uint64(len(destination))
	if offset > resource.allocationSize || length > resource.allocationSize-offset {
		return fmt.Errorf(
			"download range offset %d size %d exceeds allocation size %d",
			offset,
			length,
			resource.allocationSize,
		)
	}
	if uint64(uintptr(offset)) != offset {
		return fmt.Errorf("download offset %d exceeds the platform pointer range", offset)
	}
	pointer, err := state.ops.mapMemory(
		state.deviceFns,
		state.device,
		resource.memory,
		0,
		resource.allocationSize,
	)
	if err != nil {
		return fmt.Errorf("map download transfer memory: %w", err)
	}
	if pointer == nil {
		state.ops.unmapMemory(state.deviceFns, state.device, resource.memory)
		return fmt.Errorf("map download transfer memory returned a nil pointer")
	}
	defer state.ops.unmapMemory(state.deviceFns, state.device, resource.memory)

	if resource.properties&vk.MemoryPropertyHostCoherentBit == 0 {
		invalidateOffset, invalidateSize, err := nonCoherentRange(
			offset,
			length,
			resource.allocationSize,
			state.nonCoherentAtomSize,
		)
		if err != nil {
			return err
		}
		if state.ops.invalidateMappedMemoryRanges == nil {
			return fmt.Errorf("invalidate download transfer memory: operation unavailable")
		}
		ranges := []vk.MappedMemoryRange{{
			SType:  vk.StructureTypeMappedMemoryRange,
			Memory: resource.memory,
			Offset: invalidateOffset,
			Size:   invalidateSize,
		}}
		if err := state.ops.invalidateMappedMemoryRanges(state.deviceFns, state.device, ranges); err != nil {
			return fmt.Errorf("invalidate download transfer memory: %w", err)
		}
	}

	copy(destination, unsafe.Slice((*byte)(unsafe.Add(pointer, uintptr(offset))), len(destination)))
	return nil
}

func nonCoherentRange(offset, size, allocationSize, atomSize uint64) (uint64, uint64, error) {
	if size == 0 {
		return 0, 0, fmt.Errorf("noncoherent range size must be greater than zero")
	}
	if offset > allocationSize || size > allocationSize-offset {
		return 0, 0, fmt.Errorf(
			"noncoherent range offset %d size %d exceeds allocation size %d",
			offset,
			size,
			allocationSize,
		)
	}
	if atomSize == 0 {
		atomSize = 1
	}
	start := offset - offset%atomSize
	end := offset + size
	alignedEnd := end
	if remainder := end % atomSize; remainder != 0 {
		increase := atomSize - remainder
		if increase > allocationSize-end {
			alignedEnd = allocationSize
		} else {
			alignedEnd += increase
		}
	}
	return start, alignedEnd - start, nil
}
