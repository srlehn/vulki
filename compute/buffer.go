package compute

import (
	"fmt"
	"unsafe"

	"github.com/srlehn/vulki/vk"
)

// Buffer wraps a device-local Vulkan buffer with a host-visible staging buffer for transfers.
type Buffer struct {
	DeviceBuffer  vk.Buffer
	DeviceMemory  vk.DeviceMemory
	StagingBuffer vk.Buffer
	StagingMemory vk.DeviceMemory
	Size          uint64
}

// CreateBuffer creates a device-local buffer (for compute) and a host-visible staging buffer (for transfers).
func (c *Context) CreateBuffer(size uint64, extraUsage uint32) (*Buffer, error) {
	b := &Buffer{Size: size}

	// Create device-local buffer.
	deviceBufInfo := vk.BufferCreateInfo{
		SType:       vk.StructureTypeBufferCreateInfo,
		Size:        size,
		Usage:       extraUsage | vk.BufferUsageTransferSrcBit | vk.BufferUsageTransferDstBit,
		SharingMode: vk.SharingModeExclusive,
	}
	var err error
	b.DeviceBuffer, err = c.DevFuncs.CreateBuffer(c.Device, &deviceBufInfo)
	if err != nil {
		return nil, err
	}

	reqs := c.DevFuncs.GetBufferMemoryRequirements(c.Device, b.DeviceBuffer)
	memIdx, err := findMemoryType(c, reqs.MemoryTypeBits, vk.MemoryPropertyDeviceLocalBit)
	if err != nil {
		c.DevFuncs.DestroyBuffer(c.Device, b.DeviceBuffer)
		return nil, fmt.Errorf("compute: no device-local memory: %w", err)
	}

	allocInfo := vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  reqs.Size,
		MemoryTypeIndex: memIdx,
	}
	b.DeviceMemory, err = c.DevFuncs.AllocateMemory(c.Device, &allocInfo)
	if err != nil {
		c.DevFuncs.DestroyBuffer(c.Device, b.DeviceBuffer)
		return nil, err
	}
	if err := c.DevFuncs.BindBufferMemory(c.Device, b.DeviceBuffer, b.DeviceMemory, 0); err != nil {
		b.destroyDevice(c)
		return nil, err
	}

	// Create host-visible staging buffer.
	stagingBufInfo := vk.BufferCreateInfo{
		SType:       vk.StructureTypeBufferCreateInfo,
		Size:        size,
		Usage:       vk.BufferUsageTransferSrcBit | vk.BufferUsageTransferDstBit,
		SharingMode: vk.SharingModeExclusive,
	}
	b.StagingBuffer, err = c.DevFuncs.CreateBuffer(c.Device, &stagingBufInfo)
	if err != nil {
		b.destroyDevice(c)
		return nil, err
	}

	stagingReqs := c.DevFuncs.GetBufferMemoryRequirements(c.Device, b.StagingBuffer)
	stagingMemIdx, err := findMemoryType(c, stagingReqs.MemoryTypeBits, vk.MemoryPropertyHostVisibleBit|vk.MemoryPropertyHostCoherentBit)
	if err != nil {
		c.DevFuncs.DestroyBuffer(c.Device, b.StagingBuffer)
		b.destroyDevice(c)
		return nil, fmt.Errorf("compute: no host-visible memory: %w", err)
	}

	stagingAllocInfo := vk.MemoryAllocateInfo{
		SType:           vk.StructureTypeMemoryAllocateInfo,
		AllocationSize:  stagingReqs.Size,
		MemoryTypeIndex: stagingMemIdx,
	}
	b.StagingMemory, err = c.DevFuncs.AllocateMemory(c.Device, &stagingAllocInfo)
	if err != nil {
		c.DevFuncs.DestroyBuffer(c.Device, b.StagingBuffer)
		b.destroyDevice(c)
		return nil, err
	}
	if err := c.DevFuncs.BindBufferMemory(c.Device, b.StagingBuffer, b.StagingMemory, 0); err != nil {
		b.Destroy(c)
		return nil, err
	}

	return b, nil
}

// Upload copies data from the host to the device-local buffer via the staging buffer.
func (b *Buffer) Upload(ctx *Context, data []byte) error {
	// Map staging memory and copy data in.
	ptr, err := ctx.DevFuncs.MapMemory(ctx.Device, b.StagingMemory, 0, uint64(len(data)))
	if err != nil {
		return err
	}
	dst := unsafe.Slice((*byte)(ptr), len(data))
	copy(dst, data)
	ctx.DevFuncs.UnmapMemory(ctx.Device, b.StagingMemory)

	// Record and submit a copy command.
	return ctx.copyBuffer(b.StagingBuffer, b.DeviceBuffer, uint64(len(data)))
}

// Download copies data from the device-local buffer to the host via the staging buffer.
func (b *Buffer) Download(ctx *Context, size uint64) ([]byte, error) {
	// Record and submit a copy command.
	if err := ctx.copyBuffer(b.DeviceBuffer, b.StagingBuffer, size); err != nil {
		return nil, err
	}

	// Map staging and read out.
	ptr, err := ctx.DevFuncs.MapMemory(ctx.Device, b.StagingMemory, 0, size)
	if err != nil {
		return nil, err
	}
	src := unsafe.Slice((*byte)(ptr), size)
	out := make([]byte, size)
	copy(out, src)
	ctx.DevFuncs.UnmapMemory(ctx.Device, b.StagingMemory)

	return out, nil
}

// Destroy releases all buffer and memory resources.
func (b *Buffer) Destroy(ctx *Context) {
	if b.StagingBuffer != 0 {
		ctx.DevFuncs.DestroyBuffer(ctx.Device, b.StagingBuffer)
		b.StagingBuffer = 0
	}
	if b.StagingMemory != 0 {
		ctx.DevFuncs.FreeMemory(ctx.Device, b.StagingMemory)
		b.StagingMemory = 0
	}
	b.destroyDevice(ctx)
}

func (b *Buffer) destroyDevice(ctx *Context) {
	if b.DeviceBuffer != 0 {
		ctx.DevFuncs.DestroyBuffer(ctx.Device, b.DeviceBuffer)
		b.DeviceBuffer = 0
	}
	if b.DeviceMemory != 0 {
		ctx.DevFuncs.FreeMemory(ctx.Device, b.DeviceMemory)
		b.DeviceMemory = 0
	}
}

// copyBuffer records and submits a single buffer copy command.
func (ctx *Context) copyBuffer(src, dst vk.Buffer, size uint64) error {
	poolInfo := vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: ctx.QueueFamily,
	}
	pool, err := ctx.DevFuncs.CreateCommandPool(ctx.Device, &poolInfo)
	if err != nil {
		return err
	}
	defer ctx.DevFuncs.DestroyCommandPool(ctx.Device, pool)

	allocInfo := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        pool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}
	cbs, err := ctx.DevFuncs.AllocateCommandBuffers(ctx.Device, &allocInfo)
	if err != nil {
		return err
	}
	cb := cbs[0]

	beginInfo := vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
		Flags: vk.CommandBufferUsageOneTimeSubmitBit,
	}
	if err := ctx.DevFuncs.BeginCommandBuffer(cb, &beginInfo); err != nil {
		return err
	}

	region := vk.BufferCopy{Size: size}
	ctx.DevFuncs.CmdCopyBuffer(cb, src, dst, []vk.BufferCopy{region})

	if err := ctx.DevFuncs.EndCommandBuffer(cb); err != nil {
		return err
	}

	fenceInfo := vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}
	fence, err := ctx.DevFuncs.CreateFence(ctx.Device, &fenceInfo)
	if err != nil {
		return err
	}
	defer ctx.DevFuncs.DestroyFence(ctx.Device, fence)

	submitInfo := vk.SubmitInfo{
		SType:              vk.StructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    &cb,
	}
	if err := ctx.DevFuncs.QueueSubmit(ctx.Queue, []vk.SubmitInfo{submitInfo}, fence); err != nil {
		return err
	}

	return ctx.DevFuncs.WaitForFences(ctx.Device, []vk.Fence{fence}, true, ^uint64(0))
}

func findMemoryType(ctx *Context, typeBits uint32, properties vk.MemoryPropertyFlags) (uint32, error) {
	for i := uint32(0); i < ctx.MemProps.MemoryTypeCount; i++ {
		if typeBits&(1<<i) != 0 && ctx.MemProps.MemoryTypes[i].PropertyFlags&properties == properties {
			return i, nil
		}
	}
	return 0, fmt.Errorf("no suitable memory type found")
}
