package vk

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"
)

// DeviceFuncs holds device-level Vulkan function pointers.
type DeviceFuncs struct {
	getDeviceQueue               func(device Device, queueFamilyIndex, queueIndex uint32, pQueue *Queue)
	createBuffer                 func(device Device, pCreateInfo *BufferCreateInfo, pAllocator uintptr, pBuffer *Buffer) Result
	destroyBuffer                func(device Device, buffer Buffer, pAllocator uintptr)
	getBufferMemoryRequirements  func(device Device, buffer Buffer, pReqs *MemoryRequirements)
	allocateMemory               func(device Device, pAllocateInfo *MemoryAllocateInfo, pAllocator uintptr, pMemory *DeviceMemory) Result
	freeMemory                   func(device Device, memory DeviceMemory, pAllocator uintptr)
	bindBufferMemory             func(device Device, buffer Buffer, memory DeviceMemory, offset uint64) Result
	mapMemory                    func(device Device, memory DeviceMemory, offset, size uint64, flags uint32, ppData *unsafe.Pointer) Result
	unmapMemory                  func(device Device, memory DeviceMemory)
	createShaderModule           func(device Device, pCreateInfo *ShaderModuleCreateInfo, pAllocator uintptr, pModule *ShaderModule) Result
	destroyShaderModule          func(device Device, module ShaderModule, pAllocator uintptr)
	createDescriptorSetLayout    func(device Device, pCreateInfo *DescriptorSetLayoutCreateInfo, pAllocator uintptr, pLayout *DescriptorSetLayout) Result
	destroyDescriptorSetLayout   func(device Device, layout DescriptorSetLayout, pAllocator uintptr)
	createPipelineLayout         func(device Device, pCreateInfo *PipelineLayoutCreateInfo, pAllocator uintptr, pLayout *PipelineLayout) Result
	destroyPipelineLayout        func(device Device, layout PipelineLayout, pAllocator uintptr)
	createComputePipelines       func(device Device, pipelineCache uintptr, createInfoCount uint32, pCreateInfos *ComputePipelineCreateInfo, pAllocator uintptr, pPipelines *Pipeline) Result
	destroyPipeline              func(device Device, pipeline Pipeline, pAllocator uintptr)
	createDescriptorPool         func(device Device, pCreateInfo *DescriptorPoolCreateInfo, pAllocator uintptr, pPool *DescriptorPool) Result
	destroyDescriptorPool        func(device Device, pool DescriptorPool, pAllocator uintptr)
	allocateDescriptorSets       func(device Device, pAllocateInfo *DescriptorSetAllocateInfo, pSets *DescriptorSet) Result
	updateDescriptorSets         func(device Device, writeCount uint32, pWrites *WriteDescriptorSet, copyCount uint32, pCopies uintptr)
	createCommandPool            func(device Device, pCreateInfo *CommandPoolCreateInfo, pAllocator uintptr, pPool *CommandPool) Result
	destroyCommandPool           func(device Device, pool CommandPool, pAllocator uintptr)
	allocateCommandBuffers       func(device Device, pAllocateInfo *CommandBufferAllocateInfo, pBuffers *CommandBuffer) Result
	beginCommandBuffer           func(cb CommandBuffer, pBeginInfo *CommandBufferBeginInfo) Result
	endCommandBuffer             func(cb CommandBuffer) Result
	cmdBindPipeline              func(cb CommandBuffer, bindPoint PipelineBindPoint, pipeline Pipeline)
	cmdBindDescriptorSets        func(cb CommandBuffer, bindPoint PipelineBindPoint, layout PipelineLayout, firstSet, setCount uint32, pSets *DescriptorSet, dynOffsetCount uint32, pDynOffsets uintptr)
	cmdDispatch                  func(cb CommandBuffer, groupCountX, groupCountY, groupCountZ uint32)
	cmdCopyBuffer                func(cb CommandBuffer, src, dst Buffer, regionCount uint32, pRegions *BufferCopy)
	cmdPipelineBarrier           func(cb CommandBuffer, srcStage, dstStage PipelineStageFlags, dependencyFlags uint32, memBarrierCount uint32, pMemBarriers *MemoryBarrier, bufBarrierCount uint32, pBufBarriers *BufferMemoryBarrier, imgBarrierCount uint32, pImgBarriers uintptr)
	createFence                  func(device Device, pCreateInfo *FenceCreateInfo, pAllocator uintptr, pFence *Fence) Result
	destroyFence                 func(device Device, fence Fence, pAllocator uintptr)
	waitForFences                func(device Device, fenceCount uint32, pFences *Fence, waitAll uint32, timeout uint64) Result
	resetFences                  func(device Device, fenceCount uint32, pFences *Fence) Result
	resetCommandBuffer           func(cb CommandBuffer, flags CommandBufferResetFlags) Result
	queueSubmit                  func(queue Queue, submitCount uint32, pSubmits *SubmitInfo, fence Fence) Result
	flushMappedMemoryRanges      func(device Device, rangeCount uint32, pRanges *MappedMemoryRange) Result
	invalidateMappedMemoryRanges func(device Device, rangeCount uint32, pRanges *MappedMemoryRange) Result
	deviceWaitIdle               func(device Device) Result
	cmdUpdateBuffer              func(cb CommandBuffer, dst Buffer, offset uint64, dataSize uint64, pData unsafe.Pointer)
	destroyDevice                func(device Device, pAllocator uintptr)
}

// LoadDeviceFuncs resolves device-level functions via vkGetDeviceProcAddr. If
// resolution fails after vkDestroyDevice was loaded, it returns a non-nil table
// with the error so the caller can destroy its device.
func LoadDeviceFuncs(instFuncs *InstanceFuncs, device Device) (*DeviceFuncs, error) {
	if instFuncs == nil || instFuncs.getDeviceProcAddr == nil {
		return nil, fmt.Errorf("vk: instance functions are not loaded")
	}
	if device == 0 {
		return nil, fmt.Errorf("vk: null device")
	}
	f := &DeviceFuncs{}

	resolve := func(target interface{}, name string) error {
		addr := instFuncs.GetDeviceProcAddr(device, name)
		if addr == 0 {
			return fmt.Errorf("vk: device function %s not found", name)
		}
		purego.RegisterFunc(target, addr)
		return nil
	}

	type entry struct {
		target interface{}
		name   string
	}
	// Resolve destruction first so the caller can release its device after a
	// later resolution failure.
	if err := resolve(&f.destroyDevice, "vkDestroyDevice"); err != nil {
		return nil, err
	}

	entries := []entry{
		{&f.getDeviceQueue, "vkGetDeviceQueue"},
		{&f.createBuffer, "vkCreateBuffer"},
		{&f.destroyBuffer, "vkDestroyBuffer"},
		{&f.getBufferMemoryRequirements, "vkGetBufferMemoryRequirements"},
		{&f.allocateMemory, "vkAllocateMemory"},
		{&f.freeMemory, "vkFreeMemory"},
		{&f.bindBufferMemory, "vkBindBufferMemory"},
		{&f.mapMemory, "vkMapMemory"},
		{&f.unmapMemory, "vkUnmapMemory"},
		{&f.createShaderModule, "vkCreateShaderModule"},
		{&f.destroyShaderModule, "vkDestroyShaderModule"},
		{&f.createDescriptorSetLayout, "vkCreateDescriptorSetLayout"},
		{&f.destroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout"},
		{&f.createPipelineLayout, "vkCreatePipelineLayout"},
		{&f.destroyPipelineLayout, "vkDestroyPipelineLayout"},
		{&f.createComputePipelines, "vkCreateComputePipelines"},
		{&f.destroyPipeline, "vkDestroyPipeline"},
		{&f.createDescriptorPool, "vkCreateDescriptorPool"},
		{&f.destroyDescriptorPool, "vkDestroyDescriptorPool"},
		{&f.allocateDescriptorSets, "vkAllocateDescriptorSets"},
		{&f.updateDescriptorSets, "vkUpdateDescriptorSets"},
		{&f.createCommandPool, "vkCreateCommandPool"},
		{&f.destroyCommandPool, "vkDestroyCommandPool"},
		{&f.allocateCommandBuffers, "vkAllocateCommandBuffers"},
		{&f.beginCommandBuffer, "vkBeginCommandBuffer"},
		{&f.endCommandBuffer, "vkEndCommandBuffer"},
		{&f.cmdBindPipeline, "vkCmdBindPipeline"},
		{&f.cmdBindDescriptorSets, "vkCmdBindDescriptorSets"},
		{&f.cmdDispatch, "vkCmdDispatch"},
		{&f.cmdCopyBuffer, "vkCmdCopyBuffer"},
		{&f.cmdPipelineBarrier, "vkCmdPipelineBarrier"},
		{&f.createFence, "vkCreateFence"},
		{&f.destroyFence, "vkDestroyFence"},
		{&f.waitForFences, "vkWaitForFences"},
		{&f.resetFences, "vkResetFences"},
		{&f.resetCommandBuffer, "vkResetCommandBuffer"},
		{&f.queueSubmit, "vkQueueSubmit"},
		{&f.flushMappedMemoryRanges, "vkFlushMappedMemoryRanges"},
		{&f.invalidateMappedMemoryRanges, "vkInvalidateMappedMemoryRanges"},
		{&f.deviceWaitIdle, "vkDeviceWaitIdle"},
		{&f.cmdUpdateBuffer, "vkCmdUpdateBuffer"},
	}

	for _, e := range entries {
		if err := resolve(e.target, e.name); err != nil {
			return f, err
		}
	}

	return f, nil
}

// --- Wrapper methods ---

func (f *DeviceFuncs) GetDeviceQueue(device Device, familyIndex, queueIndex uint32) Queue {
	var q Queue
	f.getDeviceQueue(device, familyIndex, queueIndex, &q)
	return q
}

func (f *DeviceFuncs) CreateBuffer(device Device, info *BufferCreateInfo) (Buffer, error) {
	if info == nil {
		return 0, fmt.Errorf("vk: vkCreateBuffer requires create info")
	}
	var buf Buffer
	res := f.createBuffer(device, info, 0, &buf)
	if res != Success {
		return 0, resultError("vkCreateBuffer", res)
	}
	return buf, nil
}

func (f *DeviceFuncs) DestroyBuffer(device Device, buffer Buffer) {
	f.destroyBuffer(device, buffer, 0)
}

func (f *DeviceFuncs) GetBufferMemoryRequirements(device Device, buffer Buffer) MemoryRequirements {
	var reqs MemoryRequirements
	f.getBufferMemoryRequirements(device, buffer, &reqs)
	return reqs
}

func (f *DeviceFuncs) AllocateMemory(device Device, info *MemoryAllocateInfo) (DeviceMemory, error) {
	if info == nil {
		return 0, fmt.Errorf("vk: vkAllocateMemory requires allocation info")
	}
	var mem DeviceMemory
	res := f.allocateMemory(device, info, 0, &mem)
	if res != Success {
		return 0, resultError("vkAllocateMemory", res)
	}
	return mem, nil
}

func (f *DeviceFuncs) FreeMemory(device Device, memory DeviceMemory) {
	f.freeMemory(device, memory, 0)
}

func (f *DeviceFuncs) BindBufferMemory(device Device, buffer Buffer, memory DeviceMemory, offset uint64) error {
	res := f.bindBufferMemory(device, buffer, memory, offset)
	if res != Success {
		return resultError("vkBindBufferMemory", res)
	}
	return nil
}

func (f *DeviceFuncs) MapMemory(device Device, memory DeviceMemory, offset, size uint64) (unsafe.Pointer, error) {
	var ptr unsafe.Pointer
	res := f.mapMemory(device, memory, offset, size, 0, &ptr)
	if res != Success {
		return nil, resultError("vkMapMemory", res)
	}
	return ptr, nil
}

func (f *DeviceFuncs) UnmapMemory(device Device, memory DeviceMemory) {
	f.unmapMemory(device, memory)
}

func (f *DeviceFuncs) CreateShaderModule(device Device, info *ShaderModuleCreateInfo) (ShaderModule, error) {
	if info == nil {
		return 0, fmt.Errorf("vk: vkCreateShaderModule requires create info")
	}
	var mod ShaderModule
	res := f.createShaderModule(device, info, 0, &mod)
	if res != Success {
		return 0, resultError("vkCreateShaderModule", res)
	}
	return mod, nil
}

func (f *DeviceFuncs) DestroyShaderModule(device Device, module ShaderModule) {
	f.destroyShaderModule(device, module, 0)
}

func (f *DeviceFuncs) CreateDescriptorSetLayout(device Device, info *DescriptorSetLayoutCreateInfo) (DescriptorSetLayout, error) {
	if info == nil {
		return 0, fmt.Errorf("vk: vkCreateDescriptorSetLayout requires create info")
	}
	var layout DescriptorSetLayout
	res := f.createDescriptorSetLayout(device, info, 0, &layout)
	if res != Success {
		return 0, resultError("vkCreateDescriptorSetLayout", res)
	}
	return layout, nil
}

func (f *DeviceFuncs) DestroyDescriptorSetLayout(device Device, layout DescriptorSetLayout) {
	f.destroyDescriptorSetLayout(device, layout, 0)
}

func (f *DeviceFuncs) CreatePipelineLayout(device Device, info *PipelineLayoutCreateInfo) (PipelineLayout, error) {
	if info == nil {
		return 0, fmt.Errorf("vk: vkCreatePipelineLayout requires create info")
	}
	var layout PipelineLayout
	res := f.createPipelineLayout(device, info, 0, &layout)
	if res != Success {
		return 0, resultError("vkCreatePipelineLayout", res)
	}
	return layout, nil
}

func (f *DeviceFuncs) DestroyPipelineLayout(device Device, layout PipelineLayout) {
	f.destroyPipelineLayout(device, layout, 0)
}

// CreateComputePipelines creates compute pipelines without a pipeline cache or
// allocation callbacks. It destroys every non-null result if creation fails.
func (f *DeviceFuncs) CreateComputePipelines(device Device, infos []ComputePipelineCreateInfo) ([]Pipeline, error) {
	if f == nil || f.createComputePipelines == nil || f.destroyPipeline == nil {
		return nil, fmt.Errorf("vk: device functions are not loaded")
	}
	if len(infos) == 0 {
		return nil, fmt.Errorf("vkCreateComputePipelines requires at least one create info")
	}
	pipelines := make([]Pipeline, len(infos))
	res := f.createComputePipelines(device, 0, uint32(len(infos)), &infos[0], 0, &pipelines[0])
	if res != Success {
		for _, pipeline := range pipelines {
			if pipeline != 0 {
				f.destroyPipeline(device, pipeline, 0)
			}
		}
		return nil, resultError("vkCreateComputePipelines", res)
	}
	return pipelines, nil
}

func (f *DeviceFuncs) DestroyPipeline(device Device, pipeline Pipeline) {
	f.destroyPipeline(device, pipeline, 0)
}

func (f *DeviceFuncs) CreateDescriptorPool(device Device, info *DescriptorPoolCreateInfo) (DescriptorPool, error) {
	if info == nil {
		return 0, fmt.Errorf("vk: vkCreateDescriptorPool requires create info")
	}
	var pool DescriptorPool
	res := f.createDescriptorPool(device, info, 0, &pool)
	if res != Success {
		return 0, resultError("vkCreateDescriptorPool", res)
	}
	return pool, nil
}

func (f *DeviceFuncs) DestroyDescriptorPool(device Device, pool DescriptorPool) {
	f.destroyDescriptorPool(device, pool, 0)
}

func (f *DeviceFuncs) AllocateDescriptorSets(device Device, info *DescriptorSetAllocateInfo) ([]DescriptorSet, error) {
	if info == nil || info.DescriptorSetCount == 0 {
		return nil, fmt.Errorf("vkAllocateDescriptorSets requires at least one descriptor set")
	}
	sets := make([]DescriptorSet, info.DescriptorSetCount)
	res := f.allocateDescriptorSets(device, info, &sets[0])
	if res != Success {
		return nil, resultError("vkAllocateDescriptorSets", res)
	}
	return sets, nil
}

// WriteDescriptorSets updates descriptor sets with writes. Descriptor copies
// are not supported by this wrapper.
func (f *DeviceFuncs) WriteDescriptorSets(device Device, writes []WriteDescriptorSet) {
	if len(writes) == 0 {
		return
	}
	f.updateDescriptorSets(device, uint32(len(writes)), &writes[0], 0, 0)
}

func (f *DeviceFuncs) CreateCommandPool(device Device, info *CommandPoolCreateInfo) (CommandPool, error) {
	if info == nil {
		return 0, fmt.Errorf("vk: vkCreateCommandPool requires create info")
	}
	var pool CommandPool
	res := f.createCommandPool(device, info, 0, &pool)
	if res != Success {
		return 0, resultError("vkCreateCommandPool", res)
	}
	return pool, nil
}

func (f *DeviceFuncs) DestroyCommandPool(device Device, pool CommandPool) {
	f.destroyCommandPool(device, pool, 0)
}

func (f *DeviceFuncs) AllocateCommandBuffers(device Device, info *CommandBufferAllocateInfo) ([]CommandBuffer, error) {
	if info == nil || info.CommandBufferCount == 0 {
		return nil, fmt.Errorf("vkAllocateCommandBuffers requires at least one command buffer")
	}
	bufs := make([]CommandBuffer, info.CommandBufferCount)
	res := f.allocateCommandBuffers(device, info, &bufs[0])
	if res != Success {
		return nil, resultError("vkAllocateCommandBuffers", res)
	}
	return bufs, nil
}

func (f *DeviceFuncs) BeginCommandBuffer(cb CommandBuffer, info *CommandBufferBeginInfo) error {
	if info == nil {
		return fmt.Errorf("vk: vkBeginCommandBuffer requires begin info")
	}
	res := f.beginCommandBuffer(cb, info)
	if res != Success {
		return resultError("vkBeginCommandBuffer", res)
	}
	return nil
}

func (f *DeviceFuncs) EndCommandBuffer(cb CommandBuffer) error {
	res := f.endCommandBuffer(cb)
	if res != Success {
		return resultError("vkEndCommandBuffer", res)
	}
	return nil
}

func (f *DeviceFuncs) CmdBindPipeline(cb CommandBuffer, bindPoint PipelineBindPoint, pipeline Pipeline) {
	f.cmdBindPipeline(cb, bindPoint, pipeline)
}

func (f *DeviceFuncs) CmdBindDescriptorSets(cb CommandBuffer, bindPoint PipelineBindPoint, layout PipelineLayout, firstSet uint32, sets []DescriptorSet) {
	if len(sets) == 0 {
		return
	}
	f.cmdBindDescriptorSets(cb, bindPoint, layout, firstSet, uint32(len(sets)), &sets[0], 0, 0)
}

func (f *DeviceFuncs) CmdDispatch(cb CommandBuffer, groupCountX, groupCountY, groupCountZ uint32) {
	f.cmdDispatch(cb, groupCountX, groupCountY, groupCountZ)
}

func (f *DeviceFuncs) CmdCopyBuffer(cb CommandBuffer, src, dst Buffer, regions []BufferCopy) {
	if len(regions) == 0 {
		return
	}
	f.cmdCopyBuffer(cb, src, dst, uint32(len(regions)), &regions[0])
}

// CmdPipelineBarrierBuffers records buffer memory barriers with no dependency
// flags, memory barriers, or image barriers.
func (f *DeviceFuncs) CmdPipelineBarrierBuffers(cb CommandBuffer, srcStage, dstStage PipelineStageFlags, bufBarriers []BufferMemoryBarrier) {
	var pBuf *BufferMemoryBarrier
	if len(bufBarriers) > 0 {
		pBuf = &bufBarriers[0]
	}
	f.cmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nil, uint32(len(bufBarriers)), pBuf, 0, 0)
}

func (f *DeviceFuncs) CreateFence(device Device, info *FenceCreateInfo) (Fence, error) {
	if info == nil {
		return 0, fmt.Errorf("vk: vkCreateFence requires create info")
	}
	var fence Fence
	res := f.createFence(device, info, 0, &fence)
	if res != Success {
		return 0, resultError("vkCreateFence", res)
	}
	return fence, nil
}

func (f *DeviceFuncs) DestroyFence(device Device, fence Fence) {
	f.destroyFence(device, fence, 0)
}

func (f *DeviceFuncs) WaitForFences(device Device, fences []Fence, waitAll bool, timeout uint64) error {
	if len(fences) == 0 {
		return fmt.Errorf("vkWaitForFences requires at least one fence")
	}
	var wa uint32
	if waitAll {
		wa = 1
	}
	res := f.waitForFences(device, uint32(len(fences)), &fences[0], wa, timeout)
	if res != Success {
		return resultError("vkWaitForFences", res)
	}
	return nil
}

func (f *DeviceFuncs) ResetFences(device Device, fences []Fence) error {
	if len(fences) == 0 {
		return fmt.Errorf("vkResetFences requires at least one fence")
	}
	res := f.resetFences(device, uint32(len(fences)), &fences[0])
	if res != Success {
		return resultError("vkResetFences", res)
	}
	return nil
}

func (f *DeviceFuncs) ResetCommandBuffer(cb CommandBuffer, flags CommandBufferResetFlags) error {
	res := f.resetCommandBuffer(cb, flags)
	if res != Success {
		return resultError("vkResetCommandBuffer", res)
	}
	return nil
}

func (f *DeviceFuncs) QueueSubmit(queue Queue, submits []SubmitInfo, fence Fence) error {
	if len(submits) == 0 {
		res := f.queueSubmit(queue, 0, nil, fence)
		if res != Success {
			return resultError("vkQueueSubmit", res)
		}
		return nil
	}
	res := f.queueSubmit(queue, uint32(len(submits)), &submits[0], fence)
	if res != Success {
		return resultError("vkQueueSubmit", res)
	}
	return nil
}

func (f *DeviceFuncs) FlushMappedMemoryRanges(device Device, ranges []MappedMemoryRange) error {
	if len(ranges) == 0 {
		return nil
	}
	res := f.flushMappedMemoryRanges(device, uint32(len(ranges)), &ranges[0])
	if res != Success {
		return resultError("vkFlushMappedMemoryRanges", res)
	}
	return nil
}

func (f *DeviceFuncs) InvalidateMappedMemoryRanges(device Device, ranges []MappedMemoryRange) error {
	if len(ranges) == 0 {
		return nil
	}
	res := f.invalidateMappedMemoryRanges(device, uint32(len(ranges)), &ranges[0])
	if res != Success {
		return resultError("vkInvalidateMappedMemoryRanges", res)
	}
	return nil
}

func (f *DeviceFuncs) DeviceWaitIdle(device Device) error {
	res := f.deviceWaitIdle(device)
	if res != Success {
		return resultError("vkDeviceWaitIdle", res)
	}
	return nil
}

func (f *DeviceFuncs) CmdUpdateBuffer(cb CommandBuffer, dst Buffer, offset uint64, data []byte) {
	if len(data) == 0 {
		return
	}
	f.cmdUpdateBuffer(cb, dst, offset, uint64(len(data)), unsafe.Pointer(&data[0]))
}

func (f *DeviceFuncs) DestroyDevice(device Device) {
	if f != nil && f.destroyDevice != nil && device != 0 {
		f.destroyDevice(device, 0)
	}
}
