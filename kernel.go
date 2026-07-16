package vulki

import (
	"fmt"
	"sync"
	"syscall"
	"unsafe"

	"github.com/srlehn/vulki/shader"
	"github.com/srlehn/vulki/vk"
)

// BufferAccess describes how a kernel accesses a storage-buffer binding.
type BufferAccess uint8

const (
	// BufferReadOnly declares a read-only storage-buffer binding.
	BufferReadOnly BufferAccess = iota + 1
	// BufferReadWrite declares a read-write storage-buffer binding.
	BufferReadWrite
)

// BindingLayout declares one storage-buffer binding in descriptor set zero.
type BindingLayout struct {
	Binding uint32
	Access  BufferAccess
}

// KernelOptions describes a reusable WGSL compute kernel.
type KernelOptions struct {
	WGSL       string
	EntryPoint string
	Bindings   []BindingLayout
}

// Workgroups is a three-dimensional compute dispatch size.
type Workgroups struct {
	X uint32
	Y uint32
	Z uint32
}

// Kernel is a reusable compute pipeline and binding layout owned by a Device.
type Kernel struct {
	mu               sync.Mutex
	device           *Device
	childID          uint64
	shaderModule     vk.ShaderModule
	descriptorLayout vk.DescriptorSetLayout
	pipelineLayout   vk.PipelineLayout
	pipeline         vk.Pipeline
	layouts          map[uint32]BufferAccess
	bindingSets      int
	closed           bool
}

// BufferBinding associates a declared binding number with a Buffer.
type BufferBinding struct {
	binding uint32
	buffer  *Buffer
}

// BindBuffer creates one buffer binding for Kernel.NewBindings.
func BindBuffer(binding uint32, buffer *Buffer) BufferBinding {
	return BufferBinding{binding: binding, buffer: buffer}
}

// BindingSet is a concrete set of buffers bound to one Kernel.
type BindingSet struct {
	mu        sync.Mutex
	device    *Device
	kernel    *Kernel
	childID   uint64
	pool      vk.DescriptorPool
	set       vk.DescriptorSet
	buffers   []*Buffer
	handles   []vk.Buffer
	resources submissionResources
	recorders int
	closed    bool
}

type kernelOps struct {
	createShaderModule      func(*vk.DeviceFuncs, vk.Device, *vk.ShaderModuleCreateInfo) (vk.ShaderModule, error)
	destroyShaderModule     func(*vk.DeviceFuncs, vk.Device, vk.ShaderModule)
	createDescriptorLayout  func(*vk.DeviceFuncs, vk.Device, *vk.DescriptorSetLayoutCreateInfo) (vk.DescriptorSetLayout, error)
	destroyDescriptorLayout func(*vk.DeviceFuncs, vk.Device, vk.DescriptorSetLayout)
	createPipelineLayout    func(*vk.DeviceFuncs, vk.Device, *vk.PipelineLayoutCreateInfo) (vk.PipelineLayout, error)
	destroyPipelineLayout   func(*vk.DeviceFuncs, vk.Device, vk.PipelineLayout)
	createComputePipelines  func(*vk.DeviceFuncs, vk.Device, []vk.ComputePipelineCreateInfo) ([]vk.Pipeline, error)
	destroyPipeline         func(*vk.DeviceFuncs, vk.Device, vk.Pipeline)
	createDescriptorPool    func(*vk.DeviceFuncs, vk.Device, *vk.DescriptorPoolCreateInfo) (vk.DescriptorPool, error)
	destroyDescriptorPool   func(*vk.DeviceFuncs, vk.Device, vk.DescriptorPool)
	allocateDescriptorSets  func(*vk.DeviceFuncs, vk.Device, *vk.DescriptorSetAllocateInfo) ([]vk.DescriptorSet, error)
	writeDescriptorSets     func(*vk.DeviceFuncs, vk.Device, []vk.WriteDescriptorSet)
	bindPipeline            func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineBindPoint, vk.Pipeline)
	bindDescriptorSets      func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineBindPoint, vk.PipelineLayout, uint32, []vk.DescriptorSet)
	dispatch                func(*vk.DeviceFuncs, vk.CommandBuffer, uint32, uint32, uint32)
}

var directKernelOps = kernelOps{
	createShaderModule: func(functions *vk.DeviceFuncs, device vk.Device, info *vk.ShaderModuleCreateInfo) (vk.ShaderModule, error) {
		return functions.CreateShaderModule(device, info)
	},
	destroyShaderModule: func(functions *vk.DeviceFuncs, device vk.Device, module vk.ShaderModule) {
		functions.DestroyShaderModule(device, module)
	},
	createDescriptorLayout: func(functions *vk.DeviceFuncs, device vk.Device, info *vk.DescriptorSetLayoutCreateInfo) (vk.DescriptorSetLayout, error) {
		return functions.CreateDescriptorSetLayout(device, info)
	},
	destroyDescriptorLayout: func(functions *vk.DeviceFuncs, device vk.Device, layout vk.DescriptorSetLayout) {
		functions.DestroyDescriptorSetLayout(device, layout)
	},
	createPipelineLayout: func(functions *vk.DeviceFuncs, device vk.Device, info *vk.PipelineLayoutCreateInfo) (vk.PipelineLayout, error) {
		return functions.CreatePipelineLayout(device, info)
	},
	destroyPipelineLayout: func(functions *vk.DeviceFuncs, device vk.Device, layout vk.PipelineLayout) {
		functions.DestroyPipelineLayout(device, layout)
	},
	createComputePipelines: func(functions *vk.DeviceFuncs, device vk.Device, infos []vk.ComputePipelineCreateInfo) ([]vk.Pipeline, error) {
		return functions.CreateComputePipelines(device, infos)
	},
	destroyPipeline: func(functions *vk.DeviceFuncs, device vk.Device, pipeline vk.Pipeline) {
		functions.DestroyPipeline(device, pipeline)
	},
	createDescriptorPool: func(functions *vk.DeviceFuncs, device vk.Device, info *vk.DescriptorPoolCreateInfo) (vk.DescriptorPool, error) {
		return functions.CreateDescriptorPool(device, info)
	},
	destroyDescriptorPool: func(functions *vk.DeviceFuncs, device vk.Device, pool vk.DescriptorPool) {
		functions.DestroyDescriptorPool(device, pool)
	},
	allocateDescriptorSets: func(functions *vk.DeviceFuncs, device vk.Device, info *vk.DescriptorSetAllocateInfo) ([]vk.DescriptorSet, error) {
		return functions.AllocateDescriptorSets(device, info)
	},
	writeDescriptorSets: func(functions *vk.DeviceFuncs, device vk.Device, writes []vk.WriteDescriptorSet) {
		functions.WriteDescriptorSets(device, writes)
	},
	bindPipeline: func(functions *vk.DeviceFuncs, command vk.CommandBuffer, point vk.PipelineBindPoint, pipeline vk.Pipeline) {
		functions.CmdBindPipeline(command, point, pipeline)
	},
	bindDescriptorSets: func(functions *vk.DeviceFuncs, command vk.CommandBuffer, point vk.PipelineBindPoint, layout vk.PipelineLayout, first uint32, sets []vk.DescriptorSet) {
		functions.CmdBindDescriptorSets(command, point, layout, first, sets)
	},
	dispatch: func(functions *vk.DeviceFuncs, command vk.CommandBuffer, x, y, z uint32) {
		functions.CmdDispatch(command, x, y, z)
	},
}

// NewKernel compiles WGSL and creates a reusable compute kernel. EntryPoint
// defaults to main.
func (d *Device) NewKernel(options KernelOptions) (*Kernel, error) {
	if options.WGSL == "" {
		return nil, fmt.Errorf("vulki: kernel WGSL must not be empty")
	}
	if len(options.Bindings) == 0 {
		return nil, fmt.Errorf("vulki: kernel requires at least one buffer binding")
	}
	entryPoint := options.EntryPoint
	if entryPoint == "" {
		entryPoint = "main"
	}
	entryName, err := syscall.BytePtrFromString(entryPoint)
	if err != nil {
		return nil, fmt.Errorf("vulki: invalid kernel entry point: %w", err)
	}

	layouts := make(map[uint32]BufferAccess, len(options.Bindings))
	nativeLayouts := make([]vk.DescriptorSetLayoutBinding, len(options.Bindings))
	for index, binding := range options.Bindings {
		if binding.Access != BufferReadOnly && binding.Access != BufferReadWrite {
			return nil, fmt.Errorf("vulki: binding %d has invalid buffer access %d", binding.Binding, binding.Access)
		}
		if _, exists := layouts[binding.Binding]; exists {
			return nil, fmt.Errorf("vulki: duplicate kernel binding %d", binding.Binding)
		}
		layouts[binding.Binding] = binding.Access
		nativeLayouts[index] = vk.DescriptorSetLayoutBinding{
			Binding:         binding.Binding,
			DescriptorType:  vk.DescriptorTypeStorageBuffer,
			DescriptorCount: 1,
			StageFlags:      vk.ShaderStageComputeBit,
		}
	}

	spirv, err := shader.Compile(options.WGSL)
	if err != nil {
		return nil, fmt.Errorf("vulki: compile kernel WGSL: %w", err)
	}
	state, err := d.beginOperation()
	if err != nil {
		return nil, err
	}
	defer d.endOperation()

	kernel := &Kernel{device: d, layouts: layouts}
	moduleInfo := vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uintptr(len(spirv)),
		PCode:    unsafe.Pointer(&spirv[0]),
	}
	kernel.shaderModule, err = state.kernelOps.createShaderModule(state.deviceFns, state.device, &moduleInfo)
	if err != nil {
		return nil, fmt.Errorf("vulki: create kernel shader module: %w", err)
	}

	descriptorInfo := vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(nativeLayouts)),
		PBindings:    &nativeLayouts[0],
	}
	kernel.descriptorLayout, err = state.kernelOps.createDescriptorLayout(state.deviceFns, state.device, &descriptorInfo)
	if err != nil {
		kernel.closeNative(state)
		return nil, fmt.Errorf("vulki: create kernel descriptor layout: %w", err)
	}
	pipelineLayoutInfo := vk.PipelineLayoutCreateInfo{
		SType:          vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount: 1,
		PSetLayouts:    &kernel.descriptorLayout,
	}
	kernel.pipelineLayout, err = state.kernelOps.createPipelineLayout(state.deviceFns, state.device, &pipelineLayoutInfo)
	if err != nil {
		kernel.closeNative(state)
		return nil, fmt.Errorf("vulki: create kernel pipeline layout: %w", err)
	}
	stage := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: kernel.shaderModule,
		PName:  entryName,
	}
	pipelines, err := state.kernelOps.createComputePipelines(state.deviceFns, state.device, []vk.ComputePipelineCreateInfo{{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stage,
		Layout: kernel.pipelineLayout,
	}})
	if err != nil {
		kernel.closeNative(state)
		return nil, fmt.Errorf("vulki: create compute pipeline: %w", err)
	}
	kernel.pipeline = pipelines[0]
	kernel.childID, err = d.addChild(kernel)
	if err != nil {
		kernel.closeNative(state)
		return nil, err
	}
	return kernel, nil
}

// NewBindings binds one Buffer to every binding declared by the Kernel.
func (k *Kernel) NewBindings(bindings ...BufferBinding) (*BindingSet, error) {
	if k == nil || k.device == nil {
		return nil, fmt.Errorf("vulki: kernel is closed")
	}
	if len(bindings) != len(k.layouts) {
		return nil, fmt.Errorf("vulki: got %d bindings, want %d", len(bindings), len(k.layouts))
	}
	seen := make(map[uint32]struct{}, len(bindings))
	for _, binding := range bindings {
		if _, ok := k.layouts[binding.binding]; !ok {
			return nil, fmt.Errorf("vulki: kernel has no binding %d", binding.binding)
		}
		if _, duplicate := seen[binding.binding]; duplicate {
			return nil, fmt.Errorf("vulki: duplicate concrete binding %d", binding.binding)
		}
		seen[binding.binding] = struct{}{}
		if binding.buffer == nil || binding.buffer.device != k.device {
			return nil, fmt.Errorf("vulki: binding %d uses a buffer from another device", binding.binding)
		}
	}

	state, err := k.device.beginOperation()
	if err != nil {
		return nil, err
	}
	defer k.device.endOperation()
	k.mu.Lock()
	if k.closed {
		k.mu.Unlock()
		return nil, fmt.Errorf("vulki: kernel is closed")
	}
	k.bindingSets++
	k.mu.Unlock()

	set := &BindingSet{device: k.device, kernel: k}
	nativeBuffers := make([]vk.DescriptorBufferInfo, len(bindings))
	writes := make([]vk.WriteDescriptorSet, len(bindings))
	for index, binding := range bindings {
		buffer := binding.buffer
		buffer.mu.Lock()
		if buffer.closed || buffer.buffer == 0 {
			buffer.mu.Unlock()
			set.releaseReferences()
			return nil, fmt.Errorf("vulki: binding %d uses a closed buffer", binding.binding)
		}
		buffer.references++
		handle := buffer.buffer
		size := buffer.size
		buffer.mu.Unlock()
		set.buffers = append(set.buffers, buffer)
		set.handles = append(set.handles, handle)
		access := submissionRead
		if k.layouts[binding.binding] == BufferReadWrite {
			access = submissionWrite
		}
		set.resources.add(buffer, access)
		nativeBuffers[index] = vk.DescriptorBufferInfo{Buffer: handle, Range: size}
		writes[index] = vk.WriteDescriptorSet{
			SType:           vk.StructureTypeWriteDescriptorSet,
			DstBinding:      binding.binding,
			DescriptorCount: 1,
			DescriptorType:  vk.DescriptorTypeStorageBuffer,
			PBufferInfo:     &nativeBuffers[index],
		}
	}

	poolSize := vk.DescriptorPoolSize{Type: vk.DescriptorTypeStorageBuffer, DescriptorCount: uint32(len(bindings))}
	poolInfo := vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    &poolSize,
	}
	set.pool, err = state.kernelOps.createDescriptorPool(state.deviceFns, state.device, &poolInfo)
	if err != nil {
		set.releaseReferences()
		return nil, fmt.Errorf("vulki: create binding pool: %w", err)
	}
	allocation := vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     set.pool,
		DescriptorSetCount: 1,
		PSetLayouts:        &k.descriptorLayout,
	}
	sets, err := state.kernelOps.allocateDescriptorSets(state.deviceFns, state.device, &allocation)
	if err != nil {
		set.closeNative(state)
		set.releaseReferences()
		return nil, fmt.Errorf("vulki: allocate binding set: %w", err)
	}
	set.set = sets[0]
	for index := range writes {
		writes[index].DstSet = set.set
	}
	state.kernelOps.writeDescriptorSets(state.deviceFns, state.device, writes)
	set.childID, err = k.device.addChild(set)
	if err != nil {
		set.closeNative(state)
		set.releaseReferences()
		return nil, err
	}
	return set, nil
}

// DispatchAndWait records one compute dispatch and blocks until it completes.
func (d *Device) DispatchAndWait(kernel *Kernel, bindings *BindingSet, groups Workgroups) error {
	if kernel == nil || bindings == nil || kernel.device != d || bindings.device != d || bindings.kernel != kernel {
		return fmt.Errorf("vulki: dispatch resources do not belong to this device and kernel")
	}
	if groups.X == 0 || groups.Y == 0 || groups.Z == 0 {
		return fmt.Errorf("vulki: workgroup counts must be greater than zero")
	}
	limits := d.Info().Limits.MaxComputeWorkGroupCount
	counts := [3]uint32{groups.X, groups.Y, groups.Z}
	for index, count := range counts {
		if limits[index] != 0 && count > limits[index] {
			return fmt.Errorf("vulki: workgroup count %d exceeds dimension %d limit %d", count, index, limits[index])
		}
	}

	state, err := d.beginOperation()
	if err != nil {
		return err
	}
	defer d.endOperation()
	bindings.mu.Lock()
	if bindings.closed {
		bindings.mu.Unlock()
		return fmt.Errorf("vulki: binding set is closed")
	}
	submission := &transientSubmission{device: d}
	submission.retainBindingsLocked(bindings)
	handles := bindings.handles
	resources := bindings.resources
	set := bindings.set
	bindings.mu.Unlock()
	cleanupSubmission := true
	defer func() {
		if cleanupSubmission {
			submission.cleanup(state)
		}
	}()

	poolInfo := vk.CommandPoolCreateInfo{SType: vk.StructureTypeCommandPoolCreateInfo, QueueFamilyIndex: state.queueFamily}
	submission.pool, err = state.ops.createCommandPool(state.deviceFns, state.device, &poolInfo)
	if err != nil {
		return fmt.Errorf("vulki: create dispatch command pool: %w", err)
	}
	allocation := vk.CommandBufferAllocateInfo{
		SType: vk.StructureTypeCommandBufferAllocateInfo, CommandPool: submission.pool,
		Level: vk.CommandBufferLevelPrimary, CommandBufferCount: 1,
	}
	commands, err := state.ops.allocateCommandBuffers(state.deviceFns, state.device, &allocation)
	if err != nil {
		return fmt.Errorf("vulki: allocate dispatch command buffer: %w", err)
	}
	command := commands[0]
	begin := vk.CommandBufferBeginInfo{SType: vk.StructureTypeCommandBufferBeginInfo, Flags: vk.CommandBufferUsageOneTimeSubmitBit}
	if err := state.ops.beginCommandBuffer(state.deviceFns, command, &begin); err != nil {
		return fmt.Errorf("vulki: begin dispatch command buffer: %w", err)
	}
	barriers := make([]vk.BufferMemoryBarrier, len(handles))
	for index, buffer := range handles {
		barriers[index] = vk.BufferMemoryBarrier{
			SType:               vk.StructureTypeBufferMemoryBarrier,
			SrcAccessMask:       vk.AccessTransferWriteBit | vk.AccessShaderWriteBit,
			DstAccessMask:       vk.AccessShaderReadBit | vk.AccessShaderWriteBit,
			SrcQueueFamilyIndex: ^uint32(0), DstQueueFamilyIndex: ^uint32(0),
			Buffer: buffer, Size: vk.WholeSize,
		}
	}
	state.ops.bufferBarriers(
		state.deviceFns, command,
		vk.PipelineStageTransferBit|vk.PipelineStageComputeShaderBit,
		vk.PipelineStageComputeShaderBit,
		barriers,
	)
	state.kernelOps.bindPipeline(state.deviceFns, command, vk.PipelineBindPointCompute, kernel.pipeline)
	state.kernelOps.bindDescriptorSets(
		state.deviceFns, command, vk.PipelineBindPointCompute,
		kernel.pipelineLayout, 0, []vk.DescriptorSet{set},
	)
	state.kernelOps.dispatch(state.deviceFns, command, groups.X, groups.Y, groups.Z)
	if err := state.ops.endCommandBuffer(state.deviceFns, command); err != nil {
		return fmt.Errorf("vulki: end dispatch command buffer: %w", err)
	}
	fenceInfo := vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}
	submission.fence, err = state.ops.createFence(state.deviceFns, state.device, &fenceInfo)
	if err != nil {
		return fmt.Errorf("vulki: create dispatch fence: %w", err)
	}
	submission.reservation, err = d.acquireSubmission(resources)
	if err != nil {
		return fmt.Errorf("vulki: reserve dispatch resources: %w", err)
	}
	submit := vk.SubmitInfo{SType: vk.StructureTypeSubmitInfo, CommandBufferCount: 1, PCommandBuffers: &command}
	if err := d.submitQueue(state, []vk.SubmitInfo{submit}, submission.fence); err != nil {
		return fmt.Errorf("vulki: submit dispatch: %w", err)
	}
	if err := state.ops.waitForFences(
		state.deviceFns,
		state.device,
		[]vk.Fence{submission.fence},
		true,
		^uint64(0),
	); err != nil {
		err = classifyDeviceError(err)
		d.retainUnknownTransientSubmission(submission, err)
		cleanupSubmission = false
		return fmt.Errorf("vulki: wait for dispatch: %w", err)
	}
	return nil
}

// Close releases the binding set. Repeated calls return nil.
func (set *BindingSet) Close() error {
	return set.close(false)
}

func (set *BindingSet) closeFromDevice() error {
	return set.close(true)
}

func (set *BindingSet) close(fromDevice bool) error {
	if set == nil {
		return nil
	}
	set.mu.Lock()
	defer set.mu.Unlock()
	if set.closed {
		return nil
	}
	if set.recorders > 0 {
		return fmt.Errorf("vulki: binding set is used by %d recorders", set.recorders)
	}
	var state *deviceState
	if fromDevice {
		set.device.mu.Lock()
		state = set.device.state
		set.device.mu.Unlock()
		if state == nil {
			return fmt.Errorf("vulki: binding-set owner is closed")
		}
	} else {
		var err error
		state, err = set.device.beginOperation()
		if err != nil {
			return err
		}
		defer set.device.endOperation()
	}
	set.closeNative(state)
	set.closed = true
	set.releaseReferences()
	if !fromDevice {
		set.device.removeChild(set.childID)
	}
	return nil
}

func (set *BindingSet) closeNative(state *deviceState) {
	if set.pool != 0 {
		state.kernelOps.destroyDescriptorPool(state.deviceFns, state.device, set.pool)
		set.pool = 0
		set.set = 0
	}
}

func (set *BindingSet) releaseReferences() {
	for _, buffer := range set.buffers {
		buffer.mu.Lock()
		if buffer.references > 0 {
			buffer.references--
		}
		buffer.mu.Unlock()
	}
	set.buffers = nil
	set.handles = nil
	set.resources = nil
	if set.kernel != nil {
		set.kernel.mu.Lock()
		if set.kernel.bindingSets > 0 {
			set.kernel.bindingSets--
		}
		set.kernel.mu.Unlock()
	}
}

// Close releases the kernel. It returns an error while binding sets still
// reference it. Repeated calls return nil.
func (k *Kernel) Close() error {
	return k.close(false)
}

func (k *Kernel) closeFromDevice() error {
	return k.close(true)
}

func (k *Kernel) close(fromDevice bool) error {
	if k == nil {
		return nil
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.closed {
		return nil
	}
	if k.bindingSets > 0 {
		return fmt.Errorf("vulki: kernel is used by %d binding sets", k.bindingSets)
	}
	var state *deviceState
	if fromDevice {
		k.device.mu.Lock()
		state = k.device.state
		k.device.mu.Unlock()
		if state == nil {
			return fmt.Errorf("vulki: kernel owner is closed")
		}
	} else {
		var err error
		state, err = k.device.beginOperation()
		if err != nil {
			return err
		}
		defer k.device.endOperation()
	}
	k.closeNative(state)
	k.closed = true
	if !fromDevice {
		k.device.removeChild(k.childID)
	}
	return nil
}

func (k *Kernel) closeNative(state *deviceState) {
	if k.pipeline != 0 {
		state.kernelOps.destroyPipeline(state.deviceFns, state.device, k.pipeline)
		k.pipeline = 0
	}
	if k.pipelineLayout != 0 {
		state.kernelOps.destroyPipelineLayout(state.deviceFns, state.device, k.pipelineLayout)
		k.pipelineLayout = 0
	}
	if k.descriptorLayout != 0 {
		state.kernelOps.destroyDescriptorLayout(state.deviceFns, state.device, k.descriptorLayout)
		k.descriptorLayout = 0
	}
	if k.shaderModule != 0 {
		state.kernelOps.destroyShaderModule(state.deviceFns, state.device, k.shaderModule)
		k.shaderModule = 0
	}
}

func (d *Device) validateWorkgroups(groups Workgroups) error {
	if groups.X == 0 || groups.Y == 0 || groups.Z == 0 {
		return fmt.Errorf("vulki: workgroup counts must be greater than zero")
	}
	limits := d.Info().Limits.MaxComputeWorkGroupCount
	counts := [3]uint32{groups.X, groups.Y, groups.Z}
	for index, count := range counts {
		if limits[index] != 0 && count > limits[index] {
			return fmt.Errorf("vulki: workgroup count %d exceeds dimension %d limit %d", count, index, limits[index])
		}
	}
	return nil
}
