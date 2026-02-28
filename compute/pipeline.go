package compute

import (
	"unsafe"

	"vkpg/vk"
)

// Pipeline holds a compute pipeline and its associated descriptor resources.
type Pipeline struct {
	ShaderModule      vk.ShaderModule
	DescSetLayout     vk.DescriptorSetLayout
	PipelineLayout    vk.PipelineLayout
	Pipeline          vk.Pipeline
	DescPool          vk.DescriptorPool
	DescSet           vk.DescriptorSet
}

// BufferBinding associates a binding index with a buffer.
type BufferBinding struct {
	Binding uint32
	Buffer  *Buffer
}

// CreateComputePipeline creates a shader module, descriptor set layout, pipeline layout,
// compute pipeline, descriptor pool, and descriptor set, and binds the provided buffers.
func (c *Context) CreateComputePipeline(spirv []byte, bindings []BufferBinding) (*Pipeline, error) {
	p := &Pipeline{}
	var err error

	// Create shader module.
	moduleInfo := vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uintptr(len(spirv)),
		PCode:    unsafe.Pointer(&spirv[0]),
	}
	p.ShaderModule, err = c.DevFuncs.CreateShaderModule(c.Device, &moduleInfo)
	if err != nil {
		return nil, err
	}

	// Descriptor set layout bindings.
	layoutBindings := make([]vk.DescriptorSetLayoutBinding, len(bindings))
	for i, b := range bindings {
		layoutBindings[i] = vk.DescriptorSetLayoutBinding{
			Binding:         b.Binding,
			DescriptorType:  vk.DescriptorTypeStorageBuffer,
			DescriptorCount: 1,
			StageFlags:      vk.ShaderStageComputeBit,
		}
	}
	descLayoutInfo := vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(layoutBindings)),
		PBindings:    &layoutBindings[0],
	}
	p.DescSetLayout, err = c.DevFuncs.CreateDescriptorSetLayout(c.Device, &descLayoutInfo)
	if err != nil {
		p.Destroy(c)
		return nil, err
	}

	// Pipeline layout.
	pipeLayoutInfo := vk.PipelineLayoutCreateInfo{
		SType:          vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount: 1,
		PSetLayouts:    &p.DescSetLayout,
	}
	p.PipelineLayout, err = c.DevFuncs.CreatePipelineLayout(c.Device, &pipeLayoutInfo)
	if err != nil {
		p.Destroy(c)
		return nil, err
	}

	// Compute pipeline.
	entryPoint := append([]byte("main"), 0)
	stageInfo := vk.PipelineShaderStageCreateInfo{
		SType:  vk.StructureTypePipelineShaderStageCreateInfo,
		Stage:  vk.ShaderStageComputeBit,
		Module: p.ShaderModule,
		PName:  &entryPoint[0],
	}
	pipelineInfo := vk.ComputePipelineCreateInfo{
		SType:  vk.StructureTypeComputePipelineCreateInfo,
		Stage:  stageInfo,
		Layout: p.PipelineLayout,
	}
	pipelines, err := c.DevFuncs.CreateComputePipelines(c.Device, []vk.ComputePipelineCreateInfo{pipelineInfo})
	if err != nil {
		p.Destroy(c)
		return nil, err
	}
	p.Pipeline = pipelines[0]

	// Descriptor pool.
	poolSize := vk.DescriptorPoolSize{
		Type:            vk.DescriptorTypeStorageBuffer,
		DescriptorCount: uint32(len(bindings)),
	}
	descPoolInfo := vk.DescriptorPoolCreateInfo{
		SType:         vk.StructureTypeDescriptorPoolCreateInfo,
		MaxSets:       1,
		PoolSizeCount: 1,
		PPoolSizes:    &poolSize,
	}
	p.DescPool, err = c.DevFuncs.CreateDescriptorPool(c.Device, &descPoolInfo)
	if err != nil {
		p.Destroy(c)
		return nil, err
	}

	// Allocate descriptor set.
	descAllocInfo := vk.DescriptorSetAllocateInfo{
		SType:              vk.StructureTypeDescriptorSetAllocateInfo,
		DescriptorPool:     p.DescPool,
		DescriptorSetCount: 1,
		PSetLayouts:        &p.DescSetLayout,
	}
	sets, err := c.DevFuncs.AllocateDescriptorSets(c.Device, &descAllocInfo)
	if err != nil {
		p.Destroy(c)
		return nil, err
	}
	p.DescSet = sets[0]

	// Update descriptor set with buffer bindings.
	writes := make([]vk.WriteDescriptorSet, len(bindings))
	bufInfos := make([]vk.DescriptorBufferInfo, len(bindings))
	for i, b := range bindings {
		bufInfos[i] = vk.DescriptorBufferInfo{
			Buffer: b.Buffer.DeviceBuffer,
			Offset: 0,
			Range:  b.Buffer.Size,
		}
		writes[i] = vk.WriteDescriptorSet{
			SType:           vk.StructureTypeWriteDescriptorSet,
			DstSet:          p.DescSet,
			DstBinding:      b.Binding,
			DescriptorCount: 1,
			DescriptorType:  vk.DescriptorTypeStorageBuffer,
			PBufferInfo:     &bufInfos[i],
		}
	}
	c.DevFuncs.UpdateDescriptorSets(c.Device, writes)

	return p, nil
}

// Destroy releases all pipeline resources.
func (p *Pipeline) Destroy(ctx *Context) {
	if p.Pipeline != 0 {
		ctx.DevFuncs.DestroyPipeline(ctx.Device, p.Pipeline)
		p.Pipeline = 0
	}
	if p.DescPool != 0 {
		ctx.DevFuncs.DestroyDescriptorPool(ctx.Device, p.DescPool)
		p.DescPool = 0
	}
	if p.PipelineLayout != 0 {
		ctx.DevFuncs.DestroyPipelineLayout(ctx.Device, p.PipelineLayout)
		p.PipelineLayout = 0
	}
	if p.DescSetLayout != 0 {
		ctx.DevFuncs.DestroyDescriptorSetLayout(ctx.Device, p.DescSetLayout)
		p.DescSetLayout = 0
	}
	if p.ShaderModule != 0 {
		ctx.DevFuncs.DestroyShaderModule(ctx.Device, p.ShaderModule)
		p.ShaderModule = 0
	}
}
