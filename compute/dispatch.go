package compute

import "github.com/srlehn/vulki/vk"

// Dispatch records and submits a compute dispatch command.
func (c *Context) Dispatch(pipeline *Pipeline, groupCountX, groupCountY, groupCountZ uint32) error {
	poolInfo := vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: c.QueueFamily,
	}
	pool, err := c.DevFuncs.CreateCommandPool(c.Device, &poolInfo)
	if err != nil {
		return err
	}
	defer c.DevFuncs.DestroyCommandPool(c.Device, pool)

	allocInfo := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        pool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}
	cbs, err := c.DevFuncs.AllocateCommandBuffers(c.Device, &allocInfo)
	if err != nil {
		return err
	}
	cb := cbs[0]

	beginInfo := vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
		Flags: vk.CommandBufferUsageOneTimeSubmitBit,
	}
	if err := c.DevFuncs.BeginCommandBuffer(cb, &beginInfo); err != nil {
		return err
	}

	c.DevFuncs.CmdBindPipeline(cb, vk.PipelineBindPointCompute, pipeline.Pipeline)
	c.DevFuncs.CmdBindDescriptorSets(cb, vk.PipelineBindPointCompute, pipeline.PipelineLayout, 0, []vk.DescriptorSet{pipeline.DescSet})
	c.DevFuncs.CmdDispatch(cb, groupCountX, groupCountY, groupCountZ)

	if err := c.DevFuncs.EndCommandBuffer(cb); err != nil {
		return err
	}

	fenceInfo := vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}
	fence, err := c.DevFuncs.CreateFence(c.Device, &fenceInfo)
	if err != nil {
		return err
	}
	defer c.DevFuncs.DestroyFence(c.Device, fence)

	submitInfo := vk.SubmitInfo{
		SType:              vk.StructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    &cb,
	}
	return c.submitAndWait([]vk.SubmitInfo{submitInfo}, fence)
}
