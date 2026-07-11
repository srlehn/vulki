package compute

import "github.com/srlehn/vulki/vk"

// CommandRecorder batches multiple dispatches, barriers, and buffer updates
// into a single command buffer for efficient GPU submission.
type CommandRecorder struct {
	ctx     *Context
	pool    vk.CommandPool
	cb      vk.CommandBuffer
	curPipe *Pipeline
}

// NewCommandRecorder allocates a command pool and command buffer, and begins recording.
func (c *Context) NewCommandRecorder() (*CommandRecorder, error) {
	poolInfo := vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		QueueFamilyIndex: c.QueueFamily,
	}
	pool, err := c.DevFuncs.CreateCommandPool(c.Device, &poolInfo)
	if err != nil {
		return nil, err
	}

	allocInfo := vk.CommandBufferAllocateInfo{
		SType:              vk.StructureTypeCommandBufferAllocateInfo,
		CommandPool:        pool,
		Level:              vk.CommandBufferLevelPrimary,
		CommandBufferCount: 1,
	}
	cbs, err := c.DevFuncs.AllocateCommandBuffers(c.Device, &allocInfo)
	if err != nil {
		c.DevFuncs.DestroyCommandPool(c.Device, pool)
		return nil, err
	}

	beginInfo := vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
		Flags: vk.CommandBufferUsageOneTimeSubmitBit,
	}
	if err := c.DevFuncs.BeginCommandBuffer(cbs[0], &beginInfo); err != nil {
		c.DevFuncs.DestroyCommandPool(c.Device, pool)
		return nil, err
	}

	return &CommandRecorder{ctx: c, pool: pool, cb: cbs[0]}, nil
}

// Bind sets the current pipeline for subsequent Dispatch calls.
func (r *CommandRecorder) Bind(p *Pipeline) {
	r.curPipe = p
	r.ctx.DevFuncs.CmdBindPipeline(r.cb, vk.PipelineBindPointCompute, p.Pipeline)
	r.ctx.DevFuncs.CmdBindDescriptorSets(r.cb, vk.PipelineBindPointCompute, p.PipelineLayout, 0, []vk.DescriptorSet{p.DescSet})
}

// Dispatch records a compute dispatch with the currently bound pipeline.
func (r *CommandRecorder) Dispatch(gx, gy, gz uint32) {
	r.ctx.DevFuncs.CmdDispatch(r.cb, gx, gy, gz)
}

// Barrier inserts a compute-to-compute pipeline barrier on the given buffers.
func (r *CommandRecorder) Barrier(bufs ...vk.Buffer) {
	barriers := make([]vk.BufferMemoryBarrier, len(bufs))
	for i, b := range bufs {
		barriers[i] = vk.BufferMemoryBarrier{
			SType:               vk.StructureTypeBufferMemoryBarrier,
			SrcAccessMask:       vk.AccessShaderWriteBit,
			DstAccessMask:       vk.AccessShaderReadBit | vk.AccessShaderWriteBit,
			SrcQueueFamilyIndex: ^uint32(0),
			DstQueueFamilyIndex: ^uint32(0),
			Buffer:              b,
			Offset:              0,
			Size:                vk.WholeSize,
		}
	}
	r.ctx.DevFuncs.CmdPipelineBarrier(r.cb, vk.PipelineStageComputeShaderBit, vk.PipelineStageComputeShaderBit, barriers)
}

// BarrierTransferToCompute inserts a transfer-to-compute barrier for UpdateBuffer results.
func (r *CommandRecorder) BarrierTransferToCompute(bufs ...vk.Buffer) {
	barriers := make([]vk.BufferMemoryBarrier, len(bufs))
	for i, b := range bufs {
		barriers[i] = vk.BufferMemoryBarrier{
			SType:               vk.StructureTypeBufferMemoryBarrier,
			SrcAccessMask:       vk.AccessTransferWriteBit,
			DstAccessMask:       vk.AccessShaderReadBit,
			SrcQueueFamilyIndex: ^uint32(0),
			DstQueueFamilyIndex: ^uint32(0),
			Buffer:              b,
			Offset:              0,
			Size:                vk.WholeSize,
		}
	}
	r.ctx.DevFuncs.CmdPipelineBarrier(r.cb, vk.PipelineStageTransferBit, vk.PipelineStageComputeShaderBit, barriers)
}

// UpdateBuffer writes small data (<=64KB) into a buffer from the command stream.
func (r *CommandRecorder) UpdateBuffer(buf vk.Buffer, offset uint64, data []byte) {
	r.ctx.DevFuncs.CmdUpdateBuffer(r.cb, buf, offset, data)
}

// Submit ends recording, submits the command buffer, waits for completion, and cleans up.
func (r *CommandRecorder) Submit() error {
	if err := r.ctx.DevFuncs.EndCommandBuffer(r.cb); err != nil {
		r.ctx.DevFuncs.DestroyCommandPool(r.ctx.Device, r.pool)
		return err
	}

	fenceInfo := vk.FenceCreateInfo{SType: vk.StructureTypeFenceCreateInfo}
	fence, err := r.ctx.DevFuncs.CreateFence(r.ctx.Device, &fenceInfo)
	if err != nil {
		r.ctx.DevFuncs.DestroyCommandPool(r.ctx.Device, r.pool)
		return err
	}

	submitInfo := vk.SubmitInfo{
		SType:              vk.StructureTypeSubmitInfo,
		CommandBufferCount: 1,
		PCommandBuffers:    &r.cb,
	}
	err = r.ctx.DevFuncs.QueueSubmit(r.ctx.Queue, []vk.SubmitInfo{submitInfo}, fence)
	if err == nil {
		err = r.ctx.DevFuncs.WaitForFences(r.ctx.Device, []vk.Fence{fence}, true, ^uint64(0))
	}

	r.ctx.DevFuncs.DestroyFence(r.ctx.Device, fence)
	r.ctx.DevFuncs.DestroyCommandPool(r.ctx.Device, r.pool)
	return err
}
