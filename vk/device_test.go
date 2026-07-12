package vk

import (
	"errors"
	"strings"
	"testing"
	"unsafe"
)

func TestDeviceCreateWrappersRejectNilInfo(t *testing.T) {
	functions := &DeviceFuncs{}
	tests := []struct {
		name string
		call func() error
		op   string
	}{
		{
			name: "buffer",
			call: func() error {
				_, err := functions.CreateBuffer(Device(1), nil)
				return err
			},
			op: "vkCreateBuffer",
		},
		{
			name: "memory",
			call: func() error {
				_, err := functions.AllocateMemory(Device(1), nil)
				return err
			},
			op: "vkAllocateMemory",
		},
		{
			name: "shader module",
			call: func() error {
				_, err := functions.CreateShaderModule(Device(1), nil)
				return err
			},
			op: "vkCreateShaderModule",
		},
		{
			name: "descriptor set layout",
			call: func() error {
				_, err := functions.CreateDescriptorSetLayout(Device(1), nil)
				return err
			},
			op: "vkCreateDescriptorSetLayout",
		},
		{
			name: "pipeline layout",
			call: func() error {
				_, err := functions.CreatePipelineLayout(Device(1), nil)
				return err
			},
			op: "vkCreatePipelineLayout",
		},
		{
			name: "descriptor pool",
			call: func() error {
				_, err := functions.CreateDescriptorPool(Device(1), nil)
				return err
			},
			op: "vkCreateDescriptorPool",
		},
		{
			name: "command pool",
			call: func() error {
				_, err := functions.CreateCommandPool(Device(1), nil)
				return err
			},
			op: "vkCreateCommandPool",
		},
		{
			name: "command buffer begin",
			call: func() error {
				return functions.BeginCommandBuffer(CommandBuffer(1), nil)
			},
			op: "vkBeginCommandBuffer",
		},
		{
			name: "fence",
			call: func() error {
				_, err := functions.CreateFence(Device(1), nil)
				return err
			},
			op: "vkCreateFence",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.call()
			if err == nil || !strings.Contains(err.Error(), test.op) {
				t.Fatalf("error = %v, want operation %s", err, test.op)
			}
		})
	}
}

func TestCreateComputePipelinesCleansPartialFailure(t *testing.T) {
	var destroyed []Pipeline
	functions := &DeviceFuncs{
		createComputePipelines: func(_ Device, _ uintptr, count uint32, _ *ComputePipelineCreateInfo, _ uintptr, pipelines *Pipeline) Result {
			results := unsafe.Slice(pipelines, count)
			results[0] = Pipeline(11)
			results[2] = Pipeline(33)
			return ErrorOutOfDeviceMemory
		},
		destroyPipeline: func(_ Device, pipeline Pipeline, _ uintptr) {
			destroyed = append(destroyed, pipeline)
		},
	}

	pipelines, err := functions.CreateComputePipelines(
		Device(1),
		make([]ComputePipelineCreateInfo, 3),
	)
	if pipelines != nil {
		t.Fatalf("pipelines = %v, want nil", pipelines)
	}
	var vkErr *Error
	if !errors.As(err, &vkErr) || vkErr.Result != ErrorOutOfDeviceMemory {
		t.Fatalf("error = %v, want ErrorOutOfDeviceMemory", err)
	}
	if len(destroyed) != 2 || destroyed[0] != 11 || destroyed[1] != 33 {
		t.Fatalf("destroyed = %v, want [11 33]", destroyed)
	}
}

func TestNarrowSliceWrappersHandleEmptyInput(t *testing.T) {
	barrierCalls := 0
	writeCalls := 0
	submitCalls := 0
	functions := &DeviceFuncs{
		cmdPipelineBarrier: func(_ CommandBuffer, _, _ PipelineStageFlags, _ uint32, memoryCount uint32, memory *MemoryBarrier, bufferCount uint32, buffers *BufferMemoryBarrier, imageCount uint32, images uintptr) {
			barrierCalls++
			if memoryCount != 0 || memory != nil || bufferCount != 0 || buffers != nil || imageCount != 0 || images != 0 {
				t.Fatal("empty buffer barrier passed non-empty barrier arguments")
			}
		},
		updateDescriptorSets: func(_ Device, _ uint32, _ *WriteDescriptorSet, _ uint32, _ uintptr) {
			writeCalls++
		},
		queueSubmit: func(_ Queue, count uint32, submits *SubmitInfo, fence Fence) Result {
			submitCalls++
			if count != 0 || submits != nil || fence != Fence(9) {
				t.Fatal("empty queue submission passed incorrect arguments")
			}
			return Success
		},
	}

	functions.CmdPipelineBarrierBuffers(CommandBuffer(1), PipelineStageTransferBit, PipelineStageComputeShaderBit, nil)
	functions.WriteDescriptorSets(Device(1), nil)
	if err := functions.QueueSubmit(Queue(1), nil, Fence(9)); err != nil {
		t.Fatalf("QueueSubmit: %v", err)
	}
	if barrierCalls != 1 || writeCalls != 0 || submitCalls != 1 {
		t.Fatalf("calls = barrier:%d write:%d submit:%d", barrierCalls, writeCalls, submitCalls)
	}
}
