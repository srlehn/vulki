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
	if barrierCalls != 0 || writeCalls != 0 || submitCalls != 1 {
		t.Fatalf("calls = barrier:%d write:%d submit:%d", barrierCalls, writeCalls, submitCalls)
	}
}

func TestCmdUpdateBufferValidation(t *testing.T) {
	calls := 0
	functions := &DeviceFuncs{cmdUpdateBuffer: func(_ CommandBuffer, _ Buffer, _ uint64, size uint64, _ unsafe.Pointer) {
		calls++
		if size != 4 {
			t.Fatalf("size = %d, want 4", size)
		}
	}}

	valid := make([]uint32, 1)
	validBytes := unsafe.Slice((*byte)(unsafe.Pointer(&valid[0])), 4)
	unalignedStorage := make([]byte, 5)
	unalignedOffset := 0
	if uintptr(unsafe.Pointer(&unalignedStorage[0]))%4 == 0 {
		unalignedOffset = 1
	}
	unalignedBytes := unalignedStorage[unalignedOffset : unalignedOffset+4]
	tests := []struct {
		name   string
		cb     CommandBuffer
		dst    Buffer
		offset uint64
		data   []byte
	}{
		{name: "null command buffer", dst: Buffer(1), data: validBytes},
		{name: "null destination", cb: CommandBuffer(1), data: validBytes},
		{name: "unaligned offset", cb: CommandBuffer(1), dst: Buffer(1), offset: 2, data: validBytes},
		{name: "unaligned size", cb: CommandBuffer(1), dst: Buffer(1), data: validBytes[:3]},
		{name: "oversized", cb: CommandBuffer(1), dst: Buffer(1), data: make([]byte, 65540)},
		{name: "unaligned address", cb: CommandBuffer(1), dst: Buffer(1), data: unalignedBytes},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := functions.CmdUpdateBuffer(test.cb, test.dst, test.offset, test.data); err == nil {
				t.Fatal("CmdUpdateBuffer succeeded")
			}
		})
	}

	if err := functions.CmdUpdateBuffer(CommandBuffer(1), Buffer(1), 0, nil); err != nil {
		t.Fatalf("empty CmdUpdateBuffer: %v", err)
	}
	if err := functions.CmdUpdateBuffer(CommandBuffer(1), Buffer(1), 0, validBytes); err != nil {
		t.Fatalf("valid CmdUpdateBuffer: %v", err)
	}
	if calls != 1 {
		t.Fatalf("driver calls = %d, want 1", calls)
	}
}

func TestLoadDeviceFuncsFailureOwnershipMatrix(t *testing.T) {
	for failureIndex := 0; ; failureIndex++ {
		lookupIndex := 0
		destroyed := false
		instanceFunctions := &InstanceFuncs{
			getDeviceProcAddr: func(Device, *byte) uintptr {
				current := lookupIndex
				lookupIndex++
				if current == failureIndex {
					return 0
				}
				return 1
			},
		}

		functions, err := loadDeviceFuncs(instanceFunctions, Device(1), func(target any, _ uintptr) {
			if destroy, ok := target.(*func(Device, uintptr)); ok {
				*destroy = func(Device, uintptr) { destroyed = true }
			}
		})
		if functions != nil {
			functions.DestroyDevice(Device(1))
			if !destroyed {
				t.Fatalf("failure at resolution index %d returned an unusable cleanup table", failureIndex)
			}
		}
		if err == nil {
			if functions == nil {
				t.Fatal("successful load returned nil functions")
			}
			if failureIndex != lookupIndex {
				t.Fatalf("resolved %d functions after %d failure cases", lookupIndex, failureIndex)
			}
			break
		}
		if failureIndex == 0 && functions != nil {
			t.Fatal("missing vkDestroyDevice returned a cleanup table")
		}
		if failureIndex > 0 && functions == nil {
			t.Fatalf("failure at resolution index %d returned no cleanup table", failureIndex)
		}
	}
}
