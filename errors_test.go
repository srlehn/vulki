package vulki

import (
	"errors"
	"fmt"
	"testing"
	"unsafe"

	"github.com/srlehn/vulki/vk"
)

func TestClassifyDeviceError(t *testing.T) {
	if err := classifyDeviceError(nil); err != nil {
		t.Fatalf("classify nil = %v, want nil", err)
	}
	plain := errors.New("plain failure")
	if err := classifyDeviceError(plain); err != plain {
		t.Fatalf("classify plain error = %v, want the unchanged error", err)
	}
	hostOOM := &vk.Error{Op: "vkAllocateMemory", Result: vk.ErrorOutOfHostMemory}
	if err := classifyDeviceError(hostOOM); err != error(hostOOM) {
		t.Fatalf("classify host OOM = %v, want the unchanged error", err)
	}

	deviceOOM := &vk.Error{Op: "vkAllocateMemory", Result: vk.ErrorOutOfDeviceMemory}
	classified := classifyDeviceError(deviceOOM)
	if !errors.Is(classified, ErrOutOfDeviceMemory) {
		t.Fatalf("device OOM error %v does not match ErrOutOfDeviceMemory", classified)
	}
	if errors.Is(classified, ErrDeviceLost) || errors.Is(classified, ErrDeviceUnavailable) {
		t.Fatalf("device OOM error %v matches unrelated sentinels", classified)
	}
	if classified.Error() != deviceOOM.Error() {
		t.Fatalf("classified message = %q, want %q", classified.Error(), deviceOOM.Error())
	}
	var vkError *vk.Error
	if !errors.As(classified, &vkError) || vkError.Result != vk.ErrorOutOfDeviceMemory {
		t.Fatalf("classified error %v lost the vk.Error cause", classified)
	}
	if again := classifyDeviceError(classified); again != classified {
		t.Fatalf("reclassification produced a new error: %v", again)
	}

	lost := classifyDeviceError(fmt.Errorf(
		"wrapped: %w",
		&vk.Error{Op: "vkWaitForFences", Result: vk.ErrorDeviceLost},
	))
	if !errors.Is(lost, ErrDeviceLost) {
		t.Fatalf("wrapped device loss %v does not match ErrDeviceLost", lost)
	}
}

func TestNewBufferReportsOutOfDeviceMemory(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	device.state.ops.allocateMemory = func(*vk.DeviceFuncs, vk.Device, *vk.MemoryAllocateInfo) (vk.DeviceMemory, error) {
		return 0, &vk.Error{Op: "vkAllocateMemory", Result: vk.ErrorOutOfDeviceMemory}
	}
	_, err := device.NewBuffer(16)
	if !errors.Is(err, ErrOutOfDeviceMemory) {
		t.Fatalf("NewBuffer error %v does not match ErrOutOfDeviceMemory", err)
	}
	if errors.Is(err, ErrDeviceLost) || errors.Is(err, ErrDeviceUnavailable) {
		t.Fatalf("NewBuffer error %v matches unrelated sentinels", err)
	}
	if err := device.Err(); err != nil {
		t.Fatalf("Device.Err after allocation failure = %v, want nil", err)
	}
}

func TestDeviceErrReportsLostDeviceAndRefusesSubmissions(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	if err := device.Err(); err != nil {
		t.Fatalf("healthy Device.Err = %v, want nil", err)
	}
	nextFence := 0
	device.state.ops.endCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer) error { return nil }
	device.state.ops.createFence = func(*vk.DeviceFuncs, vk.Device, *vk.FenceCreateInfo) (vk.Fence, error) {
		nextFence++
		return vk.Fence(nextFence), nil
	}
	device.state.ops.destroyFence = func(*vk.DeviceFuncs, vk.Device, vk.Fence) {}
	device.state.ops.queueSubmit = func(*vk.DeviceFuncs, vk.Queue, []vk.SubmitInfo, vk.Fence) error { return nil }
	device.state.ops.waitForFences = func(*vk.DeviceFuncs, vk.Device, []vk.Fence, bool, uint64) error {
		return &vk.Error{Op: "vkWaitForFences", Result: vk.ErrorDeviceLost}
	}
	device.state.deviceFns = &vk.DeviceFuncs{}
	device.state.hooks.deviceWaitIdle = func(*vk.DeviceFuncs, vk.Device) error { return nil }
	device.state.hooks.destroyDevice = func(*vk.DeviceFuncs, vk.Device) {}

	first := &Recorder{
		device: device, state: recorderRecording,
		resources: submissionResources{{buffer: &Buffer{device: device}, access: submissionWrite}},
	}
	var err error
	first.childID, err = device.addChild(first)
	if err != nil {
		t.Fatalf("add first recorder: %v", err)
	}
	err = first.SubmitAndWait()
	if !errors.Is(err, ErrDeviceLost) {
		t.Fatalf("SubmitAndWait error %v does not match ErrDeviceLost", err)
	}
	if errors.Is(err, ErrDeviceUnavailable) {
		t.Fatalf("first failure %v matches ErrDeviceUnavailable, want cause only", err)
	}

	health := device.Err()
	if !errors.Is(health, ErrDeviceUnavailable) {
		t.Fatalf("Device.Err %v does not match ErrDeviceUnavailable", health)
	}
	if !errors.Is(health, ErrDeviceLost) {
		t.Fatalf("Device.Err %v does not expose the ErrDeviceLost cause", health)
	}

	second := &Recorder{
		device: device, state: recorderRecording,
		resources: submissionResources{{buffer: &Buffer{device: device}, access: submissionRead}},
	}
	second.childID, err = device.addChild(second)
	if err != nil {
		t.Fatalf("add second recorder: %v", err)
	}
	err = second.SubmitAndWait()
	if !errors.Is(err, ErrDeviceUnavailable) || !errors.Is(err, ErrDeviceLost) {
		t.Fatalf("refused submission error %v does not match both sentinels", err)
	}
	if err := device.Close(); err != nil {
		t.Fatalf("Device.Close: %v", err)
	}
}

func TestBufferUploadClassifiesDeviceLoss(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	buffer, err := device.NewBuffer(4)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	mapped := make([]byte, 4)
	device.state.ops.mapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error) {
		return unsafe.Pointer(&mapped[0]), nil
	}
	device.state.ops.unmapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) {}
	device.state.ops.bufferBarriers = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineStageFlags, vk.PipelineStageFlags, []vk.BufferMemoryBarrier) {
	}
	device.state.ops.copyBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.Buffer, vk.Buffer, []vk.BufferCopy) {}
	device.state.ops.createCommandPool = func(*vk.DeviceFuncs, vk.Device, *vk.CommandPoolCreateInfo) (vk.CommandPool, error) {
		return vk.CommandPool(20), nil
	}
	device.state.ops.allocateCommandBuffers = func(*vk.DeviceFuncs, vk.Device, *vk.CommandBufferAllocateInfo) ([]vk.CommandBuffer, error) {
		return []vk.CommandBuffer{21}, nil
	}
	device.state.ops.beginCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer, *vk.CommandBufferBeginInfo) error { return nil }
	device.state.ops.endCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer) error { return nil }
	device.state.ops.createFence = func(*vk.DeviceFuncs, vk.Device, *vk.FenceCreateInfo) (vk.Fence, error) {
		return vk.Fence(22), nil
	}
	device.state.ops.destroyCommandPool = func(*vk.DeviceFuncs, vk.Device, vk.CommandPool) {}
	device.state.ops.destroyFence = func(*vk.DeviceFuncs, vk.Device, vk.Fence) {}
	device.state.ops.queueSubmit = func(*vk.DeviceFuncs, vk.Queue, []vk.SubmitInfo, vk.Fence) error { return nil }
	device.state.ops.waitForFences = func(*vk.DeviceFuncs, vk.Device, []vk.Fence, bool, uint64) error {
		return &vk.Error{Op: "vkWaitForFences", Result: vk.ErrorDeviceLost}
	}
	device.state.deviceFns = &vk.DeviceFuncs{}
	device.state.hooks.deviceWaitIdle = func(*vk.DeviceFuncs, vk.Device) error { return nil }
	device.state.hooks.destroyDevice = func(*vk.DeviceFuncs, vk.Device) {}

	err = buffer.Upload([]byte{1, 2, 3, 4})
	if !errors.Is(err, ErrDeviceLost) {
		t.Fatalf("Upload error %v does not match ErrDeviceLost", err)
	}
	if !errors.Is(device.Err(), ErrDeviceLost) {
		t.Fatalf("Device.Err %v does not expose the upload device loss", device.Err())
	}
	if err := device.Close(); err != nil {
		t.Fatalf("Device.Close: %v", err)
	}
}

func TestDispatchAndWaitClassifiesDeviceLoss(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	buffer := &Buffer{device: device, references: 1}
	kernel := &Kernel{device: device, pipeline: 3, pipelineLayout: 4, bindingSets: 1}
	bindings := &BindingSet{
		device:    device,
		kernel:    kernel,
		set:       5,
		buffers:   []*Buffer{buffer},
		handles:   []vk.Buffer{6},
		resources: submissionResources{{buffer: buffer, access: submissionRead}},
	}
	var err error
	bindings.childID, err = device.addChild(bindings)
	if err != nil {
		t.Fatalf("add binding set: %v", err)
	}
	device.state.ops.createCommandPool = func(*vk.DeviceFuncs, vk.Device, *vk.CommandPoolCreateInfo) (vk.CommandPool, error) {
		return 20, nil
	}
	device.state.ops.allocateCommandBuffers = func(*vk.DeviceFuncs, vk.Device, *vk.CommandBufferAllocateInfo) ([]vk.CommandBuffer, error) {
		return []vk.CommandBuffer{21}, nil
	}
	device.state.ops.beginCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer, *vk.CommandBufferBeginInfo) error { return nil }
	device.state.ops.bufferBarriers = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineStageFlags, vk.PipelineStageFlags, []vk.BufferMemoryBarrier) {
	}
	device.state.ops.endCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer) error { return nil }
	device.state.kernelOps.bindPipeline = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineBindPoint, vk.Pipeline) {}
	device.state.kernelOps.bindDescriptorSets = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineBindPoint, vk.PipelineLayout, uint32, []vk.DescriptorSet) {
	}
	device.state.kernelOps.dispatch = func(*vk.DeviceFuncs, vk.CommandBuffer, uint32, uint32, uint32) {}
	device.state.ops.createFence = func(*vk.DeviceFuncs, vk.Device, *vk.FenceCreateInfo) (vk.Fence, error) {
		return 22, nil
	}
	device.state.ops.destroyCommandPool = func(*vk.DeviceFuncs, vk.Device, vk.CommandPool) {}
	device.state.ops.destroyFence = func(*vk.DeviceFuncs, vk.Device, vk.Fence) {}
	device.state.ops.queueSubmit = func(*vk.DeviceFuncs, vk.Queue, []vk.SubmitInfo, vk.Fence) error { return nil }
	device.state.ops.waitForFences = func(*vk.DeviceFuncs, vk.Device, []vk.Fence, bool, uint64) error {
		return &vk.Error{Op: "vkWaitForFences", Result: vk.ErrorDeviceLost}
	}
	device.state.deviceFns = &vk.DeviceFuncs{}
	device.state.hooks.deviceWaitIdle = func(*vk.DeviceFuncs, vk.Device) error { return nil }
	device.state.hooks.destroyDevice = func(*vk.DeviceFuncs, vk.Device) {}

	err = device.DispatchAndWait(kernel, bindings, Workgroups{X: 1, Y: 1, Z: 1})
	if !errors.Is(err, ErrDeviceLost) {
		t.Fatalf("DispatchAndWait error %v does not match ErrDeviceLost", err)
	}
	if !errors.Is(device.Err(), ErrDeviceLost) {
		t.Fatalf("Device.Err %v does not expose the dispatch device loss", device.Err())
	}
	if err := device.Close(); err != nil {
		t.Fatalf("Device.Close: %v", err)
	}
}
