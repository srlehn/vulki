package vulki

import (
	"bytes"
	"errors"
	"reflect"
	"testing"
	"unsafe"

	"github.com/srlehn/vulki/vk"
)

const inPlaceDoubleWGSL = `
struct Values {
    data: array<f32, 64>,
}

@group(0) @binding(0) var<storage, read_write> values: Values;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < 64u {
        values.data[id.x] = values.data[id.x] * 2.0;
    }
}
`

func TestRecorderBatchesUpdateAndDispatchesDirectVulkan(t *testing.T) {
	device, err := Open()
	if err != nil {
		t.Skipf("direct Vulkan device unavailable: %v", err)
	}
	defer device.Close()
	buffer, err := device.NewBuffer(64 * 4)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	defer buffer.Close()
	kernel, err := device.NewKernel(KernelOptions{
		WGSL:     inPlaceDoubleWGSL,
		Bindings: []BindingLayout{{Binding: 0, Access: BufferReadWrite}},
	})
	if err != nil {
		t.Fatalf("NewKernel: %v", err)
	}
	defer kernel.Close()
	bindings, err := kernel.NewBindings(BindBuffer(0, buffer))
	if err != nil {
		t.Fatalf("NewBindings: %v", err)
	}
	defer bindings.Close()
	recorder, err := device.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	defer recorder.Close()

	values := make([]float32, 64)
	for index := range values {
		values[index] = float32(index + 1)
	}
	if err := recorder.Update(buffer, 0, encodeFloat32s(values)); err != nil {
		t.Fatalf("Update: %v", err)
	}
	if err := recorder.Dispatch(kernel, bindings, Workgroups{X: 1, Y: 1, Z: 1}); err != nil {
		t.Fatalf("first Dispatch: %v", err)
	}
	if err := recorder.Barrier(buffer); err != nil {
		t.Fatalf("Barrier: %v", err)
	}
	if err := recorder.Dispatch(kernel, bindings, Workgroups{X: 1, Y: 1, Z: 1}); err != nil {
		t.Fatalf("second Dispatch: %v", err)
	}
	if err := bindings.Close(); err == nil {
		t.Fatal("binding set closed while retained by recorder")
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("SubmitAndWait: %v", err)
	}
	if err := recorder.SubmitAndWait(); err == nil {
		t.Fatal("second SubmitAndWait succeeded")
	}

	encoded := make([]byte, 64*4)
	if err := buffer.Download(encoded); err != nil {
		t.Fatalf("Download: %v", err)
	}
	results := decodeFloat32s(encoded)
	for index, result := range results {
		if want := values[index] * 4; result != want {
			t.Fatalf("result[%d] = %v, want %v", index, result, want)
		}
	}
}

func TestRecorderArbitraryUploadAndDownloadDirectVulkan(t *testing.T) {
	device, err := Open()
	if err != nil {
		t.Skipf("direct Vulkan device unavailable: %v", err)
	}
	defer device.Close()
	buffer, err := device.NewBuffer(70200)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	defer buffer.Close()
	recorder, err := device.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	defer recorder.Close()

	input := make([]byte, 70001)
	for index := range input {
		input[index] = byte(index*17 + 5)
	}
	patch := make([]byte, 127)
	for index := range patch {
		patch[index] = byte(index*29 + 11)
	}
	want := append(append(make([]byte, 0, len(input)+len(patch)), input...), patch...)
	if err := recorder.Upload(buffer, 3, input); err != nil {
		t.Fatalf("first Upload: %v", err)
	}
	if err := recorder.Upload(buffer, 3+uint64(len(input)), patch); err != nil {
		t.Fatalf("second Upload: %v", err)
	}
	clear(input)
	clear(patch)
	output := make([]byte, len(want))
	if err := recorder.Download(buffer, 3, output); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if err := recorder.SubmitAndWait(); err != nil {
		t.Fatalf("SubmitAndWait: %v", err)
	}
	if !bytes.Equal(output, want) {
		t.Fatal("recorded arbitrary-size round trip differs")
	}
}

func TestRecorderRejectsInvalidUpdatesBeforeBackend(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	buffer, err := device.NewBuffer(16)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	recorder := &Recorder{device: device, state: recorderRecording}
	tests := []struct {
		name   string
		offset uint64
		data   []byte
	}{
		{name: "offset alignment", offset: 2, data: make([]byte, 4)},
		{name: "size alignment", data: make([]byte, 3)},
		{name: "bounds", offset: 12, data: make([]byte, 8)},
		{name: "maximum", data: make([]byte, 65540)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := recorder.Update(buffer, test.offset, test.data); err == nil {
				t.Fatal("invalid update succeeded")
			}
		})
	}
}

func TestRecorderRejectsInvalidTransfersBeforeBackend(t *testing.T) {
	first, _, firstCreates := fakeBufferDevice("")
	second, _, _ := fakeBufferDevice("")
	buffer, err := first.NewBuffer(16)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	other, err := second.NewBuffer(16)
	if err != nil {
		t.Fatalf("other NewBuffer: %v", err)
	}
	defer other.Close()
	recorder := &Recorder{device: first, state: recorderRecording}
	tests := []struct {
		name string
		call func() error
	}{
		{name: "nil upload", call: func() error { return recorder.Upload(nil, 0, []byte{1}) }},
		{name: "wrong owner upload", call: func() error { return recorder.Upload(other, 0, []byte{1}) }},
		{name: "upload bounds", call: func() error { return recorder.Upload(buffer, 15, []byte{1, 2}) }},
		{name: "nil download", call: func() error { return recorder.Download(nil, 0, make([]byte, 1)) }},
		{name: "wrong owner download", call: func() error { return recorder.Download(other, 0, make([]byte, 1)) }},
		{name: "download bounds", call: func() error { return recorder.Download(buffer, 17, nil) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.call(); err == nil {
				t.Fatal("invalid transfer succeeded")
			}
		})
	}
	if got := *firstCreates; got != 1 {
		t.Fatalf("invalid transfers created %d buffers, want only storage buffer", got)
	}
	if err := buffer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := recorder.Upload(buffer, 0, nil); err == nil {
		t.Fatal("upload to closed buffer succeeded")
	}
	if err := recorder.Download(buffer, 0, nil); err == nil {
		t.Fatal("download from closed buffer succeeded")
	}
}

func TestRecorderUploadCleansFailedTransferAllocation(t *testing.T) {
	tests := []struct {
		failure string
		want    []string
	}{
		{failure: "staging-create"},
		{failure: "staging-memory-type", want: []string{"destroy-buffer:2"}},
		{failure: "staging-allocate", want: []string{"destroy-buffer:2"}},
		{failure: "staging-bind", want: []string{"destroy-buffer:2", "free-memory:2"}},
	}
	for _, test := range tests {
		t.Run(test.failure, func(t *testing.T) {
			device, events, _ := fakeBufferDevice(test.failure)
			buffer, err := device.NewBuffer(16)
			if err != nil {
				t.Fatalf("NewBuffer: %v", err)
			}
			defer buffer.Close()
			recorder := &Recorder{device: device, state: recorderRecording}
			if err := recorder.Upload(buffer, 0, []byte{1}); err == nil {
				t.Fatal("Upload succeeded")
			}
			if !reflect.DeepEqual(*events, test.want) {
				t.Fatalf("cleanup = %v, want %v", *events, test.want)
			}
		})
	}
}

func TestRecorderAbortReleasesTransferAndBuffer(t *testing.T) {
	device, events, _ := fakeBufferDevice("")
	buffer, err := device.NewBuffer(16)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	mapped := make([]byte, 16)
	device.state.ops.mapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error) {
		return unsafe.Pointer(&mapped[0]), nil
	}
	device.state.ops.unmapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) {}
	device.state.ops.bufferBarriers = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineStageFlags, vk.PipelineStageFlags, []vk.BufferMemoryBarrier) {
	}
	device.state.ops.copyBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.Buffer, vk.Buffer, []vk.BufferCopy) {}
	recorder := &Recorder{device: device, state: recorderRecording}
	if err := recorder.Upload(buffer, 0, []byte{1, 2, 3}); err != nil {
		t.Fatalf("Upload: %v", err)
	}
	if err := buffer.Close(); err == nil {
		t.Fatal("buffer closed while retained by recorder")
	}
	if err := recorder.Abort(); err != nil {
		t.Fatalf("Abort: %v", err)
	}
	if got := len(device.idleTransfers); got != 1 {
		t.Fatalf("idle transfers after Abort = %d, want 1", got)
	}
	if err := buffer.Close(); err != nil {
		t.Fatalf("buffer Close after Abort: %v", err)
	}
	want := []string{"destroy-buffer:1", "free-memory:1"}
	if !reflect.DeepEqual(*events, want) {
		t.Fatalf("cleanup = %v, want %v", *events, want)
	}
	device.drainTransferPool(device.state)
	want = append(want, "destroy-buffer:2", "free-memory:2")
	if !reflect.DeepEqual(*events, want) {
		t.Fatalf("cleanup after pool drain = %v, want %v", *events, want)
	}
}

func TestRecorderUploadCleansMapFailure(t *testing.T) {
	device, events, _ := fakeBufferDevice("")
	buffer, err := device.NewBuffer(16)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	defer buffer.Close()
	device.state.ops.mapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error) {
		return nil, errors.New("injected map failure")
	}
	recorder := &Recorder{device: device, state: recorderRecording}
	if err := recorder.Upload(buffer, 0, []byte{1}); err == nil {
		t.Fatal("Upload succeeded")
	}
	want := []string{"destroy-buffer:2", "free-memory:2"}
	if !reflect.DeepEqual(*events, want) {
		t.Fatalf("cleanup = %v, want %v", *events, want)
	}
}

func TestRecorderMultipleUploadsHoldExclusiveTransferLeases(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	buffer, err := device.NewBuffer(16)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	mapped := make([]byte, 16)
	device.state.ops.mapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error) {
		return unsafe.Pointer(&mapped[0]), nil
	}
	device.state.ops.unmapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) {}
	device.state.ops.bufferBarriers = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineStageFlags, vk.PipelineStageFlags, []vk.BufferMemoryBarrier) {
	}
	device.state.ops.copyBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.Buffer, vk.Buffer, []vk.BufferCopy) {}
	recorder := &Recorder{device: device, state: recorderRecording}
	if err := recorder.Upload(buffer, 0, []byte{1}); err != nil {
		t.Fatalf("first Upload: %v", err)
	}
	if err := recorder.Upload(buffer, 1, []byte{2}); err != nil {
		t.Fatalf("second Upload: %v", err)
	}
	if len(recorder.transfers) != 2 {
		t.Fatalf("recorder transfers = %d, want 2", len(recorder.transfers))
	}
	first := recorder.transfers[0].resource
	second := recorder.transfers[1].resource
	if first == second || first.buffer == second.buffer || first.memory == second.memory {
		t.Fatal("multiple uploads share an active transfer lease")
	}
	if err := recorder.Abort(); err != nil {
		t.Fatalf("Abort: %v", err)
	}
	if got := len(device.idleTransfers); got != 2 {
		t.Fatalf("idle transfers after Abort = %d, want 2", got)
	}
	if err := buffer.Close(); err != nil {
		t.Fatalf("buffer Close: %v", err)
	}
	device.drainTransferPool(device.state)
}

func TestRecorderReusesTransfersAfterKnownCompletion(t *testing.T) {
	device, _, createCount := fakeBufferDevice("")
	buffer, err := device.NewBuffer(16)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	mapped := make([]byte, 16)
	device.state.ops.mapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error) {
		return unsafe.Pointer(&mapped[0]), nil
	}
	device.state.ops.unmapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) {}
	device.state.ops.bufferBarriers = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineStageFlags, vk.PipelineStageFlags, []vk.BufferMemoryBarrier) {
	}
	device.state.ops.copyBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.Buffer, vk.Buffer, []vk.BufferCopy) {}
	device.state.ops.endCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer) error { return nil }
	nextFence := vk.Fence(20)
	device.state.ops.createFence = func(*vk.DeviceFuncs, vk.Device, *vk.FenceCreateInfo) (vk.Fence, error) {
		nextFence++
		return nextFence, nil
	}
	device.state.ops.destroyFence = func(*vk.DeviceFuncs, vk.Device, vk.Fence) {}
	device.state.ops.queueSubmit = func(*vk.DeviceFuncs, vk.Queue, []vk.SubmitInfo, vk.Fence) error { return nil }
	device.state.ops.waitForFences = func(*vk.DeviceFuncs, vk.Device, []vk.Fence, bool, uint64) error { return nil }

	run := func() {
		t.Helper()
		recorder := &Recorder{device: device, state: recorderRecording}
		if err := recorder.Upload(buffer, 0, []byte{1, 2, 3, 4}); err != nil {
			t.Fatalf("Upload: %v", err)
		}
		destination := make([]byte, 4)
		if err := recorder.Download(buffer, 0, destination); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if err := recorder.SubmitAndWait(); err != nil {
			t.Fatalf("SubmitAndWait: %v", err)
		}
	}

	run()
	if got := *createCount; got != 3 {
		t.Fatalf("creations after warmup = %d, want one storage and two transfer buffers", got)
	}
	run()
	if got := *createCount; got != 3 {
		t.Fatalf("warmed recorder created another transfer buffer: creations=%d", got)
	}
	if got := len(device.idleTransfers); got != 2 {
		t.Fatalf("idle transfers after repeated submission = %d, want 2", got)
	}
	if err := buffer.Close(); err != nil {
		t.Fatalf("buffer Close: %v", err)
	}
	device.drainTransferPool(device.state)
}

func TestRecorderSubmitFailureReturnsUnsubmittedTransfer(t *testing.T) {
	submitErr := errors.New("injected submit failure")
	device, _, _ := fakeBufferDevice("")
	buffer, err := device.NewBuffer(16)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	mapped := make([]byte, 16)
	device.state.ops.mapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error) {
		return unsafe.Pointer(&mapped[0]), nil
	}
	device.state.ops.unmapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) {}
	device.state.ops.bufferBarriers = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineStageFlags, vk.PipelineStageFlags, []vk.BufferMemoryBarrier) {
	}
	device.state.ops.copyBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.Buffer, vk.Buffer, []vk.BufferCopy) {}
	device.state.ops.endCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer) error { return nil }
	device.state.ops.createFence = func(*vk.DeviceFuncs, vk.Device, *vk.FenceCreateInfo) (vk.Fence, error) {
		return vk.Fence(12), nil
	}
	destroyedFences := 0
	device.state.ops.destroyFence = func(*vk.DeviceFuncs, vk.Device, vk.Fence) { destroyedFences++ }
	device.state.ops.queueSubmit = func(*vk.DeviceFuncs, vk.Queue, []vk.SubmitInfo, vk.Fence) error {
		return submitErr
	}

	recorder := &Recorder{device: device, state: recorderRecording}
	if err := recorder.Upload(buffer, 0, []byte{1}); err != nil {
		t.Fatalf("Upload: %v", err)
	}
	if err := recorder.SubmitAndWait(); !errors.Is(err, submitErr) {
		t.Fatalf("SubmitAndWait error = %v, want submit failure", err)
	}
	if recorder.state != recorderAborted {
		t.Fatalf("recorder state = %d, want aborted", recorder.state)
	}
	if destroyedFences != 1 {
		t.Fatalf("destroyed fences = %d, want 1", destroyedFences)
	}
	if got := len(device.idleTransfers); got != 1 {
		t.Fatalf("idle transfers after submit failure = %d, want 1", got)
	}
	if err := buffer.Close(); err != nil {
		t.Fatalf("buffer Close: %v", err)
	}
	device.drainTransferPool(device.state)
}

func TestRecorderWaitFailureRetainsResourcesUntilDeviceCleanup(t *testing.T) {
	waitErr := errors.New("injected wait failure")
	device, events, _ := fakeBufferDevice("")
	buffer, err := device.NewBuffer(16)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	mapped := make([]byte, 16)
	device.state.ops.mapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error) {
		return unsafe.Pointer(&mapped[0]), nil
	}
	device.state.ops.unmapMemory = func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) {}
	device.state.ops.bufferBarriers = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.PipelineStageFlags, vk.PipelineStageFlags, []vk.BufferMemoryBarrier) {
	}
	device.state.ops.copyBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer, vk.Buffer, vk.Buffer, []vk.BufferCopy) {}
	device.state.ops.endCommandBuffer = func(*vk.DeviceFuncs, vk.CommandBuffer) error { return nil }
	device.state.ops.createFence = func(*vk.DeviceFuncs, vk.Device, *vk.FenceCreateInfo) (vk.Fence, error) {
		return vk.Fence(11), nil
	}
	destroyedFences := 0
	device.state.ops.destroyFence = func(*vk.DeviceFuncs, vk.Device, vk.Fence) { destroyedFences++ }
	device.state.ops.queueSubmit = func(*vk.DeviceFuncs, vk.Queue, []vk.SubmitInfo, vk.Fence) error { return nil }
	device.state.ops.waitForFences = func(*vk.DeviceFuncs, vk.Device, []vk.Fence, bool, uint64) error {
		return waitErr
	}
	destroyedPools := 0
	device.state.ops.destroyCommandPool = func(*vk.DeviceFuncs, vk.Device, vk.CommandPool) { destroyedPools++ }
	waitIdleCalls := 0
	destroyedDevices := 0
	device.state.deviceFns = &vk.DeviceFuncs{}
	device.state.hooks.deviceWaitIdle = func(*vk.DeviceFuncs, vk.Device) error {
		waitIdleCalls++
		return nil
	}
	device.state.hooks.destroyDevice = func(*vk.DeviceFuncs, vk.Device) { destroyedDevices++ }

	recorder := &Recorder{
		device:  device,
		pool:    vk.CommandPool(7),
		command: vk.CommandBuffer(9),
		state:   recorderRecording,
	}
	recorder.childID, err = device.addChild(recorder)
	if err != nil {
		t.Fatalf("add recorder child: %v", err)
	}
	if err := recorder.Upload(buffer, 0, []byte{1, 2, 3}); err != nil {
		t.Fatalf("Upload: %v", err)
	}
	if err := recorder.SubmitAndWait(); !errors.Is(err, waitErr) {
		t.Fatalf("SubmitAndWait error = %v, want wait failure", err)
	}
	if recorder.state != recorderCompletionUnknown {
		t.Fatalf("recorder state = %d, want completion unknown", recorder.state)
	}
	if destroyedFences != 0 || destroyedPools != 0 || len(*events) != 0 {
		t.Fatalf("native resources released after uncertain wait: fences=%d pools=%d events=%v",
			destroyedFences, destroyedPools, *events)
	}
	if got := len(device.idleTransfers); got != 0 {
		t.Fatalf("idle transfers after uncertain wait = %d, want 0", got)
	}
	if err := buffer.Close(); err == nil {
		t.Fatal("buffer closed while uncertain submission retained it")
	}
	if err := recorder.Abort(); err != nil {
		t.Fatalf("Abort after uncertain wait: %v", err)
	}
	if destroyedFences != 0 || destroyedPools != 0 || len(*events) != 0 {
		t.Fatal("Abort released resources with unknown completion")
	}

	if err := device.Close(); err != nil {
		t.Fatalf("Device.Close: %v", err)
	}
	if destroyedFences != 1 || destroyedPools != 1 {
		t.Fatalf("device cleanup counts: fences=%d pools=%d, want 1 each", destroyedFences, destroyedPools)
	}
	if waitIdleCalls != 1 || destroyedDevices != 1 {
		t.Fatalf("device lifecycle counts: waits=%d destroys=%d, want 1 each", waitIdleCalls, destroyedDevices)
	}
	want := []string{
		"destroy-buffer:2", "free-memory:2",
		"destroy-buffer:1", "free-memory:1",
	}
	if !reflect.DeepEqual(*events, want) {
		t.Fatalf("device cleanup events = %v, want %v", *events, want)
	}
	if err := buffer.Close(); err != nil {
		t.Fatalf("buffer Close after device cleanup: %v", err)
	}
}

func TestRecorderAbortStateIsIdempotent(t *testing.T) {
	destroyed := 0
	device, _, _ := fakeBufferDevice("")
	device.state.ops.destroyCommandPool = func(*vk.DeviceFuncs, vk.Device, vk.CommandPool) {
		destroyed++
	}
	recorder := &Recorder{
		device:  device,
		pool:    1,
		command: 2,
		state:   recorderRecording,
	}
	if err := recorder.Abort(); err != nil {
		t.Fatalf("Abort: %v", err)
	}
	if err := recorder.Abort(); err != nil {
		t.Fatalf("second Abort: %v", err)
	}
	if destroyed != 1 {
		t.Fatalf("command pool destructions = %d, want 1", destroyed)
	}
	if err := recorder.Barrier(); err == nil {
		t.Fatal("Barrier after Abort succeeded")
	}
	if err := recorder.Dispatch(nil, nil, Workgroups{X: 1, Y: 1, Z: 1}); err == nil {
		t.Fatal("Dispatch after Abort succeeded")
	}
	if err := recorder.SubmitAndWait(); err == nil {
		t.Fatal("SubmitAndWait after Abort succeeded")
	}
}
