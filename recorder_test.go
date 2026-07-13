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
	if err := buffer.Close(); err != nil {
		t.Fatalf("buffer Close after Abort: %v", err)
	}
	want := []string{"destroy-buffer:2", "free-memory:2", "destroy-buffer:1", "free-memory:1"}
	if !reflect.DeepEqual(*events, want) {
		t.Fatalf("cleanup = %v, want %v", *events, want)
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
