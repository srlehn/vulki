package vulki

import (
	"testing"

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
