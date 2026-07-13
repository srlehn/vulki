package vulki_test

import (
	"encoding/binary"

	"github.com/srlehn/vulki"
)

const doubleWGSL = `
@group(0) @binding(0) var<storage, read_write> values: array<u32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    values[id.x] = values[id.x] * 2u;
}
`

func ExampleDevice_DispatchAndWait() {
	device, err := vulki.Open()
	if err != nil {
		return
	}
	defer device.Close()

	buffer, err := device.NewBuffer(4)
	if err != nil {
		return
	}
	defer buffer.Close()
	input := make([]byte, 4)
	binary.LittleEndian.PutUint32(input, 21)
	if err := buffer.Upload(input); err != nil {
		return
	}

	kernel, err := device.NewKernel(vulki.KernelOptions{
		WGSL: doubleWGSL,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadWrite},
		},
	})
	if err != nil {
		return
	}
	defer kernel.Close()
	bindings, err := kernel.NewBindings(vulki.BindBuffer(0, buffer))
	if err != nil {
		return
	}
	defer bindings.Close()

	_ = device.DispatchAndWait(kernel, bindings, vulki.Workgroups{X: 1, Y: 1, Z: 1})
}

func ExampleRecorder() {
	device, err := vulki.Open()
	if err != nil {
		return
	}
	defer device.Close()
	buffer, err := device.NewBuffer(4)
	if err != nil {
		return
	}
	defer buffer.Close()
	kernel, err := device.NewKernel(vulki.KernelOptions{
		WGSL: doubleWGSL,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadWrite},
		},
	})
	if err != nil {
		return
	}
	defer kernel.Close()
	bindings, err := kernel.NewBindings(vulki.BindBuffer(0, buffer))
	if err != nil {
		return
	}
	defer bindings.Close()
	recorder, err := device.NewRecorder()
	if err != nil {
		return
	}
	defer recorder.Close()

	input := make([]byte, 4)
	binary.LittleEndian.PutUint32(input, 21)
	if err := recorder.Update(buffer, 0, input); err != nil {
		return
	}
	if err := recorder.Dispatch(kernel, bindings, vulki.Workgroups{X: 1, Y: 1, Z: 1}); err != nil {
		return
	}
	if err := recorder.Barrier(buffer); err != nil {
		return
	}
	if err := recorder.Dispatch(kernel, bindings, vulki.Workgroups{X: 1, Y: 1, Z: 1}); err != nil {
		return
	}
	_ = recorder.SubmitAndWait()
}
