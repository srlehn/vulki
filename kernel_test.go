package vulki

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/srlehn/vulki/vk"
)

const doubleKernelWGSL = `
struct Values {
    data: array<f32, 64>,
}

@group(0) @binding(0) var<storage, read> input: Values;
@group(0) @binding(1) var<storage, read_write> output: Values;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < 64u {
        output.data[id.x] = input.data[id.x] * 2.0;
    }
}
`

func TestKernelReusesTwoBindingSetsDirectVulkan(t *testing.T) {
	device, err := Open()
	if err != nil {
		t.Skipf("direct Vulkan device unavailable: %v", err)
	}
	defer device.Close()

	kernel, err := device.NewKernel(KernelOptions{
		WGSL: doubleKernelWGSL,
		Bindings: []BindingLayout{
			{Binding: 0, Access: BufferReadOnly},
			{Binding: 1, Access: BufferReadWrite},
		},
	})
	if err != nil {
		t.Fatalf("NewKernel: %v", err)
	}
	defer kernel.Close()

	for run := 1; run <= 2; run++ {
		input, err := device.NewBuffer(64 * 4)
		if err != nil {
			t.Fatalf("run %d input buffer: %v", run, err)
		}
		defer input.Close()
		output, err := device.NewBuffer(64 * 4)
		if err != nil {
			t.Fatalf("run %d output buffer: %v", run, err)
		}
		defer output.Close()

		values := make([]float32, 64)
		for index := range values {
			values[index] = float32(run*100 + index)
		}
		if err := input.Upload(encodeFloat32s(values)); err != nil {
			t.Fatalf("run %d Upload: %v", run, err)
		}
		bindings, err := kernel.NewBindings(
			BindBuffer(0, input),
			BindBuffer(1, output),
		)
		if err != nil {
			t.Fatalf("run %d NewBindings: %v", run, err)
		}
		if err := input.Close(); err == nil {
			t.Fatalf("run %d input closed while bound", run)
		}
		if err := kernel.Close(); err == nil {
			t.Fatalf("run %d kernel closed while bound", run)
		}
		if err := device.DispatchAndWait(kernel, bindings, Workgroups{X: 1, Y: 1, Z: 1}); err != nil {
			t.Fatalf("run %d DispatchAndWait: %v", run, err)
		}

		resultBytes := make([]byte, 64*4)
		if err := output.Download(resultBytes); err != nil {
			t.Fatalf("run %d Download: %v", run, err)
		}
		results := decodeFloat32s(resultBytes)
		for index, result := range results {
			if want := values[index] * 2; result != want {
				t.Fatalf("run %d result[%d] = %v, want %v", run, index, result, want)
			}
		}
		if err := bindings.Close(); err != nil {
			t.Fatalf("run %d bindings Close: %v", run, err)
		}
		if err := input.Close(); err != nil {
			t.Fatalf("run %d input Close: %v", run, err)
		}
		if err := output.Close(); err != nil {
			t.Fatalf("run %d output Close: %v", run, err)
		}
	}
}

func TestKernelBindingValidationRejectsWrongOwner(t *testing.T) {
	first, _, _ := fakeBufferDevice("")
	second, _, _ := fakeBufferDevice("")
	kernel := &Kernel{
		device:  first,
		layouts: map[uint32]BufferAccess{0: BufferReadOnly},
	}
	buffer := &Buffer{device: second, size: 4, buffer: 1}
	if _, err := kernel.NewBindings(BindBuffer(0, buffer)); err == nil {
		t.Fatal("binding a buffer from another device succeeded")
	}
}

func TestKernelOptionsRejectInvalidLayoutsBeforeCompilation(t *testing.T) {
	var device Device
	if _, err := device.NewKernel(KernelOptions{}); err == nil {
		t.Fatal("empty kernel succeeded")
	}
	if _, err := device.NewKernel(KernelOptions{
		WGSL: "invalid but must not be compiled",
		Bindings: []BindingLayout{
			{Binding: 2, Access: BufferReadOnly},
			{Binding: 2, Access: BufferReadWrite},
		},
	}); err == nil {
		t.Fatal("duplicate kernel layout succeeded")
	}
	if _, err := device.NewKernel(KernelOptions{
		WGSL:     "invalid but must not be compiled",
		Bindings: []BindingLayout{{Binding: 0, Access: BufferAccess(99)}},
	}); err == nil {
		t.Fatal("invalid buffer access succeeded")
	}
}

func TestNewKernelCleansPartialResources(t *testing.T) {
	tests := []struct {
		failure string
		want    []string
	}{
		{failure: "shader"},
		{failure: "descriptor", want: []string{"shader:1"}},
		{failure: "layout", want: []string{"descriptor:2", "shader:1"}},
		{failure: "pipeline", want: []string{"layout:3", "descriptor:2", "shader:1"}},
	}

	for _, test := range tests {
		t.Run(test.failure, func(t *testing.T) {
			device, _, _ := fakeBufferDevice("")
			var cleanup []string
			device.state.kernelOps = fakeKernelOperations(test.failure, &cleanup)
			kernel, err := device.NewKernel(KernelOptions{
				WGSL:     doubleKernelWGSL,
				Bindings: []BindingLayout{{Binding: 0, Access: BufferReadOnly}},
			})
			if err == nil {
				if kernel != nil {
					_ = kernel.Close()
				}
				t.Fatal("NewKernel succeeded")
			}
			if kernel != nil {
				t.Fatalf("kernel = %#v, want nil", kernel)
			}
			if !reflect.DeepEqual(cleanup, test.want) {
				t.Fatalf("cleanup = %v, want %v", cleanup, test.want)
			}
		})
	}
}

func TestNewBindingsCleansPoolAndReferences(t *testing.T) {
	for _, failure := range []string{"pool", "sets"} {
		t.Run(failure, func(t *testing.T) {
			device, _, _ := fakeBufferDevice("")
			var cleanup []string
			device.state.kernelOps = fakeKernelOperations(failure, &cleanup)
			kernel := &Kernel{
				device:           device,
				descriptorLayout: 2,
				layouts:          map[uint32]BufferAccess{0: BufferReadOnly},
			}
			buffer, err := device.NewBuffer(4)
			if err != nil {
				t.Fatalf("NewBuffer: %v", err)
			}

			bindings, err := kernel.NewBindings(BindBuffer(0, buffer))
			if err == nil {
				if bindings != nil {
					_ = bindings.Close()
				}
				t.Fatal("NewBindings succeeded")
			}
			if bindings != nil {
				t.Fatalf("bindings = %#v, want nil", bindings)
			}
			if kernel.bindingSets != 0 || buffer.references != 0 {
				t.Fatalf("references after failure: kernel=%d buffer=%d", kernel.bindingSets, buffer.references)
			}
			if failure == "sets" {
				want := []string{"pool:5"}
				if !reflect.DeepEqual(cleanup, want) {
					t.Fatalf("cleanup = %v, want %v", cleanup, want)
				}
			}
			if err := buffer.Close(); err != nil {
				t.Fatalf("buffer Close after failed bindings: %v", err)
			}
		})
	}
}

func TestDispatchRejectsInvalidResourcesAndWorkgroups(t *testing.T) {
	first, _, _ := fakeBufferDevice("")
	second, _, _ := fakeBufferDevice("")
	kernel := &Kernel{device: first}
	bindings := &BindingSet{device: first, kernel: kernel}
	if err := second.DispatchAndWait(kernel, bindings, Workgroups{X: 1, Y: 1, Z: 1}); err == nil {
		t.Fatal("wrong-device dispatch succeeded")
	}
	if err := first.DispatchAndWait(kernel, bindings, Workgroups{}); err == nil {
		t.Fatal("zero-workgroup dispatch succeeded")
	}
	first.info.Limits.MaxComputeWorkGroupCount = [3]uint32{1, 1, 1}
	if err := first.DispatchAndWait(kernel, bindings, Workgroups{X: 2, Y: 1, Z: 1}); err == nil {
		t.Fatal("oversized dispatch succeeded")
	}
}

func encodeFloat32s(values []float32) []byte {
	encoded := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(encoded[index*4:], math.Float32bits(value))
	}
	return encoded
}

func decodeFloat32s(encoded []byte) []float32 {
	values := make([]float32, len(encoded)/4)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(encoded[index*4:]))
	}
	return values
}

func fakeKernelOperations(failure string, cleanup *[]string) kernelOps {
	fail := func(name string) error {
		return errors.New("injected " + name + " failure")
	}
	return kernelOps{
		createShaderModule: func(*vk.DeviceFuncs, vk.Device, *vk.ShaderModuleCreateInfo) (vk.ShaderModule, error) {
			if failure == "shader" {
				return 0, fail("shader")
			}
			return 1, nil
		},
		destroyShaderModule: func(_ *vk.DeviceFuncs, _ vk.Device, handle vk.ShaderModule) {
			*cleanup = append(*cleanup, fmt.Sprintf("shader:%d", handle))
		},
		createDescriptorLayout: func(*vk.DeviceFuncs, vk.Device, *vk.DescriptorSetLayoutCreateInfo) (vk.DescriptorSetLayout, error) {
			if failure == "descriptor" {
				return 0, fail("descriptor")
			}
			return 2, nil
		},
		destroyDescriptorLayout: func(_ *vk.DeviceFuncs, _ vk.Device, handle vk.DescriptorSetLayout) {
			*cleanup = append(*cleanup, fmt.Sprintf("descriptor:%d", handle))
		},
		createPipelineLayout: func(*vk.DeviceFuncs, vk.Device, *vk.PipelineLayoutCreateInfo) (vk.PipelineLayout, error) {
			if failure == "layout" {
				return 0, fail("layout")
			}
			return 3, nil
		},
		destroyPipelineLayout: func(_ *vk.DeviceFuncs, _ vk.Device, handle vk.PipelineLayout) {
			*cleanup = append(*cleanup, fmt.Sprintf("layout:%d", handle))
		},
		createComputePipelines: func(*vk.DeviceFuncs, vk.Device, []vk.ComputePipelineCreateInfo) ([]vk.Pipeline, error) {
			if failure == "pipeline" {
				return nil, fail("pipeline")
			}
			return []vk.Pipeline{4}, nil
		},
		destroyPipeline: func(_ *vk.DeviceFuncs, _ vk.Device, handle vk.Pipeline) {
			*cleanup = append(*cleanup, fmt.Sprintf("pipeline:%d", handle))
		},
		createDescriptorPool: func(*vk.DeviceFuncs, vk.Device, *vk.DescriptorPoolCreateInfo) (vk.DescriptorPool, error) {
			if failure == "pool" {
				return 0, fail("pool")
			}
			return 5, nil
		},
		destroyDescriptorPool: func(_ *vk.DeviceFuncs, _ vk.Device, handle vk.DescriptorPool) {
			*cleanup = append(*cleanup, fmt.Sprintf("pool:%d", handle))
		},
		allocateDescriptorSets: func(*vk.DeviceFuncs, vk.Device, *vk.DescriptorSetAllocateInfo) ([]vk.DescriptorSet, error) {
			if failure == "sets" {
				return nil, fail("sets")
			}
			return []vk.DescriptorSet{6}, nil
		},
		writeDescriptorSets: func(*vk.DeviceFuncs, vk.Device, []vk.WriteDescriptorSet) {},
	}
}
