package registration

import (
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/srlehn/vulki"
)

type gpuPipeline struct {
	kernel   *vulki.Kernel
	bindings *vulki.BindingSet
}

type gpuBinding struct {
	binding uint32
	access  vulki.BufferAccess
	buffer  *vulki.Buffer
}

func readBinding(binding uint32, buffer *vulki.Buffer) gpuBinding {
	return gpuBinding{binding: binding, access: vulki.BufferReadOnly, buffer: buffer}
}

func readWriteBinding(binding uint32, buffer *vulki.Buffer) gpuBinding {
	return gpuBinding{binding: binding, access: vulki.BufferReadWrite, buffer: buffer}
}

func (c *Correlator) newPipeline(name, source string, bindings []gpuBinding) (*gpuPipeline, error) {
	kernel := c.kernels[name]
	if kernel == nil {
		layouts := make([]vulki.BindingLayout, len(bindings))
		for index, binding := range bindings {
			layouts[index] = vulki.BindingLayout{Binding: binding.binding, Access: binding.access}
		}
		var err error
		kernel, err = c.device.NewKernel(vulki.KernelOptions{WGSL: source, Bindings: layouts})
		if err != nil {
			return nil, fmt.Errorf("registration: create %s kernel: %w", name, err)
		}
		c.kernels[name] = kernel
		c.kernelOrder = append(c.kernelOrder, kernel)
	}
	nativeBindings := make([]vulki.BufferBinding, len(bindings))
	for index, binding := range bindings {
		nativeBindings[index] = vulki.BindBuffer(binding.binding, binding.buffer)
	}
	set, err := kernel.NewBindings(nativeBindings...)
	if err != nil {
		return nil, fmt.Errorf("registration: bind %s kernel: %w", name, err)
	}
	c.bindings = append(c.bindings, set)
	return &gpuPipeline{kernel: kernel, bindings: set}, nil
}

type gpuRecorder struct {
	recorder *vulki.Recorder
	current  *gpuPipeline
	err      error
}

func newGPURecorder(device *vulki.Device) (*gpuRecorder, error) {
	recorder, err := device.NewRecorder()
	if err != nil {
		return nil, err
	}
	return &gpuRecorder{recorder: recorder}, nil
}

func (r *gpuRecorder) Upload(buffer *vulki.Buffer, data []byte) error {
	if r == nil || r.recorder == nil {
		return fmt.Errorf("registration: GPU recorder is closed")
	}
	if r.err != nil {
		return r.err
	}
	if err := r.recorder.Upload(buffer, 0, data); err != nil {
		r.err = err
		return err
	}
	return nil
}

func (r *gpuRecorder) Update(buffer *vulki.Buffer, data []byte) {
	if r == nil || r.recorder == nil || r.err != nil {
		return
	}
	r.err = r.recorder.Update(buffer, 0, data)
}

func (r *gpuRecorder) Bind(pipeline *gpuPipeline) {
	if r == nil || r.err != nil {
		return
	}
	if pipeline == nil || pipeline.kernel == nil || pipeline.bindings == nil {
		r.err = fmt.Errorf("registration: invalid GPU pipeline")
		return
	}
	r.current = pipeline
}

func (r *gpuRecorder) Dispatch(x, y, z uint32) {
	if r == nil || r.err != nil {
		return
	}
	if r.current == nil {
		r.err = fmt.Errorf("registration: no GPU pipeline is bound")
		return
	}
	r.err = r.recorder.Dispatch(
		r.current.kernel,
		r.current.bindings,
		vulki.Workgroups{X: x, Y: y, Z: z},
	)
}

func (r *gpuRecorder) Barrier(buffers ...*vulki.Buffer) {
	if r == nil || r.err != nil {
		return
	}
	r.err = r.recorder.Barrier(buffers...)
}

func (r *gpuRecorder) Download(buffer *vulki.Buffer, destination []byte) error {
	if r == nil || r.recorder == nil {
		return fmt.Errorf("registration: GPU recorder is closed")
	}
	if r.err != nil {
		return r.err
	}
	if err := r.recorder.Download(buffer, 0, destination); err != nil {
		r.err = err
		return err
	}
	return nil
}

func (r *gpuRecorder) SubmitAndWait() error {
	if r == nil || r.recorder == nil {
		return fmt.Errorf("registration: GPU recorder is closed")
	}
	if r.err != nil {
		_ = r.recorder.Abort()
		return r.err
	}
	return r.recorder.SubmitAndWait()
}

func (r *gpuRecorder) Abort() {
	if r != nil && r.recorder != nil {
		_ = r.recorder.Abort()
	}
}

func newElementBuffer[T any](device *vulki.Device, count int) (*vulki.Buffer, error) {
	if count <= 0 {
		return nil, fmt.Errorf("registration: GPU buffer element count must be greater than zero")
	}
	var value T
	elementSize := uint64(unsafe.Sizeof(value))
	if elementSize == 0 || uint64(count) > ^uint64(0)/elementSize {
		return nil, fmt.Errorf("registration: GPU buffer size overflows uint64")
	}
	return device.NewBuffer(uint64(count) * elementSize)
}

func decodeFloat32Slice(encoded []byte) []float32 {
	values := make([]float32, len(encoded)/4)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(encoded[index*4:]))
	}
	return values
}
