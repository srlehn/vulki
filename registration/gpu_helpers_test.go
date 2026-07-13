package registration

import (
	"fmt"
	"testing"
	"unsafe"

	"github.com/srlehn/vulki"
	"github.com/srlehn/vulki/vk"
)

type testGPUContext struct {
	device *vulki.Device
}

func newTestGPUContext(t *testing.T) *testGPUContext {
	t.Helper()
	device, err := vulki.Open()
	if err != nil {
		t.Skipf("Vulkan unavailable: %v", err)
	}
	return &testGPUContext{device: device}
}

func (c *testGPUContext) Close() {
	if c != nil && c.device != nil {
		_ = c.device.Close()
		c.device = nil
	}
}

func (c *testGPUContext) CreateBuffer(size uint64, _ vk.BufferUsageFlags) (*vulki.Buffer, error) {
	return c.device.NewBuffer(size)
}

func (c *testGPUContext) NewCommandRecorder() (*gpuRecorder, error) {
	return newGPURecorder(c.device)
}

func (c *testGPUContext) Dispatch(pipeline *gpuPipeline, x, y, z uint32) error {
	if pipeline == nil {
		return fmt.Errorf("registration: invalid test GPU pipeline")
	}
	return c.device.DispatchAndWait(
		pipeline.kernel,
		pipeline.bindings,
		vulki.Workgroups{X: x, Y: y, Z: z},
	)
}

type testTypedBuffer[T any] struct {
	Buf   *vulki.Buffer
	count int
}

func newTestTypedBuffer[T any](ctx *testGPUContext, count int, _ vk.BufferUsageFlags) (*testTypedBuffer[T], error) {
	buffer, err := newElementBuffer[T](ctx.device, count)
	if err != nil {
		return nil, err
	}
	return &testTypedBuffer[T]{Buf: buffer, count: count}, nil
}

func (b *testTypedBuffer[T]) UploadSlice(_ *testGPUContext, values []T) error {
	if b == nil || b.Buf == nil || len(values) > b.count {
		return fmt.Errorf("registration: invalid test GPU upload")
	}
	return b.Buf.Upload(testValueBytes(values))
}

func (b *testTypedBuffer[T]) DownloadSlice(_ *testGPUContext) ([]T, error) {
	if b == nil || b.Buf == nil {
		return nil, fmt.Errorf("registration: invalid test GPU download")
	}
	encoded, err := b.Buf.Bytes()
	if err != nil {
		return nil, err
	}
	return testBytesToValues[T](encoded, b.count), nil
}

func (b *testTypedBuffer[T]) Destroy(_ *testGPUContext) {
	if b != nil && b.Buf != nil {
		_ = b.Buf.Close()
		b.Buf = nil
	}
}

func testValueBytes[T any](values []T) []byte {
	if len(values) == 0 {
		return nil
	}
	size := int(unsafe.Sizeof(values[0])) * len(values)
	encoded := make([]byte, size)
	copy(encoded, unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), size))
	return encoded
}

func testBytesToValues[T any](encoded []byte, count int) []T {
	values := make([]T, count)
	if count == 0 {
		return values
	}
	size := int(unsafe.Sizeof(values[0])) * count
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), size), encoded[:size])
	return values
}

type testBufferBinding struct {
	binding uint32
	buffer  *vulki.Buffer
}

func (r *gpuRecorder) UpdateBuffer(buffer *vulki.Buffer, offset uint64, data []byte) {
	if r == nil || r.recorder == nil || r.err != nil {
		return
	}
	r.err = r.recorder.Update(buffer, offset, data)
}

func (r *gpuRecorder) BarrierTransferToCompute(_ ...*vulki.Buffer) {}

func (r *gpuRecorder) Submit() error {
	return r.SubmitAndWait()
}

func (p *gpuPipeline) Destroy(_ *testGPUContext) {
	if p == nil {
		return
	}
	if p.bindings != nil {
		_ = p.bindings.Close()
		p.bindings = nil
	}
	if p.kernel != nil {
		_ = p.kernel.Close()
		p.kernel = nil
	}
}
