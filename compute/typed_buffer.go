package compute

import (
	"unsafe"

	"github.com/srlehn/vulki/vk"
)

// TypedBuffer wraps a Buffer with compile-time type safety for element access.
type TypedBuffer[T any] struct {
	Buf   *Buffer
	Count int
}

// NewTypedBuffer creates a typed buffer that holds count elements of type T.
func NewTypedBuffer[T any](ctx *Context, count int, usage uint32) (*TypedBuffer[T], error) {
	var zero T
	elemSize := unsafe.Sizeof(zero)
	size := uint64(count) * uint64(elemSize)
	buf, err := ctx.CreateBuffer(size, usage)
	if err != nil {
		return nil, err
	}
	return &TypedBuffer[T]{Buf: buf, Count: count}, nil
}

// UploadSlice copies a Go slice to the device buffer via staging.
func (tb *TypedBuffer[T]) UploadSlice(ctx *Context, data []T) error {
	return tb.Buf.Upload(ctx, sliceToBytes(data))
}

// DownloadSlice copies the device buffer contents back to a Go slice.
func (tb *TypedBuffer[T]) DownloadSlice(ctx *Context) ([]T, error) {
	raw, err := tb.Buf.Download(ctx, tb.Buf.Size)
	if err != nil {
		return nil, err
	}
	return bytesToSlice[T](raw, tb.Count), nil
}

// Destroy releases the underlying buffer.
func (tb *TypedBuffer[T]) Destroy(ctx *Context) {
	tb.Buf.Destroy(ctx)
}

// DeviceBuffer returns the underlying vk.Buffer handle for descriptor bindings.
func (tb *TypedBuffer[T]) DeviceBuffer() vk.Buffer {
	return tb.Buf.DeviceBuffer
}

func sliceToBytes[T any](data []T) []byte {
	if len(data) == 0 {
		return nil
	}
	var zero T
	elemSize := unsafe.Sizeof(zero)
	return unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), uintptr(len(data))*elemSize)
}

func bytesToSlice[T any](data []byte, count int) []T {
	if len(data) == 0 {
		return nil
	}
	out := make([]T, count)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), len(data)), data)
	return out
}
