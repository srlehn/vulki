package compute

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/srlehn/vulki/vk"
)

// TypedBuffer wraps a Buffer with compile-time type safety for element access.
type TypedBuffer[T any] struct {
	Buf   *Buffer
	count int
}

// NewTypedBuffer creates a typed buffer that holds count elements of type T.
func NewTypedBuffer[T any](ctx *Context, count int, usage uint32) (*TypedBuffer[T], error) {
	if err := validateGPUValueType[T](); err != nil {
		return nil, err
	}
	if count <= 0 {
		return nil, fmt.Errorf("compute: typed buffer count must be greater than zero")
	}

	var zero T
	elemSize := unsafe.Sizeof(zero)
	if elemSize == 0 {
		return nil, fmt.Errorf("compute: zero-sized type %T cannot be stored in a GPU buffer", zero)
	}
	if uint64(count) > ^uint64(0)/uint64(elemSize) {
		return nil, fmt.Errorf("compute: typed buffer size overflows uint64")
	}
	size := uint64(count) * uint64(elemSize)
	buf, err := ctx.CreateBuffer(size, usage)
	if err != nil {
		return nil, err
	}
	return &TypedBuffer[T]{Buf: buf, count: count}, nil
}

// UploadSlice copies a Go slice to the device buffer via staging.
func (tb *TypedBuffer[T]) UploadSlice(ctx *Context, data []T) error {
	if err := tb.validate(); err != nil {
		return err
	}
	if len(data) > tb.count {
		return fmt.Errorf("compute: upload element count %d exceeds typed buffer count %d", len(data), tb.count)
	}
	return tb.Buf.Upload(ctx, sliceToBytes(data))
}

// DownloadSlice copies the device buffer contents back to a Go slice.
func (tb *TypedBuffer[T]) DownloadSlice(ctx *Context) ([]T, error) {
	if err := tb.validate(); err != nil {
		return nil, err
	}
	var zero T
	size := uint64(tb.count) * uint64(unsafe.Sizeof(zero))
	if size > tb.Buf.Size() {
		return nil, fmt.Errorf("compute: typed buffer requires %d bytes but buffer has %d", size, tb.Buf.Size())
	}
	raw, err := tb.Buf.Download(ctx, size)
	if err != nil {
		return nil, err
	}
	return bytesToSlice[T](raw, tb.count), nil
}

// Len returns the fixed number of elements allocated for the buffer.
func (tb *TypedBuffer[T]) Len() int {
	if tb == nil {
		return 0
	}
	return tb.count
}

// Destroy releases the underlying buffer.
func (tb *TypedBuffer[T]) Destroy(ctx *Context) {
	if tb != nil && tb.Buf != nil {
		tb.Buf.Destroy(ctx)
	}
}

// DeviceBuffer returns the underlying vk.Buffer handle for descriptor bindings.
func (tb *TypedBuffer[T]) DeviceBuffer() vk.Buffer {
	if tb == nil || tb.Buf == nil {
		return 0
	}
	return tb.Buf.DeviceBuffer
}

func (tb *TypedBuffer[T]) validate() error {
	if tb == nil || tb.Buf == nil || tb.count <= 0 {
		return fmt.Errorf("compute: invalid typed buffer")
	}
	return validateGPUValueType[T]()
}

func validateGPUValueType[T any]() error {
	t := reflect.TypeOf((*T)(nil)).Elem()
	if gpuTypeContainsPointer(t) {
		return fmt.Errorf("compute: type %s contains Go pointers and cannot be stored in a GPU buffer", t)
	}
	return nil
}

func gpuTypeContainsPointer(t reflect.Type) bool {
	switch t.Kind() {
	case reflect.Array:
		return gpuTypeContainsPointer(t.Elem())
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			if gpuTypeContainsPointer(t.Field(i).Type) {
				return true
			}
		}
		return false
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map,
		reflect.Pointer, reflect.Slice, reflect.String, reflect.UnsafePointer:
		return true
	default:
		return false
	}
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
	if len(data) == 0 || count <= 0 {
		return nil
	}
	var zero T
	byteLen := uintptr(count) * unsafe.Sizeof(zero)
	out := make([]T, count)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), byteLen), data)
	return out
}
