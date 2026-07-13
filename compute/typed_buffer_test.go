package compute

import (
	"reflect"
	"testing"
)

func TestGPUTypeContainsPointer(t *testing.T) {
	tests := []struct {
		name   string
		typeOf reflect.Type
		want   bool
	}{
		{name: "float array", typeOf: reflect.TypeFor[[2]float32](), want: false},
		{name: "numeric struct", typeOf: reflect.TypeFor[struct {
			X uint32
			Y float32
		}](), want: false},
		{name: "string", typeOf: reflect.TypeFor[string](), want: true},
		{name: "slice", typeOf: reflect.TypeFor[[]uint32](), want: true},
		{name: "nested pointer", typeOf: reflect.TypeFor[struct{ Values [2]*uint32 }](), want: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := gpuTypeContainsPointer(test.typeOf); got != test.want {
				t.Fatalf("gpuTypeContainsPointer(%s) = %v, want %v", test.typeOf, got, test.want)
			}
		})
	}
}

func TestTypedBufferRejectsPointerTypeBeforeAllocation(t *testing.T) {
	if _, err := NewTypedBuffer[string](nil, 1, 0); err == nil {
		t.Fatal("NewTypedBuffer[string] succeeded")
	}
}

func TestBufferTransferBoundsAreCheckedBeforeVulkanCalls(t *testing.T) {
	buffer := &Buffer{size: 4}
	ctx := &Context{}
	if err := buffer.Upload(ctx, make([]byte, 5)); err == nil {
		t.Fatal("oversized upload succeeded")
	}
	if _, err := buffer.Download(ctx, 5); err == nil {
		t.Fatal("oversized download succeeded")
	}
}
