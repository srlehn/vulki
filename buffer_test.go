package vulki

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/srlehn/vulki/vk"
)

func TestNewBufferCleansPartialAllocation(t *testing.T) {
	tests := []struct {
		failure string
		want    []string
	}{
		{failure: "create"},
		{failure: "memory-type", want: []string{"destroy-buffer:1"}},
		{failure: "allocate", want: []string{"destroy-buffer:1"}},
		{failure: "bind", want: []string{"destroy-buffer:1", "free-memory:1"}},
	}

	for _, test := range tests {
		t.Run(test.failure, func(t *testing.T) {
			device, events, _ := fakeBufferDevice(test.failure)
			buffer, err := device.NewBuffer(64)
			if err == nil {
				if buffer != nil {
					_ = buffer.Close()
				}
				t.Fatal("NewBuffer succeeded")
			}
			if buffer != nil {
				t.Fatalf("buffer = %#v, want nil", buffer)
			}
			if !reflect.DeepEqual(*events, test.want) {
				t.Fatalf("cleanup = %v, want %v", *events, test.want)
			}
		})
	}
}

func TestBufferStagingIsLazyAndCloseIsIdempotent(t *testing.T) {
	device, events, createCount := fakeBufferDevice("")
	buffer, err := device.NewBuffer(64)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	if got := *createCount; got != 1 {
		t.Fatalf("buffer creations = %d, want only the storage buffer", got)
	}
	if err := buffer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	want := []string{"destroy-buffer:1", "free-memory:1"}
	if !reflect.DeepEqual(*events, want) {
		t.Fatalf("cleanup = %v, want %v", *events, want)
	}
	if err := buffer.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if !reflect.DeepEqual(*events, want) {
		t.Fatalf("second Close repeated cleanup: %v", *events)
	}
}

func TestBufferCleansPartialStagingAllocation(t *testing.T) {
	tests := []struct {
		failure string
		want    []string
	}{
		{failure: "staging-create", want: []string{"destroy-buffer:1", "free-memory:1"}},
		{failure: "staging-memory-type", want: []string{"destroy-buffer:2", "destroy-buffer:1", "free-memory:1"}},
		{failure: "staging-allocate", want: []string{"destroy-buffer:2", "destroy-buffer:1", "free-memory:1"}},
		{failure: "staging-bind", want: []string{"destroy-buffer:2", "free-memory:2", "destroy-buffer:1", "free-memory:1"}},
	}

	for _, test := range tests {
		t.Run(test.failure, func(t *testing.T) {
			device, events, _ := fakeBufferDevice(test.failure)
			buffer, err := device.NewBuffer(64)
			if err != nil {
				t.Fatalf("NewBuffer: %v", err)
			}
			if err := buffer.Upload([]byte{1, 2, 3, 4}); err == nil {
				t.Fatal("Upload succeeded")
			}
			if err := buffer.Close(); err != nil {
				t.Fatalf("Close: %v", err)
			}
			if !reflect.DeepEqual(*events, test.want) {
				t.Fatalf("cleanup = %v, want %v", *events, test.want)
			}
		})
	}
}

func TestBufferRejectsBoundsBeforeTransferAllocation(t *testing.T) {
	device, _, createCount := fakeBufferDevice("")
	buffer, err := device.NewBuffer(8)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	defer buffer.Close()

	tests := []struct {
		name string
		call func() error
	}{
		{name: "upload offset", call: func() error { return buffer.UploadAt(9, nil) }},
		{name: "upload length", call: func() error { return buffer.UploadAt(7, []byte{1, 2}) }},
		{name: "download offset", call: func() error { return buffer.DownloadAt(9, nil) }},
		{name: "download length", call: func() error { return buffer.DownloadAt(7, make([]byte, 2)) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.call(); err == nil {
				t.Fatal("out-of-bounds transfer succeeded")
			}
		})
	}
	if got := *createCount; got != 1 {
		t.Fatalf("buffer creations = %d, bounds errors allocated staging", got)
	}
}

func TestBufferUseAfterClose(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	buffer, err := device.NewBuffer(8)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	if err := buffer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := buffer.Upload(nil); err == nil {
		t.Fatal("Upload on closed buffer succeeded")
	}
	if err := buffer.Download(nil); err == nil {
		t.Fatal("Download on closed buffer succeeded")
	}
}

func TestBufferRoundTripDirectVulkan(t *testing.T) {
	device, err := Open()
	if err != nil {
		t.Skipf("direct Vulkan device unavailable: %v", err)
	}
	defer device.Close()

	buffer, err := device.NewBuffer(32)
	if err != nil {
		t.Fatalf("NewBuffer: %v", err)
	}
	defer buffer.Close()

	input := make([]byte, 32)
	for index := range input {
		input[index] = byte(index*7 + 3)
	}
	if err := buffer.Upload(input); err != nil {
		t.Fatalf("Upload: %v", err)
	}
	output, err := buffer.Bytes()
	if err != nil {
		t.Fatalf("Bytes: %v", err)
	}
	if !bytes.Equal(output, input) {
		t.Fatalf("full round trip = %v, want %v", output, input)
	}

	patch := []byte{91, 92, 93, 94, 95}
	if err := buffer.UploadAt(11, patch); err != nil {
		t.Fatalf("UploadAt: %v", err)
	}
	partial := make([]byte, len(patch))
	if err := buffer.DownloadAt(11, partial); err != nil {
		t.Fatalf("DownloadAt: %v", err)
	}
	if !bytes.Equal(partial, patch) {
		t.Fatalf("partial round trip = %v, want %v", partial, patch)
	}
}

func TestNewBufferRejectsZeroAndClosedDevice(t *testing.T) {
	device, _, createCount := fakeBufferDevice("")
	if _, err := device.NewBuffer(0); err == nil {
		t.Fatal("zero-sized NewBuffer succeeded")
	}
	if got := *createCount; got != 0 {
		t.Fatalf("zero-sized buffer reached backend %d times", got)
	}
	device.mu.Lock()
	device.state = nil
	device.closed = true
	device.mu.Unlock()
	if _, err := device.NewBuffer(1); err == nil {
		t.Fatal("NewBuffer on closed device succeeded")
	}
}

func TestNewBufferRejectsStorageLimit(t *testing.T) {
	device, _, createCount := fakeBufferDevice("")
	device.info.Limits.MaxStorageBufferSize = 8
	if _, err := device.NewBuffer(9); err == nil {
		t.Fatal("oversized NewBuffer succeeded")
	}
	if got := *createCount; got != 0 {
		t.Fatalf("oversized buffer reached backend %d times", got)
	}
}

func fakeBufferDevice(failure string) (*Device, *[]string, *int) {
	events := new([]string)
	createCount := new(int)
	nextMemory := 0
	allocationCount := 0
	transferBuffers := make(map[vk.Buffer]bool)
	operations := deviceOps{
		createBuffer: func(_ *vk.DeviceFuncs, _ vk.Device, info *vk.BufferCreateInfo) (vk.Buffer, error) {
			if failure == "create" || failure == "staging-create" && *createCount == 1 {
				return 0, errors.New("injected create failure")
			}
			(*createCount)++
			buffer := vk.Buffer(*createCount)
			transferBuffers[buffer] = info.Usage&vk.BufferUsageStorageBufferBit == 0
			return buffer, nil
		},
		destroyBuffer: func(_ *vk.DeviceFuncs, _ vk.Device, buffer vk.Buffer) {
			*events = append(*events, fmt.Sprintf("destroy-buffer:%d", buffer))
		},
		bufferMemoryRequirements: func(_ *vk.DeviceFuncs, _ vk.Device, buffer vk.Buffer) vk.MemoryRequirements {
			bits := uint32(1)
			if failure == "memory-type" && !transferBuffers[buffer] {
				bits = 2
			}
			if transferBuffers[buffer] {
				bits = 2
				if failure == "staging-memory-type" {
					bits = 1
				}
			}
			return vk.MemoryRequirements{Size: 64, MemoryTypeBits: bits}
		},
		allocateMemory: func(*vk.DeviceFuncs, vk.Device, *vk.MemoryAllocateInfo) (vk.DeviceMemory, error) {
			allocationCount++
			if failure == "allocate" || failure == "staging-allocate" && allocationCount == 2 {
				return 0, errors.New("injected allocation failure")
			}
			nextMemory++
			return vk.DeviceMemory(nextMemory), nil
		},
		freeMemory: func(_ *vk.DeviceFuncs, _ vk.Device, memory vk.DeviceMemory) {
			*events = append(*events, fmt.Sprintf("free-memory:%d", memory))
		},
		bindBufferMemory: func(_ *vk.DeviceFuncs, _ vk.Device, buffer vk.Buffer, _ vk.DeviceMemory, _ uint64) error {
			if failure == "bind" || failure == "staging-bind" && transferBuffers[buffer] {
				return errors.New("injected bind failure")
			}
			return nil
		},
	}
	state := &deviceState{
		device: vk.Device(1),
		memory: vk.PhysicalDeviceMemoryProperties{
			MemoryTypeCount: 2,
			MemoryTypes: [vk.MaxMemoryTypes]vk.MemoryType{
				{PropertyFlags: vk.MemoryPropertyDeviceLocalBit},
				{PropertyFlags: vk.MemoryPropertyHostVisibleBit | vk.MemoryPropertyHostCoherentBit},
			},
		},
		ops: operations,
	}
	device := &Device{state: state, closeDone: make(chan struct{})}
	device.cond = sync.NewCond(&device.mu)
	return device, events, createCount
}
