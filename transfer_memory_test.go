package vulki

import (
	"errors"
	"reflect"
	"strings"
	"testing"
	"unsafe"

	"github.com/srlehn/vulki/vk"
)

func TestSelectTransferMemoryTypeRTXPreferences(t *testing.T) {
	properties := vk.PhysicalDeviceMemoryProperties{MemoryTypeCount: 6}
	properties.MemoryTypes[3].PropertyFlags =
		vk.MemoryPropertyHostVisibleBit | vk.MemoryPropertyHostCoherentBit
	properties.MemoryTypes[4].PropertyFlags =
		vk.MemoryPropertyHostVisibleBit |
			vk.MemoryPropertyHostCoherentBit |
			vk.MemoryPropertyHostCachedBit
	properties.MemoryTypes[5].PropertyFlags =
		vk.MemoryPropertyDeviceLocalBit |
			vk.MemoryPropertyHostVisibleBit |
			vk.MemoryPropertyHostCoherentBit
	typeBits := uint32(1<<3 | 1<<4 | 1<<5)

	uploadIndex, uploadFlags, err := selectTransferMemoryType(properties, typeBits, transferUpload)
	if err != nil {
		t.Fatalf("select upload memory: %v", err)
	}
	if uploadIndex != 3 || uploadFlags != properties.MemoryTypes[3].PropertyFlags {
		t.Fatalf("upload selection = type %d flags %#x, want type 3 flags %#x",
			uploadIndex, uint32(uploadFlags), uint32(properties.MemoryTypes[3].PropertyFlags))
	}

	downloadIndex, downloadFlags, err := selectTransferMemoryType(properties, typeBits, transferDownload)
	if err != nil {
		t.Fatalf("select download memory: %v", err)
	}
	if downloadIndex != 4 || downloadFlags != properties.MemoryTypes[4].PropertyFlags {
		t.Fatalf("download selection = type %d flags %#x, want type 4 flags %#x",
			downloadIndex, uint32(downloadFlags), uint32(properties.MemoryTypes[4].PropertyFlags))
	}
}

func TestSelectTransferMemoryTypeFallbacks(t *testing.T) {
	visible := vk.MemoryPropertyHostVisibleBit
	coherent := vk.MemoryPropertyHostCoherentBit
	cached := vk.MemoryPropertyHostCachedBit
	tests := []struct {
		name      string
		direction transferDirection
		flags     []vk.MemoryPropertyFlags
		typeBits  uint32
		wantIndex uint32
	}{
		{
			name:      "cached noncoherent download",
			direction: transferDownload,
			flags:     []vk.MemoryPropertyFlags{visible | coherent, visible | cached},
			typeBits:  0b11,
			wantIndex: 1,
		},
		{
			name:      "coherent uncached download",
			direction: transferDownload,
			flags:     []vk.MemoryPropertyFlags{visible | coherent, visible | cached | coherent},
			typeBits:  0b01,
			wantIndex: 0,
		},
		{
			name:      "cached coherent upload fallback",
			direction: transferUpload,
			flags:     []vk.MemoryPropertyFlags{visible | cached, visible | cached | coherent},
			typeBits:  0b11,
			wantIndex: 1,
		},
		{
			name:      "lowest index wins a tie",
			direction: transferDownload,
			flags:     []vk.MemoryPropertyFlags{visible | cached | coherent, visible | cached | coherent},
			typeBits:  0b11,
			wantIndex: 0,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			properties := vk.PhysicalDeviceMemoryProperties{MemoryTypeCount: uint32(len(test.flags))}
			for index, flags := range test.flags {
				properties.MemoryTypes[index].PropertyFlags = flags
			}
			index, flags, err := selectTransferMemoryType(properties, test.typeBits, test.direction)
			if err != nil {
				t.Fatalf("select memory: %v", err)
			}
			if index != test.wantIndex || flags != test.flags[test.wantIndex] {
				t.Fatalf("selection = type %d flags %#x, want type %d flags %#x",
					index, uint32(flags), test.wantIndex, uint32(test.flags[test.wantIndex]))
			}
		})
	}
}

func TestSelectTransferMemoryTypeRejectsIncompatibleTypes(t *testing.T) {
	properties := vk.PhysicalDeviceMemoryProperties{MemoryTypeCount: 2}
	properties.MemoryTypes[0].PropertyFlags =
		vk.MemoryPropertyHostVisibleBit |
			vk.MemoryPropertyHostCoherentBit |
			vk.MemoryPropertyHostCachedBit
	properties.MemoryTypes[1].PropertyFlags = vk.MemoryPropertyDeviceLocalBit

	_, _, err := selectTransferMemoryType(properties, 0b10, transferDownload)
	if err == nil {
		t.Fatal("selected an incompatible memory type")
	}
	if !strings.Contains(err.Error(), "download") || !strings.Contains(err.Error(), "0x2") {
		t.Fatalf("selection error = %q, want direction and type bits", err)
	}
}

func TestCreateTransferResourceUsesDirectionSpecificPolicy(t *testing.T) {
	properties := vk.PhysicalDeviceMemoryProperties{MemoryTypeCount: 2}
	properties.MemoryTypes[0].PropertyFlags =
		vk.MemoryPropertyHostVisibleBit | vk.MemoryPropertyHostCoherentBit
	properties.MemoryTypes[1].PropertyFlags =
		vk.MemoryPropertyHostVisibleBit |
			vk.MemoryPropertyHostCoherentBit |
			vk.MemoryPropertyHostCachedBit

	var usages []vk.BufferUsageFlags
	var memoryTypes []uint32
	nextBuffer := vk.Buffer(10)
	nextMemory := vk.DeviceMemory(20)
	state := &deviceState{
		device: vk.Device(1),
		memory: properties,
		ops: deviceOps{
			createBuffer: func(_ *vk.DeviceFuncs, _ vk.Device, info *vk.BufferCreateInfo) (vk.Buffer, error) {
				usages = append(usages, info.Usage)
				nextBuffer++
				return nextBuffer, nil
			},
			destroyBuffer: func(*vk.DeviceFuncs, vk.Device, vk.Buffer) {},
			bufferMemoryRequirements: func(*vk.DeviceFuncs, vk.Device, vk.Buffer) vk.MemoryRequirements {
				return vk.MemoryRequirements{Size: 128, MemoryTypeBits: 0b11}
			},
			allocateMemory: func(_ *vk.DeviceFuncs, _ vk.Device, info *vk.MemoryAllocateInfo) (vk.DeviceMemory, error) {
				memoryTypes = append(memoryTypes, info.MemoryTypeIndex)
				nextMemory++
				return nextMemory, nil
			},
			freeMemory:       func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) {},
			bindBufferMemory: func(*vk.DeviceFuncs, vk.Device, vk.Buffer, vk.DeviceMemory, uint64) error { return nil },
		},
	}

	upload, err := createTransferResource(state, transferUpload, 64)
	if err != nil {
		t.Fatalf("create upload resource: %v", err)
	}
	download, err := createTransferResource(state, transferDownload, 64)
	if err != nil {
		t.Fatalf("create download resource: %v", err)
	}
	if !reflect.DeepEqual(usages, []vk.BufferUsageFlags{
		vk.BufferUsageTransferSrcBit,
		vk.BufferUsageTransferDstBit,
	}) {
		t.Fatalf("buffer usages = %#v", usages)
	}
	if !reflect.DeepEqual(memoryTypes, []uint32{0, 1}) {
		t.Fatalf("memory type indices = %v, want [0 1]", memoryTypes)
	}
	if upload.direction != transferUpload || upload.properties != properties.MemoryTypes[0].PropertyFlags {
		t.Fatalf("upload resource metadata = %#v", upload)
	}
	if download.direction != transferDownload || download.properties != properties.MemoryTypes[1].PropertyFlags {
		t.Fatalf("download resource metadata = %#v", download)
	}
	if upload.capacity != 64 || upload.allocationSize != 128 || download.capacity != 64 || download.allocationSize != 128 {
		t.Fatalf("resource sizes: upload=%#v download=%#v", upload, download)
	}
}

func TestNonCoherentRange(t *testing.T) {
	tests := []struct {
		name                 string
		offset, size         uint64
		allocation, atomSize uint64
		wantOffset, wantSize uint64
	}{
		{name: "below atom", size: 1, allocation: 128, atomSize: 64, wantSize: 64},
		{name: "equal atom", size: 64, allocation: 128, atomSize: 64, wantSize: 64},
		{name: "above atom", size: 65, allocation: 128, atomSize: 64, wantSize: 128},
		{name: "nonzero offset", offset: 65, size: 1, allocation: 256, atomSize: 64, wantOffset: 64, wantSize: 64},
		{name: "allocation end", offset: 64, size: 36, allocation: 100, atomSize: 64, wantOffset: 64, wantSize: 36},
		{name: "zero atom in injected state", offset: 3, size: 2, allocation: 8, wantOffset: 3, wantSize: 2},
		{
			name:       "overflow-safe allocation end",
			offset:     ^uint64(0) - 1,
			size:       1,
			allocation: ^uint64(0),
			atomSize:   64,
			wantOffset: ^uint64(0) - 63,
			wantSize:   63,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			offset, size, err := nonCoherentRange(test.offset, test.size, test.allocation, test.atomSize)
			if err != nil {
				t.Fatalf("nonCoherentRange: %v", err)
			}
			if offset != test.wantOffset || size != test.wantSize {
				t.Fatalf("range = offset %d size %d, want offset %d size %d",
					offset, size, test.wantOffset, test.wantSize)
			}
		})
	}

	for _, test := range []struct {
		name                     string
		offset, size, allocation uint64
	}{
		{name: "zero size", size: 0, allocation: 1},
		{name: "offset beyond allocation", offset: 2, size: 1, allocation: 1},
		{name: "size beyond allocation", offset: 1, size: 2, allocation: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, _, err := nonCoherentRange(test.offset, test.size, test.allocation, 64); err == nil {
				t.Fatal("invalid range succeeded")
			}
		})
	}
}

func TestReadTransferMemoryCoherentSkipsInvalidation(t *testing.T) {
	mapped := make([]byte, 16)
	copy(mapped, []byte{0, 1, 2, 3, 4, 5, 6, 7})
	destination := make([]byte, 4)
	mapCalls := 0
	unmapCalls := 0
	invalidateCalls := 0
	state := &deviceState{ops: deviceOps{
		mapMemory: func(_ *vk.DeviceFuncs, _ vk.Device, _ vk.DeviceMemory, offset, size uint64) (unsafe.Pointer, error) {
			mapCalls++
			if offset != 0 || size != uint64(len(mapped)) {
				t.Fatalf("mapped range = offset %d size %d", offset, size)
			}
			return unsafe.Pointer(&mapped[0]), nil
		},
		unmapMemory: func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) { unmapCalls++ },
		invalidateMappedMemoryRanges: func(*vk.DeviceFuncs, vk.Device, []vk.MappedMemoryRange) error {
			invalidateCalls++
			return nil
		},
	}}
	resource := &transferResource{
		memory:         3,
		allocationSize: uint64(len(mapped)),
		properties:     vk.MemoryPropertyHostVisibleBit | vk.MemoryPropertyHostCoherentBit,
		direction:      transferDownload,
	}
	if err := readTransferMemory(state, resource, 2, destination); err != nil {
		t.Fatalf("readTransferMemory: %v", err)
	}
	if !reflect.DeepEqual(destination, []byte{2, 3, 4, 5}) {
		t.Fatalf("destination = %v", destination)
	}
	if mapCalls != 1 || unmapCalls != 1 || invalidateCalls != 0 {
		t.Fatalf("calls: map=%d unmap=%d invalidate=%d", mapCalls, unmapCalls, invalidateCalls)
	}
}

func TestReadTransferMemoryInvalidatesBeforeCopy(t *testing.T) {
	mapped := make([]byte, 100)
	destination := make([]byte, 36)
	for index := range destination {
		destination[index] = 0xaa
	}
	var events []string
	state := &deviceState{
		nonCoherentAtomSize: 64,
		ops: deviceOps{
			mapMemory: func(_ *vk.DeviceFuncs, _ vk.Device, _ vk.DeviceMemory, offset, size uint64) (unsafe.Pointer, error) {
				events = append(events, "map")
				if offset != 0 || size != 100 {
					t.Fatalf("mapped range = offset %d size %d", offset, size)
				}
				return unsafe.Pointer(&mapped[0]), nil
			},
			unmapMemory: func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) {
				events = append(events, "unmap")
				if destination[0] != 1 {
					t.Fatal("destination was not copied before unmap")
				}
			},
			invalidateMappedMemoryRanges: func(_ *vk.DeviceFuncs, _ vk.Device, ranges []vk.MappedMemoryRange) error {
				events = append(events, "invalidate")
				if destination[0] != 0xaa {
					t.Fatal("destination changed before invalidation")
				}
				want := []vk.MappedMemoryRange{{
					SType:  vk.StructureTypeMappedMemoryRange,
					Memory: 7,
					Offset: 64,
					Size:   36,
				}}
				if !reflect.DeepEqual(ranges, want) {
					t.Fatalf("invalidate ranges = %#v, want %#v", ranges, want)
				}
				for index := range destination {
					mapped[64+index] = byte(index + 1)
				}
				return nil
			},
		},
	}
	resource := &transferResource{
		memory:         7,
		allocationSize: 100,
		properties:     vk.MemoryPropertyHostVisibleBit | vk.MemoryPropertyHostCachedBit,
		direction:      transferDownload,
	}
	if err := readTransferMemory(state, resource, 64, destination); err != nil {
		t.Fatalf("readTransferMemory: %v", err)
	}
	if want := []string{"map", "invalidate", "unmap"}; !reflect.DeepEqual(events, want) {
		t.Fatalf("events = %v, want %v", events, want)
	}
	for index, value := range destination {
		if want := byte(index + 1); value != want {
			t.Fatalf("destination[%d] = %d, want %d", index, value, want)
		}
	}
}

func TestReadTransferMemoryInvalidationFailureLeavesDestination(t *testing.T) {
	invalidateErr := errors.New("injected invalidate failure")
	mapped := []byte{1, 2, 3, 4}
	destination := []byte{9, 9, 9, 9}
	unmapped := false
	state := &deviceState{
		nonCoherentAtomSize: 4,
		ops: deviceOps{
			mapMemory: func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error) {
				return unsafe.Pointer(&mapped[0]), nil
			},
			unmapMemory: func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) { unmapped = true },
			invalidateMappedMemoryRanges: func(*vk.DeviceFuncs, vk.Device, []vk.MappedMemoryRange) error {
				return invalidateErr
			},
		},
	}
	resource := &transferResource{
		memory:         1,
		allocationSize: 4,
		properties:     vk.MemoryPropertyHostVisibleBit | vk.MemoryPropertyHostCachedBit,
		direction:      transferDownload,
	}
	err := readTransferMemory(state, resource, 0, destination)
	if !errors.Is(err, invalidateErr) {
		t.Fatalf("read error = %v, want invalidation failure", err)
	}
	if !unmapped {
		t.Fatal("mapped memory was not unmapped after invalidation failure")
	}
	if !reflect.DeepEqual(destination, []byte{9, 9, 9, 9}) {
		t.Fatalf("destination changed after invalidation failure: %v", destination)
	}
}

func TestReadTransferMemoryMapFailuresLeaveDestination(t *testing.T) {
	mapErr := errors.New("injected map failure")
	tests := []struct {
		name       string
		mapMemory  func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error)
		wantErr    error
		wantUnmaps int
	}{
		{
			name: "map error",
			mapMemory: func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error) {
				return nil, mapErr
			},
			wantErr: mapErr,
		},
		{
			name: "nil mapped pointer",
			mapMemory: func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error) {
				return nil, nil
			},
			wantUnmaps: 1,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			destination := []byte{9, 9, 9, 9}
			unmaps := 0
			state := &deviceState{ops: deviceOps{
				mapMemory: test.mapMemory,
				unmapMemory: func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) {
					unmaps++
				},
			}}
			resource := &transferResource{
				memory:         1,
				allocationSize: 4,
				properties:     vk.MemoryPropertyHostVisibleBit | vk.MemoryPropertyHostCoherentBit,
				direction:      transferDownload,
			}
			err := readTransferMemory(state, resource, 0, destination)
			if err == nil {
				t.Fatal("mapped read succeeded")
			}
			if test.wantErr != nil && !errors.Is(err, test.wantErr) {
				t.Fatalf("read error = %v, want %v", err, test.wantErr)
			}
			if unmaps != test.wantUnmaps {
				t.Fatalf("unmap calls = %d, want %d", unmaps, test.wantUnmaps)
			}
			if !reflect.DeepEqual(destination, []byte{9, 9, 9, 9}) {
				t.Fatalf("destination changed after map failure: %v", destination)
			}
		})
	}
}

func TestReadTransferMemoryRejectsMissingInvalidationOperation(t *testing.T) {
	mapped := []byte{1, 2, 3, 4}
	destination := []byte{9, 9, 9, 9}
	unmapped := false
	state := &deviceState{
		nonCoherentAtomSize: 4,
		ops: deviceOps{
			mapMemory: func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory, uint64, uint64) (unsafe.Pointer, error) {
				return unsafe.Pointer(&mapped[0]), nil
			},
			unmapMemory: func(*vk.DeviceFuncs, vk.Device, vk.DeviceMemory) { unmapped = true },
		},
	}
	resource := &transferResource{
		memory:         1,
		allocationSize: 4,
		properties:     vk.MemoryPropertyHostVisibleBit | vk.MemoryPropertyHostCachedBit,
		direction:      transferDownload,
	}
	err := readTransferMemory(state, resource, 0, destination)
	if err == nil || !strings.Contains(err.Error(), "operation unavailable") {
		t.Fatalf("read error = %v, want unavailable invalidation operation", err)
	}
	if !unmapped {
		t.Fatal("memory remained mapped after missing invalidation operation")
	}
	if !reflect.DeepEqual(destination, []byte{9, 9, 9, 9}) {
		t.Fatalf("destination changed: %v", destination)
	}
}
