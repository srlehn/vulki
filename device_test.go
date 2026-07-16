package vulki

import (
	"errors"
	"reflect"
	"runtime"
	"testing"

	"github.com/srlehn/vulki/vk"
)

func TestOpenCleansPartialState(t *testing.T) {
	tests := []struct {
		failure string
		want    []string
	}{
		{failure: "open"},
		{failure: "global", want: []string{"loader"}},
		{failure: "instance-create", want: []string{"loader"}},
		{failure: "instance-load", want: []string{"instance", "loader"}},
		{failure: "enumerate", want: []string{"instance", "loader"}},
		{failure: "no-devices", want: []string{"instance", "loader"}},
		{failure: "no-queue", want: []string{"instance", "loader"}},
		{failure: "device-create", want: []string{"instance", "loader"}},
		{failure: "device-load", want: []string{"device", "instance", "loader"}},
		{failure: "null-queue", want: []string{"device", "instance", "loader"}},
	}

	for _, test := range tests {
		t.Run(test.failure, func(t *testing.T) {
			var cleanup []string
			device, err := openWithHooks(fakeOpenHooks(test.failure, nil, &cleanup))
			if err == nil {
				if device != nil {
					_ = device.Close()
				}
				t.Fatal("Open succeeded")
			}
			if device != nil {
				t.Fatalf("device = %#v, want nil", device)
			}
			if !reflect.DeepEqual(cleanup, test.want) {
				t.Fatalf("cleanup = %v, want %v", cleanup, test.want)
			}
		})
	}
}

func TestDeviceCloseContinuesAfterWaitError(t *testing.T) {
	waitErr := errors.New("wait failed")
	var cleanup []string
	device, err := openWithHooks(fakeOpenHooks("", waitErr, &cleanup))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	err = device.Close()
	if !errors.Is(err, waitErr) {
		t.Fatalf("Close error = %v, want wait error", err)
	}
	want := []string{"wait", "device", "instance", "loader"}
	if !reflect.DeepEqual(cleanup, want) {
		t.Fatalf("cleanup = %v, want %v", cleanup, want)
	}
	if err := device.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if !reflect.DeepEqual(cleanup, want) {
		t.Fatalf("second Close repeated cleanup: %v", cleanup)
	}
}

func TestDeviceClosedReportsLifecycle(t *testing.T) {
	if !(*Device)(nil).Closed() {
		t.Fatal("nil Device reported open")
	}
	if !new(Device).Closed() {
		t.Fatal("zero Device reported open")
	}
	var cleanup []string
	device, err := openWithHooks(fakeOpenHooks("", nil, &cleanup))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if device.Closed() {
		t.Fatal("open Device reported closed")
	}
	if err := device.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !device.Closed() {
		t.Fatal("closed Device reported open")
	}
}

func TestDeviceClosesChildrenInReverseCreationOrder(t *testing.T) {
	var cleanup []string
	device, err := openWithHooks(fakeOpenHooks("", nil, &cleanup))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	for _, name := range []string{"first", "second", "third"} {
		if _, err := device.addChild(&fakeChild{name: name, events: &cleanup}); err != nil {
			t.Fatalf("add child %s: %v", name, err)
		}
	}

	if err := device.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	want := []string{
		"wait",
		"child:third",
		"child:second",
		"child:first",
		"device",
		"instance",
		"loader",
	}
	if !reflect.DeepEqual(cleanup, want) {
		t.Fatalf("cleanup = %v, want %v", cleanup, want)
	}
	if _, err := device.addChild(&fakeChild{}); err == nil {
		t.Fatal("adding child to closed device succeeded")
	}
}

func TestDeviceCloseReportsChildErrorAfterCleanup(t *testing.T) {
	childErr := errors.New("child close failed")
	var cleanup []string
	device, err := openWithHooks(fakeOpenHooks("", nil, &cleanup))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := device.addChild(&fakeChild{name: "bad", events: &cleanup, err: childErr}); err != nil {
		t.Fatalf("add child: %v", err)
	}
	if _, err := device.addChild(&fakeChild{name: "good", events: &cleanup}); err != nil {
		t.Fatalf("add child: %v", err)
	}

	err = device.Close()
	if !errors.Is(err, childErr) {
		t.Fatalf("Close error = %v, want child error", err)
	}
	want := []string{"wait", "child:good", "child:bad", "device", "instance", "loader"}
	if !reflect.DeepEqual(cleanup, want) {
		t.Fatalf("cleanup = %v, want %v", cleanup, want)
	}
}

func TestDeviceInfoIsSnapshot(t *testing.T) {
	var cleanup []string
	device, err := openWithHooks(fakeOpenHooks("", nil, &cleanup))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer device.Close()

	info := device.Info()
	if info.Implementation != "Vulkan" || info.AdapterName != "Fake GPU" {
		t.Fatalf("Info = %#v", info)
	}
	if info.DeviceType != DeviceTypeDiscreteGPU {
		t.Fatalf("Info device type = %d, want %d", info.DeviceType, DeviceTypeDiscreteGPU)
	}
	if info.VendorID != 0x1234 || info.DeviceID != 0x5678 {
		t.Fatalf("Info IDs = %#v", info)
	}
	if info.Limits.MaxStorageBufferSize != 4096 || info.Limits.MaxComputeWorkGroupCount != [3]uint32{7, 8, 9} {
		t.Fatalf("Info limits = %#v", info.Limits)
	}
	if device.state.nonCoherentAtomSize != 64 {
		t.Fatalf("internal noncoherent atom size = %d, want 64", device.state.nonCoherentAtomSize)
	}

	info.AdapterName = "changed"
	if got := device.Info().AdapterName; got != "Fake GPU" {
		t.Fatalf("mutating snapshot changed device info to %q", got)
	}
}

func TestDeviceInfoPreservesPhysicalDeviceType(t *testing.T) {
	tests := []struct {
		name   string
		native vk.PhysicalDeviceType
		want   DeviceType
	}{
		{name: "other", native: vk.PhysicalDeviceTypeOther, want: DeviceTypeOther},
		{name: "integrated GPU", native: vk.PhysicalDeviceTypeIntegratedGPU, want: DeviceTypeIntegratedGPU},
		{name: "discrete GPU", native: vk.PhysicalDeviceTypeDiscreteGPU, want: DeviceTypeDiscreteGPU},
		{name: "virtual GPU", native: vk.PhysicalDeviceTypeVirtualGPU, want: DeviceTypeVirtualGPU},
		{name: "CPU", native: vk.PhysicalDeviceTypeCPU, want: DeviceTypeCPU},
		{name: "future value", native: vk.PhysicalDeviceType(99), want: DeviceType(99)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			info := deviceInfoSnapshot(vk.PhysicalDeviceProperties{DeviceType: test.native}, vk.PhysicalDeviceMemoryProperties{})
			if info.DeviceType != test.want {
				t.Fatalf("DeviceType = %d, want %d", info.DeviceType, test.want)
			}
		})
	}
}

func TestDeviceInfoSumsDeviceLocalHeaps(t *testing.T) {
	memory := vk.PhysicalDeviceMemoryProperties{
		MemoryHeapCount: 3,
		MemoryHeaps: [vk.MaxMemoryHeaps]vk.MemoryHeap{
			{Size: 8 << 30, Flags: vk.MemoryHeapDeviceLocalBit},
			{Size: 16 << 30},
			{Size: 4 << 30, Flags: vk.MemoryHeapDeviceLocalBit},
			{Size: 1 << 30, Flags: vk.MemoryHeapDeviceLocalBit},
		},
	}
	info := deviceInfoSnapshot(vk.PhysicalDeviceProperties{}, memory)
	if want := uint64(12 << 30); info.DeviceLocalMemoryBytes != want {
		t.Fatalf("DeviceLocalMemoryBytes = %d, want %d", info.DeviceLocalMemoryBytes, want)
	}

	memory.MemoryHeapCount = vk.MaxMemoryHeaps + 5
	info = deviceInfoSnapshot(vk.PhysicalDeviceProperties{}, memory)
	if want := uint64(13 << 30); info.DeviceLocalMemoryBytes != want {
		t.Fatalf("clamped DeviceLocalMemoryBytes = %d, want %d", info.DeviceLocalMemoryBytes, want)
	}
}

func TestZeroDeviceIsClosedAndSafe(t *testing.T) {
	var device Device
	if err := device.Close(); err != nil {
		t.Fatalf("zero Device.Close: %v", err)
	}
	if err := device.Close(); err != nil {
		t.Fatalf("second zero Device.Close: %v", err)
	}
	if info := (*Device)(nil).Info(); info != (DeviceInfo{}) {
		t.Fatalf("nil Device.Info = %#v", info)
	}
}

func TestDeviceCloseWaitsForActiveOperation(t *testing.T) {
	var cleanup []string
	device, err := openWithHooks(fakeOpenHooks("", nil, &cleanup))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := device.beginOperation(); err != nil {
		t.Fatalf("begin operation: %v", err)
	}

	closed := make(chan error, 1)
	go func() {
		closed <- device.Close()
	}()
	for {
		device.mu.Lock()
		closing := device.closing
		device.mu.Unlock()
		if closing {
			break
		}
		runtime.Gosched()
	}
	select {
	case err := <-closed:
		t.Fatalf("Close returned during active operation: %v", err)
	default:
	}

	device.endOperation()
	if err := <-closed; err != nil {
		t.Fatalf("Close: %v", err)
	}
	want := []string{"wait", "device", "instance", "loader"}
	if !reflect.DeepEqual(cleanup, want) {
		t.Fatalf("cleanup = %v, want %v", cleanup, want)
	}
}

func TestOpenDirectVulkan(t *testing.T) {
	device, err := Open()
	if err != nil {
		t.Skipf("direct Vulkan device unavailable: %v", err)
	}
	info := device.Info()
	if info.Implementation != "Vulkan" || info.AdapterName == "" {
		_ = device.Close()
		t.Fatalf("Info = %#v", info)
	}
	if err := device.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

type fakeChild struct {
	name   string
	events *[]string
	err    error
}

func (child *fakeChild) closeFromDevice() error {
	if child.events != nil {
		*child.events = append(*child.events, "child:"+child.name)
	}
	return child.err
}

func fakeOpenHooks(failure string, waitErr error, cleanup *[]string) openHooks {
	failureErr := errors.New("injected " + failure + " failure")
	return openHooks{
		openLoader: func() (*vk.Loader, error) {
			if failure == "open" {
				return nil, failureErr
			}
			return &vk.Loader{}, nil
		},
		closeLoader: func(*vk.Loader) {
			*cleanup = append(*cleanup, "loader")
		},
		loadGlobal: func(*vk.Loader) (*vk.GlobalFuncs, error) {
			if failure == "global" {
				return nil, failureErr
			}
			return &vk.GlobalFuncs{}, nil
		},
		createInstance: func(*vk.GlobalFuncs, *vk.InstanceCreateInfo) (vk.Instance, error) {
			if failure == "instance-create" {
				return 0, failureErr
			}
			return vk.Instance(11), nil
		},
		loadInstance: func(*vk.Loader, vk.Instance) (*vk.InstanceFuncs, error) {
			if failure == "instance-load" {
				return &vk.InstanceFuncs{}, failureErr
			}
			return &vk.InstanceFuncs{}, nil
		},
		destroyInstance: func(*vk.InstanceFuncs, vk.Instance) {
			*cleanup = append(*cleanup, "instance")
		},
		enumerateDevices: func(*vk.InstanceFuncs, vk.Instance) ([]vk.PhysicalDevice, error) {
			if failure == "enumerate" {
				return nil, failureErr
			}
			if failure == "no-devices" {
				return nil, nil
			}
			return []vk.PhysicalDevice{21}, nil
		},
		queueFamilies: func(*vk.InstanceFuncs, vk.PhysicalDevice) []vk.QueueFamilyProperties {
			if failure == "no-queue" {
				return []vk.QueueFamilyProperties{{QueueCount: 1}}
			}
			return []vk.QueueFamilyProperties{{QueueFlags: vk.QueueComputeBit, QueueCount: 1}}
		},
		deviceProperties: func(*vk.InstanceFuncs, vk.PhysicalDevice) vk.PhysicalDeviceProperties {
			properties := vk.PhysicalDeviceProperties{
				APIVersion:    1,
				DriverVersion: 2,
				VendorID:      0x1234,
				DeviceID:      0x5678,
				DeviceType:    vk.PhysicalDeviceTypeDiscreteGPU,
				Limits: vk.PhysicalDeviceLimits{
					MaxStorageBufferRange:          4096,
					MaxComputeWorkGroupCount:       [3]uint32{7, 8, 9},
					MaxComputeWorkGroupInvocations: 10,
					MaxComputeWorkGroupSize:        [3]uint32{11, 12, 13},
					NonCoherentAtomSize:            64,
				},
			}
			copy(properties.DeviceName[:], "Fake GPU")
			return properties
		},
		memoryProperties: func(*vk.InstanceFuncs, vk.PhysicalDevice) vk.PhysicalDeviceMemoryProperties {
			return vk.PhysicalDeviceMemoryProperties{}
		},
		createDevice: func(*vk.InstanceFuncs, vk.PhysicalDevice, *vk.DeviceCreateInfo) (vk.Device, error) {
			if failure == "device-create" {
				return 0, failureErr
			}
			return vk.Device(31), nil
		},
		loadDevice: func(*vk.InstanceFuncs, vk.Device) (*vk.DeviceFuncs, error) {
			if failure == "device-load" {
				return &vk.DeviceFuncs{}, failureErr
			}
			return &vk.DeviceFuncs{}, nil
		},
		destroyDevice: func(*vk.DeviceFuncs, vk.Device) {
			*cleanup = append(*cleanup, "device")
		},
		deviceWaitIdle: func(*vk.DeviceFuncs, vk.Device) error {
			*cleanup = append(*cleanup, "wait")
			return waitErr
		},
		getDeviceQueue: func(*vk.DeviceFuncs, vk.Device, uint32, uint32) vk.Queue {
			if failure == "null-queue" {
				return 0
			}
			return vk.Queue(41)
		},
	}
}
