package vk

import (
	"errors"
	"strings"
	"testing"
	"unsafe"
)

func TestEnumeratePhysicalDevicesRetriesIncomplete(t *testing.T) {
	call := 0
	functions := &InstanceFuncs{
		enumeratePhysicalDevices: func(_ Instance, count *uint32, devices *PhysicalDevice) Result {
			call++
			switch call {
			case 1:
				*count = 1
				return Success
			case 2:
				*count = 1
				return Incomplete
			case 3:
				*count = 2
				return Success
			case 4:
				*count = 2
				result := unsafe.Slice(devices, 2)
				result[0] = PhysicalDevice(11)
				result[1] = PhysicalDevice(22)
				return Success
			default:
				t.Fatalf("unexpected enumeration call %d", call)
				return ErrorUnknown
			}
		},
	}

	devices, err := functions.EnumeratePhysicalDevices(Instance(1))
	if err != nil {
		t.Fatalf("EnumeratePhysicalDevices: %v", err)
	}
	if call != 4 {
		t.Fatalf("enumeration calls = %d, want 4", call)
	}
	if len(devices) != 2 || devices[0] != 11 || devices[1] != 22 {
		t.Fatalf("devices = %v, want [11 22]", devices)
	}
}

func TestDeviceFunctionReturnsStructuredResult(t *testing.T) {
	functions := &DeviceFuncs{
		deviceWaitIdle: func(Device) Result {
			return ErrorDeviceLost
		},
	}

	err := functions.DeviceWaitIdle(Device(1))
	var vkErr *Error
	if !errors.As(err, &vkErr) {
		t.Fatalf("DeviceWaitIdle error = %v, want *Error", err)
	}
	if vkErr.Op != "vkDeviceWaitIdle" || vkErr.Result != ErrorDeviceLost {
		t.Fatalf("DeviceWaitIdle error = %#v", vkErr)
	}
}

func TestCreateDeviceRejectsNilInfo(t *testing.T) {
	functions := &InstanceFuncs{}
	_, err := functions.CreateDevice(PhysicalDevice(1), nil)
	if err == nil || !strings.Contains(err.Error(), "vkCreateDevice") {
		t.Fatalf("CreateDevice error = %v, want operation context", err)
	}
}

func TestEnumeratePhysicalDevicesRejectsInvalidState(t *testing.T) {
	if _, err := (*InstanceFuncs)(nil).EnumeratePhysicalDevices(Instance(1)); err == nil {
		t.Fatal("nil InstanceFuncs enumeration succeeded")
	}
	functions := &InstanceFuncs{enumeratePhysicalDevices: func(Instance, *uint32, *PhysicalDevice) Result {
		return Success
	}}
	if _, err := functions.EnumeratePhysicalDevices(0); err == nil {
		t.Fatal("null instance enumeration succeeded")
	}
}

func TestQueueFamilyEnumerationClampsInvalidDriverCount(t *testing.T) {
	functions := &InstanceFuncs{
		getPhysicalDeviceQueueFamilyProps: func(_ PhysicalDevice, count *uint32, properties *QueueFamilyProperties) {
			if properties == nil {
				*count = 1
				return
			}
			*properties = QueueFamilyProperties{QueueCount: 4}
			*count = 2
		},
	}

	properties := functions.GetPhysicalDeviceQueueFamilyProperties(PhysicalDevice(1))
	if len(properties) != 1 || properties[0].QueueCount != 4 {
		t.Fatalf("properties = %v, want one clamped result", properties)
	}
}
