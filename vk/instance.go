package vk

import (
	"fmt"
	"strings"

	"github.com/ebitengine/purego"
)

// GlobalFuncs holds Vulkan functions available without an instance.
type GlobalFuncs struct {
	createInstance func(pCreateInfo *InstanceCreateInfo, pAllocator uintptr, pInstance *Instance) Result
}

// InstanceFuncs holds instance-level Vulkan function pointers.
type InstanceFuncs struct {
	destroyInstance                   func(instance Instance, pAllocator uintptr)
	enumeratePhysicalDevices          func(instance Instance, pCount *uint32, pDevices *PhysicalDevice) Result
	getPhysicalDeviceProperties       func(device PhysicalDevice, pProperties *PhysicalDeviceProperties)
	getPhysicalDeviceQueueFamilyProps func(device PhysicalDevice, pCount *uint32, pProps *QueueFamilyProperties)
	getPhysicalDeviceMemoryProperties func(device PhysicalDevice, pProps *PhysicalDeviceMemoryProperties)
	createDevice                      func(physicalDevice PhysicalDevice, pCreateInfo *DeviceCreateInfo, pAllocator uintptr, pDevice *Device) Result
	getDeviceProcAddr                 func(device Device, pName *byte) uintptr
}

// LoadGlobalFuncs resolves functions available without a Vulkan instance.
func LoadGlobalFuncs(l *Loader) (*GlobalFuncs, error) {
	if l == nil {
		return nil, fmt.Errorf("vk: nil loader")
	}
	f := &GlobalFuncs{}
	addr := l.GetInstanceProcAddr(0, "vkCreateInstance")
	if addr == 0 {
		return nil, fmt.Errorf("vk: vkCreateInstance not found")
	}
	purego.RegisterFunc(&f.createInstance, addr)
	return f, nil
}

// LoadInstanceFuncs resolves instance-level functions. If resolution fails
// after vkDestroyInstance was loaded, it returns a non-nil table with the error
// so the caller can destroy its instance.
func LoadInstanceFuncs(l *Loader, instance Instance) (*InstanceFuncs, error) {
	return loadInstanceFuncs(l, instance, purego.RegisterFunc)
}

func loadInstanceFuncs(l *Loader, instance Instance, register func(any, uintptr)) (*InstanceFuncs, error) {
	if l == nil {
		return nil, fmt.Errorf("vk: nil loader")
	}
	if instance == 0 {
		return nil, fmt.Errorf("vk: null instance")
	}
	f := &InstanceFuncs{}

	resolve := func(target any, name string) error {
		addr := l.GetInstanceProcAddr(instance, name)
		if addr == 0 {
			return fmt.Errorf("vk: %s not found", name)
		}
		register(target, addr)
		return nil
	}

	// Resolve destruction first so the caller can release its instance after a
	// later resolution failure.
	if err := resolve(&f.destroyInstance, "vkDestroyInstance"); err != nil {
		return nil, err
	}

	entries := []struct {
		target any
		name   string
	}{
		{&f.enumeratePhysicalDevices, "vkEnumeratePhysicalDevices"},
		{&f.getPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties"},
		{&f.getPhysicalDeviceQueueFamilyProps, "vkGetPhysicalDeviceQueueFamilyProperties"},
		{&f.getPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties"},
		{&f.createDevice, "vkCreateDevice"},
		{&f.getDeviceProcAddr, "vkGetDeviceProcAddr"},
	}

	for _, e := range entries {
		if err := resolve(e.target, e.name); err != nil {
			return f, err
		}
	}

	return f, nil
}

// CreateInstance creates a Vulkan instance with nil allocation callbacks.
func (f *GlobalFuncs) CreateInstance(info *InstanceCreateInfo) (Instance, error) {
	if f == nil || f.createInstance == nil {
		return 0, fmt.Errorf("vk: global functions are not loaded")
	}
	if info == nil {
		return 0, fmt.Errorf("vk: vkCreateInstance requires create info")
	}
	var inst Instance
	res := f.createInstance(info, 0, &inst)
	if res != Success {
		return 0, resultError("vkCreateInstance", res)
	}
	return inst, nil
}

// DestroyInstance destroys instance with nil allocation callbacks. It is a
// no-op for a nil function table or null instance.
func (f *InstanceFuncs) DestroyInstance(instance Instance) {
	if f != nil && f.destroyInstance != nil && instance != 0 {
		f.destroyInstance(instance, 0)
	}
}

// EnumeratePhysicalDevices returns the physical devices visible to instance.
// It retries when Vulkan reports Incomplete.
func (f *InstanceFuncs) EnumeratePhysicalDevices(instance Instance) ([]PhysicalDevice, error) {
	if f == nil || f.enumeratePhysicalDevices == nil {
		return nil, fmt.Errorf("vk: instance functions are not loaded")
	}
	if instance == 0 {
		return nil, fmt.Errorf("vk: vkEnumeratePhysicalDevices requires an instance")
	}
	for {
		var count uint32
		res := f.enumeratePhysicalDevices(instance, &count, nil)
		if res != Success && res != Incomplete {
			return nil, resultError("vkEnumeratePhysicalDevices", res)
		}
		if count == 0 {
			return nil, nil
		}

		devices := make([]PhysicalDevice, count)
		res = f.enumeratePhysicalDevices(instance, &count, &devices[0])
		switch res {
		case Success:
			if count > uint32(len(devices)) {
				return nil, fmt.Errorf(
					"vk: vkEnumeratePhysicalDevices returned count %d above capacity %d",
					count, len(devices),
				)
			}
			return devices[:count], nil
		case Incomplete:
			continue
		default:
			return nil, resultError("vkEnumeratePhysicalDevices", res)
		}
	}
}

// GetPhysicalDeviceProperties returns the Vulkan 1.0 properties for device.
func (f *InstanceFuncs) GetPhysicalDeviceProperties(device PhysicalDevice) PhysicalDeviceProperties {
	var props PhysicalDeviceProperties
	if f == nil || f.getPhysicalDeviceProperties == nil || device == 0 {
		return props
	}
	f.getPhysicalDeviceProperties(device, &props)
	return props
}

// GetPhysicalDeviceQueueFamilyProperties returns the queue-family properties
// reported for device.
func (f *InstanceFuncs) GetPhysicalDeviceQueueFamilyProperties(device PhysicalDevice) []QueueFamilyProperties {
	if f == nil || f.getPhysicalDeviceQueueFamilyProps == nil || device == 0 {
		return nil
	}
	var count uint32
	f.getPhysicalDeviceQueueFamilyProps(device, &count, nil)
	if count == 0 {
		return nil
	}
	props := make([]QueueFamilyProperties, count)
	f.getPhysicalDeviceQueueFamilyProps(device, &count, &props[0])
	if count > uint32(len(props)) {
		count = uint32(len(props))
	}
	return props[:count]
}

// GetPhysicalDeviceMemoryProperties returns the memory properties for device.
func (f *InstanceFuncs) GetPhysicalDeviceMemoryProperties(device PhysicalDevice) PhysicalDeviceMemoryProperties {
	var props PhysicalDeviceMemoryProperties
	if f == nil || f.getPhysicalDeviceMemoryProperties == nil || device == 0 {
		return props
	}
	f.getPhysicalDeviceMemoryProperties(device, &props)
	return props
}

// CreateDevice creates a logical device with nil allocation callbacks. The
// caller owns the returned device and must destroy it.
func (f *InstanceFuncs) CreateDevice(physicalDevice PhysicalDevice, info *DeviceCreateInfo) (Device, error) {
	if info == nil {
		return 0, fmt.Errorf("vk: vkCreateDevice requires create info")
	}
	if physicalDevice == 0 {
		return 0, fmt.Errorf("vk: vkCreateDevice requires a physical device")
	}
	if f == nil || f.createDevice == nil {
		return 0, fmt.Errorf("vk: instance functions are not loaded")
	}
	var dev Device
	res := f.createDevice(physicalDevice, info, 0, &dev)
	if res != Success {
		return 0, resultError("vkCreateDevice", res)
	}
	return dev, nil
}

// GetDeviceProcAddr resolves a device-level Vulkan function. It returns zero
// for invalid input or a missing function.
func (f *InstanceFuncs) GetDeviceProcAddr(device Device, name string) uintptr {
	if f == nil || f.getDeviceProcAddr == nil || device == 0 || name == "" || strings.IndexByte(name, 0) >= 0 {
		return 0
	}
	cname := append([]byte(name), 0)
	return f.getDeviceProcAddr(device, &cname[0])
}
