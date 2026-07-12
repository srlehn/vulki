package vk

import (
	"fmt"

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
	if l == nil {
		return nil, fmt.Errorf("vk: nil loader")
	}
	if instance == 0 {
		return nil, fmt.Errorf("vk: null instance")
	}
	f := &InstanceFuncs{}

	resolve := func(target interface{}, name string) error {
		addr := l.GetInstanceProcAddr(instance, name)
		if addr == 0 {
			return fmt.Errorf("vk: %s not found", name)
		}
		purego.RegisterFunc(target, addr)
		return nil
	}

	// Resolve destruction first so the caller can release its instance after a
	// later resolution failure.
	if err := resolve(&f.destroyInstance, "vkDestroyInstance"); err != nil {
		return nil, err
	}

	entries := []struct {
		target interface{}
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

func (f *InstanceFuncs) DestroyInstance(instance Instance) {
	if f.destroyInstance != nil {
		f.destroyInstance(instance, 0)
	}
}

func (f *InstanceFuncs) EnumeratePhysicalDevices(instance Instance) ([]PhysicalDevice, error) {
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

func (f *InstanceFuncs) GetPhysicalDeviceProperties(device PhysicalDevice) PhysicalDeviceProperties {
	var props PhysicalDeviceProperties
	f.getPhysicalDeviceProperties(device, &props)
	return props
}

func (f *InstanceFuncs) GetPhysicalDeviceQueueFamilyProperties(device PhysicalDevice) []QueueFamilyProperties {
	var count uint32
	f.getPhysicalDeviceQueueFamilyProps(device, &count, nil)
	if count == 0 {
		return nil
	}
	props := make([]QueueFamilyProperties, count)
	f.getPhysicalDeviceQueueFamilyProps(device, &count, &props[0])
	return props[:count]
}

func (f *InstanceFuncs) GetPhysicalDeviceMemoryProperties(device PhysicalDevice) PhysicalDeviceMemoryProperties {
	var props PhysicalDeviceMemoryProperties
	f.getPhysicalDeviceMemoryProperties(device, &props)
	return props
}

func (f *InstanceFuncs) CreateDevice(physicalDevice PhysicalDevice, info *DeviceCreateInfo) (Device, error) {
	var dev Device
	res := f.createDevice(physicalDevice, info, 0, &dev)
	if res != Success {
		return 0, resultError("vkCreateDevice", res)
	}
	return dev, nil
}

func (f *InstanceFuncs) GetDeviceProcAddr(device Device, name string) uintptr {
	cname := append([]byte(name), 0)
	return f.getDeviceProcAddr(device, &cname[0])
}
