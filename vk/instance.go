package vk

import (
	"fmt"
	"unsafe"

	"github.com/ebitengine/purego"
)

// InstanceFuncs holds instance-level Vulkan function pointers.
type InstanceFuncs struct {
	createInstance                    func(pCreateInfo *InstanceCreateInfo, pAllocator uintptr, pInstance *Instance) Result
	destroyInstance                   func(instance Instance, pAllocator uintptr)
	enumeratePhysicalDevices          func(instance Instance, pCount *uint32, pDevices *PhysicalDevice) Result
	getPhysicalDeviceProperties       func(device PhysicalDevice, pProperties *PhysicalDeviceProperties)
	getPhysicalDeviceQueueFamilyProps func(device PhysicalDevice, pCount *uint32, pProps *QueueFamilyProperties)
	getPhysicalDeviceMemoryProperties func(device PhysicalDevice, pProps *PhysicalDeviceMemoryProperties)
	createDevice                      func(physicalDevice PhysicalDevice, pCreateInfo *DeviceCreateInfo, pAllocator uintptr, pDevice *Device) Result
	getDeviceProcAddr                 func(device Device, pName *byte) uintptr
}

// LoadGlobalFuncs resolves only vkCreateInstance (available without an instance).
func LoadGlobalFuncs(l *Loader) (*InstanceFuncs, error) {
	f := &InstanceFuncs{}
	addr := l.GetInstanceProcAddr(0, "vkCreateInstance")
	if addr == 0 {
		return nil, fmt.Errorf("vk: vkCreateInstance not found")
	}
	purego.RegisterFunc(&f.createInstance, addr)
	return f, nil
}

// LoadInstanceFuncs resolves all instance-level functions using a real instance handle.
func LoadInstanceFuncs(l *Loader, instance Instance) (*InstanceFuncs, error) {
	f := &InstanceFuncs{}

	resolve := func(target interface{}, name string) error {
		addr := l.GetInstanceProcAddr(instance, name)
		if addr == 0 {
			return fmt.Errorf("vk: %s not found", name)
		}
		purego.RegisterFunc(target, addr)
		return nil
	}

	entries := []struct {
		target interface{}
		name   string
	}{
		{&f.createInstance, "vkCreateInstance"},
		{&f.destroyInstance, "vkDestroyInstance"},
		{&f.enumeratePhysicalDevices, "vkEnumeratePhysicalDevices"},
		{&f.getPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties"},
		{&f.getPhysicalDeviceQueueFamilyProps, "vkGetPhysicalDeviceQueueFamilyProperties"},
		{&f.getPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties"},
		{&f.createDevice, "vkCreateDevice"},
		{&f.getDeviceProcAddr, "vkGetDeviceProcAddr"},
	}

	for _, e := range entries {
		if err := resolve(e.target, e.name); err != nil {
			return nil, err
		}
	}

	return f, nil
}

func (f *InstanceFuncs) CreateInstance(info *InstanceCreateInfo) (Instance, error) {
	var inst Instance
	res := f.createInstance(info, 0, &inst)
	if res != Success {
		return 0, fmt.Errorf("vkCreateInstance failed: %d", res)
	}
	return inst, nil
}

func (f *InstanceFuncs) DestroyInstance(instance Instance) {
	if f.destroyInstance != nil {
		f.destroyInstance(instance, 0)
	}
}

func (f *InstanceFuncs) EnumeratePhysicalDevices(instance Instance) ([]PhysicalDevice, error) {
	var count uint32
	res := f.enumeratePhysicalDevices(instance, &count, nil)
	if res != Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices (count) failed: %d", res)
	}
	if count == 0 {
		return nil, nil
	}
	devices := make([]PhysicalDevice, count)
	res = f.enumeratePhysicalDevices(instance, &count, &devices[0])
	if res != Success {
		return nil, fmt.Errorf("vkEnumeratePhysicalDevices failed: %d", res)
	}
	return devices[:count], nil
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
		return 0, fmt.Errorf("vkCreateDevice failed: %d", res)
	}
	return dev, nil
}

func (f *InstanceFuncs) GetDeviceProcAddr(device Device, name string) uintptr {
	_ = unsafe.Sizeof(0) // keep unsafe import
	cname := append([]byte(name), 0)
	return f.getDeviceProcAddr(device, &cname[0])
}
