package compute

import (
	"fmt"
	"syscall"

	"vkpg/vk"
)

// Context holds the Vulkan instance, device, and queue for compute operations.
type Context struct {
	Loader       *vk.Loader
	InstFuncs    *vk.InstanceFuncs
	DevFuncs     *vk.DeviceFuncs
	Instance     vk.Instance
	PhysDevice   vk.PhysicalDevice
	Device       vk.Device
	Queue        vk.Queue
	QueueFamily  uint32
	MemProps     vk.PhysicalDeviceMemoryProperties
}

// NewContext creates a Vulkan instance, selects a physical device with a compute
// queue, and creates a logical device.
func NewContext() (*Context, error) {
	loader, err := vk.Open()
	if err != nil {
		return nil, err
	}

	// Load global functions (only vkCreateInstance available without instance).
	globalFuncs, err := vk.LoadGlobalFuncs(loader)
	if err != nil {
		loader.Close()
		return nil, err
	}

	appName, _ := syscall.BytePtrFromString("vkpg")
	engineName, _ := syscall.BytePtrFromString("vkpg")

	appInfo := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   appName,
		ApplicationVersion: 1,
		PEngineName:        engineName,
		EngineVersion:      1,
		ApiVersion:         (1 << 22) | (1 << 12), // Vulkan 1.1
	}
	createInfo := vk.InstanceCreateInfo{
		SType:            vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo: &appInfo,
	}

	instance, err := globalFuncs.CreateInstance(&createInfo)
	if err != nil {
		loader.Close()
		return nil, err
	}

	// Load all instance functions with the real instance handle.
	instFuncs, err := vk.LoadInstanceFuncs(loader, instance)
	if err != nil {
		loader.Close()
		return nil, err
	}

	// Enumerate physical devices.
	physDevices, err := instFuncs.EnumeratePhysicalDevices(instance)
	if err != nil {
		instFuncs.DestroyInstance(instance)
		loader.Close()
		return nil, err
	}
	if len(physDevices) == 0 {
		instFuncs.DestroyInstance(instance)
		loader.Close()
		return nil, fmt.Errorf("compute: no Vulkan physical devices found")
	}

	// Find a device with a compute queue.
	var physDevice vk.PhysicalDevice
	var queueFamily uint32
	found := false
	for _, pd := range physDevices {
		families := instFuncs.GetPhysicalDeviceQueueFamilyProperties(pd)
		for i, f := range families {
			if f.QueueFlags&vk.QueueComputeBit != 0 {
				physDevice = pd
				queueFamily = uint32(i)
				found = true
				break
			}
		}
		if found {
			break
		}
	}
	if !found {
		instFuncs.DestroyInstance(instance)
		loader.Close()
		return nil, fmt.Errorf("compute: no device with compute queue found")
	}

	// Create logical device with one compute queue.
	priority := float32(1.0)
	queueCreateInfo := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: queueFamily,
		QueueCount:       1,
		PQueuePriorities: &priority,
	}
	deviceCreateInfo := vk.DeviceCreateInfo{
		SType:                vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    &queueCreateInfo,
	}

	device, err := instFuncs.CreateDevice(physDevice, &deviceCreateInfo)
	if err != nil {
		instFuncs.DestroyInstance(instance)
		loader.Close()
		return nil, err
	}

	devFuncs, err := vk.LoadDeviceFuncs(instFuncs, device)
	if err != nil {
		instFuncs.DestroyInstance(instance)
		loader.Close()
		return nil, err
	}

	queue := devFuncs.GetDeviceQueue(device, queueFamily, 0)
	memProps := instFuncs.GetPhysicalDeviceMemoryProperties(physDevice)

	return &Context{
		Loader:      loader,
		InstFuncs:   instFuncs,
		DevFuncs:    devFuncs,
		Instance:    instance,
		PhysDevice:  physDevice,
		Device:      device,
		Queue:       queue,
		QueueFamily: queueFamily,
		MemProps:    memProps,
	}, nil
}

// Close destroys all Vulkan resources in reverse order.
func (c *Context) Close() {
	if c.Device != 0 {
		c.DevFuncs.DeviceWaitIdle(c.Device)
	}
	if c.Instance != 0 {
		c.InstFuncs.DestroyInstance(c.Instance)
	}
	if c.Loader != nil {
		c.Loader.Close()
	}
}
