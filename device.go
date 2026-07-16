package vulki

import (
	"bytes"
	"errors"
	"fmt"
	"sync"
	"syscall"

	"github.com/srlehn/vulki/vk"
)

// Device owns a logical compute device and the resources created from it.
// Its zero value is closed and safe to close again.
type Device struct {
	mu                          sync.Mutex
	cond                        *sync.Cond
	queueMu                     sync.Mutex
	submissionMu                sync.Mutex
	transferMu                  sync.Mutex
	submissionCond              *sync.Cond
	state                       *deviceState
	info                        DeviceInfo
	children                    []childRecord
	pendingSubmissions          []*submissionReservation
	activeSubmissions           []*submissionReservation
	unknownTransientSubmissions []*transientSubmission
	submissionErr               error
	idleTransfers               [transferDirectionCount][]*transferResource
	nextChild                   uint64
	idleTransferBytes           uint64
	closing                     bool
	closed                      bool
	closeDone                   chan struct{}
	closeErr                    error
	active                      int
}

// DeviceType identifies the broad class of a physical compute device.
// Values match Vulkan's VkPhysicalDeviceType constants.
type DeviceType uint32

const (
	// DeviceTypeOther identifies a device that does not fit another class.
	DeviceTypeOther DeviceType = 0
	// DeviceTypeIntegratedGPU identifies an integrated GPU.
	DeviceTypeIntegratedGPU DeviceType = 1
	// DeviceTypeDiscreteGPU identifies a discrete GPU.
	DeviceTypeDiscreteGPU DeviceType = 2
	// DeviceTypeVirtualGPU identifies a virtual GPU.
	DeviceTypeVirtualGPU DeviceType = 3
	// DeviceTypeCPU identifies a CPU Vulkan implementation.
	DeviceTypeCPU DeviceType = 4
)

// DeviceInfo is an immutable snapshot of the selected compute device.
type DeviceInfo struct {
	// Implementation names the active native compute implementation.
	Implementation string
	// AdapterName is the native physical-device name.
	AdapterName string
	// DeviceType identifies the broad class of the physical device.
	DeviceType DeviceType
	// APIVersion is the Vulkan API version reported by the adapter.
	APIVersion uint32
	// DriverVersion is the implementation-defined native driver version.
	DriverVersion uint32
	// VendorID is the PCI vendor identifier when the implementation reports one.
	VendorID uint32
	// DeviceID is the implementation-defined physical-device identifier.
	DeviceID uint32
	// Limits contains the portable compute limits used by Vulki.
	Limits Limits
}

// Limits contains the portable compute limits used by Vulki.
type Limits struct {
	// MaxStorageBufferSize is the maximum storage-buffer range in bytes.
	MaxStorageBufferSize uint64
	// MaxComputeWorkGroupCount is the maximum dispatch count per dimension.
	MaxComputeWorkGroupCount [3]uint32
	// MaxComputeWorkGroupInvocations is the maximum invocations in one workgroup.
	MaxComputeWorkGroupInvocations uint32
	// MaxComputeWorkGroupSize is the maximum workgroup size per dimension.
	MaxComputeWorkGroupSize [3]uint32
}

type deviceState struct {
	hooks               openHooks
	loader              *vk.Loader
	instanceFns         *vk.InstanceFuncs
	deviceFns           *vk.DeviceFuncs
	instance            vk.Instance
	physical            vk.PhysicalDevice
	device              vk.Device
	queue               vk.Queue
	queueFamily         uint32
	memory              vk.PhysicalDeviceMemoryProperties
	nonCoherentAtomSize uint64
	ops                 deviceOps
	kernelOps           kernelOps
}

type deviceChild interface {
	closeFromDevice() error
}

type childRecord struct {
	id    uint64
	child deviceChild
}

type openHooks struct {
	openLoader       func() (*vk.Loader, error)
	closeLoader      func(*vk.Loader)
	loadGlobal       func(*vk.Loader) (*vk.GlobalFuncs, error)
	createInstance   func(*vk.GlobalFuncs, *vk.InstanceCreateInfo) (vk.Instance, error)
	loadInstance     func(*vk.Loader, vk.Instance) (*vk.InstanceFuncs, error)
	destroyInstance  func(*vk.InstanceFuncs, vk.Instance)
	enumerateDevices func(*vk.InstanceFuncs, vk.Instance) ([]vk.PhysicalDevice, error)
	queueFamilies    func(*vk.InstanceFuncs, vk.PhysicalDevice) []vk.QueueFamilyProperties
	deviceProperties func(*vk.InstanceFuncs, vk.PhysicalDevice) vk.PhysicalDeviceProperties
	memoryProperties func(*vk.InstanceFuncs, vk.PhysicalDevice) vk.PhysicalDeviceMemoryProperties
	createDevice     func(*vk.InstanceFuncs, vk.PhysicalDevice, *vk.DeviceCreateInfo) (vk.Device, error)
	loadDevice       func(*vk.InstanceFuncs, vk.Device) (*vk.DeviceFuncs, error)
	destroyDevice    func(*vk.DeviceFuncs, vk.Device)
	deviceWaitIdle   func(*vk.DeviceFuncs, vk.Device) error
	getDeviceQueue   func(*vk.DeviceFuncs, vk.Device, uint32, uint32) vk.Queue
}

var directVulkanHooks = openHooks{
	openLoader:  vk.Open,
	closeLoader: func(loader *vk.Loader) { loader.Close() },
	loadGlobal:  vk.LoadGlobalFuncs,
	createInstance: func(functions *vk.GlobalFuncs, info *vk.InstanceCreateInfo) (vk.Instance, error) {
		return functions.CreateInstance(info)
	},
	loadInstance: vk.LoadInstanceFuncs,
	destroyInstance: func(functions *vk.InstanceFuncs, instance vk.Instance) {
		if functions != nil {
			functions.DestroyInstance(instance)
		}
	},
	enumerateDevices: func(functions *vk.InstanceFuncs, instance vk.Instance) ([]vk.PhysicalDevice, error) {
		return functions.EnumeratePhysicalDevices(instance)
	},
	queueFamilies: func(functions *vk.InstanceFuncs, device vk.PhysicalDevice) []vk.QueueFamilyProperties {
		return functions.GetPhysicalDeviceQueueFamilyProperties(device)
	},
	deviceProperties: func(functions *vk.InstanceFuncs, device vk.PhysicalDevice) vk.PhysicalDeviceProperties {
		return functions.GetPhysicalDeviceProperties(device)
	},
	memoryProperties: func(functions *vk.InstanceFuncs, device vk.PhysicalDevice) vk.PhysicalDeviceMemoryProperties {
		return functions.GetPhysicalDeviceMemoryProperties(device)
	},
	createDevice: func(functions *vk.InstanceFuncs, physical vk.PhysicalDevice, info *vk.DeviceCreateInfo) (vk.Device, error) {
		return functions.CreateDevice(physical, info)
	},
	loadDevice: vk.LoadDeviceFuncs,
	destroyDevice: func(functions *vk.DeviceFuncs, device vk.Device) {
		if functions != nil {
			functions.DestroyDevice(device)
		}
	},
	deviceWaitIdle: func(functions *vk.DeviceFuncs, device vk.Device) error {
		return functions.DeviceWaitIdle(device)
	},
	getDeviceQueue: func(functions *vk.DeviceFuncs, device vk.Device, family, index uint32) vk.Queue {
		return functions.GetDeviceQueue(device, family, index)
	},
}

// Open acquires the first Vulkan physical device with a compute queue and
// creates one logical compute device. The caller must close the returned
// Device.
func Open() (*Device, error) {
	return openWithHooks(directVulkanHooks)
}

func openWithHooks(hooks openHooks) (_ *Device, err error) {
	state := &deviceState{hooks: hooks, ops: directDeviceOps, kernelOps: directKernelOps}
	complete := false
	defer func() {
		if !complete {
			state.release()
		}
	}()

	state.loader, err = hooks.openLoader()
	if err != nil {
		return nil, fmt.Errorf("vulki: open Vulkan loader: %w", err)
	}

	global, err := hooks.loadGlobal(state.loader)
	if err != nil {
		return nil, fmt.Errorf("vulki: load global Vulkan functions: %w", err)
	}

	appName, err := syscall.BytePtrFromString("vulki")
	if err != nil {
		return nil, fmt.Errorf("vulki: encode application name: %w", err)
	}
	engineName, err := syscall.BytePtrFromString("vulki")
	if err != nil {
		return nil, fmt.Errorf("vulki: encode engine name: %w", err)
	}
	appInfo := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   appName,
		ApplicationVersion: 1,
		PEngineName:        engineName,
		EngineVersion:      1,
		APIVersion:         (1 << 22) | (1 << 12),
	}
	instanceInfo := vk.InstanceCreateInfo{
		SType:            vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo: &appInfo,
	}
	state.instance, err = hooks.createInstance(global, &instanceInfo)
	if err != nil {
		return nil, fmt.Errorf("vulki: create Vulkan instance: %w", err)
	}

	state.instanceFns, err = hooks.loadInstance(state.loader, state.instance)
	if err != nil {
		return nil, fmt.Errorf("vulki: load instance Vulkan functions: %w", err)
	}

	physicalDevices, err := hooks.enumerateDevices(state.instanceFns, state.instance)
	if err != nil {
		return nil, fmt.Errorf("vulki: enumerate physical devices: %w", err)
	}
	if len(physicalDevices) == 0 {
		return nil, fmt.Errorf("vulki: no Vulkan physical devices found")
	}

	found := false
	for _, physical := range physicalDevices {
		families := hooks.queueFamilies(state.instanceFns, physical)
		for index, family := range families {
			if family.QueueFlags&vk.QueueComputeBit == 0 || family.QueueCount == 0 {
				continue
			}
			state.physical = physical
			state.queueFamily = uint32(index)
			found = true
			break
		}
		if found {
			break
		}
	}
	if !found {
		return nil, fmt.Errorf("vulki: no physical device with a compute queue found")
	}

	priority := float32(1)
	queueInfo := vk.DeviceQueueCreateInfo{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: state.queueFamily,
		QueueCount:       1,
		PQueuePriorities: &priority,
	}
	deviceInfo := vk.DeviceCreateInfo{
		SType:                vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount: 1,
		PQueueCreateInfos:    &queueInfo,
	}
	state.device, err = hooks.createDevice(state.instanceFns, state.physical, &deviceInfo)
	if err != nil {
		return nil, fmt.Errorf("vulki: create logical device: %w", err)
	}

	state.deviceFns, err = hooks.loadDevice(state.instanceFns, state.device)
	if err != nil {
		return nil, fmt.Errorf("vulki: load device Vulkan functions: %w", err)
	}
	state.queue = hooks.getDeviceQueue(state.deviceFns, state.device, state.queueFamily, 0)
	if state.queue == 0 {
		return nil, fmt.Errorf("vulki: Vulkan returned a null compute queue")
	}
	state.memory = hooks.memoryProperties(state.instanceFns, state.physical)
	properties := hooks.deviceProperties(state.instanceFns, state.physical)
	state.nonCoherentAtomSize = properties.Limits.NonCoherentAtomSize

	device := &Device{
		state:     state,
		info:      deviceInfoSnapshot(properties),
		closeDone: make(chan struct{}),
	}
	device.cond = sync.NewCond(&device.mu)
	complete = true
	return device, nil
}

// Info returns a copy of the immutable device information captured by Open.
func (d *Device) Info() DeviceInfo {
	if d == nil {
		return DeviceInfo{}
	}
	d.mu.Lock()
	if d.cond == nil {
		d.cond = sync.NewCond(&d.mu)
	}
	defer d.mu.Unlock()
	return d.info
}

// Closed reports whether the Device is nil, closing, or closed.
func (d *Device) Closed() bool {
	if d == nil {
		return true
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.state == nil || d.closing || d.closed
}

// Close waits for active queue work, closes remaining child resources in
// reverse creation order, and releases the native device. Cleanup continues
// after a wait or child error. Repeated calls after cleanup return nil.
func (d *Device) Close() error {
	if d == nil {
		return nil
	}

	d.mu.Lock()
	if d.closeDone == nil {
		d.closeDone = make(chan struct{})
	}
	if d.closed {
		d.mu.Unlock()
		return nil
	}
	if d.closing {
		done := d.closeDone
		d.mu.Unlock()
		<-done
		d.mu.Lock()
		err := d.closeErr
		d.mu.Unlock()
		return err
	}
	d.closing = true
	for d.active > 0 {
		d.cond.Wait()
	}
	children := append([]childRecord(nil), d.children...)
	state := d.state
	d.mu.Unlock()

	d.queueMu.Lock()
	var closeErrors []error
	if state != nil {
		if err := state.waitIdle(); err != nil {
			closeErrors = append(closeErrors, err)
		}
		d.cleanupUnknownTransientSubmissions(state)
	}
	for index := len(children) - 1; index >= 0; index-- {
		if err := children[index].child.closeFromDevice(); err != nil {
			closeErrors = append(closeErrors, err)
		}
	}
	if state != nil {
		d.drainTransferPool(state)
		state.release()
	}
	d.queueMu.Unlock()
	closeErr := errors.Join(closeErrors...)

	d.mu.Lock()
	d.state = nil
	d.children = nil
	d.closeErr = closeErr
	d.closing = false
	d.closed = true
	close(d.closeDone)
	d.mu.Unlock()
	return closeErr
}

func (d *Device) beginOperation() (*deviceState, error) {
	if d == nil {
		return nil, fmt.Errorf("vulki: device is closed")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.cond == nil {
		d.cond = sync.NewCond(&d.mu)
	}
	if d.state == nil || d.closing || d.closed {
		return nil, fmt.Errorf("vulki: device is closed")
	}
	d.active++
	return d.state, nil
}

func (d *Device) endOperation() {
	if d == nil {
		return
	}
	d.mu.Lock()
	if d.active > 0 {
		d.active--
	}
	if d.active == 0 && d.cond != nil {
		d.cond.Broadcast()
	}
	d.mu.Unlock()
}

func (d *Device) addChild(child deviceChild) (uint64, error) {
	if d == nil || child == nil {
		return 0, fmt.Errorf("vulki: invalid device child")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.state == nil || d.closing || d.closed {
		return 0, fmt.Errorf("vulki: device is closed")
	}
	d.nextChild++
	id := d.nextChild
	d.children = append(d.children, childRecord{id: id, child: child})
	return id, nil
}

func (d *Device) removeChild(id uint64) {
	if d == nil || id == 0 {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	for index := range d.children {
		if d.children[index].id != id {
			continue
		}
		copy(d.children[index:], d.children[index+1:])
		d.children[len(d.children)-1] = childRecord{}
		d.children = d.children[:len(d.children)-1]
		return
	}
}

func (state *deviceState) waitIdle() error {
	if state == nil || state.device == 0 || state.deviceFns == nil {
		return nil
	}
	if err := state.hooks.deviceWaitIdle(state.deviceFns, state.device); err != nil {
		return fmt.Errorf("vulki: wait for device idle: %w", err)
	}
	return nil
}

func (state *deviceState) release() {
	if state == nil {
		return
	}
	if state.device != 0 {
		state.hooks.destroyDevice(state.deviceFns, state.device)
		state.device = 0
		state.deviceFns = nil
	}
	if state.instance != 0 {
		state.hooks.destroyInstance(state.instanceFns, state.instance)
		state.instance = 0
		state.instanceFns = nil
	}
	if state.loader != nil {
		state.hooks.closeLoader(state.loader)
		state.loader = nil
	}
}

func deviceInfoSnapshot(properties vk.PhysicalDeviceProperties) DeviceInfo {
	name := properties.DeviceName[:]
	if end := bytes.IndexByte(name, 0); end >= 0 {
		name = name[:end]
	}
	return DeviceInfo{
		Implementation: "Vulkan",
		AdapterName:    string(name),
		DeviceType:     DeviceType(properties.DeviceType),
		APIVersion:     properties.APIVersion,
		DriverVersion:  properties.DriverVersion,
		VendorID:       properties.VendorID,
		DeviceID:       properties.DeviceID,
		Limits: Limits{
			MaxStorageBufferSize:           uint64(properties.Limits.MaxStorageBufferRange),
			MaxComputeWorkGroupCount:       properties.Limits.MaxComputeWorkGroupCount,
			MaxComputeWorkGroupInvocations: properties.Limits.MaxComputeWorkGroupInvocations,
			MaxComputeWorkGroupSize:        properties.Limits.MaxComputeWorkGroupSize,
		},
	}
}
