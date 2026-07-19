package vk

import (
	"testing"
	"unsafe"
)

func TestCoreStructABI64(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("vk supports only 64-bit targets")
	}

	testStructSize(t, "ApplicationInfo", unsafe.Sizeof(ApplicationInfo{}), 48)
	testStructSize(t, "InstanceCreateInfo", unsafe.Sizeof(InstanceCreateInfo{}), 64)
	testStructSize(t, "BufferCreateInfo", unsafe.Sizeof(BufferCreateInfo{}), 56)
	testStructSize(t, "PipelineCacheCreateInfo", unsafe.Sizeof(PipelineCacheCreateInfo{}), 40)
	testStructSize(t, "QueryPoolCreateInfo", unsafe.Sizeof(QueryPoolCreateInfo{}), 32)
	testStructSize(t, "PipelineShaderStageCreateInfo", unsafe.Sizeof(PipelineShaderStageCreateInfo{}), 48)
	testStructSize(t, "ComputePipelineCreateInfo", unsafe.Sizeof(ComputePipelineCreateInfo{}), 96)
	testStructSize(t, "WriteDescriptorSet", unsafe.Sizeof(WriteDescriptorSet{}), 64)
	testStructSize(t, "BufferMemoryBarrier", unsafe.Sizeof(BufferMemoryBarrier{}), 56)
	testStructSize(t, "PhysicalDeviceLimits", unsafe.Sizeof(PhysicalDeviceLimits{}), 504)
	testStructSize(t, "PhysicalDeviceSparseProperties", unsafe.Sizeof(PhysicalDeviceSparseProperties{}), 20)
	testStructSize(t, "PhysicalDeviceProperties", unsafe.Sizeof(PhysicalDeviceProperties{}), 824)
}

func TestCoreHandleABI64(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("vk supports only 64-bit targets")
	}

	testStructSize(t, "Instance", unsafe.Sizeof(Instance(0)), 8)
	testStructSize(t, "PhysicalDevice", unsafe.Sizeof(PhysicalDevice(0)), 8)
	testStructSize(t, "Device", unsafe.Sizeof(Device(0)), 8)
	testStructSize(t, "Buffer", unsafe.Sizeof(Buffer(0)), 8)
	testStructSize(t, "DeviceMemory", unsafe.Sizeof(DeviceMemory(0)), 8)
	testStructSize(t, "PipelineCache", unsafe.Sizeof(PipelineCache(0)), 8)
	testStructSize(t, "Pipeline", unsafe.Sizeof(Pipeline(0)), 8)
	testStructSize(t, "QueryPool", unsafe.Sizeof(QueryPool(0)), 8)
	if got := unsafe.Alignof(PhysicalDeviceLimits{}); got != 8 {
		t.Fatalf("alignof(PhysicalDeviceLimits) = %d, want 8", got)
	}
	if got := unsafe.Alignof(PhysicalDeviceProperties{}); got != 8 {
		t.Fatalf("alignof(PhysicalDeviceProperties) = %d, want 8", got)
	}
}

func TestPipelineCacheCreateInfoOffsets64(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("vk supports only 64-bit targets")
	}

	info := PipelineCacheCreateInfo{}
	testFieldOffset(t, "PipelineCacheCreateInfo.PNext", unsafe.Offsetof(info.PNext), 8)
	testFieldOffset(t, "PipelineCacheCreateInfo.Flags", unsafe.Offsetof(info.Flags), 16)
	testFieldOffset(t, "PipelineCacheCreateInfo.InitialDataSize", unsafe.Offsetof(info.InitialDataSize), 24)
	testFieldOffset(t, "PipelineCacheCreateInfo.PInitialData", unsafe.Offsetof(info.PInitialData), 32)
}

func TestQueryPoolCreateInfoOffsets64(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("vk supports only 64-bit targets")
	}

	info := QueryPoolCreateInfo{}
	testFieldOffset(t, "QueryPoolCreateInfo.PNext", unsafe.Offsetof(info.PNext), 8)
	testFieldOffset(t, "QueryPoolCreateInfo.Flags", unsafe.Offsetof(info.Flags), 16)
	testFieldOffset(t, "QueryPoolCreateInfo.QueryType", unsafe.Offsetof(info.QueryType), 20)
	testFieldOffset(t, "QueryPoolCreateInfo.QueryCount", unsafe.Offsetof(info.QueryCount), 24)
	testFieldOffset(t, "QueryPoolCreateInfo.PipelineStatistics", unsafe.Offsetof(info.PipelineStatistics), 28)
}

func TestPhysicalDevicePropertiesOffsets64(t *testing.T) {
	if unsafe.Sizeof(uintptr(0)) != 8 {
		t.Skip("vk supports only 64-bit targets")
	}

	properties := PhysicalDeviceProperties{}
	testFieldOffset(t, "PhysicalDeviceProperties.DeviceName", unsafe.Offsetof(properties.DeviceName), 20)
	testFieldOffset(t, "PhysicalDeviceProperties.PipelineCacheUUID", unsafe.Offsetof(properties.PipelineCacheUUID), 276)
	testFieldOffset(t, "PhysicalDeviceProperties.Limits", unsafe.Offsetof(properties.Limits), 296)
	testFieldOffset(t, "PhysicalDeviceProperties.SparseProperties", unsafe.Offsetof(properties.SparseProperties), 800)

	limits := PhysicalDeviceLimits{}
	testFieldOffset(t, "PhysicalDeviceLimits.BufferImageGranularity", unsafe.Offsetof(limits.BufferImageGranularity), 48)
	testFieldOffset(t, "PhysicalDeviceLimits.MaxComputeWorkGroupCount", unsafe.Offsetof(limits.MaxComputeWorkGroupCount), 220)
	testFieldOffset(t, "PhysicalDeviceLimits.MinMemoryMapAlignment", unsafe.Offsetof(limits.MinMemoryMapAlignment), 304)
	testFieldOffset(t, "PhysicalDeviceLimits.NonCoherentAtomSize", unsafe.Offsetof(limits.NonCoherentAtomSize), 496)

	if got := len(properties.DeviceName); got != 256 {
		t.Fatalf("len(PhysicalDeviceProperties.DeviceName) = %d, want 256", got)
	}
	if got := len(properties.PipelineCacheUUID); got != 16 {
		t.Fatalf("len(PhysicalDeviceProperties.PipelineCacheUUID) = %d, want 16", got)
	}
	if got := len(limits.MaxComputeWorkGroupCount); got != 3 {
		t.Fatalf("len(PhysicalDeviceLimits.MaxComputeWorkGroupCount) = %d, want 3", got)
	}
}

func testStructSize(t *testing.T, name string, got, want uintptr) {
	t.Helper()
	if got != want {
		t.Fatalf("sizeof(%s) = %d, want %d", name, got, want)
	}
}

func testFieldOffset(t *testing.T, name string, got, want uintptr) {
	t.Helper()
	if got != want {
		t.Fatalf("offsetof(%s) = %d, want %d", name, got, want)
	}
}
