package vk

import "unsafe"

// Handle types — all opaque pointers represented as uintptr.
type (
	Instance            uintptr
	PhysicalDevice      uintptr
	Device              uintptr
	Queue               uintptr
	CommandPool         uintptr
	CommandBuffer       uintptr
	Buffer              uintptr
	DeviceMemory        uintptr
	ShaderModule        uintptr
	PipelineLayout      uintptr
	Pipeline            uintptr
	DescriptorSetLayout uintptr
	DescriptorPool      uintptr
	DescriptorSet       uintptr
	Fence               uintptr
)

// Result codes.
type Result int32

const (
	Success    Result = 0
	NotReady   Result = 1
	Timeout    Result = 2
	Incomplete Result = 5
)

// Structure types (sType values).
const (
	StructureTypeApplicationInfo                = 0
	StructureTypeInstanceCreateInfo             = 1
	StructureTypeDeviceQueueCreateInfo          = 2
	StructureTypeDeviceCreateInfo               = 3
	StructureTypeSubmitInfo                     = 4
	StructureTypeMemoryAllocateInfo             = 5
	StructureTypeMappedMemoryRange              = 6
	StructureTypeBufferCreateInfo               = 12
	StructureTypeShaderModuleCreateInfo         = 16
	StructureTypeComputePipelineCreateInfo      = 29
	StructureTypePipelineShaderStageCreateInfo  = 18
	StructureTypePipelineLayoutCreateInfo       = 30
	StructureTypeDescriptorSetLayoutCreateInfo  = 32
	StructureTypeDescriptorPoolCreateInfo       = 33
	StructureTypeDescriptorSetAllocateInfo      = 34
	StructureTypeWriteDescriptorSet             = 35
	StructureTypeCommandPoolCreateInfo          = 39
	StructureTypeCommandBufferAllocateInfo      = 40
	StructureTypeCommandBufferBeginInfo         = 42
	StructureTypeFenceCreateInfo                = 8
	StructureTypeBufferMemoryBarrier            = 44
	StructureTypeMemoryBarrier                  = 46
)

// Buffer usage flags.
const (
	BufferUsageTransferSrcBit = 0x00000001
	BufferUsageTransferDstBit = 0x00000002
	BufferUsageStorageBufferBit = 0x00000020
)

// Memory property flags.
type MemoryPropertyFlags uint32

const (
	MemoryPropertyDeviceLocalBit  MemoryPropertyFlags = 0x00000001
	MemoryPropertyHostVisibleBit  MemoryPropertyFlags = 0x00000002
	MemoryPropertyHostCoherentBit MemoryPropertyFlags = 0x00000004
)

// Sharing mode.
const (
	SharingModeExclusive = 0
)

// Descriptor type.
const (
	DescriptorTypeStorageBuffer = 7
)

// Pipeline bind point.
const (
	PipelineBindPointCompute = 1
)

// Shader stage flags.
const (
	ShaderStageComputeBit = 0x00000020
)

// Command buffer level.
const (
	CommandBufferLevelPrimary = 0
)

// Command buffer usage flags.
const (
	CommandBufferUsageOneTimeSubmitBit = 0x00000001
)

// Queue flags.
const (
	QueueComputeBit = 0x00000002
)

// Access flags.
const (
	AccessShaderWriteBit  = 0x00000040
	AccessTransferReadBit = 0x00000800
	AccessShaderReadBit   = 0x00000020
	AccessTransferWriteBit = 0x00001000
	AccessHostReadBit     = 0x00002000
)

// Pipeline stage flags.
const (
	PipelineStageComputeShaderBit = 0x00000800
	PipelineStageTransferBit      = 0x00001000
	PipelineStageHostBit          = 0x00004000
)

// Whole size sentinel.
const WholeSize = ^uint64(0)

// ---- Structs matching Vulkan C layout ----

type ApplicationInfo struct {
	SType              uint32
	PNext              uintptr
	PApplicationName   *byte
	ApplicationVersion uint32
	PEngineName        *byte
	EngineVersion      uint32
	ApiVersion         uint32
}

type InstanceCreateInfo struct {
	SType                   uint32
	PNext                   uintptr
	Flags                   uint32
	PApplicationInfo        *ApplicationInfo
	EnabledLayerCount       uint32
	PpEnabledLayerNames     uintptr
	EnabledExtensionCount   uint32
	PpEnabledExtensionNames uintptr
}

type DeviceQueueCreateInfo struct {
	SType            uint32
	PNext            uintptr
	Flags            uint32
	QueueFamilyIndex uint32
	QueueCount       uint32
	PQueuePriorities *float32
}

type DeviceCreateInfo struct {
	SType                   uint32
	PNext                   uintptr
	Flags                   uint32
	QueueCreateInfoCount    uint32
	PQueueCreateInfos       *DeviceQueueCreateInfo
	EnabledLayerCount       uint32
	PpEnabledLayerNames     uintptr
	EnabledExtensionCount   uint32
	PpEnabledExtensionNames uintptr
	PEnabledFeatures        uintptr
}

type BufferCreateInfo struct {
	SType                 uint32
	PNext                 uintptr
	Flags                 uint32
	Size                  uint64
	Usage                 uint32
	SharingMode           uint32
	QueueFamilyIndexCount uint32
	PQueueFamilyIndices   uintptr
}

type MemoryAllocateInfo struct {
	SType           uint32
	PNext           uintptr
	AllocationSize  uint64
	MemoryTypeIndex uint32
}

type ShaderModuleCreateInfo struct {
	SType    uint32
	PNext    uintptr
	Flags    uint32
	CodeSize uintptr
	PCode    unsafe.Pointer
}

type DescriptorSetLayoutBinding struct {
	Binding            uint32
	DescriptorType     uint32
	DescriptorCount    uint32
	StageFlags         uint32
	PImmutableSamplers uintptr
}

type DescriptorSetLayoutCreateInfo struct {
	SType        uint32
	PNext        uintptr
	Flags        uint32
	BindingCount uint32
	PBindings    *DescriptorSetLayoutBinding
}

type PipelineLayoutCreateInfo struct {
	SType                  uint32
	PNext                  uintptr
	Flags                  uint32
	SetLayoutCount         uint32
	PSetLayouts            *DescriptorSetLayout
	PushConstantRangeCount uint32
	PPushConstantRanges    uintptr
}

type SpecializationInfo struct {
	MapEntryCount uint32
	PMapEntries   uintptr
	DataSize      uintptr
	PData         uintptr
}

type PipelineShaderStageCreateInfo struct {
	SType               uint32
	PNext               uintptr
	Flags               uint32
	Stage               uint32
	Module              ShaderModule
	PName               *byte
	PSpecializationInfo *SpecializationInfo
}

type ComputePipelineCreateInfo struct {
	SType              uint32
	PNext              uintptr
	Flags              uint32
	Stage              PipelineShaderStageCreateInfo
	Layout             PipelineLayout
	BasePipelineHandle Pipeline
	BasePipelineIndex  int32
}

type DescriptorPoolSize struct {
	Type            uint32
	DescriptorCount uint32
}

type DescriptorPoolCreateInfo struct {
	SType         uint32
	PNext         uintptr
	Flags         uint32
	MaxSets       uint32
	PoolSizeCount uint32
	PPoolSizes    *DescriptorPoolSize
}

type DescriptorSetAllocateInfo struct {
	SType              uint32
	PNext              uintptr
	DescriptorPool     DescriptorPool
	DescriptorSetCount uint32
	PSetLayouts        *DescriptorSetLayout
}

type DescriptorBufferInfo struct {
	Buffer Buffer
	Offset uint64
	Range  uint64
}

type WriteDescriptorSet struct {
	SType            uint32
	PNext            uintptr
	DstSet           DescriptorSet
	DstBinding       uint32
	DstArrayElement  uint32
	DescriptorCount  uint32
	DescriptorType   uint32
	PImageInfo       uintptr
	PBufferInfo      *DescriptorBufferInfo
	PTexelBufferView uintptr
}

type CommandPoolCreateInfo struct {
	SType            uint32
	PNext            uintptr
	Flags            uint32
	QueueFamilyIndex uint32
}

type CommandBufferAllocateInfo struct {
	SType              uint32
	PNext              uintptr
	CommandPool        CommandPool
	Level              uint32
	CommandBufferCount uint32
}

type CommandBufferBeginInfo struct {
	SType            uint32
	PNext            uintptr
	Flags            uint32
	PInheritanceInfo uintptr
}

type SubmitInfo struct {
	SType                uint32
	PNext                uintptr
	WaitSemaphoreCount   uint32
	PWaitSemaphores      uintptr
	PWaitDstStageMask    uintptr
	CommandBufferCount   uint32
	PCommandBuffers      *CommandBuffer
	SignalSemaphoreCount uint32
	PSignalSemaphores    uintptr
}

type FenceCreateInfo struct {
	SType uint32
	PNext uintptr
	Flags uint32
}

type MemoryRequirements struct {
	Size           uint64
	Alignment      uint64
	MemoryTypeBits uint32
}

type MemoryType struct {
	PropertyFlags MemoryPropertyFlags
	HeapIndex     uint32
}

type MemoryHeap struct {
	Size  uint64
	Flags uint32
}

const MaxMemoryTypes = 32
const MaxMemoryHeaps = 16

type PhysicalDeviceMemoryProperties struct {
	MemoryTypeCount uint32
	MemoryTypes     [MaxMemoryTypes]MemoryType
	MemoryHeapCount uint32
	MemoryHeaps     [MaxMemoryHeaps]MemoryHeap
}

type MappedMemoryRange struct {
	SType  uint32
	PNext  uintptr
	Memory DeviceMemory
	Offset uint64
	Size   uint64
}

type BufferMemoryBarrier struct {
	SType               uint32
	PNext               uintptr
	SrcAccessMask       uint32
	DstAccessMask       uint32
	SrcQueueFamilyIndex uint32
	DstQueueFamilyIndex uint32
	Buffer              Buffer
	Offset              uint64
	Size                uint64
}

type MemoryBarrier struct {
	SType         uint32
	PNext         uintptr
	SrcAccessMask uint32
	DstAccessMask uint32
}

type BufferCopy struct {
	SrcOffset uint64
	DstOffset uint64
	Size      uint64
}

type QueueFamilyProperties struct {
	QueueFlags                  uint32
	QueueCount                  uint32
	TimestampValidBits          uint32
	MinImageTransferGranularity [3]uint32
}

type PhysicalDeviceProperties struct {
	ApiVersion    uint32
	DriverVersion uint32
	VendorID      uint32
	DeviceID      uint32
	DeviceType    uint32
	DeviceName    [256]byte
	_             [16]byte // pipelineCacheUUID
	Limits        [504]byte
	SparseProperties [20]byte
}

// CommandPoolResetReleaseResourcesBit for resetting command pools.
const CommandPoolResetReleaseResourcesBit = 0x00000001
