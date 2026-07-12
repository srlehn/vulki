package vk

import (
	"fmt"
	"unsafe"
)

// Vulkan handles are represented as uintptr on the supported 64-bit targets.
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

// Result is a Vulkan result or status code.
type Result int32

// Bool32 is Vulkan's 32-bit boolean representation.
type Bool32 uint32

// PhysicalDeviceType identifies a broad class of Vulkan device.
type PhysicalDeviceType uint32

// Physical device types.
const (
	PhysicalDeviceTypeOther         PhysicalDeviceType = 0
	PhysicalDeviceTypeIntegratedGPU PhysicalDeviceType = 1
	PhysicalDeviceTypeDiscreteGPU   PhysicalDeviceType = 2
	PhysicalDeviceTypeVirtualGPU    PhysicalDeviceType = 3
	PhysicalDeviceTypeCPU           PhysicalDeviceType = 4
)

// SampleCountFlags selects supported sample counts.
type SampleCountFlags uint32

const (
	Success                   Result = 0
	NotReady                  Result = 1
	Timeout                   Result = 2
	Incomplete                Result = 5
	ErrorOutOfHostMemory      Result = -1
	ErrorOutOfDeviceMemory    Result = -2
	ErrorInitializationFailed Result = -3
	ErrorDeviceLost           Result = -4
	ErrorMemoryMapFailed      Result = -5
	ErrorLayerNotPresent      Result = -6
	ErrorExtensionNotPresent  Result = -7
	ErrorFeatureNotPresent    Result = -8
	ErrorIncompatibleDriver   Result = -9
	ErrorTooManyObjects       Result = -10
	ErrorFormatNotSupported   Result = -11
	ErrorFragmentedPool       Result = -12
	ErrorUnknown              Result = -13
)

// String returns the Vulkan name for a known result and its numeric value for
// an unknown result.
func (r Result) String() string {
	switch r {
	case Success:
		return "VK_SUCCESS"
	case NotReady:
		return "VK_NOT_READY"
	case Timeout:
		return "VK_TIMEOUT"
	case Incomplete:
		return "VK_INCOMPLETE"
	case ErrorOutOfHostMemory:
		return "VK_ERROR_OUT_OF_HOST_MEMORY"
	case ErrorOutOfDeviceMemory:
		return "VK_ERROR_OUT_OF_DEVICE_MEMORY"
	case ErrorInitializationFailed:
		return "VK_ERROR_INITIALIZATION_FAILED"
	case ErrorDeviceLost:
		return "VK_ERROR_DEVICE_LOST"
	case ErrorMemoryMapFailed:
		return "VK_ERROR_MEMORY_MAP_FAILED"
	case ErrorLayerNotPresent:
		return "VK_ERROR_LAYER_NOT_PRESENT"
	case ErrorExtensionNotPresent:
		return "VK_ERROR_EXTENSION_NOT_PRESENT"
	case ErrorFeatureNotPresent:
		return "VK_ERROR_FEATURE_NOT_PRESENT"
	case ErrorIncompatibleDriver:
		return "VK_ERROR_INCOMPATIBLE_DRIVER"
	case ErrorTooManyObjects:
		return "VK_ERROR_TOO_MANY_OBJECTS"
	case ErrorFormatNotSupported:
		return "VK_ERROR_FORMAT_NOT_SUPPORTED"
	case ErrorFragmentedPool:
		return "VK_ERROR_FRAGMENTED_POOL"
	case ErrorUnknown:
		return "VK_ERROR_UNKNOWN"
	default:
		return fmt.Sprintf("VkResult(%d)", int32(r))
	}
}

// Error reports a non-success Vulkan result and preserves its code for
// inspection with errors.As. The result may be a status such as Timeout rather
// than a driver failure.
type Error struct {
	Op     string
	Result Result
}

func (e *Error) Error() string {
	if e == nil {
		return "vk: operation returned a non-success result"
	}
	return fmt.Sprintf("vk: %s returned %s", e.Op, e.Result)
}

func resultError(op string, result Result) error {
	return &Error{Op: op, Result: result}
}

// StructureType identifies a Vulkan structure passed through the API.
type StructureType uint32

// Structure types (sType values).
const (
	StructureTypeApplicationInfo               StructureType = 0
	StructureTypeInstanceCreateInfo            StructureType = 1
	StructureTypeDeviceQueueCreateInfo         StructureType = 2
	StructureTypeDeviceCreateInfo              StructureType = 3
	StructureTypeSubmitInfo                    StructureType = 4
	StructureTypeMemoryAllocateInfo            StructureType = 5
	StructureTypeMappedMemoryRange             StructureType = 6
	StructureTypeFenceCreateInfo               StructureType = 8
	StructureTypeBufferCreateInfo              StructureType = 12
	StructureTypeShaderModuleCreateInfo        StructureType = 16
	StructureTypeComputePipelineCreateInfo     StructureType = 29
	StructureTypePipelineShaderStageCreateInfo StructureType = 18
	StructureTypePipelineLayoutCreateInfo      StructureType = 30
	StructureTypeDescriptorSetLayoutCreateInfo StructureType = 32
	StructureTypeDescriptorPoolCreateInfo      StructureType = 33
	StructureTypeDescriptorSetAllocateInfo     StructureType = 34
	StructureTypeWriteDescriptorSet            StructureType = 35
	StructureTypeCommandPoolCreateInfo         StructureType = 39
	StructureTypeCommandBufferAllocateInfo     StructureType = 40
	StructureTypeCommandBufferBeginInfo        StructureType = 42
	StructureTypeBufferMemoryBarrier           StructureType = 44
	StructureTypeMemoryBarrier                 StructureType = 46
)

// BufferUsageFlags controls how a buffer may be used.
type BufferUsageFlags uint32

// Buffer usage flags.
const (
	BufferUsageTransferSrcBit   BufferUsageFlags = 0x00000001
	BufferUsageTransferDstBit   BufferUsageFlags = 0x00000002
	BufferUsageUniformBufferBit BufferUsageFlags = 0x00000010
	BufferUsageStorageBufferBit BufferUsageFlags = 0x00000020
)

// MemoryPropertyFlags selects physical-memory properties.
type MemoryPropertyFlags uint32

const (
	MemoryPropertyDeviceLocalBit  MemoryPropertyFlags = 0x00000001
	MemoryPropertyHostVisibleBit  MemoryPropertyFlags = 0x00000002
	MemoryPropertyHostCoherentBit MemoryPropertyFlags = 0x00000004
)

// SharingMode controls buffer access across queue families.
type SharingMode uint32

// Sharing mode.
const (
	SharingModeExclusive SharingMode = 0
)

// DescriptorType identifies the resource represented by a descriptor.
type DescriptorType uint32

// Descriptor type.
const (
	DescriptorTypeStorageBuffer DescriptorType = 7
)

// PipelineBindPoint identifies the pipeline type bound to a command buffer.
type PipelineBindPoint uint32

// Pipeline bind point.
const (
	PipelineBindPointCompute PipelineBindPoint = 1
)

// ShaderStageFlags selects shader stages.
type ShaderStageFlags uint32

// Shader stage flags.
const (
	ShaderStageComputeBit ShaderStageFlags = 0x00000020
)

// CommandBufferLevel identifies a primary or secondary command buffer.
type CommandBufferLevel uint32

// Command buffer level.
const (
	CommandBufferLevelPrimary CommandBufferLevel = 0
)

// CommandBufferUsageFlags describes command buffer recording usage.
type CommandBufferUsageFlags uint32

// Command buffer usage flags.
const (
	CommandBufferUsageOneTimeSubmitBit CommandBufferUsageFlags = 0x00000001
)

// CommandBufferResetFlags controls command buffer reset behavior.
type CommandBufferResetFlags uint32

// QueueFlags describes operations supported by a queue family.
type QueueFlags uint32

// Queue flags.
const (
	QueueComputeBit QueueFlags = 0x00000002
)

// AccessFlags describes memory accesses for synchronization.
type AccessFlags uint32

// Access flags.
const (
	AccessShaderReadBit    AccessFlags = 0x00000020
	AccessShaderWriteBit   AccessFlags = 0x00000040
	AccessTransferReadBit  AccessFlags = 0x00000800
	AccessTransferWriteBit AccessFlags = 0x00001000
	AccessHostReadBit      AccessFlags = 0x00002000
	AccessMemoryReadBit    AccessFlags = 0x00008000
	AccessMemoryWriteBit   AccessFlags = 0x00010000
)

// PipelineStageFlags selects pipeline stages for synchronization.
type PipelineStageFlags uint32

// Pipeline stage flags.
const (
	PipelineStageComputeShaderBit PipelineStageFlags = 0x00000800
	PipelineStageTransferBit      PipelineStageFlags = 0x00001000
	PipelineStageHostBit          PipelineStageFlags = 0x00004000
)

// WholeSize selects the complete remaining size of a Vulkan resource.
const WholeSize = ^uint64(0)

// ApplicationInfo mirrors VkApplicationInfo.
type ApplicationInfo struct {
	SType              StructureType
	PNext              uintptr
	PApplicationName   *byte
	ApplicationVersion uint32
	PEngineName        *byte
	EngineVersion      uint32
	APIVersion         uint32
}

// InstanceCreateInfo mirrors VkInstanceCreateInfo.
type InstanceCreateInfo struct {
	SType                   StructureType
	PNext                   uintptr
	Flags                   uint32
	PApplicationInfo        *ApplicationInfo
	EnabledLayerCount       uint32
	PPEnabledLayerNames     uintptr
	EnabledExtensionCount   uint32
	PPEnabledExtensionNames uintptr
}

// DeviceQueueCreateInfo mirrors VkDeviceQueueCreateInfo.
type DeviceQueueCreateInfo struct {
	SType            StructureType
	PNext            uintptr
	Flags            uint32
	QueueFamilyIndex uint32
	QueueCount       uint32
	PQueuePriorities *float32
}

// DeviceCreateInfo mirrors VkDeviceCreateInfo.
type DeviceCreateInfo struct {
	SType                   StructureType
	PNext                   uintptr
	Flags                   uint32
	QueueCreateInfoCount    uint32
	PQueueCreateInfos       *DeviceQueueCreateInfo
	EnabledLayerCount       uint32
	PPEnabledLayerNames     uintptr
	EnabledExtensionCount   uint32
	PPEnabledExtensionNames uintptr
	PEnabledFeatures        uintptr
}

// BufferCreateInfo mirrors VkBufferCreateInfo.
type BufferCreateInfo struct {
	SType                 StructureType
	PNext                 uintptr
	Flags                 uint32
	Size                  uint64
	Usage                 BufferUsageFlags
	SharingMode           SharingMode
	QueueFamilyIndexCount uint32
	PQueueFamilyIndices   uintptr
}

// MemoryAllocateInfo mirrors VkMemoryAllocateInfo.
type MemoryAllocateInfo struct {
	SType           StructureType
	PNext           uintptr
	AllocationSize  uint64
	MemoryTypeIndex uint32
}

// ShaderModuleCreateInfo mirrors VkShaderModuleCreateInfo.
type ShaderModuleCreateInfo struct {
	SType    StructureType
	PNext    uintptr
	Flags    uint32
	CodeSize uintptr
	PCode    unsafe.Pointer
}

// DescriptorSetLayoutBinding mirrors VkDescriptorSetLayoutBinding.
type DescriptorSetLayoutBinding struct {
	Binding            uint32
	DescriptorType     DescriptorType
	DescriptorCount    uint32
	StageFlags         ShaderStageFlags
	PImmutableSamplers uintptr
}

// DescriptorSetLayoutCreateInfo mirrors VkDescriptorSetLayoutCreateInfo.
type DescriptorSetLayoutCreateInfo struct {
	SType        StructureType
	PNext        uintptr
	Flags        uint32
	BindingCount uint32
	PBindings    *DescriptorSetLayoutBinding
}

// PipelineLayoutCreateInfo mirrors VkPipelineLayoutCreateInfo.
type PipelineLayoutCreateInfo struct {
	SType                  StructureType
	PNext                  uintptr
	Flags                  uint32
	SetLayoutCount         uint32
	PSetLayouts            *DescriptorSetLayout
	PushConstantRangeCount uint32
	PPushConstantRanges    uintptr
}

// SpecializationInfo mirrors VkSpecializationInfo.
type SpecializationInfo struct {
	MapEntryCount uint32
	PMapEntries   uintptr
	DataSize      uintptr
	PData         uintptr
}

// PipelineShaderStageCreateInfo mirrors VkPipelineShaderStageCreateInfo.
type PipelineShaderStageCreateInfo struct {
	SType               StructureType
	PNext               uintptr
	Flags               uint32
	Stage               ShaderStageFlags
	Module              ShaderModule
	PName               *byte
	PSpecializationInfo *SpecializationInfo
}

// ComputePipelineCreateInfo mirrors VkComputePipelineCreateInfo.
type ComputePipelineCreateInfo struct {
	SType              StructureType
	PNext              uintptr
	Flags              uint32
	Stage              PipelineShaderStageCreateInfo
	Layout             PipelineLayout
	BasePipelineHandle Pipeline
	BasePipelineIndex  int32
}

// DescriptorPoolSize mirrors VkDescriptorPoolSize.
type DescriptorPoolSize struct {
	Type            DescriptorType
	DescriptorCount uint32
}

// DescriptorPoolCreateInfo mirrors VkDescriptorPoolCreateInfo.
type DescriptorPoolCreateInfo struct {
	SType         StructureType
	PNext         uintptr
	Flags         uint32
	MaxSets       uint32
	PoolSizeCount uint32
	PPoolSizes    *DescriptorPoolSize
}

// DescriptorSetAllocateInfo mirrors VkDescriptorSetAllocateInfo.
type DescriptorSetAllocateInfo struct {
	SType              StructureType
	PNext              uintptr
	DescriptorPool     DescriptorPool
	DescriptorSetCount uint32
	PSetLayouts        *DescriptorSetLayout
}

// DescriptorBufferInfo mirrors VkDescriptorBufferInfo.
type DescriptorBufferInfo struct {
	Buffer Buffer
	Offset uint64
	Range  uint64
}

// WriteDescriptorSet mirrors VkWriteDescriptorSet.
type WriteDescriptorSet struct {
	SType            StructureType
	PNext            uintptr
	DstSet           DescriptorSet
	DstBinding       uint32
	DstArrayElement  uint32
	DescriptorCount  uint32
	DescriptorType   DescriptorType
	PImageInfo       uintptr
	PBufferInfo      *DescriptorBufferInfo
	PTexelBufferView uintptr
}

// CommandPoolCreateInfo mirrors VkCommandPoolCreateInfo.
type CommandPoolCreateInfo struct {
	SType            StructureType
	PNext            uintptr
	Flags            uint32
	QueueFamilyIndex uint32
}

// CommandBufferAllocateInfo mirrors VkCommandBufferAllocateInfo.
type CommandBufferAllocateInfo struct {
	SType              StructureType
	PNext              uintptr
	CommandPool        CommandPool
	Level              CommandBufferLevel
	CommandBufferCount uint32
}

// CommandBufferBeginInfo mirrors VkCommandBufferBeginInfo.
type CommandBufferBeginInfo struct {
	SType            StructureType
	PNext            uintptr
	Flags            CommandBufferUsageFlags
	PInheritanceInfo uintptr
}

// SubmitInfo mirrors VkSubmitInfo.
type SubmitInfo struct {
	SType                StructureType
	PNext                uintptr
	WaitSemaphoreCount   uint32
	PWaitSemaphores      uintptr
	PWaitDstStageMask    uintptr
	CommandBufferCount   uint32
	PCommandBuffers      *CommandBuffer
	SignalSemaphoreCount uint32
	PSignalSemaphores    uintptr
}

// FenceCreateInfo mirrors VkFenceCreateInfo.
type FenceCreateInfo struct {
	SType StructureType
	PNext uintptr
	Flags uint32
}

// MemoryRequirements mirrors VkMemoryRequirements.
type MemoryRequirements struct {
	Size           uint64
	Alignment      uint64
	MemoryTypeBits uint32
}

// MemoryType mirrors VkMemoryType.
type MemoryType struct {
	PropertyFlags MemoryPropertyFlags
	HeapIndex     uint32
}

// MemoryHeap mirrors VkMemoryHeap.
type MemoryHeap struct {
	Size  uint64
	Flags uint32
}

const (
	// MaxMemoryTypes is Vulkan's maximum number of physical-device memory types.
	MaxMemoryTypes = 32
	// MaxMemoryHeaps is Vulkan's maximum number of physical-device memory heaps.
	MaxMemoryHeaps = 16
)

// PhysicalDeviceMemoryProperties mirrors VkPhysicalDeviceMemoryProperties.
type PhysicalDeviceMemoryProperties struct {
	MemoryTypeCount uint32
	MemoryTypes     [MaxMemoryTypes]MemoryType
	MemoryHeapCount uint32
	MemoryHeaps     [MaxMemoryHeaps]MemoryHeap
}

// MappedMemoryRange mirrors VkMappedMemoryRange.
type MappedMemoryRange struct {
	SType  StructureType
	PNext  uintptr
	Memory DeviceMemory
	Offset uint64
	Size   uint64
}

// BufferMemoryBarrier mirrors VkBufferMemoryBarrier.
type BufferMemoryBarrier struct {
	SType               StructureType
	PNext               uintptr
	SrcAccessMask       AccessFlags
	DstAccessMask       AccessFlags
	SrcQueueFamilyIndex uint32
	DstQueueFamilyIndex uint32
	Buffer              Buffer
	Offset              uint64
	Size                uint64
}

// MemoryBarrier mirrors VkMemoryBarrier.
type MemoryBarrier struct {
	SType         StructureType
	PNext         uintptr
	SrcAccessMask AccessFlags
	DstAccessMask AccessFlags
}

// BufferCopy mirrors VkBufferCopy.
type BufferCopy struct {
	SrcOffset uint64
	DstOffset uint64
	Size      uint64
}

// QueueFamilyProperties mirrors VkQueueFamilyProperties.
type QueueFamilyProperties struct {
	QueueFlags                  QueueFlags
	QueueCount                  uint32
	TimestampValidBits          uint32
	MinImageTransferGranularity [3]uint32
}

// PhysicalDeviceLimits reports implementation-dependent Vulkan limits.
type PhysicalDeviceLimits struct {
	MaxImageDimension1D                             uint32
	MaxImageDimension2D                             uint32
	MaxImageDimension3D                             uint32
	MaxImageDimensionCube                           uint32
	MaxImageArrayLayers                             uint32
	MaxTexelBufferElements                          uint32
	MaxUniformBufferRange                           uint32
	MaxStorageBufferRange                           uint32
	MaxPushConstantsSize                            uint32
	MaxMemoryAllocationCount                        uint32
	MaxSamplerAllocationCount                       uint32
	BufferImageGranularity                          uint64
	SparseAddressSpaceSize                          uint64
	MaxBoundDescriptorSets                          uint32
	MaxPerStageDescriptorSamplers                   uint32
	MaxPerStageDescriptorUniformBuffers             uint32
	MaxPerStageDescriptorStorageBuffers             uint32
	MaxPerStageDescriptorSampledImages              uint32
	MaxPerStageDescriptorStorageImages              uint32
	MaxPerStageDescriptorInputAttachments           uint32
	MaxPerStageResources                            uint32
	MaxDescriptorSetSamplers                        uint32
	MaxDescriptorSetUniformBuffers                  uint32
	MaxDescriptorSetUniformBuffersDynamic           uint32
	MaxDescriptorSetStorageBuffers                  uint32
	MaxDescriptorSetStorageBuffersDynamic           uint32
	MaxDescriptorSetSampledImages                   uint32
	MaxDescriptorSetStorageImages                   uint32
	MaxDescriptorSetInputAttachments                uint32
	MaxVertexInputAttributes                        uint32
	MaxVertexInputBindings                          uint32
	MaxVertexInputAttributeOffset                   uint32
	MaxVertexInputBindingStride                     uint32
	MaxVertexOutputComponents                       uint32
	MaxTessellationGenerationLevel                  uint32
	MaxTessellationPatchSize                        uint32
	MaxTessellationControlPerVertexInputComponents  uint32
	MaxTessellationControlPerVertexOutputComponents uint32
	MaxTessellationControlPerPatchOutputComponents  uint32
	MaxTessellationControlTotalOutputComponents     uint32
	MaxTessellationEvaluationInputComponents        uint32
	MaxTessellationEvaluationOutputComponents       uint32
	MaxGeometryShaderInvocations                    uint32
	MaxGeometryInputComponents                      uint32
	MaxGeometryOutputComponents                     uint32
	MaxGeometryOutputVertices                       uint32
	MaxGeometryTotalOutputComponents                uint32
	MaxFragmentInputComponents                      uint32
	MaxFragmentOutputAttachments                    uint32
	MaxFragmentDualSrcAttachments                   uint32
	MaxFragmentCombinedOutputResources              uint32
	MaxComputeSharedMemorySize                      uint32
	MaxComputeWorkGroupCount                        [3]uint32
	MaxComputeWorkGroupInvocations                  uint32
	MaxComputeWorkGroupSize                         [3]uint32
	SubPixelPrecisionBits                           uint32
	SubTexelPrecisionBits                           uint32
	MipmapPrecisionBits                             uint32
	MaxDrawIndexedIndexValue                        uint32
	MaxDrawIndirectCount                            uint32
	MaxSamplerLODBias                               float32
	MaxSamplerAnisotropy                            float32
	MaxViewports                                    uint32
	MaxViewportDimensions                           [2]uint32
	ViewportBoundsRange                             [2]float32
	ViewportSubPixelBits                            uint32
	MinMemoryMapAlignment                           uintptr
	MinTexelBufferOffsetAlignment                   uint64
	MinUniformBufferOffsetAlignment                 uint64
	MinStorageBufferOffsetAlignment                 uint64
	MinTexelOffset                                  int32
	MaxTexelOffset                                  uint32
	MinTexelGatherOffset                            int32
	MaxTexelGatherOffset                            uint32
	MinInterpolationOffset                          float32
	MaxInterpolationOffset                          float32
	SubPixelInterpolationOffsetBits                 uint32
	MaxFramebufferWidth                             uint32
	MaxFramebufferHeight                            uint32
	MaxFramebufferLayers                            uint32
	FramebufferColorSampleCounts                    SampleCountFlags
	FramebufferDepthSampleCounts                    SampleCountFlags
	FramebufferStencilSampleCounts                  SampleCountFlags
	FramebufferNoAttachmentsSampleCounts            SampleCountFlags
	MaxColorAttachments                             uint32
	SampledImageColorSampleCounts                   SampleCountFlags
	SampledImageIntegerSampleCounts                 SampleCountFlags
	SampledImageDepthSampleCounts                   SampleCountFlags
	SampledImageStencilSampleCounts                 SampleCountFlags
	StorageImageSampleCounts                        SampleCountFlags
	MaxSampleMaskWords                              uint32
	TimestampComputeAndGraphics                     Bool32
	TimestampPeriod                                 float32
	MaxClipDistances                                uint32
	MaxCullDistances                                uint32
	MaxCombinedClipAndCullDistances                 uint32
	DiscreteQueuePriorities                         uint32
	PointSizeRange                                  [2]float32
	LineWidthRange                                  [2]float32
	PointSizeGranularity                            float32
	LineWidthGranularity                            float32
	StrictLines                                     Bool32
	StandardSampleLocations                         Bool32
	OptimalBufferCopyOffsetAlignment                uint64
	OptimalBufferCopyRowPitchAlignment              uint64
	NonCoherentAtomSize                             uint64
}

// PhysicalDeviceSparseProperties reports supported sparse residency behavior.
type PhysicalDeviceSparseProperties struct {
	ResidencyStandard2DBlockShape            Bool32
	ResidencyStandard2DMultisampleBlockShape Bool32
	ResidencyStandard3DBlockShape            Bool32
	ResidencyAlignedMipSize                  Bool32
	ResidencyNonResidentStrict               Bool32
}

// PhysicalDeviceProperties reports core properties of a Vulkan physical device.
type PhysicalDeviceProperties struct {
	APIVersion        uint32
	DriverVersion     uint32
	VendorID          uint32
	DeviceID          uint32
	DeviceType        PhysicalDeviceType
	DeviceName        [256]byte
	PipelineCacheUUID [16]byte
	Limits            PhysicalDeviceLimits
	SparseProperties  PhysicalDeviceSparseProperties
}
