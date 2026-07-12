// Package vk provides the small Vulkan subset used by Vulki's direct compute
// implementation.
//
// The package is low-level and experimental. It covers the Vulkan 1.0 instance,
// device, buffer, descriptor, compute-pipeline, command-buffer, fence, and
// memory operations needed by Vulki. It is not a complete Vulkan binding.
// Handles, structures, and function methods follow Vulkan ownership and
// host-synchronization rules. Allocation callbacks are always nil, and only
// 64-bit targets are supported. Most applications should use a higher-level
// package that owns Vulkan resources and enforces their lifetimes.
package vk
