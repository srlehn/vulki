// Package vk provides the small Vulkan subset used by Vulki's direct compute
// implementation.
//
// The package is low-level and experimental. Its handles, structures, and
// function methods follow Vulkan ownership and host-synchronization rules.
// Allocation callbacks are currently always nil, and only 64-bit targets are
// supported. Most applications should use a higher-level package that owns
// Vulkan resources and enforces their lifetimes.
package vk
