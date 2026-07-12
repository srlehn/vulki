// Package vk provides the small Vulkan subset used by Vulki's direct compute
// implementation.
//
// The package is low-level and experimental. Its handles, structures, and
// function methods follow Vulkan ownership and host-synchronization rules.
// Allocation callbacks are currently always nil, and only 64-bit targets are
// supported. Most applications should use the high-level Vulki package after
// the publication API migration is complete.
package vk
