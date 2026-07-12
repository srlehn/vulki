// Package vulki provides explicit, cgo-free GPU compute resources.
//
// Open acquires a compute device. Resources created from a Device remember
// their owner and may be closed explicitly; closing the Device closes any
// remaining children in reverse creation order. The current implementation
// uses Vulkan directly through the low-level vk package.
package vulki
