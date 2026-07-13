// Package compute provides the lower-level Vulkan-backed resource layer used
// by Vulki's experimental image-registration implementation.
//
// New application code should normally use the root github.com/srlehn/vulki
// package, whose Device, Buffer, Kernel, BindingSet, and Recorder types keep
// native handles private and bind resources to their owner. Package compute is
// retained while the image-registration implementation is migrated.
//
// A Context owns its Vulkan loader, instance, logical device, queue, and child
// resources. Callers must close resources and then close the Context. Upload,
// download, dispatch, and recorder submission operations are synchronous unless
// their documentation states otherwise. The package is cgo-free and currently
// implements only the direct Vulkan backend.
package compute
