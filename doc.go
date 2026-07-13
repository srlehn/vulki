// Package vulki provides explicit, cgo-free GPU compute resources.
//
// Open acquires a compute device. Resources created from a Device remember
// their owner and may be closed explicitly; closing the Device closes any
// remaining children in reverse creation order. The current implementation
// uses Vulkan directly through the low-level vk package.
//
// Upload, download, DispatchAndWait, and Recorder.SubmitAndWait block until the
// requested queue work completes. Device serializes submissions to its queue.
// Individual child resources and recorders must not be used concurrently with
// Close; Recorder also rejects use after submission or abort.
//
// The public API is experimental. The direct Vulkan path is cgo-free and has
// runtime evidence on Linux; other native platforms are not yet verified.
package vulki
