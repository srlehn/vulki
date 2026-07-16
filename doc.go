// Package vulki provides explicit, cgo-free GPU compute resources.
//
// Open acquires a compute device. Resources created from a Device remember
// their owner and may be closed explicitly; closing the Device closes any
// remaining children in reverse creation order. The current implementation
// uses Vulkan directly through the low-level vk package.
//
// Buffer uploads and downloads, DispatchAndWait, and Recorder.SubmitAndWait
// block until the requested queue work completes. Recorder uploads copy their
// input immediately, while recorded download destinations become valid after
// SubmitAndWait succeeds. Device serializes calls that access its Vulkan queue,
// while submissions using disjoint buffers or only shared read-only buffers may
// remain in flight concurrently. Overlapping writes retain submission order.
// Individual child resources and recorders must not be used concurrently with
// Close; Recorder also rejects use after submission or abort. If a submitted
// fence cannot establish completion, later submissions fail and Device.Close
// retains the uncertain resources through its device-idle cleanup.
//
// The public API is experimental. The direct Vulkan path is cgo-free and has
// runtime evidence on Linux. All other platforms are unsupported backlog work.
package vulki
