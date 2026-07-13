// Package imgproc provides experimental image registration by phase
// correlation.
//
// NewCorrelator creates the package's only long-lived public resource. With no
// options it prefers the direct Vulkan implementation and falls back to the CPU
// reference implementation. WithBackend can require either backend. A
// Correlator owns the resources it creates; callers must close it when it is no
// longer needed.
//
// PhaseCorrelate accepts two equal-sized images and blocks until registration
// finishes. It estimates the rotation, scale, and translation that map image A
// to image B. The operation converts non-RGBA inputs internally and does not
// mutate either source image. A Correlator is not safe for concurrent method
// calls.
//
// The CPU path is the portability and correctness reference. The direct Vulkan
// path is cgo-free and has runtime evidence on Linux. The registration API and
// numerical behavior remain experimental.
package imgproc
