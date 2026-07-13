package main

import (
	"fmt"
	"image"
	"runtime"
	"testing"

	"github.com/srlehn/vulki/internal/testutils"
	"github.com/srlehn/vulki/registration"
)

var benchmarkResult *registration.Result

// BenchmarkPhaseCorrelate reports allocations directly in MiB, so callers do
// not need the testing package's raw -benchmem output.
func BenchmarkPhaseCorrelate(b *testing.B) {
	imageA := loadBenchmarkImage(b, "snake.png")
	imageB := loadBenchmarkImage(b, "snake_rot_12deg.png")

	maxWidth := max(imageA.Bounds().Dx(), imageB.Bounds().Dx())
	maxHeight := max(imageA.Bounds().Dy(), imageB.Bounds().Dy())
	backends := []struct {
		name    string
		backend registration.Backend
	}{
		{name: "Vulkan", backend: registration.BackendVulkan},
		{name: "CPU", backend: registration.BackendCPU},
	}
	var vulkanMillisecondsPerOp, cpuMillisecondsPerOp float64

	for _, backend := range backends {
		b.Run(backend.name, func(b *testing.B) {
			correlator, err := registration.NewCorrelator(
				maxWidth,
				maxHeight,
				registration.WithBackend(backend.backend),
			)
			if err != nil {
				if backend.backend == registration.BackendVulkan {
					b.Skipf("Vulkan unavailable: %v", err)
				}
				b.Fatalf("create %s correlator: %v", backend.name, err)
			}
			b.Cleanup(func() {
				if err := correlator.Close(); err != nil {
					b.Errorf("close %s correlator: %v", backend.name, err)
				}
			})
			if got := correlator.Backend(); got != backend.backend {
				b.Fatalf("backend = %q, want %q", got, backend.backend)
			}

			result, err := correlator.PhaseCorrelate(imageA, imageB)
			if err != nil {
				b.Fatalf("warm up %s correlator: %v", backend.name, err)
			}
			benchmarkResult = result

			b.StopTimer()
			b.ResetTimer()
			var memoryBefore runtime.MemStats
			runtime.ReadMemStats(&memoryBefore)
			b.StartTimer()
			for range b.N {
				result, err := correlator.PhaseCorrelate(imageA, imageB)
				if err != nil {
					b.Fatalf("phase correlate with %s: %v", backend.name, err)
				}
				benchmarkResult = result
			}
			b.StopTimer()
			var memoryAfter runtime.MemStats
			runtime.ReadMemStats(&memoryAfter)

			millisecondsPerOp := b.Elapsed().Seconds() * 1000 / float64(b.N)
			mebibytesPerOp := float64(memoryAfter.TotalAlloc-memoryBefore.TotalAlloc) /
				float64(b.N) / (1024 * 1024)
			allocationsPerOp := float64(memoryAfter.Mallocs-memoryBefore.Mallocs) / float64(b.N)
			b.ReportMetric(0, "ns/op")
			b.ReportMetric(millisecondsPerOp, "ms/op")
			b.ReportMetric(mebibytesPerOp, "MiB/op")
			b.ReportMetric(allocationsPerOp, "alloc/op")
			if backend.backend == registration.BackendVulkan {
				vulkanMillisecondsPerOp = millisecondsPerOp
			} else {
				cpuMillisecondsPerOp = millisecondsPerOp
			}
		})
	}
	if vulkanMillisecondsPerOp > 0 && cpuMillisecondsPerOp > 0 {
		fmt.Printf("CPU/Vulkan time ratio: %.2fx\n", cpuMillisecondsPerOp/vulkanMillisecondsPerOp)
	}
}

func loadBenchmarkImage(b *testing.B, name string) *image.RGBA {
	b.Helper()
	path, err := testutils.GetTestDataFilePath(name)
	if err != nil {
		b.Fatalf("locate benchmark image %q: %v", name, err)
	}
	loaded, err := loadRGBA(path)
	if err != nil {
		b.Fatalf("load benchmark image %q: %v", name, err)
	}
	return loaded
}
