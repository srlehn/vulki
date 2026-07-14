package main

import (
	"fmt"
	"image"
	"math"
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
	reference := loadBenchmarkReference(b, imageA, imageB, maxWidth, maxHeight)
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
			checkBenchmarkResult(b, backend.name, result, reference)
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

func loadBenchmarkReference(
	b *testing.B,
	imageA, imageB *image.RGBA,
	maxWidth, maxHeight int,
) *registration.Result {
	b.Helper()
	correlator, err := registration.NewCorrelator(
		maxWidth,
		maxHeight,
		registration.WithBackend(registration.BackendCPU),
	)
	if err != nil {
		b.Fatalf("create CPU reference correlator: %v", err)
	}
	result, err := correlator.PhaseCorrelate(imageA, imageB)
	if err != nil {
		_ = correlator.Close()
		b.Fatalf("compute CPU reference transform: %v", err)
	}
	if err := correlator.Close(); err != nil {
		b.Fatalf("close CPU reference correlator: %v", err)
	}
	checkBenchmarkResult(b, "CPU reference", result, nil)
	return result
}

func checkBenchmarkResult(
	b *testing.B,
	name string,
	result, reference *registration.Result,
) {
	b.Helper()
	if result == nil {
		b.Fatalf("%s result is nil", name)
	}
	values := []float64{
		result.Angle,
		result.Scale,
		result.Tx,
		result.Ty,
		result.RotationConfidence,
		result.TranslationConfidence,
	}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			b.Fatalf("%s result contains non-finite values: %#v", name, result)
		}
	}
	if result.Scale <= 0 || result.RotationConfidence <= 0 || result.TranslationConfidence <= 0 {
		b.Fatalf("%s result is not a valid transform: %#v", name, result)
	}
	if reference == nil {
		return
	}
	angleDelta := math.Mod(result.Angle-reference.Angle+180, 360) - 180
	if math.Abs(angleDelta) > 0.25 ||
		math.Abs(result.Scale-reference.Scale) > 0.01 ||
		math.Abs(result.Tx-reference.Tx) > 0.25 ||
		math.Abs(result.Ty-reference.Ty) > 0.25 {
		b.Fatalf("%s transform = %#v, want CPU reference %#v", name, result, reference)
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
