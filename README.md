# vulki

[![Go Reference](https://pkg.go.dev/badge/github.com/srlehn/vulki.svg)](https://pkg.go.dev/github.com/srlehn/vulki)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-srlehn%2Fvulki-blue.svg)](https://deepwiki.com/srlehn/vulki)
![experimental](https://img.shields.io/badge/status-experimental-orange.svg)

Vulki is a cgo-free compute library with a small owner-bound Go API, direct
Vulkan execution, WGSL shader compilation, and experimental image registration.

Vulki loads Vulkan through
[`purego`](https://github.com/ebitengine/purego), compiles WGSL to SPIR-V with
[`gogpu/naga`](https://github.com/gogpu/naga), and does not require cgo.

This module is **experimental**. APIs may change. The root compute package is
usable for development, while image-registration accuracy and portability are
still being validated.

When you find any errors please report them as issues.

## Requirements

- Go 1.26 or newer.
- A 64-bit target.
- A Vulkan 1.1 loader and compute-capable device for GPU acceleration.
- ImageMagick's `convert` command only for the registration CLI self-test.

Vulkan is optional for image registration. The automatic correlator prefers the
GPU and falls back to the CPU implementation when Vulkan is unavailable.
The GPU path stages each packed input once, keeps every intermediate on the
device, and includes both uploads and the 64-byte result readback in one Vulkan
queue submission.

The Linux Vulkan path is the only GPU path with current runtime evidence.
Windows, Android, Apple platforms, browser support, and other platform work are
backlog only and are not currently supported.

## Install

Add a package to a Go module:

```sh
go get github.com/srlehn/vulki
```

Install the example commands:

```sh
go install github.com/srlehn/vulki/cmd/demo@latest
go install github.com/srlehn/vulki/cmd/correlate@latest
```

## Packages

- The root `vulki` package provides owner-bound devices, buffers, reusable WGSL
  kernels, binding sets, synchronous dispatch, and command recording without
  exposing Vulkan handles.
- `shader` compiles WGSL source to SPIR-V.
- `vk` contains the low-level Vulkan types, constants, loader, and function
  wrappers used by the direct implementation.
- `registration` contains an experimental FFT-based phase-correlation pipeline for
  estimating rotation, scale, and translation between images.

### Migration from the legacy packages

- Replace `compute.NewContext` with `vulki.Open` and let each root resource
  remember its creating device.
- Replace Vulkan buffer flags and typed buffers with `Device.NewBuffer` plus
  explicit byte encoders matching the WGSL layout.
- Replace permanently bound compute pipelines with `Device.NewKernel` and
  reusable `Kernel.NewBindings` sets.
- Replace `shader.Compile(source, nil)` with `shader.Compile(source)` or the
  package's functional options.
- Replace the `imgproc` import path with `registration`; use its single
  `NewCorrelator` constructor and functional options.

The smallest complete compute example is in
[`cmd/demo`](cmd/demo). Its core setup looks like this:

```go
device, err := vulki.Open()
if err != nil {
    return err
}
defer device.Close()

input, err := device.NewBuffer(size)
if err != nil {
    return err
}
defer input.Close()

output, err := device.NewBuffer(size)
if err != nil {
    return err
}
defer output.Close()

kernel, err := device.NewKernel(vulki.KernelOptions{
    WGSL: wgslSource,
    Bindings: []vulki.BindingLayout{
        {Binding: 0, Access: vulki.BufferReadOnly},
        {Binding: 1, Access: vulki.BufferReadWrite},
    },
})
if err != nil {
    return err
}
defer kernel.Close()
```

Bind concrete buffers with `Kernel.NewBindings`, upload raw bytes through the
buffer, and call `Device.DispatchAndWait`. Use a `Recorder` to batch arbitrary
recorded uploads and downloads, aligned inline updates, explicit compute
barriers, and multiple dispatches into one blocking queue submission. The
complete checked example is in `cmd/demo`. Compatible submissions from separate
goroutines may remain in flight together: disjoint buffers and shared read-only
buffers can overlap, while any overlapping write remains ordered.

Failures keep their Vulkan cause inspectable with `errors.Is`:
`ErrOutOfDeviceMemory` marks recoverable device memory exhaustion,
`ErrDeviceLost` marks a lost device, and `ErrDeviceUnavailable` marks a device
that refuses further submissions after a failed fence wait. `Device.Err`
reports that unavailable state and its cause.

### Pipeline cache

`Open` creates one application-managed Vulkan pipeline cache per device and
`NewKernel` persists it after each successful pipeline creation. This is
enabled by default so compiled pipelines can be reused by later processes and
by differently named executables. The default file is
`os.UserCacheDir()/vulki/pipeline-<pipelineCacheUUID>.bin`.

Set `VULKI_PIPELINE_CACHE=off` to disable both the in-memory cache and disk
persistence. Set `VULKI_PIPELINE_CACHE_PATH` to an exact file path to override
the default location. Cache files are device- and driver-specific opaque data;
Vulki validates their standard header before use. Missing, corrupt, mismatched,
unreadable, or unwritable files are ignored, so cache failures do not change
the result of `Open` or `NewKernel`.

## Commands

Run the WGSL compute demo, which doubles 256 `float32` values and verifies the
readback:

```sh
go run ./cmd/demo
```

Estimate the transform between two PNG images:

```sh
go run ./cmd/correlate image-a.png image-b.png
```

The command uses `-backend auto` by default. Use `-backend vulkan` to require
Vulkan or `-backend cpu` to bypass Vulkan explicitly:

```sh
go run ./cmd/correlate -backend cpu image-a.png image-b.png
```

The two input images must have the same pixel dimensions. Inputs with different
dimensions are rejected until the higher-resolution-image semantics from the
reference algorithm are implemented explicitly.

Phase-correlation peaks are normalized to the theoretical `[0, 1]` range.
Matches at or below the paper's `0.03` validity threshold return
`registration.ErrLowConfidence` instead of a transform.

Run the randomized registration self-test for one PNG. Add `-save` to write a
stacked comparison image:

```sh
go run ./cmd/correlate -save image.png
```

The registration command is a research tool. Its reported transform should be
checked against known inputs before relying on it.

Library users get the same GPU-first behavior from
`registration.NewCorrelator(maxW, maxH)`. Pass
`registration.WithBackend(registration.BackendVulkan)` or
`registration.WithBackend(registration.BackendCPU)` to require a backend, or
use `registration.WithDevice(device)` to borrow an existing root device.
`Correlator.Backend` reports the implementation actually in use, and
`FallbackReason` explains an automatic CPU selection. `PhaseCorrelate` accepts
any `image.Image` implementation and converts non-RGBA inputs internally.

## Development

```sh
go test ./...
go vet ./...
```

CPU registration tests do not establish support for untested platforms. GPU
tests skip when no suitable Vulkan loader, driver, or device is available; a
skip is not GPU evidence.
When `spirv-val` is installed, the image-processing shader test also validates
every generated module against the Vulkan 1.1 SPIR-V rules.

Dependencies are vendored, so the test and build commands use the checked-in
dependency source by default.

## License

Vulki is available under the [MIT License](LICENSE).
