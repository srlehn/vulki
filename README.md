# vulki

[![Go Reference](https://pkg.go.dev/badge/github.com/srlehn/vulki/compute.svg)](https://pkg.go.dev/github.com/srlehn/vulki/compute)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![experimental](https://img.shields.io/badge/status-experimental-orange.svg)

Pure-Go Vulkan compute utilities, WGSL shader compilation, and experimental
GPU image registration.

Vulki loads Vulkan through
[`purego`](https://github.com/ebitengine/purego), compiles WGSL to SPIR-V with
[`gogpu/naga`](https://github.com/gogpu/naga), and does not require cgo.

This module is **experimental**. APIs may change. The generic compute path is
usable for development, while the full-GPU image-registration path is still
being validated and debugged.

When you find any errors please report them as issues.

## Requirements

- Go 1.26 or newer.
- A Vulkan 1.1 loader.
- A Vulkan device and driver exposing a compute queue.
- ImageMagick's `convert` command only for the registration CLI self-test.

The loader uses the platform Vulkan library name on Linux, Windows, and macOS.
On macOS, a Vulkan portability implementation such as MoltenVK is required.

## Install

Add a package to a Go module:

```sh
go get github.com/srlehn/vulki/compute
```

Install the example commands:

```sh
go install github.com/srlehn/vulki/cmd/demo@latest
go install github.com/srlehn/vulki/cmd/correlate@latest
```

## Packages

- `compute` manages a Vulkan instance, compute device, buffers, descriptor
  bindings, pipelines, command recording, uploads, and readback.
- `shader` compiles WGSL source to SPIR-V.
- `vk` contains the low-level Vulkan types, constants, loader, and function
  wrappers used by `compute`.
- `imgproc` contains an experimental FFT-based phase-correlation pipeline for
  estimating rotation, scale, and translation between images.

The smallest complete compute example is in
[`cmd/demo`](cmd/demo). Its core setup looks like this:

```go
ctx, err := compute.NewContext()
if err != nil {
    return err
}
defer ctx.Close()

spirv, err := shader.Compile(wgslSource, nil)
if err != nil {
    return err
}

input, err := ctx.CreateBuffer(size, vk.BufferUsageStorageBufferBit)
if err != nil {
    return err
}
defer input.Destroy(ctx)
```

Buffers use device-local memory and host-visible staging memory. Create a
pipeline with `Context.CreateComputePipeline`, dispatch it with
`Context.Dispatch` or a `CommandRecorder`, and download the result from the
output buffer.

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

Run the randomized registration self-test for one PNG. Add `-save` to write a
stacked comparison image:

```sh
go run ./cmd/correlate -save image.png
```

The registration command is a research tool. Its reported transform should be
checked against known inputs before relying on it.

## Development

```sh
go test ./...
go vet ./...
```

Tests that need a Vulkan compute device skip when no suitable loader, driver,
or device is available. A machine with Vulkan is required for full validation.

Dependencies are vendored, so the test and build commands use the checked-in
dependency source by default.

## License

Vulki is available under the [MIT License](LICENSE).
