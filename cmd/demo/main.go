package main

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"vkpg/compute"
	"vkpg/shader"
	"vkpg/vk"
)

//go:embed double.wgsl
var wgslSource string

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	// 1. Compile WGSL → SPIR-V.
	fmt.Println("Compiling WGSL shader...")
	spirv, err := shader.Compile(wgslSource, nil)
	if err != nil {
		return fmt.Errorf("shader compile: %w", err)
	}
	fmt.Printf("SPIR-V: %d bytes\n", len(spirv))

	// 2. Create Vulkan compute context.
	fmt.Println("Creating Vulkan compute context...")
	ctx, err := compute.NewContext()
	if err != nil {
		return fmt.Errorf("context: %w", err)
	}
	defer ctx.Close()

	// 3. Prepare input data.
	const numElements = 256
	const elementSize = 4 // float32
	const bufSize = numElements * elementSize

	inputData := make([]byte, bufSize)
	for i := 0; i < numElements; i++ {
		binary.LittleEndian.PutUint32(inputData[i*elementSize:], math.Float32bits(float32(i)))
	}

	// 4. Create input/output buffers.
	inputBuf, err := ctx.CreateBuffer(bufSize, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return fmt.Errorf("create input buffer: %w", err)
	}
	defer inputBuf.Destroy(ctx)

	outputBuf, err := ctx.CreateBuffer(bufSize, vk.BufferUsageStorageBufferBit)
	if err != nil {
		return fmt.Errorf("create output buffer: %w", err)
	}
	defer outputBuf.Destroy(ctx)

	// 5. Upload input data.
	if err := inputBuf.Upload(ctx, inputData); err != nil {
		return fmt.Errorf("upload: %w", err)
	}

	// 6. Create compute pipeline.
	pipeline, err := ctx.CreateComputePipeline(spirv, []compute.BufferBinding{
		{Binding: 0, Buffer: inputBuf},
		{Binding: 1, Buffer: outputBuf},
	})
	if err != nil {
		return fmt.Errorf("create pipeline: %w", err)
	}
	defer pipeline.Destroy(ctx)

	// 7. Dispatch compute shader.
	fmt.Println("Dispatching compute shader...")
	if err := ctx.Dispatch(pipeline, numElements/64, 1, 1); err != nil {
		return fmt.Errorf("dispatch: %w", err)
	}

	// 8. Read back results.
	result, err := outputBuf.Download(ctx, bufSize)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}

	// 9. Print results.
	fmt.Println("\nResults (first 16 elements):")
	fmt.Printf("%-10s %-10s %-10s\n", "Index", "Input", "Output")
	for i := 0; i < 16; i++ {
		in := math.Float32frombits(binary.LittleEndian.Uint32(inputData[i*elementSize:]))
		out := math.Float32frombits(binary.LittleEndian.Uint32(result[i*elementSize:]))
		fmt.Printf("%-10d %-10.1f %-10.1f\n", i, in, out)
	}

	// Verify all results.
	errors := 0
	for i := 0; i < numElements; i++ {
		in := math.Float32frombits(binary.LittleEndian.Uint32(inputData[i*elementSize:]))
		out := math.Float32frombits(binary.LittleEndian.Uint32(result[i*elementSize:]))
		if out != in*2.0 {
			errors++
			if errors <= 5 {
				fmt.Printf("MISMATCH at %d: expected %.1f, got %.1f\n", i, in*2.0, out)
			}
		}
	}
	if errors == 0 {
		fmt.Printf("\nAll %d elements verified correct!\n", numElements)
	} else {
		fmt.Printf("\n%d/%d elements incorrect\n", errors, numElements)
	}

	return nil
}
