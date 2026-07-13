package main

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/srlehn/vulki"
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
	fmt.Println("Opening compute device...")
	device, err := vulki.Open()
	if err != nil {
		return fmt.Errorf("open device: %w", err)
	}
	defer device.Close()
	fmt.Printf("Adapter: %s (%s)\n", device.Info().AdapterName, device.Info().Implementation)

	const numElements = 256
	const elementSize = 4
	const bufSize = numElements * elementSize

	inputData := make([]byte, bufSize)
	for i := range numElements {
		binary.LittleEndian.PutUint32(inputData[i*elementSize:], math.Float32bits(float32(i)))
	}

	inputBuf, err := device.NewBuffer(bufSize)
	if err != nil {
		return fmt.Errorf("create input buffer: %w", err)
	}
	defer inputBuf.Close()

	outputBuf, err := device.NewBuffer(bufSize)
	if err != nil {
		return fmt.Errorf("create output buffer: %w", err)
	}
	defer outputBuf.Close()

	if err := inputBuf.Upload(inputData); err != nil {
		return fmt.Errorf("upload: %w", err)
	}

	kernel, err := device.NewKernel(vulki.KernelOptions{
		WGSL: wgslSource,
		Bindings: []vulki.BindingLayout{
			{Binding: 0, Access: vulki.BufferReadOnly},
			{Binding: 1, Access: vulki.BufferReadWrite},
		},
	})
	if err != nil {
		return fmt.Errorf("create kernel: %w", err)
	}
	defer kernel.Close()

	bindings, err := kernel.NewBindings(
		vulki.BindBuffer(0, inputBuf),
		vulki.BindBuffer(1, outputBuf),
	)
	if err != nil {
		return fmt.Errorf("create bindings: %w", err)
	}
	defer bindings.Close()

	fmt.Println("Dispatching compute shader...")
	if err := device.DispatchAndWait(kernel, bindings, vulki.Workgroups{X: numElements / 64, Y: 1, Z: 1}); err != nil {
		return fmt.Errorf("dispatch: %w", err)
	}

	result := make([]byte, bufSize)
	if err := outputBuf.Download(result); err != nil {
		return fmt.Errorf("download: %w", err)
	}

	fmt.Println("\nResults (first 16 elements):")
	fmt.Printf("%-10s %-10s %-10s\n", "Index", "Input", "Output")
	for i := range 16 {
		in := math.Float32frombits(binary.LittleEndian.Uint32(inputData[i*elementSize:]))
		out := math.Float32frombits(binary.LittleEndian.Uint32(result[i*elementSize:]))
		fmt.Printf("%-10d %-10.1f %-10.1f\n", i, in, out)
	}

	errors := 0
	for i := range numElements {
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
