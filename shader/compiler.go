package shader

import "github.com/gogpu/naga"

// Compile compiles WGSL source code to SPIR-V binary using default options.
func Compile(wgslSource string) ([]byte, error) {
	return naga.Compile(wgslSource)
}

// CompileWithOptions compiles WGSL source code to SPIR-V with custom options.
func CompileWithOptions(wgslSource string, opts naga.CompileOptions) ([]byte, error) {
	return naga.CompileWithOptions(wgslSource, opts)
}
