package shader

import "github.com/gogpu/naga"

// Compile compiles WGSL source code to SPIR-V with custom options.
func Compile(wgslSource string, opts *naga.CompileOptions) ([]byte, error) {
	if opts == nil {
		opts = new(naga.DefaultOptions())
	}
	return naga.CompileWithOptions(wgslSource, *opts)
}
