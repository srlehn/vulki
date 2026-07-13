package shader_test

import (
	"errors"

	"github.com/srlehn/vulki/shader"
)

const computeWGSL = `
@compute @workgroup_size(1)
fn main() {}
`

func ExampleCompile() {
	_, _ = shader.Compile(computeWGSL)
}

func ExampleCompile_options() {
	_, _ = shader.Compile(
		computeWGSL,
		shader.WithSPIRVVersion(shader.SPIRV1_0),
		shader.WithDebugInfo,
	)
}

func ExampleCompileError() {
	_, err := shader.Compile("not WGSL")
	var compileError *shader.CompileError
	if errors.As(err, &compileError) {
		_ = compileError.Stage
		_ = compileError.Message
	}
}
