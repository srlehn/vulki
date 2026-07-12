package shader

import (
	"encoding/binary"
	"errors"
	"testing"
)

const testWGSL = `
@compute @workgroup_size(1)
fn main() {}
`

func TestCompileDefaultsToSPIRV13(t *testing.T) {
	module, err := Compile(testWGSL)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	if got, want := spirvVersionWord(t, module), uint32(0x00010300); got != want {
		t.Fatalf("SPIR-V version word = %#08x, want %#08x", got, want)
	}
}

func TestCompileOptions(t *testing.T) {
	module, err := Compile(testWGSL, WithSPIRVVersion(SPIRV1_0), WithDebugInfo())
	if err != nil {
		t.Fatalf("Compile with options: %v", err)
	}
	if got, want := spirvVersionWord(t, module), uint32(0x00010000); got != want {
		t.Fatalf("SPIR-V version word = %#08x, want %#08x", got, want)
	}
}

func TestCompileRejectsInvalidOption(t *testing.T) {
	_, err := Compile(testWGSL, WithSPIRVVersion(SPIRVVersion(99)))
	var compileErr *CompileError
	if !errors.As(err, &compileErr) {
		t.Fatalf("Compile error = %v, want *CompileError", err)
	}
	if compileErr.Stage != CompileStageOptions {
		t.Fatalf("Compile stage = %q, want %q", compileErr.Stage, CompileStageOptions)
	}
}

func TestCompileRejectsNilOption(t *testing.T) {
	_, err := Compile(testWGSL, nil)
	var compileErr *CompileError
	if !errors.As(err, &compileErr) {
		t.Fatalf("Compile error = %v, want *CompileError", err)
	}
	if compileErr.Stage != CompileStageOptions {
		t.Fatalf("Compile stage = %q, want %q", compileErr.Stage, CompileStageOptions)
	}
}

func TestCompileReturnsOwnedCompilerError(t *testing.T) {
	_, err := Compile("not wgsl")
	var compileErr *CompileError
	if !errors.As(err, &compileErr) {
		t.Fatalf("Compile error = %v, want *CompileError", err)
	}
	if compileErr.Stage != CompileStageCompiler {
		t.Fatalf("Compile stage = %q, want %q", compileErr.Stage, CompileStageCompiler)
	}
}

func spirvVersionWord(t *testing.T, module []byte) uint32 {
	t.Helper()
	if len(module) < 8 {
		t.Fatalf("SPIR-V module has %d bytes, want at least 8", len(module))
	}
	return binary.LittleEndian.Uint32(module[4:8])
}
