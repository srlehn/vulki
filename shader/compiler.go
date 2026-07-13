package shader

import (
	"fmt"

	"github.com/gogpu/naga"
	"github.com/gogpu/naga/spirv"
)

// SPIRVVersion identifies a supported SPIR-V output version.
type SPIRVVersion uint8

const (
	// SPIRV1_0 targets SPIR-V 1.0.
	SPIRV1_0 SPIRVVersion = iota
	// SPIRV1_3 targets SPIR-V 1.3 and is the default.
	SPIRV1_3
)

// CompileStage identifies the stage that rejected shader compilation.
type CompileStage string

const (
	// CompileStageOptions validates Compile options before invoking the compiler.
	CompileStageOptions CompileStage = "options"
	// CompileStageCompiler parses, validates, and generates the shader module.
	CompileStageCompiler CompileStage = "compiler"
)

// CompileError reports a WGSL compilation failure without exposing the
// underlying compiler implementation's error types.
type CompileError struct {
	Stage   CompileStage
	Message string
}

func (e *CompileError) Error() string {
	if e == nil {
		return "shader: compile failed"
	}
	return fmt.Sprintf("shader: %s: %s", e.Stage, e.Message)
}

type compileConfig struct {
	naga naga.CompileOptions
}

// Option configures WGSL to SPIR-V compilation.
type Option func(*compileConfig) error

// WithSPIRVVersion selects the generated SPIR-V version.
func WithSPIRVVersion(version SPIRVVersion) Option {
	return func(config *compileConfig) error {
		switch version {
		case SPIRV1_0:
			config.naga.SPIRVVersion = spirv.Version1_0
		case SPIRV1_3:
			config.naga.SPIRVVersion = spirv.Version1_3
		default:
			return fmt.Errorf("unsupported SPIR-V version %d", version)
		}
		return nil
	}
}

// WithDebugInfo includes source names and line information in the generated
// SPIR-V module.
var WithDebugInfo = withDebugInfo()

// WithoutValidation disables intermediate-representation validation before
// SPIR-V generation. Validation is enabled by default and should normally
// remain enabled.
var WithoutValidation = withoutValidation()

func withDebugInfo() Option {
	return func(config *compileConfig) error {
		config.naga.Debug = true
		return nil
	}
}

func withoutValidation() Option {
	return func(config *compileConfig) error {
		config.naga.Validate = false
		return nil
	}
}

// Compile compiles WGSL source code to SPIR-V. With no options it targets
// SPIR-V 1.3, includes no debug information, and validates the shader.
func Compile(wgslSource string, options ...Option) ([]byte, error) {
	config := compileConfig{naga: naga.DefaultOptions()}
	for i, option := range options {
		if option == nil {
			return nil, &CompileError{
				Stage:   CompileStageOptions,
				Message: fmt.Sprintf("option %d is nil", i),
			}
		}
		if err := option(&config); err != nil {
			return nil, &CompileError{
				Stage:   CompileStageOptions,
				Message: err.Error(),
			}
		}
	}

	module, err := naga.CompileWithOptions(wgslSource, config.naga)
	if err != nil {
		return nil, &CompileError{
			Stage:   CompileStageCompiler,
			Message: err.Error(),
		}
	}
	return module, nil
}
