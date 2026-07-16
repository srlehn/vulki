package shader

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"

	"github.com/gogpu/naga"
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
	module, err := Compile(testWGSL, WithSPIRVVersion(SPIRV1_0), WithDebugInfo)
	if err != nil {
		t.Fatalf("Compile with options: %v", err)
	}
	if got, want := spirvVersionWord(t, module), uint32(0x00010000); got != want {
		t.Fatalf("SPIR-V version word = %#08x, want %#08x", got, want)
	}
}

func TestCompileBooleanOptions(t *testing.T) {
	config := compileConfig{naga: naga.DefaultOptions()}
	if err := WithDebugInfo(&config); err != nil {
		t.Fatalf("WithDebugInfo: %v", err)
	}
	if err := WithoutValidation(&config); err != nil {
		t.Fatalf("WithoutValidation: %v", err)
	}
	if !config.naga.Debug {
		t.Fatal("WithDebugInfo did not enable debug information")
	}
	if config.naga.Validate {
		t.Fatal("WithoutValidation did not disable validation")
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

func TestCompileReturnsCachedModuleWithoutRecompiling(t *testing.T) {
	t.Cleanup(resetModuleCache)
	resetModuleCache()
	// The source is not valid WGSL, so a successful result proves the seeded
	// cache entry was returned without invoking the compiler.
	source := "cache probe: not wgsl"
	key := moduleCacheKey{source: source, options: naga.DefaultOptions()}
	marker := []byte{1, 2, 3, 4}
	storeCachedModule(key, marker)

	module, err := Compile(source)
	if err != nil {
		t.Fatalf("Compile with seeded cache: %v", err)
	}
	if !bytes.Equal(module, marker) {
		t.Fatalf("cached module = %v, want %v", module, marker)
	}
	module[0] = 99
	again, err := Compile(source)
	if err != nil {
		t.Fatalf("Compile after mutating cached result: %v", err)
	}
	if !bytes.Equal(again, marker) {
		t.Fatalf("mutating a returned module changed the cache: %v", again)
	}
}

func TestCompileIsolatesCachedBytes(t *testing.T) {
	t.Cleanup(resetModuleCache)
	resetModuleCache()
	first, err := Compile(testWGSL)
	if err != nil {
		t.Fatalf("first Compile: %v", err)
	}
	pristine := slices.Clone(first)
	first[0]++
	second, err := Compile(testWGSL)
	if err != nil {
		t.Fatalf("second Compile: %v", err)
	}
	if !bytes.Equal(second, pristine) {
		t.Fatal("mutating the freshly compiled module changed the cache")
	}
}

func TestCompileCachesPerResolvedOptions(t *testing.T) {
	t.Cleanup(resetModuleCache)
	resetModuleCache()
	defaultModule, err := Compile(testWGSL)
	if err != nil {
		t.Fatalf("Compile with defaults: %v", err)
	}
	oldModule, err := Compile(testWGSL, WithSPIRVVersion(SPIRV1_0))
	if err != nil {
		t.Fatalf("Compile with SPIR-V 1.0: %v", err)
	}
	if got, want := spirvVersionWord(t, defaultModule), uint32(0x00010300); got != want {
		t.Fatalf("default version word = %#08x, want %#08x", got, want)
	}
	if got, want := spirvVersionWord(t, oldModule), uint32(0x00010000); got != want {
		t.Fatalf("SPIR-V 1.0 version word = %#08x, want %#08x", got, want)
	}
	if got := cachedModuleCount(); got != 2 {
		t.Fatalf("cached modules = %d, want 2", got)
	}
}

func TestModuleCacheStaysBounded(t *testing.T) {
	t.Cleanup(resetModuleCache)
	resetModuleCache()
	for index := 0; index < maxCachedModules+8; index++ {
		key := moduleCacheKey{
			source:  fmt.Sprintf("synthetic source %d", index),
			options: naga.DefaultOptions(),
		}
		storeCachedModule(key, []byte{byte(index)})
	}
	if got := cachedModuleCount(); got > maxCachedModules {
		t.Fatalf("cached modules = %d, want at most %d", got, maxCachedModules)
	}

	resetModuleCache()
	keys := make([]moduleCacheKey, maxCachedModules)
	for index := range keys {
		keys[index] = moduleCacheKey{
			source:  fmt.Sprintf("bounded source %d", index),
			options: naga.DefaultOptions(),
		}
		storeCachedModule(keys[index], []byte{1})
	}
	storeCachedModule(keys[0], []byte{2})
	if got := cachedModuleCount(); got != maxCachedModules {
		t.Fatalf("cached modules after key update = %d, want %d", got, maxCachedModules)
	}
	updated, ok := loadCachedModule(keys[0])
	if !ok || !bytes.Equal(updated, []byte{2}) {
		t.Fatalf("updated cache entry = %v, %t, want [2], true", updated, ok)
	}
}

func TestCompileConcurrentSharedSource(t *testing.T) {
	t.Cleanup(resetModuleCache)
	resetModuleCache()
	reference, err := Compile(testWGSL)
	if err != nil {
		t.Fatalf("reference Compile: %v", err)
	}
	failures := make(chan error, 8)
	var group sync.WaitGroup
	for index := 0; index < cap(failures); index++ {
		group.Add(1)
		go func() {
			defer group.Done()
			module, err := Compile(testWGSL)
			if err != nil {
				failures <- err
				return
			}
			if !bytes.Equal(module, reference) {
				failures <- fmt.Errorf("concurrent module differs from reference")
			}
		}()
	}
	group.Wait()
	close(failures)
	for err := range failures {
		t.Fatal(err)
	}
}

func resetModuleCache() {
	moduleCacheMu.Lock()
	moduleCache = nil
	moduleCacheMu.Unlock()
}

func cachedModuleCount() int {
	moduleCacheMu.Lock()
	defer moduleCacheMu.Unlock()
	return len(moduleCache)
}

func spirvVersionWord(t *testing.T, module []byte) uint32 {
	t.Helper()
	if len(module) < 8 {
		t.Fatalf("SPIR-V module has %d bytes, want at least 8", len(module))
	}
	return binary.LittleEndian.Uint32(module[4:8])
}
