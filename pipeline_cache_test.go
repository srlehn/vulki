package vulki

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/srlehn/vulki/vk"
)

func TestDefaultPipelineCacheConfig(t *testing.T) {
	uuid := [16]byte{1, 2, 3, 4}
	t.Setenv("VULKI_PIPELINE_CACHE", "OFF")
	t.Setenv("VULKI_PIPELINE_CACHE_PATH", "/ignored")
	if config := defaultPipelineCacheConfig(uuid); config.enabled {
		t.Fatalf("disabled config = %#v", config)
	}

	t.Setenv("VULKI_PIPELINE_CACHE", "")
	override := filepath.Join(t.TempDir(), "custom.bin")
	t.Setenv("VULKI_PIPELINE_CACHE_PATH", override)
	if config := defaultPipelineCacheConfig(uuid); !config.enabled || config.path != override {
		t.Fatalf("override config = %#v", config)
	}

	root := t.TempDir()
	t.Setenv("VULKI_PIPELINE_CACHE_PATH", "")
	t.Setenv("XDG_CACHE_HOME", root)
	config := defaultPipelineCacheConfig(uuid)
	want := filepath.Join(root, "vulki", "pipeline-01020304000000000000000000000000.bin")
	if !config.enabled || config.path != want {
		t.Fatalf("default config = %#v, want path %q", config, want)
	}
}

func TestValidPipelineCacheData(t *testing.T) {
	identity := pipelineCacheIdentity{
		vendorID: 0x10de,
		deviceID: 0x2504,
		uuid:     [16]byte{1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	valid := testPipelineCacheData(identity, []byte{9, 8, 7})
	if !validPipelineCacheData(valid, identity) {
		t.Fatal("valid pipeline cache rejected")
	}

	tests := []struct {
		name string
		data func() []byte
	}{
		{name: "empty", data: func() []byte { return nil }},
		{name: "truncated", data: func() []byte { return bytes.Clone(valid[:31]) }},
		{name: "header size", data: func() []byte {
			data := bytes.Clone(valid)
			binary.LittleEndian.PutUint32(data[0:4], 31)
			return data
		}},
		{name: "header version", data: func() []byte {
			data := bytes.Clone(valid)
			binary.LittleEndian.PutUint32(data[4:8], 2)
			return data
		}},
		{name: "vendor", data: func() []byte {
			data := bytes.Clone(valid)
			data[8]++
			return data
		}},
		{name: "device", data: func() []byte {
			data := bytes.Clone(valid)
			data[12]++
			return data
		}},
		{name: "uuid", data: func() []byte {
			data := bytes.Clone(valid)
			data[16]++
			return data
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if validPipelineCacheData(test.data(), identity) {
				t.Fatal("invalid pipeline cache accepted")
			}
		})
	}
}

func TestNewPipelineCacheStateInitialDataFallbacks(t *testing.T) {
	identity := pipelineCacheIdentity{vendorID: 3, deviceID: 4, uuid: [16]byte{5}}
	valid := testPipelineCacheData(identity, []byte{1, 2, 3})

	t.Run("loads valid data", func(t *testing.T) {
		var creates [][]byte
		destroyed := vk.PipelineCache(0)
		state := newPipelineCacheState(
			nil,
			vk.Device(1),
			identity,
			pipelineCacheConfig{enabled: true, path: "cache.bin"},
			pipelineCacheDriverOps{
				create: func(_ *vk.DeviceFuncs, _ vk.Device, data []byte) (vk.PipelineCache, error) {
					creates = append(creates, bytes.Clone(data))
					return vk.PipelineCache(7), nil
				},
				destroy: func(_ *vk.DeviceFuncs, _ vk.Device, cache vk.PipelineCache) {
					destroyed = cache
				},
			},
			pipelineCacheFileOps{read: func(string) ([]byte, error) { return bytes.Clone(valid), nil }},
		)
		if state == nil || state.handle() != vk.PipelineCache(7) {
			t.Fatalf("state = %#v", state)
		}
		if len(creates) != 1 || !bytes.Equal(creates[0], valid) {
			t.Fatalf("initial create data = %v", creates)
		}
		if !state.hasLast || state.lastSize != len(valid) {
			t.Fatal("loaded data digest was not retained")
		}
		state.close(nil, vk.Device(1))
		state.close(nil, vk.Device(1))
		if destroyed != vk.PipelineCache(7) {
			t.Fatalf("destroyed cache = %d, want 7", destroyed)
		}
	})

	t.Run("rejects invalid header", func(t *testing.T) {
		var initial []byte
		invalid := bytes.Clone(valid)
		invalid[16]++
		state := newPipelineCacheState(
			nil,
			vk.Device(1),
			identity,
			pipelineCacheConfig{enabled: true, path: "cache.bin"},
			pipelineCacheDriverOps{create: func(_ *vk.DeviceFuncs, _ vk.Device, data []byte) (vk.PipelineCache, error) {
				initial = bytes.Clone(data)
				return vk.PipelineCache(8), nil
			}},
			pipelineCacheFileOps{read: func(string) ([]byte, error) { return invalid, nil }},
		)
		if state == nil || len(initial) != 0 || state.hasLast {
			t.Fatalf("state = %#v, initial = %v", state, initial)
		}
	})

	t.Run("ignores read failure", func(t *testing.T) {
		var initial []byte
		state := newPipelineCacheState(
			nil,
			vk.Device(1),
			identity,
			pipelineCacheConfig{enabled: true, path: "missing.bin"},
			pipelineCacheDriverOps{create: func(_ *vk.DeviceFuncs, _ vk.Device, data []byte) (vk.PipelineCache, error) {
				initial = bytes.Clone(data)
				return vk.PipelineCache(8), nil
			}},
			pipelineCacheFileOps{read: func(string) ([]byte, error) {
				return nil, os.ErrNotExist
			}},
		)
		if state == nil || len(initial) != 0 || state.hasLast {
			t.Fatalf("state = %#v, initial = %v", state, initial)
		}
	})

	t.Run("retries empty after initial failure", func(t *testing.T) {
		var creates [][]byte
		state := newPipelineCacheState(
			nil,
			vk.Device(1),
			identity,
			pipelineCacheConfig{enabled: true, path: "cache.bin"},
			pipelineCacheDriverOps{create: func(_ *vk.DeviceFuncs, _ vk.Device, data []byte) (vk.PipelineCache, error) {
				creates = append(creates, bytes.Clone(data))
				if len(data) > 0 {
					return 0, errors.New("injected initial-data failure")
				}
				return vk.PipelineCache(9), nil
			}},
			pipelineCacheFileOps{read: func(string) ([]byte, error) { return bytes.Clone(valid), nil }},
		)
		if state == nil || state.handle() != vk.PipelineCache(9) || state.hasLast {
			t.Fatalf("state = %#v", state)
		}
		if len(creates) != 2 || !bytes.Equal(creates[0], valid) || len(creates[1]) != 0 {
			t.Fatalf("create attempts = %v", creates)
		}
	})

	t.Run("creation failure disables cache", func(t *testing.T) {
		state := newPipelineCacheState(
			nil,
			vk.Device(1),
			identity,
			pipelineCacheConfig{enabled: true},
			pipelineCacheDriverOps{create: func(*vk.DeviceFuncs, vk.Device, []byte) (vk.PipelineCache, error) {
				return 0, errors.New("injected creation failure")
			}},
			pipelineCacheFileOps{},
		)
		if state != nil {
			t.Fatalf("state = %#v, want nil", state)
		}
	})
}

func TestPipelineCacheStatePersistsChangedDataOnly(t *testing.T) {
	identity := pipelineCacheIdentity{vendorID: 1, deviceID: 2, uuid: [16]byte{3}}
	current := testPipelineCacheData(identity, []byte{4})
	var writes [][]byte
	state := &pipelineCacheState{
		cache:    vk.PipelineCache(5),
		identity: identity,
		path:     "cache.bin",
		driver: pipelineCacheDriverOps{data: func(_ *vk.DeviceFuncs, _ vk.Device, _ vk.PipelineCache, data []byte) (uintptr, error) {
			if data == nil {
				return uintptr(len(current)), nil
			}
			copy(data, current)
			return uintptr(len(current)), nil
		}},
		files: pipelineCacheFileOps{write: func(_ string, data []byte) error {
			writes = append(writes, bytes.Clone(data))
			return nil
		}},
	}

	state.persist(nil, vk.Device(1))
	state.persist(nil, vk.Device(1))
	if len(writes) != 1 || !bytes.Equal(writes[0], current) {
		t.Fatalf("unchanged writes = %v", writes)
	}
	current = testPipelineCacheData(identity, []byte{4, 5})
	state.persist(nil, vk.Device(1))
	if len(writes) != 2 || !bytes.Equal(writes[1], current) {
		t.Fatalf("changed writes = %v", writes)
	}
}

func TestPipelineCacheStateSizePrecheckSkipsRetrieval(t *testing.T) {
	identity := pipelineCacheIdentity{vendorID: 1, deviceID: 2, uuid: [16]byte{3}}
	current := testPipelineCacheData(identity, []byte{4})
	sizeCalls, dataCalls, writes := 0, 0, 0
	state := &pipelineCacheState{
		cache:    vk.PipelineCache(5),
		identity: identity,
		path:     "cache.bin",
		driver: pipelineCacheDriverOps{data: func(_ *vk.DeviceFuncs, _ vk.Device, _ vk.PipelineCache, data []byte) (uintptr, error) {
			if data == nil {
				sizeCalls++
				return uintptr(len(current)), nil
			}
			dataCalls++
			copy(data, current)
			return uintptr(len(current)), nil
		}},
		files: pipelineCacheFileOps{write: func(string, []byte) error {
			writes++
			return nil
		}},
	}
	state.remember(current)

	state.persist(nil, vk.Device(1))
	if sizeCalls != 1 || dataCalls != 0 || writes != 0 {
		t.Fatalf("warm persist calls: size=%d data=%d writes=%d, want one size query only",
			sizeCalls, dataCalls, writes)
	}

	current = testPipelineCacheData(identity, []byte{4, 5})
	state.persist(nil, vk.Device(1))
	if dataCalls != 1 || writes != 1 {
		t.Fatalf("grown persist calls: data=%d writes=%d, want full retrieval and write",
			dataCalls, writes)
	}
	if state.lastSize != len(current) {
		t.Fatalf("lastSize = %d, want %d", state.lastSize, len(current))
	}
}

func TestPipelineCacheStateRetriesFailedWrite(t *testing.T) {
	identity := pipelineCacheIdentity{vendorID: 1, deviceID: 2, uuid: [16]byte{3}}
	data := testPipelineCacheData(identity, []byte{4})
	writes := 0
	state := &pipelineCacheState{
		cache:    vk.PipelineCache(5),
		identity: identity,
		path:     "cache.bin",
		driver: pipelineCacheDriverOps{data: func(_ *vk.DeviceFuncs, _ vk.Device, _ vk.PipelineCache, target []byte) (uintptr, error) {
			if target == nil {
				return uintptr(len(data)), nil
			}
			copy(target, data)
			return uintptr(len(data)), nil
		}},
		files: pipelineCacheFileOps{write: func(string, []byte) error {
			writes++
			return errors.New("injected write failure")
		}},
	}
	state.persist(nil, vk.Device(1))
	state.persist(nil, vk.Device(1))
	if writes != 2 || state.hasLast {
		t.Fatalf("write attempts = %d, hasLast = %t", writes, state.hasLast)
	}
}

func TestRetrievePipelineCacheDataRetriesIncomplete(t *testing.T) {
	identity := pipelineCacheIdentity{vendorID: 1, deviceID: 2, uuid: [16]byte{3}}
	first := testPipelineCacheData(identity, []byte{4})
	second := testPipelineCacheData(identity, []byte{4, 5, 6})
	dataCalls := 0
	data, err := retrievePipelineCacheData(
		nil,
		vk.Device(1),
		vk.PipelineCache(2),
		func(_ *vk.DeviceFuncs, _ vk.Device, _ vk.PipelineCache, target []byte) (uintptr, error) {
			if target == nil {
				if dataCalls == 0 {
					return uintptr(len(first)), nil
				}
				return uintptr(len(second)), nil
			}
			dataCalls++
			if dataCalls == 1 {
				copy(target, first)
				return uintptr(len(target)), &vk.Error{Op: "vkGetPipelineCacheData", Result: vk.Incomplete}
			}
			copy(target, second)
			return uintptr(len(second)), nil
		},
	)
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if !bytes.Equal(data, second) || dataCalls != 2 {
		t.Fatalf("data = %v, calls = %d", data, dataCalls)
	}
}

func TestRetrievePipelineCacheDataRejectsOversize(t *testing.T) {
	calls := 0
	_, err := retrievePipelineCacheData(
		nil,
		vk.Device(1),
		vk.PipelineCache(2),
		func(_ *vk.DeviceFuncs, _ vk.Device, _ vk.PipelineCache, data []byte) (uintptr, error) {
			calls++
			if data != nil {
				t.Fatal("oversized query allocated a data buffer")
			}
			return maxPipelineCacheDataSize + 1, nil
		},
	)
	if err == nil || calls != 1 {
		t.Fatalf("error = %v, calls = %d", err, calls)
	}
}

func TestWritePipelineCacheFileReplacesAtomically(t *testing.T) {
	directory := filepath.Join(t.TempDir(), "nested")
	path := filepath.Join(directory, "cache.bin")
	first := []byte{1, 2, 3}
	second := []byte{4, 5, 6, 7}
	if err := writePipelineCacheFile(path, first); err != nil {
		t.Fatalf("first write: %v", err)
	}
	if err := writePipelineCacheFile(path, second); err != nil {
		t.Fatalf("replacement write: %v", err)
	}
	got, err := readPipelineCacheFile(path)
	if err != nil {
		t.Fatalf("read replacement: %v", err)
	}
	if !bytes.Equal(got, second) {
		t.Fatalf("cache data = %v, want %v", got, second)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat replacement: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("cache mode = %o, want 600", info.Mode().Perm())
	}
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatalf("read cache directory: %v", err)
	}
	for _, entry := range entries {
		if strings.Contains(entry.Name(), ".tmp-") {
			t.Fatalf("temporary file remains: %s", entry.Name())
		}
	}
}

func TestReadPipelineCacheFileRejectsOversize(t *testing.T) {
	path := filepath.Join(t.TempDir(), "oversized.bin")
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create oversized cache: %v", err)
	}
	if err := file.Truncate(maxPipelineCacheDataSize + 1); err != nil {
		_ = file.Close()
		t.Fatalf("truncate oversized cache: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("close oversized cache: %v", err)
	}
	if _, err := readPipelineCacheFile(path); err == nil {
		t.Fatal("oversized cache was accepted")
	}
}

func TestDeviceDestroysPipelineCacheBeforeDevice(t *testing.T) {
	var cleanup []string
	device, err := openWithHooksAndPipelineCache(
		fakeOpenHooks("", nil, &cleanup),
		func(*vk.DeviceFuncs, vk.Device, vk.PhysicalDeviceProperties) *pipelineCacheState {
			return &pipelineCacheState{
				cache: vk.PipelineCache(51),
				driver: pipelineCacheDriverOps{destroy: func(*vk.DeviceFuncs, vk.Device, vk.PipelineCache) {
					cleanup = append(cleanup, "pipeline-cache")
				}},
			}
		},
	)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := device.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	want := []string{"wait", "pipeline-cache", "device", "instance", "loader"}
	if !reflect.DeepEqual(cleanup, want) {
		t.Fatalf("cleanup = %v, want %v", cleanup, want)
	}
}

func TestNewKernelUsesPipelineCacheAndIgnoresPersistenceFailure(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	var cleanup []string
	operations := fakeKernelOperations("", &cleanup)
	gotCache := vk.PipelineCache(0)
	operations.createComputePipelines = func(_ *vk.DeviceFuncs, _ vk.Device, cache vk.PipelineCache, _ []vk.ComputePipelineCreateInfo) ([]vk.Pipeline, error) {
		gotCache = cache
		return []vk.Pipeline{4}, nil
	}
	device.state.kernelOps = operations
	device.state.pipelineCache = &pipelineCacheState{
		cache: vk.PipelineCache(33),
		path:  "cache.bin",
		driver: pipelineCacheDriverOps{data: func(*vk.DeviceFuncs, vk.Device, vk.PipelineCache, []byte) (uintptr, error) {
			return 0, errors.New("injected retrieval failure")
		}},
		files: pipelineCacheFileOps{write: func(string, []byte) error {
			t.Fatal("write called after retrieval failure")
			return nil
		}},
	}

	kernel, err := device.NewKernel(KernelOptions{
		WGSL:     doubleKernelWGSL,
		Bindings: []BindingLayout{{Binding: 0, Access: BufferReadOnly}},
	})
	if err != nil {
		t.Fatalf("NewKernel: %v", err)
	}
	if gotCache != vk.PipelineCache(33) {
		t.Fatalf("pipeline cache = %d, want 33", gotCache)
	}
	if err := kernel.Close(); err != nil {
		t.Fatalf("Kernel.Close: %v", err)
	}
}

func TestConcurrentNewKernelDoesNotSerializePipelineCreation(t *testing.T) {
	device, _, _ := fakeBufferDevice("")
	var cleanup []string
	operations := fakeKernelOperations("", &cleanup)
	var createMu sync.Mutex
	active := 0
	maxActive := 0
	ready := make(chan struct{})
	var readyOnce sync.Once
	operations.createComputePipelines = func(_ *vk.DeviceFuncs, _ vk.Device, cache vk.PipelineCache, _ []vk.ComputePipelineCreateInfo) ([]vk.Pipeline, error) {
		if cache != vk.PipelineCache(44) {
			return nil, errors.New("wrong pipeline cache")
		}
		createMu.Lock()
		active++
		maxActive = max(maxActive, active)
		if active == 2 {
			readyOnce.Do(func() { close(ready) })
		}
		createMu.Unlock()
		defer func() {
			createMu.Lock()
			active--
			createMu.Unlock()
		}()
		select {
		case <-ready:
		case <-time.After(2 * time.Second):
			return nil, errors.New("pipeline creation was serialized")
		}
		return []vk.Pipeline{4}, nil
	}
	device.state.kernelOps = operations
	identity := pipelineCacheIdentity{vendorID: 1, deviceID: 2, uuid: [16]byte{3}}
	data := testPipelineCacheData(identity, []byte{4})
	device.state.pipelineCache = &pipelineCacheState{
		cache:    vk.PipelineCache(44),
		identity: identity,
		path:     "cache.bin",
		driver: pipelineCacheDriverOps{data: func(_ *vk.DeviceFuncs, _ vk.Device, _ vk.PipelineCache, target []byte) (uintptr, error) {
			if target == nil {
				return uintptr(len(data)), nil
			}
			copy(target, data)
			return uintptr(len(data)), nil
		}},
		files: pipelineCacheFileOps{write: func(string, []byte) error { return nil }},
	}

	results := make(chan struct {
		kernel *Kernel
		err    error
	}, 2)
	for range 2 {
		go func() {
			kernel, err := device.NewKernel(KernelOptions{
				WGSL:     doubleKernelWGSL,
				Bindings: []BindingLayout{{Binding: 0, Access: BufferReadOnly}},
			})
			results <- struct {
				kernel *Kernel
				err    error
			}{kernel: kernel, err: err}
		}()
	}
	var kernels []*Kernel
	for range 2 {
		result := <-results
		if result.err != nil {
			t.Fatalf("NewKernel: %v", result.err)
		}
		kernels = append(kernels, result.kernel)
	}
	if maxActive != 2 {
		t.Fatalf("maximum concurrent pipeline creations = %d, want 2", maxActive)
	}
	for _, kernel := range kernels {
		if err := kernel.Close(); err != nil {
			t.Fatalf("Kernel.Close: %v", err)
		}
	}
}

func TestPipelineCachePersistsAndRecoversDirectVulkan(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache", "pipelines.bin")
	t.Setenv("VULKI_PIPELINE_CACHE", "")
	t.Setenv("VULKI_PIPELINE_CACHE_PATH", path)

	compile := func(skipUnavailable, wantLoaded bool) pipelineCacheIdentity {
		device, err := Open()
		if err != nil {
			if skipUnavailable {
				t.Skipf("direct Vulkan device unavailable: %v", err)
			}
			t.Fatalf("second Open: %v", err)
		}
		if device.state.pipelineCache == nil {
			_ = device.Close()
			t.Fatal("direct Vulkan device opened without a pipeline cache")
		}
		identity := device.state.pipelineCache.identity
		if device.state.pipelineCache.hasLast != wantLoaded {
			_ = device.Close()
			t.Fatalf("loaded cache = %t, want %t", device.state.pipelineCache.hasLast, wantLoaded)
		}
		kernel, err := device.NewKernel(KernelOptions{
			WGSL:     doubleKernelWGSL,
			Bindings: []BindingLayout{{Binding: 0, Access: BufferReadOnly}},
		})
		if err != nil {
			_ = device.Close()
			t.Fatalf("NewKernel: %v", err)
		}
		if err := kernel.Close(); err != nil {
			_ = device.Close()
			t.Fatalf("Kernel.Close: %v", err)
		}
		if err := device.Close(); err != nil {
			t.Fatalf("Device.Close: %v", err)
		}
		return identity
	}

	identity := compile(true, false)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read persisted cache: %v", err)
	}
	if !validPipelineCacheData(data, identity) {
		t.Fatal("first persisted cache has an invalid header")
	}
	warmIdentity := compile(false, true)
	if warmIdentity != identity {
		t.Fatalf("warm device identity changed: %#v to %#v", identity, warmIdentity)
	}
	if err := os.WriteFile(path, []byte{1, 2, 3}, 0o600); err != nil {
		t.Fatalf("corrupt cache: %v", err)
	}
	secondIdentity := compile(false, false)
	if secondIdentity != identity {
		t.Fatalf("device identity changed: %#v to %#v", identity, secondIdentity)
	}
	recovered, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read recovered cache: %v", err)
	}
	if !validPipelineCacheData(recovered, identity) {
		t.Fatal("corrupt cache was not replaced with healthy data")
	}
}

func testPipelineCacheData(identity pipelineCacheIdentity, payload []byte) []byte {
	data := make([]byte, pipelineCacheHeaderSize+len(payload))
	binary.LittleEndian.PutUint32(data[0:4], pipelineCacheHeaderSize)
	binary.LittleEndian.PutUint32(data[4:8], pipelineCacheHeaderVersionOne)
	binary.LittleEndian.PutUint32(data[8:12], identity.vendorID)
	binary.LittleEndian.PutUint32(data[12:16], identity.deviceID)
	copy(data[16:32], identity.uuid[:])
	copy(data[pipelineCacheHeaderSize:], payload)
	return data
}
