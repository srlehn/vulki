package vulki

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"

	"github.com/srlehn/vulki/vk"
)

const (
	pipelineCacheHeaderSize       = 32
	pipelineCacheHeaderVersionOne = 1
	maxPipelineCacheDataSize      = 64 << 20
	maxPipelineCacheDataAttempts  = 4
)

type pipelineCacheIdentity struct {
	vendorID uint32
	deviceID uint32
	uuid     [16]byte
}

type pipelineCacheConfig struct {
	enabled bool
	path    string
}

type pipelineCacheDriverOps struct {
	create  func(*vk.DeviceFuncs, vk.Device, []byte) (vk.PipelineCache, error)
	destroy func(*vk.DeviceFuncs, vk.Device, vk.PipelineCache)
	data    func(*vk.DeviceFuncs, vk.Device, vk.PipelineCache, []byte) (uintptr, error)
}

type pipelineCacheFileOps struct {
	read  func(string) ([]byte, error)
	write func(string, []byte) error
}

type pipelineCacheFactory func(
	*vk.DeviceFuncs,
	vk.Device,
	vk.PhysicalDeviceProperties,
) *pipelineCacheState

type pipelineCacheState struct {
	mu       sync.Mutex
	cache    vk.PipelineCache
	identity pipelineCacheIdentity
	path     string
	driver   pipelineCacheDriverOps
	files    pipelineCacheFileOps

	lastSize int
	lastHash [sha256.Size]byte
	hasLast  bool
}

var directPipelineCacheDriverOps = pipelineCacheDriverOps{
	create: func(functions *vk.DeviceFuncs, device vk.Device, initial []byte) (vk.PipelineCache, error) {
		info := vk.PipelineCacheCreateInfo{SType: vk.StructureTypePipelineCacheCreateInfo}
		if len(initial) > 0 {
			info.InitialDataSize = uintptr(len(initial))
			info.PInitialData = unsafe.Pointer(&initial[0])
		}
		cache, err := functions.CreatePipelineCache(device, &info)
		runtime.KeepAlive(initial)
		return cache, err
	},
	destroy: func(functions *vk.DeviceFuncs, device vk.Device, cache vk.PipelineCache) {
		functions.DestroyPipelineCache(device, cache)
	},
	data: func(functions *vk.DeviceFuncs, device vk.Device, cache vk.PipelineCache, data []byte) (uintptr, error) {
		return functions.GetPipelineCacheData(device, cache, data)
	},
}

var directPipelineCacheFileOps = pipelineCacheFileOps{
	read:  readPipelineCacheFile,
	write: writePipelineCacheFile,
}

func defaultPipelineCacheConfig(uuid [16]byte) pipelineCacheConfig {
	if strings.EqualFold(strings.TrimSpace(os.Getenv("VULKI_PIPELINE_CACHE")), "off") {
		return pipelineCacheConfig{}
	}
	if path := os.Getenv("VULKI_PIPELINE_CACHE_PATH"); path != "" {
		return pipelineCacheConfig{enabled: true, path: path}
	}
	directory, err := os.UserCacheDir()
	if err != nil {
		return pipelineCacheConfig{enabled: true}
	}
	filename := "pipeline-" + hex.EncodeToString(uuid[:]) + ".bin"
	return pipelineCacheConfig{
		enabled: true,
		path:    filepath.Join(directory, "vulki", filename),
	}
}

func pipelineCacheIdentityFor(properties vk.PhysicalDeviceProperties) pipelineCacheIdentity {
	return pipelineCacheIdentity{
		vendorID: properties.VendorID,
		deviceID: properties.DeviceID,
		uuid:     properties.PipelineCacheUUID,
	}
}

func openDefaultPipelineCache(
	functions *vk.DeviceFuncs,
	device vk.Device,
	properties vk.PhysicalDeviceProperties,
) *pipelineCacheState {
	return newPipelineCacheState(
		functions,
		device,
		pipelineCacheIdentityFor(properties),
		defaultPipelineCacheConfig(properties.PipelineCacheUUID),
		directPipelineCacheDriverOps,
		directPipelineCacheFileOps,
	)
}

func newPipelineCacheState(
	functions *vk.DeviceFuncs,
	device vk.Device,
	identity pipelineCacheIdentity,
	config pipelineCacheConfig,
	driver pipelineCacheDriverOps,
	files pipelineCacheFileOps,
) *pipelineCacheState {
	if !config.enabled || driver.create == nil {
		return nil
	}

	var initial []byte
	if config.path != "" && files.read != nil {
		data, err := files.read(config.path)
		if err == nil && validPipelineCacheData(data, identity) {
			initial = data
		}
	}

	cache, err := driver.create(functions, device, initial)
	loadedInitial := err == nil && cache != 0 && len(initial) > 0
	if err != nil || cache == 0 {
		if len(initial) == 0 {
			return nil
		}
		cache, err = driver.create(functions, device, nil)
		if err != nil || cache == 0 {
			return nil
		}
	}

	state := &pipelineCacheState{
		cache:    cache,
		identity: identity,
		path:     config.path,
		driver:   driver,
		files:    files,
	}
	if loadedInitial {
		state.remember(initial)
	}
	return state
}

func (state *pipelineCacheState) handle() vk.PipelineCache {
	if state == nil {
		return 0
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	return state.cache
}

func (state *pipelineCacheState) close(functions *vk.DeviceFuncs, device vk.Device) {
	if state == nil {
		return
	}
	state.mu.Lock()
	cache := state.cache
	state.cache = 0
	state.mu.Unlock()
	if cache != 0 && state.driver.destroy != nil {
		state.driver.destroy(functions, device, cache)
	}
}

func (state *pipelineCacheState) persist(functions *vk.DeviceFuncs, device vk.Device) {
	if state == nil {
		return
	}
	state.mu.Lock()
	defer state.mu.Unlock()
	if state.cache == 0 || state.path == "" || state.driver.data == nil || state.files.write == nil {
		return
	}

	// A driver pipeline cache only grows in practice, so an unchanged data
	// size after a warm pipeline creation makes retrieval, hashing, and
	// writing unnecessary. Keep the warm path at one driver size query.
	if state.hasLast {
		size, err := state.driver.data(functions, device, state.cache, nil)
		if err == nil && int(size) == state.lastSize {
			return
		}
	}

	data, err := retrievePipelineCacheData(functions, device, state.cache, state.driver.data)
	if err != nil || !validPipelineCacheData(data, state.identity) {
		return
	}
	digest := sha256.Sum256(data)
	if state.hasLast && state.lastSize == len(data) && state.lastHash == digest {
		return
	}
	if err := state.files.write(state.path, data); err != nil {
		return
	}
	state.lastSize = len(data)
	state.lastHash = digest
	state.hasLast = true
}

func (state *pipelineCacheState) remember(data []byte) {
	state.lastSize = len(data)
	state.lastHash = sha256.Sum256(data)
	state.hasLast = true
}

func retrievePipelineCacheData(
	functions *vk.DeviceFuncs,
	device vk.Device,
	cache vk.PipelineCache,
	get func(*vk.DeviceFuncs, vk.Device, vk.PipelineCache, []byte) (uintptr, error),
) ([]byte, error) {
	for range maxPipelineCacheDataAttempts {
		size, err := get(functions, device, cache, nil)
		if err != nil {
			return nil, err
		}
		if size > maxPipelineCacheDataSize {
			return nil, fmt.Errorf("vulki: pipeline cache data size %d exceeds limit", size)
		}
		data := make([]byte, int(size))
		written, err := get(functions, device, cache, data)
		if err != nil {
			var vkErr *vk.Error
			if errors.As(err, &vkErr) && vkErr.Result == vk.Incomplete {
				continue
			}
			return nil, err
		}
		if written > uintptr(len(data)) {
			return nil, fmt.Errorf("vulki: pipeline cache wrote %d bytes into %d-byte buffer", written, len(data))
		}
		return data[:int(written)], nil
	}
	return nil, fmt.Errorf("vulki: pipeline cache kept growing during retrieval")
}

func validPipelineCacheData(data []byte, identity pipelineCacheIdentity) bool {
	if len(data) < pipelineCacheHeaderSize || len(data) > maxPipelineCacheDataSize {
		return false
	}
	return binary.LittleEndian.Uint32(data[0:4]) == pipelineCacheHeaderSize &&
		binary.LittleEndian.Uint32(data[4:8]) == pipelineCacheHeaderVersionOne &&
		binary.LittleEndian.Uint32(data[8:12]) == identity.vendorID &&
		binary.LittleEndian.Uint32(data[12:16]) == identity.deviceID &&
		bytes.Equal(data[16:32], identity.uuid[:])
}

func readPipelineCacheFile(path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	info, statErr := file.Stat()
	if statErr != nil {
		_ = file.Close()
		return nil, statErr
	}
	if info.Size() > maxPipelineCacheDataSize {
		_ = file.Close()
		return nil, fmt.Errorf("vulki: pipeline cache file exceeds size limit")
	}
	data, readErr := io.ReadAll(io.LimitReader(file, maxPipelineCacheDataSize+1))
	closeErr := file.Close()
	if readErr != nil {
		return nil, readErr
	}
	if closeErr != nil {
		return nil, closeErr
	}
	if len(data) > maxPipelineCacheDataSize {
		return nil, fmt.Errorf("vulki: pipeline cache file exceeds size limit")
	}
	return data, nil
}

func writePipelineCacheFile(path string, data []byte) error {
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(directory, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	closed := false
	renamed := false
	defer func() {
		if !closed {
			_ = temporary.Close()
		}
		if !renamed {
			_ = os.Remove(temporaryPath)
		}
	}()

	written, err := temporary.Write(data)
	if err != nil {
		return err
	}
	if written != len(data) {
		return io.ErrShortWrite
	}
	if err := temporary.Sync(); err != nil {
		return err
	}
	err = temporary.Close()
	closed = true
	if err != nil {
		return err
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return err
	}
	renamed = true
	return nil
}
