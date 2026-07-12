package vk

import (
	"fmt"
	"runtime"
	"strconv"
	"syscall"

	"github.com/ebitengine/purego"
)

// Loader holds the Vulkan library handle and vkGetInstanceProcAddr.
type Loader struct {
	lib     uintptr
	getProc func(instance Instance, pName *byte) uintptr
}

// Open loads the Vulkan shared library and resolves vkGetInstanceProcAddr.
func Open() (*Loader, error) {
	if strconv.IntSize != 64 {
		return nil, fmt.Errorf("vk: only 64-bit targets are supported")
	}

	libName := vulkanLibraryName()
	lib, err := purego.Dlopen(libName, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, fmt.Errorf("vk: failed to load %s: %w", libName, err)
	}

	sym, err := purego.Dlsym(lib, "vkGetInstanceProcAddr")
	if err != nil {
		purego.Dlclose(lib)
		return nil, fmt.Errorf("vk: failed to find vkGetInstanceProcAddr: %w", err)
	}

	var getProc func(instance Instance, pName *byte) uintptr
	purego.RegisterFunc(&getProc, sym)

	return &Loader{lib: lib, getProc: getProc}, nil
}

// GetInstanceProcAddr resolves a Vulkan function by name.
func (l *Loader) GetInstanceProcAddr(instance Instance, name string) uintptr {
	cname, _ := syscall.BytePtrFromString(name)
	return l.getProc(instance, cname)
}

// Close releases the library handle.
func (l *Loader) Close() {
	if l.lib != 0 {
		purego.Dlclose(l.lib)
		l.lib = 0
	}
}

func vulkanLibraryName() string {
	switch runtime.GOOS {
	case "windows":
		return "vulkan-1.dll"
	case "darwin":
		return "libvulkan.1.dylib"
	default:
		return "libvulkan.so.1"
	}
}
