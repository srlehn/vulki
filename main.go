package main

import (
	"fmt"
	"runtime"
	"syscall"

	"github.com/ebitengine/purego"
)

type VkInstance uintptr
type VkResult int32

const VK_SUCCESS = 0

// vkGetInstanceProcAddr signature
type vkGetInstanceProcAddrFunc func(
	instance VkInstance,
	pName *byte,
) uintptr

// vkEnumerateInstanceVersion signature
type vkEnumerateInstanceVersionFunc func(
	pApiVersion *uint32,
) VkResult

func main() {
	libName := vulkanLibraryName()

	lib, err := purego.Dlopen(libName, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		panic(fmt.Errorf("failed to load Vulkan loader: %w", err))
	}
	defer purego.Dlclose(lib)

	// --- Load vkGetInstanceProcAddr ---
	getProcAddrPtr, err := purego.Dlsym(lib, "vkGetInstanceProcAddr")
	if err != nil {
		panic(fmt.Errorf("failed to load vkGetInstanceProcAddr: %w", err))
	}

	var getProc vkGetInstanceProcAddrFunc
	purego.RegisterFunc(&getProc, getProcAddrPtr)

	// --- Resolve vkEnumerateInstanceVersion ---
	name, _ := syscall.BytePtrFromString("vkEnumerateInstanceVersion")
	addr := getProc(0, name)
	if addr == 0 {
		panic("vkEnumerateInstanceVersion not found (requires Vulkan 1.1+)")
	}

	var enumerateVersion vkEnumerateInstanceVersionFunc
	purego.RegisterFunc(&enumerateVersion, addr)

	var version uint32
	res := enumerateVersion(&version)
	if res != VK_SUCCESS {
		panic(fmt.Errorf("vkEnumerateInstanceVersion failed: %d", res))
	}

	major := version >> 22
	minor := (version >> 12) & 0x3ff
	patch := version & 0xfff

	fmt.Printf("Vulkan API version: %d.%d.%d\n", major, minor, patch)
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
