// Package shader compiles WGSL compute shaders to SPIR-V for Vulki's direct
// Vulkan path and for callers using the low-level vk package.
//
// Compile uses validated SPIR-V 1.3 output by default. Functional options can
// select another supported version, include debug information, or explicitly
// disable validation without exposing the compiler implementation's types.
// Successful compilations are cached in-process, so repeatedly compiling the
// same source with the same options does not rerun the compiler.
package shader
