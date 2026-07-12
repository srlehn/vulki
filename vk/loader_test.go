package vk

import "testing"

func TestLoaderInvalidLookupReturnsZero(t *testing.T) {
	if got := (*Loader)(nil).GetInstanceProcAddr(0, "vkCreateInstance"); got != 0 {
		t.Fatalf("nil loader lookup = %d, want 0", got)
	}

	called := false
	loader := &Loader{getProc: func(Instance, *byte) uintptr {
		called = true
		return 1
	}}
	if got := loader.GetInstanceProcAddr(0, ""); got != 0 {
		t.Fatalf("empty name lookup = %d, want 0", got)
	}
	if got := loader.GetInstanceProcAddr(0, "vkBad\x00Name"); got != 0 {
		t.Fatalf("embedded-NUL lookup = %d, want 0", got)
	}
	if called {
		t.Fatal("invalid lookup called vkGetInstanceProcAddr")
	}
}

func TestNilLoaderCloseIsSafe(t *testing.T) {
	(*Loader)(nil).Close()
}
